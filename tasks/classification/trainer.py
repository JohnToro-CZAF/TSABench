import os
import time
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
import metrics
from metrics import beautify

SUPPORTED_TASKS = ["classification", "causal", "multi_classification"]
class EarlyStopper:
    def __init__(self, patience=50, min_delta=0, greater_is_better=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.greater_is_better = greater_is_better
        if not greater_is_better:
            self.min_validation_loss = float('inf')
        else:
            self.min_validation_loss = 0.0

    def early_stop(self, validation_loss):
        if self.greater_is_better:
            if validation_loss > self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss < (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

class BaseLossFunction(nn.Module):
  def __init__(self):
    super(BaseLossFunction, self).__init__()

  def forward(self, input, output, label):
    raise NotImplementedError("forward method should be implemented")

class MultiClassificationLossFunction(BaseLossFunction):
  def __init__(self):
    super(ClassificationLossFunction, self).__init__()

  def forward(self, output, label):
    # input : (batch_size, seq_len), output : (batch_size, seq_len, num_classes), label : (batch_size)
    # get the (batch_size) tensor of positions that is different from padding token
    loss_fct = BCEWithLogitsLoss()
    loss = loss_fct(output, label)
    return loss


class ClassificationLossFunction(BaseLossFunction):
  def __init__(self):
    super(ClassificationLossFunction, self).__init__()

  def forward(self, output, label):
    # input : (batch_size, seq_len), output : (batch_size, seq_len, num_classes), label : (batch_size)
    # get the (batch_size) tensor of positions that is different from padding token
    return F.cross_entropy(output, label)

def get_loss_fn(task: str):
  if task == "classification":
    return ClassificationLossFunction()
  elif task == "multi_classification":
    return MultiClassificationLossFunction()
  else:
    raise NotImplementedError(f"Task {task} not implemented")

class TrainingArgs:
  def __init__(
      self, 
      task: str,
      learning_rate: float, 
      training_batch_size: int,
      validation_batch_size: int,
      save_model: bool = False,
      training_steps: int = None,
      metric_log_interval: int = None,
      eval_interval: int = None,
      epoch: int = None # ! DEFAULT USING TRAINING STEPS instead
    ):
    """ Training Arguments for the Trainer class

    Args:
        task (str): name of the task
        learning_rate (float): learning rate for the optimizer
        training_steps (int): number of training steps
        metric_log_interval (int): how many steps to wait before logging metrics
        training_batch_size (int): training batch size
        validation_batch_size (int): validation batch size
    """
    if epoch is not None:
      self.task = task
      self.epoch = epoch
      self.learning_rate = learning_rate
      self.training_batch_size = training_batch_size
      self.validation_batch_size = validation_batch_size
    else:
      assert task in SUPPORTED_TASKS, f"task should be one of {SUPPORTED_TASKS}"
      assert metric_log_interval <= training_steps, "metric_log_interval should be less than or equal to training"
      self.task = task
      self.learning_rate = learning_rate
      self.training_steps = training_steps
      self.eval_interval = eval_interval
      self.metric_log_interval = metric_log_interval
      self.training_batch_size = training_batch_size
      self.validation_batch_size = validation_batch_size
    
    self.save_model = save_model

class Trainer:
  def __init__(
      self, 
      model: nn.Module, 
      training_args: TrainingArgs, 
      train_loader: torch.utils.data.DataLoader,
      val_loader: torch.utils.data.DataLoader,
      test_loader: torch.utils.data.DataLoader,
      optimizer: torch.optim.Optimizer,
      metric_names: list[str],
      analysis_config: dict,
      early_stopper: EarlyStopper,
      model_type: str,
      aggregation: str,
    ):
    self.args = training_args
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.optimizer = optimizer
    self.loss_fn = get_loss_fn(self.args.task)
    self.metric_names = metric_names
    self.analysis_config = analysis_config
    self.early_stopper = early_stopper
    self.model_type = model_type
    self.metrics_log = {
        'train_loss': [],
        'train_metrics': {metric['name']: [] for metric in self.metric_names},
        'val_loss': [],
        'val_metrics': {metric['name']: [] for metric in self.metric_names},
        'test_loss': [],
        'test_metrics': {metric['name']: [] for metric in self.metric_names},
        'steps': [],
        'val_steps': [],  # Add this line
        'grad_norms': [] if self.analysis_config.get('record_gradients', False) else None,
        'grad_steps': [] if self.analysis_config.get('record_gradients', False) else None,
    }
    self.output_dir = self.analysis_config.get('output_dir', 'output/exp1')
    self.aggregation = aggregation
    os.makedirs(self.output_dir, exist_ok=True)
  
  def get_metrics_dict(self):
    return {metric["name"]: metrics.build(metric["name"], metric["args"]) for metric in self.metric_names}
  
  def test_step(self, input):
    with torch.no_grad():
      output = self.model(input) # logits only
    output = output.to("cpu")
    # outputs : (batch_size, num_classes)
    # result : (batch_size, num_classes)
    loss = self.loss_fn(output, input["label"])
    return output, loss.item()

  def test(self):
    test_loss = []
    test_metrics_dict = self.get_metrics_dict()
    for input in self.test_loader:
      output, loss = self.eval_step(input)
      test_loss.append(loss/input.size()[0])
      for metric_name, metric in test_metrics_dict.items():
        metric.update(output, input["label"])
    
    avg_test_loss = sum(test_loss) / len(test_loss)
    result_metrics = {
      metric_name: metric.value() for metric_name, metric in test_metrics_dict.items()
    }
    
    # ! For printing
    print(
      f"""Testing result:
        Test Loss: {avg_test_loss},
        Metrics: {beautify(result_metrics)}"""
    )
    
    # ! For logging analysis
    self.metrics_log['test_loss'].append(avg_test_loss)
    for metric_name, value in result_metrics.items():
        self.metrics_log['test_metrics'][metric_name].append(value)
      
  def eval_step(self, input):
    with torch.no_grad():
      output = self.model(input) # logits only
    output = output.to("cpu")
    # outputs : (batch_size, num_classes)
    # result : (batch_size, num_classes)
    loss = self.loss_fn(output, input["label"])
    return output, loss.item()

  def eval(self):
    val_loss = []
    eval_metrics_dict = self.get_metrics_dict()
    for input in self.val_loader:
      output, loss = self.eval_step(input)
      val_loss.append(loss/input.size()[0])
      for metric_name, metric in eval_metrics_dict.items():
        metric.update(output, input["label"])
    
    avg_val_loss = sum(val_loss) / len(val_loss)
    result_metrics = {
      metric_name: metric.value() for metric_name, metric in eval_metrics_dict.items()
    }
    
    # ! For printing
    print(
      f"""Validating result:
        Validation Loss: {avg_val_loss},
        Metrics: {beautify(result_metrics)}"""
    )
    
    # ! For logging analysis
    self.metrics_log['val_loss'].append(avg_val_loss)
    for metric_name, value in result_metrics.items():
        self.metrics_log['val_metrics'][metric_name].append(value)
      
    if self.early_stopper.early_stop(result_metrics['accuracy']):
      print('Early Stopping activated')
      return True

    return False
  
  def train_step(self, input):
    """
    Returns:
        loss: loss tensor float
        output: output tensor float [batch_size, num_classes]
    """
    self.optimizer.zero_grad()
    t0 = time.time()
    output = self.model(input) # output : (batch_size, num_classes), logits only
    print("Foward time: ", time.time() - t0)
    output = output.to("cpu")
    t0 = time.time()
    loss = self.loss_fn(output, input["label"])
    loss.backward()
    self.optimizer.step()
    print("Backward time: ", time.time() - t0)
    
    # Record gradients if required
    if self.analysis_config.get('record_gradients', False):
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        self.metrics_log.setdefault('grad_norms', []).append(grad_norm)
        self.metrics_log.setdefault('grad_steps', []).append(len(self.metrics_log['grad_norms']))
    return output, loss.item()

  def train(self):
    self.model.train()
    train_loss = 0
    data_metrics_dict = self.get_metrics_dict()
    print("Data Metrics: ", data_metrics_dict)
    data_iter = iter(self.train_loader)
    for step_id in tqdm.tqdm(range(self.args.training_steps)):
      try:
        input = next(data_iter)
      except StopIteration:
        # one epoch is done
        data_iter = iter(self.train_loader)
        input = next(data_iter)
      
      output, loss = self.train_step(input) # output : (batch_size, seq_len, num_classes)
      train_loss += loss
      
      # ! For logging analysis
      for metric_name, metric in data_metrics_dict.items():
          metric.update(output, input["label"])
          value = metric.value()
          self.metrics_log['train_metrics'][metric_name].append(value)
      self.metrics_log['steps'].append(step_id + 1)
      self.metrics_log['train_loss'].append(loss/input.size()[0])
      
      # ! For printing
      if (step_id + 1) % self.args.metric_log_interval == 0:
        result_metrics = {
          metric_name: metric.value() for metric_name, metric in data_metrics_dict.items()
        }
        print(
          f"""Step {step_id + 1}:
            Train Loss: {train_loss / ((step_id + 1) * self.args.training_batch_size) },
            Metrics:{beautify(result_metrics)}"""
        )
      
      if (step_id + 1) % self.args.eval_interval == 0:
        es = self.eval()
        self.metrics_log['val_steps'].append(step_id + 1)  # Record validation step
        self.save_metrics()
        if es:
          break
    # save model after training
    if self.args.save_model:
      torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
  
  def train_epoch(self): # epoch training instead
    self.model.train()
    data_metrics_dict = self.get_metrics_dict()
    print("Data Metrics: ", data_metrics_dict)
    progress_bar = tqdm.tqdm(range(self.args.epoch))
    for epoch_id in progress_bar:
      epoch_loss = 0
      data_metrics_dict = self.get_metrics_dict() # ! reset metrics for each epoch
      for input in tqdm.tqdm(self.train_loader):
        output, loss = self.train_step(input) # output : (batch_size, seq_len, num_classes)
        epoch_loss += loss/input.size()[0]
        progress_bar.set_postfix(loss=f"{loss/input.size()[0]:.4f}")
        for metric_name, metric in data_metrics_dict.items():
            metric.update(output, input["label"])
      
      # ! one epoch is done
      result_metrics = {
        metric_name: metric.value() for metric_name, metric in data_metrics_dict.items()
      }
      print(
        f"""Epoch {epoch_id}:
          Train Loss: {epoch_loss/len(self.train_loader)},
          Metrics:{beautify(result_metrics)}"""
      )
      # ! For logging analysis
      for metric_name, metric in data_metrics_dict.items():
          value = metric.value()
          self.metrics_log['train_metrics'][metric_name].append(value)
      self.metrics_log['steps'].append(epoch_id)
      self.metrics_log['train_loss'].append(epoch_loss/len(self.train_loader))

      es = self.eval()
      self.metrics_log['val_steps'].append(epoch_id)  # Record validation step
      self.save_metrics()
      if es:
        break
    if self.args.save_model:
      torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
    self.test()
    self.save_metrics()

  def save_metrics(self):
      # Save metrics to a JSON file
      metrics_file = os.path.join(self.output_dir, 'metrics.json')
      with open(metrics_file, 'w') as f:
          json.dump(self.metrics_log, f, indent=4)

      # Generate and save plots
      steps = self.metrics_log['steps']
      val_steps = self.metrics_log['val_steps']
      # Plot training loss
      plt.figure()
      plt.plot(steps, self.metrics_log['train_loss'], label='Training Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Training Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'training_loss.png'))
      plt.close()
      
      plt.figure()
      plt.plot(val_steps, self.metrics_log['val_loss'], label='Validation Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Validation Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'validation_loss.png'))
      plt.close()
      
      # Plot training and validation loss
      plt.figure()
      plt.plot(steps, self.metrics_log['train_loss'], label='Training Loss')
      plt.plot(val_steps, self.metrics_log['val_loss'], label='Validation Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Training and Validation Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'loss.png'))
      plt.close()
      
      # Plot metrics
      for metric_name in self.metrics_log['train_metrics']:
          plt.figure()
          plt.plot(steps, self.metrics_log['train_metrics'][metric_name], label=f'Train {metric_name}')
          plt.plot(val_steps, self.metrics_log['val_metrics'][metric_name], label=f'Validation {metric_name}')
          plt.xlabel('Steps')
          plt.ylabel(metric_name)
          plt.title(f'{metric_name.capitalize()} over Time')
          plt.legend()
          plt.savefig(os.path.join(self.output_dir, f'{metric_name}.png'))
          plt.close()

      # Save gradient norms if recorded
      if 'grad_norms' in self.metrics_log:
          plt.figure()
          plt.plot(self.metrics_log['grad_steps'], self.metrics_log['grad_norms'], label='Gradient Norm')
          plt.xlabel('Steps')
          plt.ylabel('Gradient Norm')
          plt.title('Gradient Norm over Time')
          plt.legend()
          plt.savefig(os.path.join(self.output_dir, 'gradient_norm.png'))
          plt.close()