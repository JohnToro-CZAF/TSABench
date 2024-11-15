import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model
from utils.tokenizer import build_tokenizer
from trainer import Trainer, TrainingArgs, EarlyStopper
from dataloader import get_dataloaders
import metrics
from metrics import beautify

def get_metrics_dict(metric_names):
    return {metric["name"]: metrics.build(metric["name"], metric["args"]) for metric in metric_names}

def checking_config(config):
  # check if the preembedding strategy is compatible with the tokenizer or not
  if config["tokenizer_config"]["tokenizer_type"] == "bpe":
    if config["model_config"]["args"]["embedding_strategy"] not in ["random", "empty"]:
      raise ValueError("BPE tokenizer only supports random or empty,  embedding strategy, got: ", config["model_config"]["args"]["embedding_strategy"])
  
  if config["tokenizer_config"]["tokenizer_type"] == "word2vec":
    if config["model_config"]["args"]["embedding_strategy"] not in ["word2vec"]:
      raise ValueError("Word2vec tokenizer only supports word2vec embedding strategy, got: ", config["model_config"]["args"]["embedding_strategy"])
  
  if config["model_config"]["args"]["embedding_strategy"] == "word2vec":
    if config["model_config"]["args"]["pretrained_path"] is None:
      raise ValueError("Please specify the path to the pretrained word2vec model")
    if config["tokenizer_config"]["tokenizer_type"] != "word2vec":
      raise ValueError("Word2Vec embedding strategy is not compatible with tokenizer, got: ", config["tokenizer_config"]["tokenizer_type"])
    
  if config["model_config"]["args"]["embedding_strategy"] in ["word2vec"]:
    # have to specify the input_dim
    if "dim_input" not in config["model_config"]["args"]:
      raise ValueError("Please specify dim_input for word2vec or glove embedding strategy")
  if config['model_config']['model_type']=='CNN':
    if 'aggregation' in config['model_config']['args']:
      raise ValueError("Aggregation method is only available for sequential models (RNN, LSTM, GRU)")
def main():
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--config", type=str, required=True)
  args = argparser.parse_args()
  print("Config file: ", args.config)
  config = json.load(open(args.config))
  print(config)
  checking_config(config)
  
  training_args = TrainingArgs(
    **config["trainer_args"]
  )

  aggregation = config['model_config']['args'].get('aggregation', 'last')
  if aggregation=='attention':
    config['model_config']['args']['attention'] = True
  model_type = config['model_config']['model_type']
  model = build_model(config["model_config"])
  model_path = os.path.join(config['analysis_config']['output_dir'], 'model.pth')
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to("cuda")
    model.eval()
  else:
    raise ValueError('Model file does not exist. Please perform the training step first')
  
  tokenizer = build_tokenizer(config["tokenizer_config"])
  train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer=tokenizer, 
    dataset_args=config["data_config"], 
    training_args=training_args
  )


  metric_names = config['metric_config']['metrics']
  metric_dict = get_metrics_dict(metric_names)
  for input, length, label in test_loader:
    input = input.to("cuda")
    with torch.no_grad():
      output = model(input)
    output = output.to("cpu")
    # outputs : (batch_size, seq_len, num_classes)
    # result : (batch_size, num_classes)
    if model_type!='CNN':
      if aggregation=='last':
        output = output[range(input.size()[0]), length - 1]
      elif aggregation=='mean':
        output = torch.mean(output, axis=1)
      elif aggregation=='max':
        output = torch.max(output, axis=1).values
    for metric_name, metric in metric_dict.items():
      metric.update(output, label)
  result_metrics = {
    metric_name: metric.value() for metric_name, metric in metric_dict.items()
    }
  print(
    f"""Test result:
        Metrics: {beautify(result_metrics)}"""
    )

  result_file = os.path.join(config['analysis_config']['output_dir'], 'test_metric.json')
  with open(result_file, 'w') as f:
    json.dump(result_metrics, f, indent=4)


if __name__ == "__main__":
  main()