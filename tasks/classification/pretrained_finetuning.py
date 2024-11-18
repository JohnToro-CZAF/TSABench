import os
import torch
import wandb
import argparse

from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from transformers import (
    Trainer, 
    TrainingArguments, 
    EvalPrediction,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset, Dataset, DatasetDict

os.environ["WANDB_ENTITY"] = "sc4001" 
os.environ["WANDB_PROJECT"] = "text-sentiment-analysis"

wandb.login()

def train_val_test_split(dataset: Dataset | DatasetDict, seed: int = 42):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if "validation" in dataset:
        val_dataset = dataset["validation"]
    else:
        train_dataset, val_dataset = train_dataset.train_test_split(test_size=0.3, seed=seed).values()
    
    return (train_dataset, val_dataset, test_dataset)

def tokenize(dataset: Dataset | DatasetDict, tokenizer_name: str, input_col_name: str = "text"):
    def _tokenize(examples):
        return tokenizer(examples[input_col_name], padding='max_length', truncation=True, max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(_tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"]).with_format("torch")
    return tokenized_datasets

def subset_dataset(dataset: Dataset | DatasetDict, 
                   size: int,
                   seed: int = 42):
    shuffled_dataset = dataset.shuffle(seed=seed)
    new_dataset = shuffled_dataset.select(range(size))
    return new_dataset


# default optimizer: AdamW
training_args = TrainingArguments(
    output_dir='./results', # output directory of results
    num_train_epochs=3, # number of train epochs
    report_to='wandb', # enable logging to W&B
    evaluation_strategy='steps', # check evaluation metrics at each epoch
    logging_steps = 10, # we will log every 10 steps
    eval_steps = 200, # we will perform evaluation every 200 steps
    save_steps = 200, # we will save the model every 200 steps
    save_total_limit = 5, # we only save the last 5 checkpoints (including the best one)
    load_best_model_at_end = True, # we will load the best model at the end of training
    metric_for_best_model = 'accuracy', # metric to see which model is better
    deepspeed='ds_config.json', # deep speed integration
    #### effective batch_size = per_device_train_batch_size x gradient_accumulation_steps ####
    #### We set effective batch_size to 32 (8 x 4) ####
    per_device_train_batch_size=int(8 / torch.cuda.device_count()), # batch size per device
    per_device_eval_batch_size=int(8 / torch.cuda.device_count()), # eval batch size per device
    gradient_accumulation_steps=4, # gradient accumulation
)


def compute_metrics(pred: EvalPrediction):
    # Extract labels and predictions
    labels = pred.label_ids
    preds = pred.predictions

    # for t5 model, the predictions is in the form of a tuple with the logits as the only element in the tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    num_classes = preds.shape[1]

    # Convert to torch tensors
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)

    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    precision = Precision(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    recall = Recall(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())

    # Calculate metrics (automatically does argmax)
    accuracy_score = accuracy(preds, labels)
    precision_score = precision(preds, labels)
    recall_score = recall(preds, labels)
    f1_score = f1(preds, labels)
    auroc_score = auroc(preds, labels)


    # Convert to CPU for serialization
    return {
        "accuracy": accuracy_score.cpu().item(),
        "precision": precision_score.cpu().item(),
        "recall": recall_score.cpu().item(),
        "f1": f1_score.cpu().item(),
        "auroc": auroc_score.cpu().item(),
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, run_name: str = None, trainer_args: TrainingArguments = None, **kwargs):
        if not trainer_args:
            # set default training arguments if not supplied
            trainer_args = training_args
        if run_name:
            trainer_args.run_name = run_name # specify the run name for wandb logging
        super().__init__(*args, compute_metrics=compute_metrics, args=trainer_args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the default compute_loss. 
        Use Cross Entropy Loss for multiclass classification (>= 2).
        """
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute cross entropy loss
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description='Small dataset experiments')
    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2', 'rotten_tomatoes'], default='imdb', help="Dataset to use")
    parser.add_argument("--model", choices=['bert', 'gpt2', 't5'], default='bert', help='Model to use')
    # we provide an option to subset the dataset to reduce training time
    parser.add_argument("--subset_yelp", type=bool, default=False, help='Whether to subset the dataset')
    # for deepspeed
    parser.add_argument("--local_rank")
    args = parser.parse_args()
    print(args)
    run_name = f"{args.model}-CompareTransformers-{args.dataset}"

    if args.subset_yelp:
        run_name += "_subset"

    # set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
        input_col_name = "text"
    elif args.dataset =='yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
        input_col_name = "text" 
    elif args.dataset == 'sst2':
        dataset = load_dataset("sst2")
        num_labels = 2
        input_col_name = "sentence"
    elif args.dataset == "rotten_tomatoes":
        dataset = load_dataset("rotten_tomatoes")
        num_labels = 2
        input_col_name = "text"
    else:
        raise NotImplementedError

    # set up model
    if args.model == 'bert':
        model_name = "bert-base"
    elif args.model == 'gpt2':
        model_name = "gpt2"
    elif args.model == 't5':
        model_name = "t5-base"
    else:
        raise NotImplementedError
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if model.config.pad_token_id == None:
        model.config.pad_token_id = model.config.eos_token_id

    tokenized_datasets = tokenize(dataset, model_name, input_col_name=input_col_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.dataset =='yelp' and args.subset_yelp == True:
        train_dataset = subset_dataset(train_dataset, size=25_000, seed=42)
        val_dataset = subset_dataset(val_dataset, size=25_000, seed=42)
        test_dataset = subset_dataset(test_dataset, size=25_000, seed=42)

    trainer = CustomTrainer(
        run_name=run_name,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()