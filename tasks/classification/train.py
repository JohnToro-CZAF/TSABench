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

def checking_config(config):
  # check if the preembedding strategy is compatible with the tokenizer or not
  # if config["tokenizer_config"]["tokenizer_type"] == "bpe":
  #   if config["model_config"]["args"]["embedding_strategy"] not in ["random", "empty"]:
  #     raise ValueError("BPE tokenizer only supports random or empty,  embedding strategy, got: ", config["model_config"]["args"]["embedding_strategy"])
  
  # if config["tokenizer_config"]["tokenizer_type"] == "word2vec":
  #   if config["model_config"]["args"]["embedding_strategy"] not in ["word2vec"]:
  #     raise ValueError("Word2vec tokenizer only supports word2vec embedding strategy, got: ", config["model_config"]["args"]["embedding_strategy"])
  
  # if config["model_config"]["args"]["embedding_strategy"] == "word2vec":
  #   if config["model_config"]["args"]["pretrained_path"] is None:
  #     raise ValueError("Please specify the path to the pretrained word2vec model")
  #   if config["tokenizer_config"]["tokenizer_type"] != "word2vec":
  #     raise ValueError("Word2Vec embedding strategy is not compatible with tokenizer, got: ", config["tokenizer_config"]["tokenizer_type"])
    
  # if config["model_config"]["args"]["embedding_strategy"] in ["word2vec"]:
  #   # have to specify the input_dim
  #   if "dim_input" not in config["model_config"]["args"]:
  #     raise ValueError("Please specify dim_input for word2vec or glove embedding strategy")

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
  save_model = config['trainer_args'].get('save_model', False)
  tokenizer = build_tokenizer(config["tokenizer_config"])
  train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer=tokenizer, 
    dataset_args=config["data_config"], 
    training_args=training_args
  )
  
  model_type = config['model_config']['model_type']
  model = build_model(config["model_config"], tokenizer)
  model.to("cuda")
  optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
  
  exp_dir = config["analysis_config"].get('output_dir', 'output/exp1')
  if os.path.exists(exp_dir):
    raise ValueError("Experiment directory already exists, please delete it or use a new directory")
  os.makedirs(exp_dir, exist_ok=True)
  # save entire config file for reproducibility
  with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

  early_stopper = EarlyStopper(patience=10,
                               min_delta=0,
                               greater_is_better=True)
  
  trainer = Trainer(
    model=model,
    training_args=training_args, 
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    metric_names=config["metric_config"]["metrics"],
    analysis_config=config["analysis_config"],
    early_stopper=early_stopper,
    model_type=model_type,
    aggregation=aggregation,
  )
  if hasattr(training_args, "epoch") or training_args.epoch is not None:
    trainer.train_epoch()
  else:
    trainer.train()

if __name__ == "__main__":
  main()
