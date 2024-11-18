from typing import Dict
from torch import nn
from models import (
    BiDeepRNN, 
    MultilayerLSTM,
    MultilayerBiLSTM,
    MultilayerGRU,
    MultilayerBiGRU,
    DeepGPT,
    DeepBERT,
    DeepT5,
    HF,
    PretrainedBERT,
    PretrainedGPT,
    PretrainedT5
)

MODULE_MAP = {
    "BiDeepRNN": BiDeepRNN,
    "LSTM": MultilayerLSTM,
    "BiLSTM": MultilayerBiLSTM,
    "GRU": MultilayerGRU,
    "BiGRU": MultilayerBiGRU,
    "GPT": DeepGPT,
    "BERT": DeepBERT,
    "T5": DeepT5,
    "HF": HF,
    "PretrainedBERT": PretrainedBERT,
    "PretrainedGPT": PretrainedGPT,
    "PretrainedT5": PretrainedT5
}

def build_model(config: Dict, tokenizer)-> nn.Module:
    if "model_type" not in config:
        raise Exception("model_type not found in config")
    model_type = config["model_type"]
    if model_type not in MODULE_MAP:
        raise Exception(f"model_type {model_type} not found in MODULE_MAP")
    return MODULE_MAP[model_type](**config["args"], tokenizer=tokenizer)