from typing import Dict
from torch import nn
from models import (
    RNN, 
    DeepRNN,
    BiDeepRNN, 
    MultilayerRNN,
    MultilayerLSTM,
    MultilayerBiLSTM,
    MultilayerGRU,
    MultilayerBiGRU,
    CNN
)

MODULE_MAP = {
    "RNN": RNN,
    "DeepRNN": DeepRNN,
    "BiDeepRNN": BiDeepRNN,
    "MultilayerRNN": MultilayerRNN,
    "LSTM": MultilayerLSTM,
    "BiLSTM": MultilayerBiLSTM,
    "GRU": MultilayerGRU,
    "BiGRU": MultilayerBiGRU,
    "CNN": CNN
}

def build_model(config: Dict, tokenizer)-> nn.Module:
    if "model_type" not in config:
        raise Exception("model_type not found in config")
    model_type = config["model_type"]
    if model_type not in MODULE_MAP:
        raise Exception(f"model_type {model_type} not found in MODULE_MAP")
    return MODULE_MAP[model_type](**config["args"], tokenizer=tokenizer)