import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .preembeddings import build_preembedding

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x