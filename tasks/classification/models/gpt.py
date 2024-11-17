from .transformer import PositionalEncoding
from .transformer import FeedForward
from .preembeddings import build_preembedding

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff, dropout=0.1):
        super(GPTLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch_size, dim_model)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class DeepGPT(nn.Module):
    def __init__(
        self, dim_input, dim_model, num_layers, num_heads, dim_ff, tokenizer,
        embedding_strategy='random', embedding_frozen=True, max_len=512, **kwargs
    ):
        super(DeepGPT, self).__init__()
        self.tokenizer = tokenizer

        # Embedding Layer
        if embedding_strategy == "empty":
            self.token_embedding = nn.Embedding(tokenizer.get_vocab_size(), dim_input)
        else:
            self.token_embedding = build_preembedding(
                strategy=embedding_strategy,
                tokenizer=tokenizer,
                embedding_dim=dim_input,
                **kwargs
            )
        if embedding_frozen:
            try:
                self.token_embedding.weight.requires_grad = False
            except AttributeError:
                self.token_embedding.embedding.weight.requires_grad = False

        self.input_linear = nn.Linear(dim_input, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model, max_len=max_len)
        self.layers = nn.ModuleList([
            GPTLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)
        self.dim_model = dim_model

        # Classification Head
        self.output_layer = nn.Linear(dim_model, kwargs.get('dim_output', 2))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        ids = input["input_ids"]
        device = ids.device

        seq_len = ids.size(1)
        embedded = self.token_embedding(ids)  # (batch_size, seq_len, dim_input)
        embedded = self.input_linear(embedded)
        embedded = self.positional_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, dim_model)

        # Generate causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))

        output = embedded
        for layer in self.layers:
            output = layer(output, src_mask=causal_mask)

        output = self.norm(output)
        output = output.transpose(0, 1)  # (batch_size, seq_len, dim_model)
        output = output[:, -1, :]  # Use last token representation
        logits = self.output_layer(output)
        return self.softmax(logits)
