from .transformer import PositionalEncoding
from .transformer import FeedForward
from .preembeddings import build_preembedding
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff, dropout=0.1):
        super(BERTLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (seq_len, batch_size, dim_model)
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class DeepBERT(nn.Module):
    def __init__(
        self, dim_input, dim_model, num_layers, num_heads, dim_ff, tokenizer,
        embedding_strategy='random', embedding_frozen=True, max_len=2048, **kwargs
    ):
        super(DeepBERT, self).__init__()
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
            BERTLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)
        self.dim_model = dim_model

        # Classification Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.output_layer = nn.Linear(dim_model, kwargs.get('dim_output', 2))

    def forward(self, input):
        assert "input_ids" in input, "input_ids is required"
        if "attention_mask" not in input:
            print("Attention mask not found in input")
            input["attention_mask"] = torch.ones(input["input_ids"].size(0), input["input_ids"].size(1))
            # import ipdb; ipdb.set_trace()
        assert "attention_mask" in input, "attention_mask is required"
        # print(input["input_ids"].shape)
        # print(input["attention_mask"].shape)
        
        ids = input["input_ids"]
        ids = ids.to("cuda")
        device = ids.device
        attention_mask = input.get("attention_mask", None)

        embedded = self.token_embedding(ids)  # (batch_size, seq_len, dim_input)
        embedded = self.input_linear(embedded)
        embedded = self.positional_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, dim_model)

        # Add classification token
        cls_tokens = self.cls_token.expand(-1, ids.size(0), -1)  # (1, batch_size, dim_model)
        src = torch.cat((cls_tokens, embedded), dim=0)  # (seq_len+1, batch_size, dim_model)
        # import ipdb; ipdb.set_trace()
        attention_mask = torch.cat((torch.ones(ids.size(0), 1), attention_mask), dim=-1)
        attention_mask = attention_mask.to(device)

        for layer in self.layers:
            src = layer(src, src_mask=None, src_key_padding_mask=attention_mask)

        src = self.norm(src)
        output = src[0]  # Classification token output
        logits = self.output_layer(output)
        return logits
