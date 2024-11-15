import torch
import torch.nn as nn
from .preembeddings import build_preembedding

class CNN(nn.Module):
    def __init__(self, dim_input, dim_output, tokenizer, filter_sizes, num_filters, dropout=0.5, embedding_strategy='random', embedding_frozen=True, **kwargs):
        super(CNN, self).__init__()
        self.embedding_strategy = embedding_strategy

        # Initialize the embedding layer
        if embedding_strategy == "empty":  # For baseline only
            self.token_embedding = nn.Embedding(tokenizer.get_vocab_size(), dim_input)
        else:
            self.token_embedding = build_preembedding(
                tokenizer=tokenizer,
                strategy=embedding_strategy,
                embedding_dim=dim_input,
                **kwargs
            )
        
        # Embedding layer: Initialize with pretrained embeddings or randomly if not provided
        if embedding_frozen:
            try:
                self.token_embedding.weight.requires_grad = False
            except:
                self.token_embedding.embedding.weight.requires_grad = False
        
        # Convolution layers with multiple filter sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, dim_input)) for fs in filter_sizes
        ])
        
        # Fully connected output layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, dim_output)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Embed the input word indices
        x = self.token_embedding(x)  # Shape: (batch_size, sentence_length, embedding_dim)
        x = x.unsqueeze(1)     # Add channel dimension for CNN, Shape: (batch_size, 1, sentence_length, embedding_dim)

        # Apply convolution + ReLU + max-pooling for each filter size
        conv_outputs = []
        for conv in self.convs:
            conv_out = nn.functional.relu(conv(x)).squeeze(3)        # Shape: (batch_size, num_filters, sentence_length - filter_size + 1)
            pool_out = nn.functional.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # Shape: (batch_size, num_filters)
            conv_outputs.append(pool_out)
        
        # Concatenate outputs of all filter sizes
        x = torch.cat(conv_outputs, 1)  # Shape: (batch_size, num_filters * len(filter_sizes))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Fully connected layer for classification
        logits = self.fc(x)
        output = self.softmax(logits)
        return output