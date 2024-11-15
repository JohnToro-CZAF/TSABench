import torch
import torch.nn as nn
from torch.autograd import Variable

from models.preembeddings import build_preembedding

class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, direction=1):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_output)
    self.direction = direction

  def forward(self, input, hidden):
    outputs = []
    if self.direction == 1:
      for i in range(input.size()[1]):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    else: 
      for i in range(input.size()[1]-1, -1, -1):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    return torch.stack(outputs, dim=1) # (batch_size, seq_len, dim_output)

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class RNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, tokenizer, embedding_strategy='random', embedding_frozen=True, **kwargs):
    super(RNN, self).__init__()
    # Initialize the embedding using the factory function
    self.tokenizer = tokenizer
    self.embedding_strategy = embedding_strategy
    if embedding_strategy == "empty": # TODO: for baseline only
      self.token_embedding = nn.Embedding(tokenizer.get_vocab_size(), dim_input)
    else:
      self.token_embedding = build_preembedding(
          tokenizer=tokenizer,
          strategy=embedding_strategy,
          embedding_dim=dim_input,
          **kwargs
      )
    if embedding_frozen:
      try:
        self.token_embedding.weight.requires_grad = False
      except:
        self.token_embedding.embedding.weight.requires_grad = False
    # frozen embeddings option
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output

    self.rnn_layer = RNNLayer(dim_input, dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def initialize(self):
    for p in self.parameters():
      # not initializing the embedding layer
      if p is not self.token_embedding.weight:
        nn.init.xavier_uniform_(p)
  
  def forward(self, input):
    hidden = self.rnn_layer.init_hidden(input.size()[0])
    hidden = hidden.to(input.device)
    embedded = self.token_embedding(input)
    outputs = self.rnn_layer(embedded, hidden)
    outputs = self.softmax(outputs)
    return outputs

  def init_hidden(self, batch_size):
    return Variable(torch.zeros((batch_size, self.dim_hidden)))

class RNNSubLayer(nn.Module):
    def __init__(self, input_size, hidden_size, direction=1):
        super(RNNSubLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.direction = direction

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        seq_len = input_seq.size(1)
        batch_size = input_seq.size(0)
        outputs = []

        indices = range(seq_len) if self.direction == 1 else range(seq_len -1, -1, -1)
        for i in indices:
            combined = torch.cat((input_seq[:, i, :], hidden), dim=1)
            hidden = self.i2h(combined)
            outputs.append(hidden.unsqueeze(1))  # Collect hidden states

        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, seq_len, hidden_size]
        return outputs  # Return all hidden states and the last hidden state

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device)

class MultilayerRNN(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, tokenizer, num_layers=1, embedding_strategy='random', embedding_frozen=True, attention=False, **kwargs):
        super(MultilayerRNN, self).__init__()
        self.embedding_strategy = embedding_strategy
        self.tokenizer = tokenizer

        # Initialize the embedding using the factory function
        if embedding_strategy == "empty":  # For baseline only
            self.token_embedding = nn.Embedding(tokenizer.get_vocab_size(), dim_input)
        else:
            self.token_embedding = build_preembedding(
                tokenizer=tokenizer,
                strategy=embedding_strategy,
                embedding_dim=dim_input,
                **kwargs
            )
        if embedding_frozen:
            try:
              self.token_embedding.weight.requires_grad = False
            except:
              self.token_embedding.embedding.weight.requires_grad = False

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers
        self.attention = attention

        # Stack RNN layers
        self.rnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = dim_input if layer == 0 else dim_hidden
            self.rnn_layers.append(RNNSubLayer(input_size, dim_hidden))

        # attention layers
        if self.attention:
          self.attention_layer = nn.Linear(dim_hidden, 1, bias=False)
        
        # Output layer
        self.fc = nn.Linear(dim_hidden, dim_output)
        self.softmax = nn.LogSoftmax(dim=-1)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad and p is not self.token_embedding.weight:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        batch_size = input.size(0)
        device = input.device

        # Embedding layer
        embedded = self.token_embedding(input)  # Shape: [batch_size, seq_len, embedding_dim]

        # Initialize hidden states for all layers
        hidden_states = [layer.init_hidden(batch_size, device) for layer in self.rnn_layers]

        # Pass through stacked RNN layers
        input_seq = embedded
        for idx, layer in enumerate(self.rnn_layers):
            outputs = layer(input_seq, hidden_states[idx])
            input_seq = outputs  # Input to the next layer

        if self.attention:
          attn_weights = self.attention_layer(outputs)
          attn_weights = torch.tanh(attn_weights)
          attn_weights = nn.functional.softmax(attn_weights, dim=1)

          attn_output = torch.sum(attn_weights * outputs, dim=1)
          logits = self.fc(attn_output)
          preds = self.softmax(logits)
          return preds
        else:
        # Output layer
          logits = self.fc(outputs)
          preds = self.softmax(logits)
          return preds