import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from .preembeddings import build_preembedding


class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden, direction=1):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_hidden)
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
    return torch.stack(outputs, dim=1) # (batch_size, seq_len, dim_hidden)

  def init_hidden(self, batch_size, device):
    return torch.zeros(batch_size, self.dim_hidden).to(device)

class DeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, num_layers, tokenizer, direction=1, embedding_strategy='random', embedding_frozen=True, **kwargs):
    super(DeepRNN, self).__init__()
    self.tokenizer = tokenizer
    
    self.embedding_strategy = embedding_strategy
    if embedding_strategy == "empty": # TODO: for baseline only
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
      except:
        self.token_embedding.embedding.weight.requires_grad = False
   
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.num_layers = num_layers
    self.direction = direction
    
    self.input_layer = RNNLayer(dim_input, dim_hidden, direction=direction)
    self.rnn_layers = nn.ModuleList([RNNLayer(dim_hidden, dim_hidden, direction=direction) for _ in range(num_layers)])
  
  def forward(self, input):
    ids = input["input_ids"]
    ids = ids.to("cuda")
    device = ids.device
    hidden = self.input_layer.init_hidden(ids.size()[0], device=device)
    embedded = self.token_embedding(ids)

    outputs = self.input_layer(embedded, hidden)
    for i in range(self.num_layers):
      hidden = self.rnn_layers[i].init_hidden(ids.size()[0], device=device)
      outputs = self.rnn_layers[i](outputs, hidden) # [bs, seq_len, dim_hidden]
    return outputs

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class BiDeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, num_layers, tokenizer, embedding_strategy='random', embedding_frozen=True, attention=False,**kwargs):
    super(BiDeepRNN, self).__init__()

    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.num_layers = num_layers
    self.attention = attention
    if self.attention:
      self.attention_layer = nn.Linear(2*dim_hidden, 1, bias=False)


    self.rnn_layers_forward = DeepRNN(dim_input, dim_hidden, num_layers, tokenizer, direction=1, embedding_strategy=embedding_strategy, embedding_frozen=embedding_frozen, **kwargs)
    self.rnn_layers_backward = DeepRNN(dim_input, dim_hidden, num_layers, tokenizer, direction=-1, embedding_strategy=embedding_strategy, embedding_frozen=embedding_frozen, **kwargs)
    self.output_layer = nn.Linear(2*dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input):
    lengths = input["lengths"]
    # Since we are dealing with Char-RNN task so we dont need to use all the output
    fs = self.rnn_layers_forward(input)
    bs = self.rnn_layers_backward(input)
    outputs = torch.cat((fs, torch.flip(bs, dims=(1,))), dim=-1)    
    # outputs : (batch_size, seq_len, 2*dim_hidden)
    if self.attention:
      attn_weights = self.attention_layer(outputs)
      attn_weights = torch.tanh(attn_weights)
      attn_weights = nn.functional.softmax(attn_weights, dim=1)

      attn_output = torch.sum(attn_weights * outputs, dim=1)
      logits = self.output_layer(attn_output)
      assert logits.size() == (input["input_ids"].size()[0], input["input_ids"].size()[1], self.dim_output)
      output = logits[range(logits.size()[0]), lengths-1, :]
      assert output.size() == (input["input_ids"].size()[0], self.dim_output)
      return output
    else:
      logits = self.output_layer(outputs)
      assert logits.size() == (input["input_ids"].size()[0], input["input_ids"].size()[1], self.dim_output)
      output = logits[range(outputs.size()[0]), lengths-1, :]
      assert output.size() == (input["input_ids"].size()[0], self.dim_output)
      return output