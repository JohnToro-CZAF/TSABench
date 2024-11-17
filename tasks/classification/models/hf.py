from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoModelForSequenceClassification

class HF(nn.Module):
  def __init__(
      self, 
      arch_name: str,
      dim_output: int,
      tokenizer,
      is_frozen: bool,
      frozen_layers: List[int]
    ):
    super(HF, self).__init__()
    # Initialize the embedding using the factory function
    self.tokenizer = tokenizer
    self.arch = arch_name
    self.dim_output = dim_output
    self.model = AutoModelForSequenceClassification.from_pretrained(arch_name, num_labels=dim_output)
    # Freeze the layers if needed
    if is_frozen:
      for i, param in enumerate(self.model.parameters()):
        print(param)
  
  def forward(self, input):
    # input: [batch_size, seq_len]
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    output = self.model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    prob   = torch.softmax(logits, dim=-1)
    return prob