import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import LlamaModel

class SlicedLlama(nn.Module):
    def __init__(self, num_labels: int = 2):
        super(SlicedLlama, self).__init__()

        self.llama = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", output_hidden_states=True)
        # freeze llama model parameters
        for param in self.llama.parameters():
            param.requires_grad = False

        self.num_layers = len(self.llama.layers)
        self.hidden_dimension = self.llama.layers[0].mlp.gate_proj.weight.shape[1]
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])
    
    def forward(self, input_ids, attention_mask: np.ndarray = None):
        outputs = self.llama(input_ids, attention_mask=attention_mask)
        all_hidden_states  = outputs[2] # shape: (num_layers, batch_size, sequence_length, hidden_dimension)
        all_hidden_states = torch.stack([tensor for tensor in all_hidden_states])
        if attention_mask is not None: # attention_mask shape: (batch_size, sequence_length)
            last_token_positions = attention_mask.argmin(-1) - 1 # (batch_size)
        else:
            non_padded_tokens = np.equal(input_ids, self.llama.config.pad_token_id) # (batch_size, sequence_length)
            last_token_positions = non_padded_tokens.argmin(-1) - 1 # (batch_size)
        
        batch_range = np.arange(all_hidden_states.shape[1])
        new_hidden_states = all_hidden_states[:, batch_range, last_token_positions] # (num_layers, batch_size, hidden_dimension)
        all_output_logits = []
        for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers):
            logits = classification_layer(hidden_state) # (batch_size, num_labels)
            all_output_logits.append(logits) # (num_layers, batch_size, num_labels)
        
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0)
        return all_output_logits_tensor # (num_layers, batch_size, num_labels)
    
def loss_fn(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Arg:
        all_layer_logits <num_layers, batch_size, num_labels>: The predicted unnormalized logits of the model for all layers.
        labels <batch_size>: Ground truth class labels
    
    Returns:
        A tuple of summed_loss (1D tensor) and all_layer_loss (num_layers)
    """
    labels = labels.clone().detach().repeat(num_layers) # (batch_size) -> (num_layers, batch_size)
    labels = labels.reshape(-1) # (num_layers, batch_size) -> 1D tensor
    logits = all_layer_logits.reshape(-1, num_labels) # (num_layers, batch_size, num_labels) -> (num_labels)
    all_layer_loss = F.cross_entropy(logits, labels.long(), reduction='none') # (num_layers)
    summed_loss = all_layer_loss.mean() # (num_layers) -> (1D tensor)
    return summed_loss, all_layer_loss # (1D tensor), # (num_layers)