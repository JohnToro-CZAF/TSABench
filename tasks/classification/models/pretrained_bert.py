import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import AutoModel

class PretrainedBERT(nn.Module):
    def __init__(self, tokenizer, num_labels: int = 2):
        super(PretrainedBERT, self).__init__()

        self.bert = AutoModel.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True)
        # freeze bert model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.num_layers = len(self.bert.encoder.layer)
        self.hidden_dimension = self.bert.encoder.layer[0].output.dense.weight.shape[0]
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])
    
    def forward(self, input):
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        
        input_ids = input_ids.to(torch.device("cuda"))
        attention_mask = attention_mask.to(torch.device("cuda"))
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        all_hidden_states  = outputs[2]
        all_hidden_states = torch.stack([tensor for tensor in all_hidden_states])

        batch_size = all_hidden_states[0].shape[0]
        first_token_positions = np.zeros(shape=(batch_size,))
        
        batch_range = np.arange(all_hidden_states.shape[1])
        new_hidden_states = all_hidden_states[:, batch_range, first_token_positions] # (num_layers, batch_size, hidden_dimension)

        all_output_logits = []
        for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers):
            logits = classification_layer(hidden_state)
            all_output_logits.append(logits)
        
        all_output_logits = torch.stack(all_output_logits, dim=0)
        return all_output_logits # (num_layers, batch_size, num_labels)
    
def loss_fn(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Arg:
        all_layer_logits <num_layers, batch_size, num_labels>: The predicted unnormalized logits of the model for all layers.
        labels <batch_size>: Ground truth class labels
    
    Returns:
        A tuple of summed_loss (1D tensor) and all_layer_loss (num_layers)
    """
    labels = labels.clone().detach().repeat(num_layers)
    labels = labels.reshape(-1)
    logits = all_layer_logits.reshape(-1, num_labels)
    all_layer_loss = F.cross_entropy(logits, labels.long(), reduction='none')
    summed_loss = all_layer_loss.mean()

    return summed_loss, all_layer_loss # (1D tensor), # (num_layers)

if __name__ == "__main__":
    model = PretrainedBERT()
    input_ids = torch.randint(0, 1000, (32, 128))
    attention_mask = torch.randint(0, 2, (32, 128))
    all_layer_logits = model(input_ids, attention_mask)
    labels = torch.randint(0, 2, (32,))
    loss = loss_fn(all_layer_logits, labels, model.num_layers, model.num_labels)
    preds = torch.argmax(all_layer_logits[-1], dim=-1)
    # print(loss)
    print(preds)