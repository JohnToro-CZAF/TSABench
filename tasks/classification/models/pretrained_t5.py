import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5EncoderModel

class PretrainedT5(nn.Module):
    def __init__(self, tokenizer, num_labels: int = 2):
        super(PretrainedT5, self).__init__()

        # Load T5 encoder model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-base", output_hidden_states=True)

        # Freeze T5 model parameters
        for param in self.t5_encoder.parameters():
            param.requires_grad = False

        # Model configuration
        self.num_layers = self.t5_encoder.config.num_layers
        self.hidden_dimension = self.t5_encoder.config.d_model
        self.num_labels = num_labels

        # Classification layers for each T5 encoder layer
        self.classification_layers = nn.ModuleList([
            nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)
        ])

    def forward(self, input):
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]

        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        # Get outputs from T5 encoder
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states  # Tuple of hidden states

        # Stack hidden states to tensor
        all_hidden_states = torch.stack(all_hidden_states)  # (num_layers+1, batch_size, seq_length, hidden_dim)

        # Exclude the embeddings layer to match the number of transformer layers
        all_hidden_states = all_hidden_states[1:]  # (num_layers, batch_size, seq_length, hidden_dim)

        # Pooling: Mean over the sequence length dimension
        new_hidden_states = all_hidden_states.mean(dim=2)  # (num_layers, batch_size, hidden_dim)

        # Apply classification layers
        all_output_logits = torch.stack([
            classification_layer(hidden_state)
            for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers)
        ], dim=0)  # (num_layers, batch_size, num_labels)

        return all_output_logits  # (num_layers, batch_size, num_labels)

def loss_fn(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Args:
        all_layer_logits (torch.Tensor): The predicted unnormalized logits for all layers (num_layers, batch_size, num_labels).
        labels (torch.Tensor): Ground truth class labels (batch_size).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the summed loss (scalar tensor) and all layer losses (num_layers).
    """
    labels = labels.clone().detach().repeat(num_layers)  # Repeat labels for each layer
    labels = labels.reshape(-1)
    logits = all_layer_logits.reshape(-1, num_labels)
    all_layer_loss = F.cross_entropy(logits, labels.long(), reduction='none')
    summed_loss = all_layer_loss.mean()
    return summed_loss, all_layer_loss  # (scalar tensor), (num_layers)

if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = PretrainedT5()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sample input sentences
    sentences = ["The quick brown fox jumps over the lazy dog.", "An apple a day keeps the doctor away."]
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Forward pass
    all_layer_logits = model(input_data)

    # Generate random labels for demonstration
    labels = torch.randint(0, model.num_labels, (input_ids.shape[0],)).to(device)

    # Calculate loss
    loss, all_layer_loss = loss_fn(all_layer_logits, labels, model.num_layers, model.num_labels)

    # Predictions from the last layer
    preds = torch.argmax(all_layer_logits[-1], dim=-1)
    print("Predictions:", preds)
