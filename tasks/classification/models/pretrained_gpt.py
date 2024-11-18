import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class PretrainedGPT(nn.Module):
    def __init__(self, tokenizer, num_labels: int = 2):
        super(PretrainedGPT, self).__init__()

        # Load GPT-2 model and tokenizer
        self.gpt = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt.config.pad_token_id = self.tokenizer.pad_token_id

        # Freeze GPT model parameters
        for param in self.gpt.parameters():
            param.requires_grad = False

        # Model configuration
        self.num_layers = self.gpt.config.n_layer
        self.hidden_dimension = self.gpt.config.hidden_size
        self.num_labels = num_labels

        # Classification layers for each GPT layer
        self.classification_layers = nn.ModuleList([
            nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)
        ])

    def forward(self, input):
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        input_ids = input_ids.to(torch.device("cuda"))

        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
        else:
            # Create attention mask if not provided
            attention_mask = (input_ids != self.gpt.config.pad_token_id).long().to(input_ids.device)

        # Get outputs from GPT model
        outputs = self.gpt(input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states  # Tuple of hidden states

        # Stack hidden states to tensor
        all_hidden_states = torch.stack(all_hidden_states)  # (num_layers+1, batch_size, seq_length, hidden_dim)

        # Exclude the embeddings layer
        all_hidden_states = all_hidden_states[1:]  # (num_layers, batch_size, seq_length, hidden_dim)

        batch_size, seq_length = input_ids.shape

        # Find positions of the last non-padded tokens
        last_token_positions = attention_mask.sum(dim=1) - 1  # (batch_size)
        last_token_positions = last_token_positions.to(torch.long)

        # Gather hidden states at the last token positions for each layer
        new_hidden_states = torch.stack([
            layer_hidden_states[torch.arange(batch_size), last_token_positions]
            for layer_hidden_states in all_hidden_states
        ])  # (num_layers, batch_size, hidden_dim)

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = PretrainedGPT()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Sample input sentences
    sentences = ["Hello, how are you?", "I am fine, thank you."]
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.gpt.device)
    attention_mask = inputs["attention_mask"].to(model.gpt.device)
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Forward pass
    all_layer_logits = model(input_data)

    # Generate random labels for demonstration
    labels = torch.randint(0, model.num_labels, (input_ids.shape[0],)).to(model.gpt.device)

    # Calculate loss
    loss, all_layer_loss = loss_fn(all_layer_logits, labels, model.num_layers, model.num_labels)

    # Predictions from the last layer
    preds = torch.argmax(all_layer_logits[-1], dim=-1)
    print("Predictions:", preds)
