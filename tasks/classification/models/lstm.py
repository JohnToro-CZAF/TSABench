import torch
import torch.nn as nn
from .preembeddings import build_preembedding
class LSTMSubLayer(nn.Module):
    def __init__(self, input_size, hidden_size, direction=1):
        super(LSTMSubLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.direction = direction

        # LSTM gates
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_seq, hidden_state):
        seq_len = input_seq.size(1)
        batch_size = input_seq.size(0)
        H, C = hidden_state  # Hidden state and cell state

        outputs = []

        indices = range(seq_len) if self.direction == 1 else range(seq_len - 1, -1, -1)
        for i in indices:
            combined = torch.cat((input_seq[:, i, :], H), dim=1)
            I = self.sigmoid(self.input_gate(combined))
            F = self.sigmoid(self.forget_gate(combined))
            G = self.tanh(self.cell_gate(combined))
            O = self.sigmoid(self.output_gate(combined))

            C = F * C + I * G
            H = O * self.tanh(C)
            outputs.append(H.unsqueeze(1))  # Shape: [batch_size, 1, hidden_size]

        if self.direction == -1:
            outputs.reverse()  # Reverse outputs if processing backward

        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, seq_len, hidden_size]
        return outputs, (H, C)

    def init_hidden(self, batch_size, device):
        H = torch.zeros(batch_size, self.hidden_size).to(device)
        C = torch.zeros(batch_size, self.hidden_size).to(device)
        return (H, C)
class MultilayerLSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, tokenizer, num_layers=1, embedding_strategy='random', embedding_frozen=True, attention=False, **kwargs):
        super(MultilayerLSTM, self).__init__()
        self.embedding_strategy = embedding_strategy
        self.tokenizer = tokenizer

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
        if embedding_frozen:
            try:
                self.token_embedding.weight.requires_grad = False
            except:
                self.token_embedding.embedding.weight.requires_grad = False

        self.attention = attention
        if self.attention:
            self.attention_layer = nn.Linear(2*dim_hidden, 1, bias=False)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        # Stack LSTM layers
        self.lstm_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = dim_input if layer == 0 else dim_hidden
            self.lstm_layers.append(LSTMSubLayer(input_size, dim_hidden))

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

        # Initialize hidden and cell states for all layers
        hidden_states = [layer.init_hidden(batch_size, device) for layer in self.lstm_layers]

        # Pass through stacked LSTM layers
        input_seq = embedded
        for idx, layer in enumerate(self.lstm_layers):
            outputs, hidden_states[idx] = layer(input_seq, hidden_states[idx])
            input_seq = outputs  # Input to the next layer

        # Apply the output layer to the last layer's outputs
        if self.attention:
            attn_weights = self.attention_layer(outputs)
            attn_weights = torch.tanh(attn_weights)
            attn_weights = nn.functional.softmax(attn_weights, dim=1)

            attn_output = torch.sum(attn_weights * outputs, dim=1)
            logits = self.fc(attn_output)
            preds = self.softmax(logits)
            return preds
        else:
            outputs = self.fc(outputs)
            outputs = self.softmax(outputs)
            return outputs

class BiLSTMSubLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMSubLayer, self).__init__()
        self.forward_layer = LSTMSubLayer(input_size, hidden_size, direction=1)
        self.backward_layer = LSTMSubLayer(input_size, hidden_size, direction=-1)

    def forward(self, input_seq, hidden_state):
        batch_size = input_seq.size(0)
        device = input_seq.device

        # Forward direction
        H_fwd, C_fwd = hidden_state[0]
        outputs_fwd, (H_fwd, C_fwd) = self.forward_layer(input_seq, (H_fwd, C_fwd))

        # Backward direction
        H_bwd, C_bwd = hidden_state[1]
        outputs_bwd, (H_bwd, C_bwd) = self.backward_layer(input_seq, (H_bwd, C_bwd))

        # Concatenate the outputs
        outputs = torch.cat((outputs_fwd, outputs_bwd), dim=2)  # Shape: [batch_size, seq_len, 2*hidden_size]

        # Update hidden states
        hidden_state = ((H_fwd, C_fwd), (H_bwd, C_bwd))

        return outputs, hidden_state

    def init_hidden(self, batch_size, device):
        H_fwd, C_fwd = self.forward_layer.init_hidden(batch_size, device)
        H_bwd, C_bwd = self.backward_layer.init_hidden(batch_size, device)
        return ((H_fwd, C_fwd), (H_bwd, C_bwd))

class MultilayerBiLSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, tokenizer, num_layers=1, embedding_strategy='random', embedding_frozen=True, attention=False,**kwargs):
        super(MultilayerBiLSTM, self).__init__()
        self.embedding_strategy = embedding_strategy
        self.tokenizer = tokenizer

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
        if self.attention:
            self.attention_layer = nn.Linear(2*dim_hidden, 1, bias=False)

        # Stack BiLSTM layers
        self.lstm_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = dim_input if layer == 0 else dim_hidden * 2
            self.lstm_layers.append(BiLSTMSubLayer(input_size, dim_hidden))

        # Output layer
        self.fc = nn.Linear(dim_hidden * 2, dim_output)
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
        hidden_states = [layer.init_hidden(batch_size, device) for layer in self.lstm_layers]

        # Pass through stacked BiLSTM layers
        input_seq = embedded
        for idx, layer in enumerate(self.lstm_layers):
            outputs, hidden_states[idx] = layer(input_seq, hidden_states[idx])
            input_seq = outputs  # Input to the next layer

        # Apply the output layer to the last layer's outputs
        if self.attention:
            attn_weights = self.attention_layer(outputs)
            attn_weights = torch.tanh(attn_weights)
            attn_weights = nn.functional.softmax(attn_weights, dim=1)

            attn_output = torch.sum(attn_weights * outputs, dim=1)
            logits = self.fc(attn_output)
            preds = self.softmax(logits)
            return preds
        else:
            outputs = self.fc(outputs)
            outputs = self.softmax(outputs)
            return outputs