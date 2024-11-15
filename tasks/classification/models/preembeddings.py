# models/preembedding.py
import numpy as np
import fasttext
from gensim.models import Word2Vec
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreEmbedding(nn.Module):
    def __init__(self, tokenizer, embedding_dim):
        super(PreEmbedding, self).__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        raise NotImplementedError("Each embedding strategy must implement the forward method.")

class RandomInitEmbedding(PreEmbedding):
    def __init__(self, tokenizer, embedding_dim):
        super(RandomInitEmbedding, self).__init__(tokenizer, embedding_dim)
        vocab_size = tokenizer.get_vocab_size()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)

class Word2VecEmbedding(nn.Module):
    """
    Train domain-specific embeddings and use them.
    """
    def __init__(self, tokenizer, embedding_dim, **kwargs):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.pad_id
        self.unk_id = tokenizer.unk_id
        assert "pretrained_path" in kwargs, "Please specify the path to the pretrained word2vec model"
        pretrained_path = kwargs["pretrained_path"]
        assert "oov_handing" in kwargs, "Please specify the OOV handling strategy"
        oov_handling = kwargs["oov_handing"]
        super(Word2VecEmbedding, self).__init__()
        word2vec_model = api.load(pretrained_path)
        w2v_embeddings = word2vec_model.vectors 
        w2v_vocab = word2vec_model.key_to_index
        
        assert embedding_dim == w2v_embeddings.shape[1], f"Input dim from model and word2vec embeddings mismatch: {embedding_dim} != {w2v_embeddings.shape[1]}"
        
        self.oov_handling = oov_handling
        print("Pretrained W2V embedding size: ", w2v_embeddings.shape)
        print("Pretrained W2V vocab size: ", len(w2v_vocab))
        
        # pretrained_embedding = np.append(pretrained_embedding, [np.random.rand(*pretrained_embedding[-1].shape)], axis=0) # unk_id
        # pretrained_embedding = np.append(pretrained_embedding, [np.zeros_like(pretrained_embedding[-1])], axis=0) # pad_id
        pretrained_embedding = np.zeros((self.vocab_size, embedding_dim))
        n_oov = 0
        # ! Start to fill-in the pretrained embedding
        for i in range(self.vocab_size):
            token = tokenizer.idx2word[i]
            if token in w2v_vocab:
                pretrained_embedding[i] = w2v_embeddings[w2v_vocab[token]]
            else:
                if i == self.unk_id:
                    pretrained_embedding[i] = np.random.rand(embedding_dim)
                elif i == self.pad_id:
                    pretrained_embedding[i] = np.zeros(embedding_dim)
                else:
                    # if the token is not in pretrained embedding
                    n_oov += 1
                    if oov_handling in ["using_unk", "average", "average_context"]:
                        # using unk during training phase and testing
                        pretrained_embedding[i] = pretrained_embedding[self.unk_id]
                    else:
                        # if not using unk, we can use the most similar token
                        # TODO: this isn;t finished
                        pretrained_embedding[i] = word2vec_model.most_similar(token)[0][1] # get the most similar token
        print(f"OOV tokens of tokenzier to pretrained {pretrained_path}: {n_oov}/{self.vocab_size}")
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float),
        )
        
        if oov_handling == "average_context":
            self.context_window = kwargs.get('context_window', 2)
            # warning if context window is not defined
            if 'context_window' not in kwargs:
                print("Warning: Context window not defined in oov handling: context_average, using default value of 2")
        
        assert embedding_dim == pretrained_embedding.shape[1], f"Embedding dim mismatch: {embedding_dim} != {pretrained_embedding.shape[1]}"

    def forward(self, input_ids):
        # Handling UNK token id and pad_id
        embeddings = self.embedding(input_ids)
        if self.oov_handling == "using_unk":
            return embeddings
        elif self.oov_handling == "average":
            mask = (input_ids != self.unk_id) & (input_ids != self.pad_id)
            mask = mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(embeddings * mask, dim=1)
            count_non_pad = torch.sum(mask, dim=1)
            avg_embeddings = sum_embeddings / (count_non_pad + 1e-8)
            avg_embeddings = avg_embeddings.unsqueeze(1).expand(-1, input_ids.size(1), -1)
            embeddings = embeddings * mask + avg_embeddings * (1-mask)
            return embeddings

        elif self.oov_handling == "average_context":
            assert hasattr(self, 'context_window'), "Context window not defined in oov handling: average_context"
            # Replace UNK embeddings with average of fixed context window
            batch_size, seq_len, embedding_dim = embeddings.size()
            # Prepare padding for context window at sequence boundaries
            padded_embeddings = F.pad(embeddings, (0, 0, self.context_window, self.context_window), mode='constant', value=0)
            padded_input_ids = F.pad(input_ids, (self.context_window, self.context_window), value=self.pad_id)

            # Iterate over each position in the sequence
            for i in range(seq_len):
                # Positions of UNK tokens at current index
                unk_positions = (input_ids[:, i] == self.unk_id)  # Shape: (batch_size)
                if unk_positions.any():
                    # Define context window indices
                    context_start = i
                    context_end = i + 2 * self.context_window + 1  # +1 because slicing is exclusive at the end
                    # Extract context embeddings and input IDs
                    context_embeddings = padded_embeddings[:, context_start:context_end, :]  # Shape: (batch_size, window_size, embedding_dim)
                    context_input_ids = padded_input_ids[:, context_start:context_end]  # Shape: (batch_size, window_size)
                    # Create mask for valid tokens in context
                    context_mask = (context_input_ids != self.unk_id) & (context_input_ids != self.pad_id)
                    context_mask = context_mask.unsqueeze(-1).float()  # Shape: (batch_size, window_size, 1)
                    # Compute average context embeddings
                    sum_context_embeddings = torch.sum(context_embeddings * context_mask, dim=1)  # Shape: (batch_size, embedding_dim)
                    count_context = torch.sum(context_mask, dim=1)  # Shape: (batch_size, 1)
                    avg_context_embeddings = sum_context_embeddings / (count_context + 1e-8)  # Shape: (batch_size, embedding_dim)
                    # Replace UNK embeddings with average context embeddings
                    embeddings[unk_positions, i, :] = avg_context_embeddings[unk_positions, :]
            return embeddings
        else:
            raise ValueError(f"Unknown OOV handling strategy: {self.oov_handling}")
    

def build_preembedding(strategy, embedding_dim, tokenizer, **kwargs):
    if strategy == 'random':
        return RandomInitEmbedding(tokenizer, embedding_dim)
    elif strategy == 'word2vec':
        return Word2VecEmbedding(tokenizer, embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
