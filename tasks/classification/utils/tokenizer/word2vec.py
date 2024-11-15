import os
import json
import nltk
import argparse
from datasets import load_dataset
import nltk
from datasets import load_dataset
from collections import Counter

import gensim.downloader as api
from nltk.tokenize import word_tokenize

class NLTKTokenizer:
    def __init__(self, dataset="rotten_tomatoes", unk_id=None, pad_id=None):
        self.dataset = dataset
        self.pad_id = pad_id  # Store pad_id for future use
        self.unk_id = unk_id  # Store unk_id for future use
        self.vocab = {"<UNK>": self.unk_id, "<PAD>": self.pad_id}  # Initialize vocab with unk and pad tokens

    @classmethod
    def from_pretrained(cls, path):
        word2vec_model = api.load(path)
        w2v_vocab = word2vec_model.key_to_index
        # get the length:
        max_len = len(w2v_vocab)
        print("max_len", max_len)
        w2v_vocab["<UNK>"] = max_len
        w2v_vocab["<PAD>"] = max_len + 1
        
        """Load a tokenizer with a pre-built vocabulary from a saved file."""
        tokenizer = cls(dataset=None, unk_id=max_len, pad_id=max_len + 1)
        tokenizer.vocab = w2v_vocab
        return tokenizer

    def build_vocab(self):
        """Build vocabulary from the given dataset."""
        dataset = load_dataset(self.dataset)
        train_dataset = dataset['train']
        vocab = Counter(self.vocab)
        for item in train_dataset:
            tokens = word_tokenize(item['text'].lower())
            vocab.update(tokens)
        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), 1)}  # Index starts at 1

    def tokenize(self, text):
        """Tokenize a given text using NLTK."""
        tokens = word_tokenize(text.lower())  # Tokenize the text into words
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]  # Get token IDs

        return {"tokens": tokens, "ids": token_ids}

    def save(self, folder_path):
        """Save the vocabulary to a file."""
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "vocab.json"), "w") as f:
            json.dump(self.vocab, f)

class NLTKTokenizerPretrainedDataset:
    def __init__(
        self,
        vocab: dict[str, int],
        idx2word: dict[int, str],
        pad_id: int,
        unk_id: int
    ):
        """Initialize the tokenizer with a pre-built vocabulary.

        Args:
            vocab (dict[str, int]): word to index mapping, for tokenizing -> preembedding indices for pretrained dataset
        """
        self.vocab = vocab
        self._idx2word = idx2word
        self.unk_id = unk_id
        self.pad_id = pad_id
    
    @property
    def idx2word(self):
        return self._idx2word
    
    @property
    def word2ix(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_dataset(cls, dataset_name):
        """Build vocabulary from the given dataset."""
        dataset = load_dataset(dataset_name)
        train_dataset = dataset['train']
        vocab = Counter()
        for item in train_dataset:
            tokens = word_tokenize(item['text'].lower())
            vocab.update(tokens)
        
        vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), 0)}  # Index starts at 1
        original_vocab_size = len(vocab)
        vocab["<UNK>"] = original_vocab_size
        vocab["<PAD>"] = original_vocab_size + 1
        
        unk_id = vocab["<UNK>"]
        pad_id = vocab["<PAD>"]
        
        idx2word = {idx: word for word, idx in vocab.items()}
        return cls(vocab, idx2word, pad_id, unk_id)
    
    def tokenize(self, text):
        """Tokenize a given text using NLTK."""
        tokens = word_tokenize(text.lower())  # Tokenize the text into words
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]  # Get token IDs

        return {"tokens": tokens, "ids": token_ids} # this have to align well with embeddings

    def save(self, folder_path):
        """Save the vocabulary to a file."""
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "vocab.json"), "w") as f:
            json.dump(self.vocab, f)