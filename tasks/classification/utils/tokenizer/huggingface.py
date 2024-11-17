import os
import json
import nltk
import argparse
from datasets import load_dataset
import nltk
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer

# def tokenize(dataset: Dataset | DatasetDict, tokenizer_name: str, input_col_name: str = "text"):
#     def _tokenize(examples):
#         return tokenizer(examples[input_col_name], padding='max_length', truncation=True, max_length=512)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenized_datasets = dataset.map(_tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"]).with_format("torch")
#     return tokenized_datasets

class HFTokenizer:
    def __init__(
        self,
        tokenizer_name: str
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pad_id = self.tokenizer._pad_token_type_id
        self.unk_id = self.tokenizer.unk_token_id
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)
    
    @classmethod
    def from_pretrained(cls, tokenizer_name):
        return cls(tokenizer_name)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        input = self.tokenizer(text)
        token_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        return {"tokens": tokens, "ids": token_ids, "attention_mask": attention_mask} # this have to align well with embeddings

    def save(self, folder_path):
        pass
    
if __name__ == "__main__":
    tokenizer = HFTokenizer("bert-base-uncased")
    print(tokenizer.tokenize("Hello, my name is John Doe."))