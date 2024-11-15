import os
import json
import tqdm
import string
import argparse
from typing import List, Dict, Tuple, Union
from collections import defaultdict, Counter

from .base_tokenizer import BaseTokenizer

class BPETokenizer(BaseTokenizer):
  def __init__(
      self, 
      name,
      max_num_vocab: int, 
      word_2_edit: Dict[str, List[str]] = defaultdict(list),
      vocab_list: List[str] = ["<UNK>"]
    ): 
    """Initialize either from pretrained or from scratch"""
    self.name = name
    self.max_num_vocab = max_num_vocab
    self.word_2_edit = word_2_edit
    
    self.word_freq = Counter()
    self.vocab_list = vocab_list
    self.vocab = set(vocab_list)
    self.vocab_ids = {v: i for i, v in enumerate(self.vocab_list)}
    self.BPE = Counter()
    if "<pad>" in self.vocab:
      self.pad_id = self.vocab_ids["<pad>"]
    self.unk_id = self.vocab_ids["<UNK>"]
    self._idx2word = {idx: word for word, idx in self.vocab_ids.items()}
  
  def get_vocab_size(self):
    return len(self.vocab_ids)
  
  @property
  def word2idx(self):
    return self.vocab_ids
  
  @property
  def idx2word(self):
    return self._idx2word
  
  @classmethod
  def from_pretrained(cls, folder_path: str):
    if os.path.isdir(folder_path):
        config_file = folder_path + "/config.json"
        state_file = folder_path + "/state.json"
        if os.path.exists(config_file):
          config_dict = json.load(open(config_file, "r"))
          print("Loading tokenizer from cache: ", folder_path)
          print("Configuration: ", config_dict)
        else:
          raise Exception("Configuration file of tokenizer does not exist: ", config_file)
        if os.path.exists(state_file):
          state_dict = json.load(open(state_file, "r"))
          return cls(name=config_dict["name"], **state_dict)
        else:
          raise Exception("State dict of tokenizer does not exist: ", state_file)
    else:
      raise Exception("Folder path to tokenizer does not exist")
  
  def save(self, folder_path: str) -> str:
    state_dict = {}
    state_file = folder_path + "/state.json"
    config_file = folder_path + "/config.json"
    state_dict["max_num_vocab"] = self.max_num_vocab
    state_dict["word_2_edit"] = self.word_2_edit
    state_dict["vocab_list"] = self.vocab_list
    with open(state_file, "w") as f:
      json.dump(state_dict, f)
    with open(config_file, "w") as f:
      json.dump({
        "vocab_size": len(self.vocab), 
        "training_maximum_vocab_size": self.max_num_vocab,
        "name": self.name
      }, f)
    print("Tokenizer saved to: ", folder_path)
  
  def n_gram(self, tokens: List[str], n: int) -> List[str]:
    return [x for x in zip(tokens[:-(n-1)], tokens[(n-1):])]

  def clean_line(self, text: str) -> str:
    """
     input: text
     output: a cleans version, without multiple spaces
    """
    return text.lower().strip().translate(str.maketrans('','', string.punctuation)).replace('\n', '')

  def merge_edit(self, edit: List[str], target: str) -> List[str]:
    result = []
    last_merged = False
    idx = 0
    while idx < len(edit):
      if idx+1 < len(edit):
        if (edit[idx] + edit[idx+1]) == target:
          result.append(target)
          if idx+1 == len(edit)-1:
            last_merged = True
          idx += 2
        else:
          result.append(edit[idx])
          idx += 1
      elif not last_merged:
        result.append(edit[idx])
        idx += 1
    return result

  def add_line(self, line: str):
    line = self.clean_line(line)
    for word in line.split(" "):
      if not word in self.word_2_edit.keys():
        self.word_2_edit[word] = list(word)
      self.word_freq[word] += 1
    self.vocab.update(set(list(line))) # list of characters

  def build_vocab(self):
    print("--------------Building vocab--------------")
    for _ in tqdm.tqdm(range(self.max_num_vocab)):
      if len(self.vocab) > self.max_num_vocab:
        print("Vocab size is larger than max_num_vocab, stopping")
        break
      for word, edit in self.word_2_edit.items():
        grams = self.n_gram(tokens=edit, n=2)
        for gram in grams:
          self.BPE[gram[0] + gram[1]] += self.word_freq[word]
          self.BPE[gram[0]] -= self.word_freq[word]
          self.BPE[gram[1]] -= self.word_freq[word]
        
      next_merge = self.BPE.most_common(1)[0][0]
      if not next_merge:
        print("No more merge, stopping early, the size of vocab is: ", len(self.vocab))
        break
      for word, edit in self.word_2_edit.items():
        if next_merge in word:
          self.word_2_edit[word] = self.merge_edit(edit, target=next_merge)
      self.vocab.add(next_merge)
      # make sure this loop is not infinite
    
    if "<pad>" not in self.vocab:
      self.vocab.add("<pad>")
      self.word_2_edit["<pad>"] = ["<pad>"]
    
    self.vocab_list = list(self.vocab)
    for v in self.vocab_list:
      if v in self.vocab_ids:
        continue
      else:
        self.vocab_ids[v] = len(self.vocab_ids)
    self.pad_id = self.vocab_ids["<pad>"]

  def tokenize(self, text: str) -> Dict[str, Union[List[str], List[int]]]:
    tokenized_words = []
    tokenized_ids = []
    for word in text.split(" "):
      if word not in self.word_2_edit.keys():
        tokenized_words.extend(["<UNK>"]) # unknown token
        tokenized_ids.extend([self.vocab_ids["<UNK>"]])
      else:
        tokenized_words.extend(self.word_2_edit[word])
        tokenized_ids.extend([self.vocab_ids[x] for x in self.word_2_edit[word]])
    return {
      "tokens": tokenized_words,
      "ids": tokenized_ids
    }
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", type=str, required=True)
  parser.add_argument("--input", type=str, required=True)
  parser.add_argument("--max-num-vocab", type=int, required=True)
  args = parser.parse_args()
  # caching the tokenizer for later use
  cache_folder = "./cache/"
  if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
  # extract the name of the file
  corpus_name = args.corpus.split("/")[-1].split(".")[0]
  if os.path.isdir(cache_folder + corpus_name):
    # load the tokenizer from cache
    tokenizer = BPETokenizer.from_pretrained(cache_folder + corpus_name)
  else:
    # create a new tokenizer
    os.makedirs(cache_folder + corpus_name)
    print("Creating a new tokenizer")
    tokenizer = BPETokenizer(name=corpus_name, max_num_vocab=args.max_num_vocab)
    with open(args.corpus, "r") as f:
      lines = f.readlines()
    for line in (lines):
      tokenizer.add_line(line)
    tokenizer.build_vocab()
    tokenizer.save(cache_folder + corpus_name)
    # save config to cache
  print(tokenizer.tokenize(args.input.lower()))
  print(tokenizer.pad_id)