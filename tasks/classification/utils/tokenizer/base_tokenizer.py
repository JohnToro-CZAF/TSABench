import os
import json
from typing import Dict, Union, List
class BaseTokenizer:
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
      raise NotImplementedError
  
  def build_vocab(self):
      raise NotImplementedError

  def tokenize(self, text: str) -> Dict[str, Union[List[str], List[int]]]:
      raise NotImplementedError