# "bos_token": "<|endoftext|>"
# "eos_token": "<|endoftext|>"

from typing import List
import tiktoken

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.eot_token = self.model.eot_token

  def encode_batch(self, input: List[str]) -> List[List[int]]:
    return self.model.encode_batch(input)

  def decode_batch(self, idx: List[List[int]]) -> List[str]:
    return self.model.decode_batch(idx)
