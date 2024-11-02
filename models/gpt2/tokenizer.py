from typing import List
import tiktoken

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.eot_token = self.model.eot_token

  def encode_batch(self, input: str | List[str]) -> List[List[int]]:
    batch = [input] if isinstance(input, str) else input
    return self.model.encode_batch(batch)

  def decode_batch(self, idx: List[List[int]]) -> List[str]:
    return self.model.decode_batch(idx)
