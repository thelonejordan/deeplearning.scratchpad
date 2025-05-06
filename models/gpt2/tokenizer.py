# "bos_token": "<|endoftext|>"
# "eos_token": "<|endoftext|>"

import tiktoken

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.eot_token: str = self.model.eot_token
    self.n_words: int = self.model.n_vocab

  def encode_batch(self, input: list[str]) -> list[list[int]]:
    return self.model.encode_batch(input)

  def decode_batch(self, idx: list[list[int]]) -> list[str]:
    return self.model.decode_batch(idx)
