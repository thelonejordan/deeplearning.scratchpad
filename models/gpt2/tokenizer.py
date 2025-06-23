# "bos_token": "<|endoftext|>"
# "eos_token": "<|endoftext|>"

import tiktoken

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.bos_id: int = self.model.eot_token
    self.eos_id: int = self.model.eot_token
    self.pad_id: int = -1
    self.n_words: int = self.model.n_vocab

  def encode_batch(self, inputs: list[str]) -> list[list[int]]:
    return self.model.encode_batch(inputs)

  def decode_batch(self, idx: list[list[int]]) -> list[str]:
    return self.model.decode_batch(idx)
