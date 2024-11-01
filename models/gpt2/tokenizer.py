from typing import List

import torch
from torch import Tensor
import tiktoken

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.eot_token = self.model.eot_token

  def encode_batch(self, input: List[str] | str, device: torch.device):
    batch = [input] if isinstance(input, str) else input
    return torch.tensor(self.model.encode_batch(batch), dtype=torch.long, device=device)

  def decode_batch(self, idx: List[List[int]]):
    return self.model.decode_batch(idx)
