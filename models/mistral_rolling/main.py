# PYTHONPATH=. python3 models/mistral_rolling/main.py

# https://github.com/mistralai/mistral-inference/blob/147c4e68279b90eb61b19bdea44e16f5539d5a5d/one_file_ref.py

from typing import Optional, List

import torch
from models.mistral_rolling.transformer import Transformer
from models.mistral_rolling.tokenizer import Tokenizer
from models.mistral_rolling.load import from_pretrained
from models.mistral_rolling.generate import generate

class Mistral:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device
  @property
  def dtype(self) -> torch.dtype: return next(self.model.parameters()).dtype

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(folder: str, max_batch_size: int=1, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
    model, tokenizer = from_pretrained(folder, max_batch_size, device, dtype)
    return Mistral(model, tokenizer)

  @torch.no_grad()
  def generate(self, prompts: List[str], max_tokens: int):
    return generate(self.model, self.tokenizer, self.device, prompts, max_tokens)


if __name__ == "__main__":
  from models.helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  model_path = "downloads/mistral-7B-v0.1"
  model = Mistral.from_pretrained(model_path, max_batch_size=3, device=device)

  max_tokens: int = 35
  context = [
    "Quantum computing is",
    "Simply put, the theory of relativity states that",
    "SpaceX and NASA have collaborated to make commercial",
  ]
  res, _logprobs = model.generate(context, max_tokens=max_tokens)
  print('-' * 50)
  for x in res:
    print(x)
    print('-' * 50)
