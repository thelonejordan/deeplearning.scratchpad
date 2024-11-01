# PYTHONPATH=. python3 models/mistral_nonrolling/main.py

# https://mistral.ai/news/announcing-mistral-7b/
# https://github.com/mistralai/mistral-inference/tree/v1.0.4

# NOTE: This implementation lacks sliding window attention & rolling KV cache

from typing import  Optional, List

import torch
from models.mistral_rolling.tokenizer import Tokenizer
from models.mistral_nonrolling.transformer import Transformer
from models.mistral_nonrolling.load import from_pretrained
from models.mistral_nonrolling.generate import generate

class Mistral:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.config = model.config

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
