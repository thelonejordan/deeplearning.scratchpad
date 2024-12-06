# PYTHONPATH=. python3 models/mistral_nonrolling/main.py

# https://mistral.ai/news/announcing-mistral-7b/
# https://github.com/mistralai/mistral-inference/tree/v1.0.4

# NOTE: This implementation lacks sliding window attention & rolling KV cache, see cache_size

from typing import List

import torch

from models.helpers import timeit
from models.mistral_nonrolling.load import build
from models.mistral_nonrolling.generate import generate
from models.mistral_nonrolling.config import MistralConfig
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_nonrolling.transformer import Transformer


class Mistral:
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: MistralConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device
  @property
  def dtype(self) -> torch.dtype: return next(self.model.parameters()).dtype

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(folder: str, max_seq_len: int, max_batch_size: int, device: torch.device):
    model, tokenizer, config = build(folder, max_seq_len, max_batch_size)
    return Mistral(model, tokenizer, config).to(device)

  @torch.no_grad()
  def generate(self, prompts: List[str], max_tokens: int):
    return generate(self.model, self.tokenizer, self.device, prompts, max_tokens)


if __name__ == "__main__":
  from models.helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  model_path = "downloads/mistral-7B-v0.1"
  model = Mistral.from_pretrained(model_path, max_seq_len=36, max_batch_size=4, device=device)

  max_tokens: int = 36
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
