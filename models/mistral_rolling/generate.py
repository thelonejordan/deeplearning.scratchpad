from __future__ import annotations

import torch

from models.helpers import Generator, timeit, SAFETENSORS
from models.mistral_rolling.load import build
from models.mistral_nonrolling.config import MistralConfig
from models.mistral_rolling.transformer import Transformer
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_nonrolling.generate import generate


class Mistral(Transformer, Generator):
  def __init__(self, *args, **kwargs):
    self.tokenizer: Tokenizer = kwargs.pop("tokenizer")
    self.config: MistralConfig = kwargs.pop("config")
    super().__init__(*args, **kwargs)

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(version: str, max_seq_len: int, max_batch_size: int, device: torch.device) -> Mistral:
    generator, _, __ = build(max_seq_len, max_batch_size, version=version, safetensors=bool(SAFETENSORS), model_class=Mistral)
    return generator.to(device)

  @property
  def args(self):
    return self, self.tokenizer, self.tokenizer.pad_id

  @torch.no_grad()
  def generate(self, prompts: list[str], max_tokens: int):
    return generate(*self.args, prompts, max_tokens)
