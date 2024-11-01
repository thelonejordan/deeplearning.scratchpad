from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import torch

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L161
def compute_hidden_dim(dim: int, multiple_of: int):
  hidden_dim = 4 * dim
  hidden_dim = int(2 * hidden_dim / 3)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

@dataclass
class LlamaConfig:
  torch_dtype: torch.dtype = torch.float16
  # from params.json
  n_heads: int = 32
  n_layers: int = 32
  dim: int = 4096
  vocab_size: int = 32000
  norm_eps: float = 1e-5
  multiple_of: int = 256
  # from args
  max_seq_len: int = 2048
  max_batch_size: int = 32
  # for post init
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.hidden_dim is None
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of)
    assert self.vocab_size > 0, self.vocab_size

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> LlamaConfig:
    return LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
