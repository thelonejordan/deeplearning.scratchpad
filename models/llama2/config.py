from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import torch

# https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py#L331
def compute_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
  hidden_dim = int(2 * (4 * dim) / 3)
  if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim) # custom dim factor multiplier
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

@dataclass
class LlamaConfig:
  torch_dtype: torch.dtype = torch.float16
  # from params.json
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = -1  # 32000
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[int] = None
  norm_eps: float = 1e-5

  # from arguments
  max_seq_len: int = 2048
  max_batch_size: int = 32

  # for post init
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.hidden_dim is None
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of, self.ffn_dim_multiplier)
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> LlamaConfig:
    config = LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    assert config.vocab_size > 0, config.vocab_size
    return config
