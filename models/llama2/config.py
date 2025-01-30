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
  vocab_size: int = -1  # (32000) defined later by tokenizer
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
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
    assert self.vocab_size > 0, self.vocab_size

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> LlamaConfig:
    return LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)

CONFIGS = {
  '7B' : dict(dim=4096, n_heads=32, n_layers=32, multiple_of=256, norm_eps=1e-05),
  '13B': dict(dim=5120, n_heads=40, n_layers=40, multiple_of=256, norm_eps=1e-05),
  '70B': dict(dim=8192, n_heads=64, n_kv_heads=8, n_layers=80, multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-05),
}
