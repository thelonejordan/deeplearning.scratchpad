from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

# https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py#L331
def compute_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
  hidden_dim = int(2 * (4 * dim) / 3)
  if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim) # custom dim factor multiplier
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

@dataclass
class LlamaConfig:
  # these changes based on model (get from params.json)
  dim: int
  n_layers: int
  n_heads: int
  n_kv_heads: Optional[int] = None
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier: Optional[float] = None
  # these are constant across all models
  vocab_size: int = 32000
  max_position_embeddings: int = 4096
  rope_theta: float = 10000.0
  norm_eps: float = 1e-5
  torch_dtype: str = "float16"
  # these are populated through __post_init__
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None
  # these are populated from runtime args
  max_seq_len: Optional[int] = None
  max_batch_size: Optional[int] = None

  def __post_init__(self):
    assert self.vocab_size > 0, self.vocab_size
    assert self.max_seq_len is not None and self.max_batch_size is not None, (self.max_seq_len, self.max_batch_size)
    assert self.max_seq_len > 0 and self.max_seq_len <= self.max_position_embeddings, self.max_seq_len
    assert self.max_batch_size > 0, self.max_batch_size

    assert self.head_dim is None and self.hidden_dim is None
    assert self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of, self.ffn_dim_multiplier)
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> LlamaConfig:
    return LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)

CONFIGS = {
  '7B' : dict(dim=4096, n_heads=32, n_layers=32),
  '13B': dict(dim=5120, n_heads=40, n_layers=40),
  '70B': dict(dim=8192, n_heads=64, n_layers=80, n_kv_heads=8, multiple_of=4096, ffn_dim_multiplier=1.3),
}
