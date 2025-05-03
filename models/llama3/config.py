from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from models.llama2.config import compute_hidden_dim

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
  vocab_size: int = 128256
  max_position_embeddings: int = 131072
  original_max_position_embeddings: int = 8192
  rope_theta: float = 500000.0
  norm_eps: float = 1e-5
  torch_dtype: str = "bfloat16"
  # these are populated through __post_init__
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None
  # these are populated from runtime args
  max_seq_len: Optional[int] = None
  max_batch_size: Optional[int] = None

  def __post_init__(self):
    assert self.vocab_size > 0, self.vocab_size
    assert self.max_seq_len is not None and self.max_batch_size is not None, (self.max_seq_len, self.max_batch_size)
    # TODO: when to use max_position_embeddings?
    assert self.max_seq_len > 0 and self.max_seq_len <= self.original_max_position_embeddings, self.max_seq_len
    assert self.max_batch_size > 0, self.max_batch_size

    assert self.head_dim is None and self.hidden_dim is None
    assert self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of, self.ffn_dim_multiplier)
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

  @staticmethod
  def build(max_seq_len: int, max_batch_size: int, **params) -> LlamaConfig:
    return LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
