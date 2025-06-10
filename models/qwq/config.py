from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

@dataclass
class QwQConfig:
  # these are constant across all models
  dim: int = 5120
  hidden_dim: int = 27648
  n_layers: int = 64
  n_heads: int = 40
  n_kv_heads: int = 8
  vocab_size: int = 152064
  max_position_embeddings: int = 32768
  rope_theta: float = 1000000
  norm_eps: float = 1e-5
  torch_dtype: str = "bfloat16"
  # TODO: are these required? maybe for Qwen2 models
  # sliding_window: int = 32768
  # max_window_layers: int = 64
  # these are populated through __post_init__
  head_dim: Optional[int] = None
  # these are populated from runtime args
  max_seq_len: Optional[int] = None
  max_batch_size: Optional[int] = None

  def __post_init__(self):
    assert self.vocab_size > 0, self.vocab_size
    assert self.max_seq_len is not None and self.max_batch_size is not None, (self.max_seq_len, self.max_batch_size)
    assert self.max_seq_len > 0 and self.max_seq_len <= self.max_position_embeddings, (
      f"max_seq_len must be between 1 and {self.max_position_embeddings}, got {self.max_seq_len}.")
    assert self.max_batch_size > 0, self.max_batch_size

    assert self.head_dim is None
    assert self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> QwQConfig:
    return QwQConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
