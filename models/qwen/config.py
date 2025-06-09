from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class QwenConfig:
  torch_dtype: str = "bfloat16"
  dim: int = 5120
  max_position_embeddings: int = 32768
  sliding_window: int = 32768
  max_window_layers: int = 64
  n_heads: int = 40
  n_kv_heads: int = 8
  n_layers: int = 64
  hidden_dim: int = 27648
  norm_eps: float = 1e-5
  rope_theta: float = 1000000
  vocab_size: int = 152064

  # from args
  max_seq_len: int = 2048
  max_batch_size: int = 32

  # from post init
  head_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.vocab_size > 0, self.vocab_size
    torch_dtype = getattr(torch, torch_dtype)

  @staticmethod
  def build(max_seq_len: int, max_batch_size: int, **params) -> QwenConfig:
    assert 1 <= max_seq_len <= 32768, f"max_seq_len must be between 1 and 32768, got {max_seq_len}."
    return QwenConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
