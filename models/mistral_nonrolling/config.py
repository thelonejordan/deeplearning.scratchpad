from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class MistralConfig:
  dim: int
  n_layers: int
  head_dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  sliding_window: int
  norm_eps: float
  vocab_size: int

  max_seq_len: int
  max_batch_size: int

  max_pos_embd: int = 32768
  rope_theta: float = 10000.0
  torch_dtype: torch.dtype = torch.bfloat16

  def __post_init__(self):
    assert self.vocab_size > 0

  @staticmethod
  def build(max_seq_len: int, max_batch_size: int, **params):
    config = MistralConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    return config