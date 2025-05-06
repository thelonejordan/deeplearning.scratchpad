from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

@dataclass
class MistralConfig:
  vocab_size: int

  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  hidden_dim: int = 14336
  norm_eps: float = 1e-05
  max_position_embeddings: int = 32768
  rope_theta: float = 10000.0
  torch_dtype: str = "bfloat16"

  n_kv_heads: Optional[int] = None
  head_dim: Optional[int] = None
  sliding_window: Optional[int] = None

  max_seq_len: Optional[int] = None
  max_batch_size: Optional[int] = None

  def __post_init__(self):
    assert self.vocab_size > 0, self.vocab_size
    assert self.max_seq_len is not None and self.max_batch_size is not None, (self.max_seq_len, self.max_batch_size)
    assert self.max_seq_len > 0 and self.max_seq_len <= self.max_position_embeddings, self.max_seq_len
    assert self.max_batch_size > 0, self.max_batch_size

    if self.head_dim is None:
      assert self.dim % self.n_heads == 0
      self.head_dim = self.dim // self.n_heads
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

  @staticmethod
  def build(max_seq_len: int, max_batch_size: int, **params) -> MistralConfig:
    return MistralConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)


CONFIGS = {
  "v0.1": {
    "7B": dict(n_kv_heads=8, sliding_window=4096, vocab_size=32000)
  },
  "v0.3": {
    "7B": dict(n_kv_heads=8, vocab_size=32768, rope_theta=1000000.0)
  },
}
