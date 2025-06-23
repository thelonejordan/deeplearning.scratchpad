from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L161
def compute_hidden_dim(dim: int, multiple_of: int):
  hidden_dim = 4 * dim
  hidden_dim = int(2 * hidden_dim / 3)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

@dataclass
class LlamaConfig:
  # these changes based on model (get from params.json)
  dim: int
  n_heads: int
  n_layers: int
  norm_eps: float = 1e-5
  # these are constant across all models
  vocab_size: int = 32000
  multiple_of: int = 256
  max_position_embeddings: int = 2048
  rope_theta: float = 10000.0
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
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of)

  @staticmethod
  def build(max_seq_len: int=2048, max_batch_size: int=32, **params) -> LlamaConfig:
    return LlamaConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)


CONFIGS = {
  '7B': dict(dim=4096, n_heads=32, n_layers=32, norm_eps=1e-05), # 6.7B
  '13B': dict(dim=5120, n_heads=40, n_layers=40, norm_eps=1e-05), # 13.0B
  '33B': dict(dim=6656, n_heads=52, n_layers=60, norm_eps=1e-05), # 32.5B
  '70B': dict(dim=8192, n_heads=64, n_layers=80, norm_eps=1e-06), # 65.2B
}
