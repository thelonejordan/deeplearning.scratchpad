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
  norm_eps: float
  vocab_size: int
  sliding_window: int

  # For rotary embeddings. If not set, will be inferred
  rope_theta: float = 1000000.0

  torch_dtype: torch.dtype = torch.bfloat16

  # from args
  max_seq_len: int = 16384
  max_batch_size: int = 0
