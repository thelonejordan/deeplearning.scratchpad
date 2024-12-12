from dataclasses import dataclass

@dataclass
class GPTConfig:
  vocab_size: int = 50257
  n_ctx: int = 1024
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  norm_eps: float = 1e-5
