from dataclasses import dataclass

@dataclass
class GPTConfig:
  vocab_size: int = 50257
  n_ctx: int = 1024
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  norm_eps: float = 1e-5

CONFIGS = {
  'gpt2':         dict(n_layer=24, n_head=16, n_embd=1024),  # 124M params
  'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
  'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
  'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}
