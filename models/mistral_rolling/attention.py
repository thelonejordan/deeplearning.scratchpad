from typing import Optional, Callable

import torch
from torch import Tensor, nn

from models.helpers import SDPA
from models.llama.attention import _attention, _fused_attention
from models.mistral_nonrolling.rope import apply_rotary_emb
from models.mistral_nonrolling.attention import repeat_kv


class Attention(nn.Module):
  _attn_fn: Callable[..., Tensor] = staticmethod(_fused_attention if bool(SDPA) else _attention)

  def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int, max_batch_size: int):
    super().__init__()
    self.n_heads, self.head_dim, self.n_kv_heads = n_heads, head_dim, n_kv_heads
    self.sliding_window = sliding_window
    self.repeats = self.n_heads // self.n_kv_heads
    self.scale = head_dim**-0.5

    self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    cache_size = (max_batch_size, sliding_window, self.n_kv_heads, head_dim)
    self.cache_k = torch.empty(cache_size, dtype=self.wq.weight.dtype)
    self.cache_v = torch.empty(cache_size, dtype=self.wq.weight.dtype)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    bsz, seqlen, _ = x.shape

    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # The cache is a rotating buffer
    scatter_pos: Tensor = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
    scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.head_dim)
    self.cache_k, self.cache_v = self.cache_k.to(xq.device), self.cache_v.to(xq.device)
    self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
    self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

    if positions.shape[0] > 1:
      key, value = repeat_kv(xk, xv, self.repeats)  # prefill
    else:
      cur_pos = positions[-1].item() + 1
      key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)  # type: ignore

    query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    output = self._attn_fn(query, key, value, mask, self.scale)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
