from typing import Optional, Callable
import math

import torch
from torch import Tensor, nn

from models.helpers import SDPA, set_device
from models.llama.rope import apply_rotary_emb
from models.llama.attention import _attention, _fused_attention

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
  bs, slen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
    x[:, :, :, None, :]
    .expand(bs, slen, n_kv_heads, n_rep, head_dim)
    .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
  )

# n_rep > 1 aka n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
def repeat_kv_alt(x: Tensor, n_rep: int) -> Tensor:
  if n_rep == 1: return x
  bs, seqlen, n_kv_heads, head_dim = x.shape
  x = x.unsqueeze(-2).expand(-1, -1, -1, n_rep, -1)
  return x.reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
  _attn_fn: Callable[..., Tensor] = staticmethod(_fused_attention if bool(SDPA) else _attention)

  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
    self.n_rep  = n_heads // n_kv_heads
    self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    with torch.device(set_device(quiet=True)):
      self.cache_k = torch.zeros(max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
      self.cache_v = torch.zeros(max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    output = self._attn_fn(xq, keys, values, mask, 1.0/math.sqrt(self.head_dim))

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output
