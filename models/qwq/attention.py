from typing import Optional, Callable
import math

import torch
from torch import Tensor, nn

from models.helpers import SDPA, set_device
from models.qwq.rope import apply_rotary_pos_emb
from models.llama.attention import _attention, _fused_attention


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  """
  This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = x.shape
  if n_rep == 1:
    return x
  x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def attention_forward(attn_fn: Callable[..., Tensor], query: Tensor, key: Tensor, value: Tensor,
                      n_rep: int, attention_mask: Optional[torch.Tensor], scaling: float) -> Tensor:
  key = repeat_kv(key, n_rep)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
  value = repeat_kv(value, n_rep)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

  output = attn_fn(query, key, value, attention_mask, scaling)
  output = output.transpose(1, 2).contiguous()
  return output


class Attention(nn.Module):
  _attn_fn: Callable[..., Tensor] = staticmethod(_fused_attention if bool(SDPA) else _attention)

  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
    self.n_rep  = n_heads // n_kv_heads
    self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    with torch.device(set_device(quiet=True)):
      self.cache_k = torch.zeros(max_batch_size, self.n_kv_heads, max_seq_len, self.head_dim)
      self.cache_v = torch.zeros(max_batch_size, self.n_kv_heads, max_seq_len, self.head_dim)

  def forward(self, x: Tensor, start_pos: int, position_embeddings: tuple[Tensor], mask: Optional[Tensor]):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    self.cache_k[:bsz, :, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, :, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, :, : start_pos + seqlen]
    values = self.cache_v[:bsz, :, : start_pos + seqlen]

    output = attention_forward(self._attn_fn, xq, keys, values, self.n_rep, mask, 1.0/math.sqrt(self.head_dim))

    output = output.view(bsz, seqlen, -1).contiguous()
    output = self.o_proj(output)
    return output
