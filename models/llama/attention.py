from typing import Optional, Callable
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.helpers import SDPA, set_device
from models.llama.rope import apply_rotary_emb_alt as apply_rotary_emb


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
  scores = torch.matmul(query, key.transpose(2, 3)) * scale
  if mask is not None:
    scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
  scores = F.softmax(scores.float(), dim=-1).type_as(query)
  output = torch.matmul(scores, value)  # (bs, n_local_heads, seqlen, head_dim)
  return output

def _fused_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
  # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)
  return output


class Attention(nn.Module):
  _attn_fn: Callable[..., Tensor] = staticmethod(_fused_attention if bool(SDPA) else _attention)

  def __init__(self, dim: int, n_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
    super().__init__()
    self.n_heads, self.head_dim = n_heads, head_dim
    self.q_proj = nn.Linear(dim, dim, bias=False)
    self.k_proj = nn.Linear(dim, dim, bias=False)
    self.v_proj = nn.Linear(dim, dim, bias=False)
    self.o_proj = nn.Linear(dim, dim, bias=False)

    with torch.device(set_device(quiet=True)):
      self.cache_k = torch.zeros(max_batch_size, max_seq_len, n_heads, head_dim)
      self.cache_v = torch.zeros(max_batch_size, max_seq_len, n_heads, head_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]=None):
    bsz, seqlen, _ = x.size()
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    output = self._attn_fn(xq, keys, values, mask, 1.0/math.sqrt(self.head_dim))

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output
