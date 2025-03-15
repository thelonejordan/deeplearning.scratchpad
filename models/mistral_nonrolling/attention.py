from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.llama.attention import _fused_attention
from models.mistral_nonrolling.rope import apply_rotary_emb


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int=2) -> Tuple[Tensor, Tensor]:
  keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
  values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
  return keys, values


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float) -> Tensor:
  scores = torch.matmul(query, key.transpose(2, 3)) * scale # (bsz, n_heads, seqlen | 1, seqlen)
  if mask is not None: scores += mask[None, None, ...]
  scores = scores.float()
  scores = F.softmax(scores, dim=-1).type_as(query)
  output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
  return output


class Attention(nn.Module):
  def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int, max_batch_size: int):
    super().__init__()
    self.head_dim, self.n_heads, self.n_kv_heads = head_dim, n_heads, n_kv_heads
    self.repeats = self.n_heads // self.n_kv_heads
    self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    cache_size = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
    self.cache_k = torch.zeros(cache_size, dtype=self.wq.weight.dtype)
    self.cache_v = torch.zeros(cache_size, dtype=self.wq.weight.dtype)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    # cache
    scatter_pos = positions[None, :, None, None].repeat(bsz, 1, self.n_kv_heads, self.head_dim)
    self.cache_k, self.cache_v = self.cache_k.to(xk), self.cache_v.to(xv)
    self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk)
    self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv)
    if positions.size(0) > 1:
      # prefill
      key, value = repeat_kv(xk, xv, self.repeats)
    else:
      cur_pos = positions[-1].item() + 1
      key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)  # type: ignore
    query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    output = _fused_attention(query, key, value, mask, self.head_dim**-0.5)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
