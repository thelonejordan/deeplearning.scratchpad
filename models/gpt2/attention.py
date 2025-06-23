from typing import Callable
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.helpers import SDPA, set_device


def _attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float) -> Tensor:
  att = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
  att = att.masked_fill(mask < 0.5, -float('inf'))  # causal mask
  y = F.softmax(att, dim=-1) @ v  # (B, H, T, T) x (B, H, T, C') -> (B, H, T, C')
  return y

def _fused_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float) -> Tensor:
  y = F.scaled_dot_product_attention(q, k, v, mask > 0.5, scale=scale)  # torch SDPA
  return y


class CausalSelfAttention(nn.Module):
  _attn_fn: Callable[..., Tensor] = staticmethod(_fused_attention if bool(SDPA) else _attention)

  def __init__(self, n_embd: int, n_head: int, n_ctx: int):
    super().__init__()
    assert n_embd % n_head == 0
    self.n_head, self.n_embd, self.head_size = n_head, n_embd, n_embd // n_head
    self.c_attn = nn.Linear(n_embd, 3 * n_embd)
    self.c_proj = nn.Linear(n_embd, n_embd)
    with torch.device(set_device(quiet=True)):
      self.bias = torch.tril(torch.ones(1, 1, n_ctx, n_ctx))

  def forward(self, x: Tensor):
    B, T, C = x.size()
    qkv = self.c_attn(x)  # (B, T, 3C)
    q, k, v = qkv.split(self.n_embd, dim=-1)  # (B, T, C)
    q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    mask = self.bias[:, :, :T, :T].to(q.device)
    y = self._attn_fn(q, k, v, mask, 1.0/math.sqrt(self.head_size))  # (B, H, T, C')
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y
