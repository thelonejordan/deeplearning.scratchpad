from typing import Callable
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.helpers import SDPA


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
    self.bias = torch.tril(torch.ones(1, 1, n_ctx, n_ctx))

  def forward(self, x: Tensor):
    B, T, C= x.size()
    qkv = self.c_attn(x)  # (B, T, 3C)
    q, k, v = qkv.split(self.n_embd, dim=-1)  # (B, T, C)
    q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, C')
    mask = self.bias[:,:,:T,:T].to(q.device)
    y = self._attn_fn(q, k, v, mask, 1.0/math.sqrt(self.head_size))  # (B, H, T, C')
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


class MLP(nn.Module):
  def __init__(self, n_embd: int):
    super().__init__()
    self.c_fc = nn.Linear(n_embd, n_embd * 4)
    self.c_proj = nn.Linear(4 * n_embd, n_embd)

  def forward(self, x: Tensor):
    x = self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))
    return x


class Block(nn.Module):
  def __init__(self, n_embd: int, n_head: int, n_ctx: int, norm_eps: float):
    super().__init__()
    self.ln_1 = nn.LayerNorm(n_embd, eps=norm_eps)
    self.attn = CausalSelfAttention(n_embd, n_head, n_ctx)
    self.ln_2 = nn.LayerNorm(n_embd, eps=norm_eps)
    self.mlp = MLP(n_embd)

  def forward(self, x: Tensor):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class Transformer(nn.Module):
  def __init__(self, n_embd: int, n_head: int, n_ctx: int, norm_eps: float, vocab_size: int, n_layer: int, **_):
    super().__init__()
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(vocab_size, n_embd),
      wpe = nn.Embedding(n_ctx, n_embd),
      h = nn.ModuleList([Block(n_embd, n_head, n_ctx, norm_eps) for _ in range(n_layer)]),
      ln_f = nn.LayerNorm(n_embd, eps=norm_eps),
    ))
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    self.apply_weight_sharing()
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def forward(self, idx: Tensor):
    pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device)
    tok_emb = self.transformer.wte(idx)  # (B, T, C)
    pos_emb = self.transformer.wpe(pos)  # (B, T, C)
    x = tok_emb + pos_emb  # (B, T, C)
    for block in self.transformer.h: x = block(x)  # (B, T, C)
    x = self.transformer.ln_f(x)  # (B, T, C)
    logits = self.lm_head(x) if self.training else self.lm_head(x[:,[-1], :])
    return logits

  def apply_weight_sharing(self):
    self.lm_head.weight = self.transformer.wte.weight
    return self

  @staticmethod
  def get_loss(x: Tensor, y: Tensor):
    return F.cross_entropy(x.view(-1, x.size(-1)), y.to(x.device).view(-1))

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.transformer.wpe.weight.numel()
    return n_params
