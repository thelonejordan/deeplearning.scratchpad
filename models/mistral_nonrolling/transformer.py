from typing import  Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = freqs_cis[:, None, :]
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)

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

def _fused_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float) -> Tensor:
  output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
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
      key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)
    query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    output = _attention(query, key, value, mask, self.head_dim**-0.5)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


class Block(nn.Module):
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
               max_seq_len: int, max_batch_size: int, norm_eps: float):
    super().__init__()
    self.attention = Attention(dim, head_dim, n_heads, n_kv_heads, max_seq_len, max_batch_size)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, eps=norm_eps)
    self.ffn_norm = RMSNorm(dim, eps=norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, vocab_size: int, n_layers: int,
               max_context_len: int, max_seq_len: int, max_batch_size: int, norm_eps: float, rope_theta: float, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList(
      [Block(dim, head_dim, hidden_dim, n_heads, n_kv_heads, max_seq_len, max_batch_size, norm_eps) for _ in range(n_layers)])
    self.norm = RMSNorm(dim, eps=norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_context_len, rope_theta)

  def forward(self, input_ids: Tensor, positions: Tensor):
    seqlen = input_ids.size(1)
    h: Tensor = self.tok_embeddings(input_ids)
    self.freqs_cis = self.freqs_cis.to(input_ids.device)
    freqs_cis = self.freqs_cis[positions]
    mask: Optional[Tensor] = None
    if seqlen > 1:
      base = torch.full((seqlen, seqlen), fill_value=1, dtype=h.dtype, device=h.device)
      mask = torch.tril(base, diagonal=0).type_as(h)
      mask = torch.triu(mask, diagonal=-self.max_seq_len)
      mask = torch.log(mask)
    for layer in self.layers: h = layer(h, freqs_cis, positions, mask)
    return self.output(self.norm(h)).float()
