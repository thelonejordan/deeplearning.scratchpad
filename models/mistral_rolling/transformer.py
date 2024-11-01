from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_FLOAT = torch.bfloat16
torch.set_default_dtype(DEFAULT_FLOAT)


@dataclass
class MistralConfig:
  dim: int
  n_layers: int
  head_dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  sliding_window: int
  norm_eps: float
  vocab_size: int

  max_batch_size: int = 0


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


class Attention(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.n_heads, self.head_dim, self.n_kv_heads = config.n_heads, config.head_dim, config.n_kv_heads
    self.sliding_window = config.sliding_window
    self.repeats = self.n_heads // self.n_kv_heads
    self.scale = config.head_dim**-0.5

    self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    cache_size = (config.max_batch_size, config.sliding_window, self.n_kv_heads, config.head_dim)
    self.cache_k = torch.empty(cache_size, dtype=DEFAULT_FLOAT)
    self.cache_v = torch.empty(cache_size, dtype=DEFAULT_FLOAT)

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
      key, value = repeat_kv(xk, xv, self.repeats) # prefill
    else:
      cur_pos = positions[-1].item() + 1
      key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)

    query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    output = self._attention(query, key, value, mask, self.scale)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

  @staticmethod
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float, use_fused: bool=True) -> Tensor:
    if use_fused:
      output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
      return output
    scores = torch.matmul(query, key.transpose(2, 3)) * scale # (bsz, n_heads, seqlen | 1, seqlen)
    if mask is not None: scores += mask[None, None, ...]
    scores = scores.float()
    scores = F.softmax(scores, dim=-1).type_as(query)
    output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
    return output


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


class TransformerBlock(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.attention = Attention(config)
    self.feed_forward = FeedForward(config.dim, config.hidden_dim)
    self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward.forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    assert config.vocab_size > 0
    self.config = config
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    self.layers = nn.ModuleList([TransformerBlock(config=config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, 128_000)

  def forward(self, input_ids: Tensor, positions: Tensor):
    seqlen = input_ids.size(1)
    h: Tensor = self.tok_embeddings(input_ids)
    self.freqs_cis = self.freqs_cis.to(input_ids.device)
    freqs_cis = self.freqs_cis[positions]
    mask: Optional[Tensor] = None
    if seqlen > 1:
      tensor = torch.full((seqlen, seqlen),dtype=h.dtype, fill_value=1, device=h.device,)
      mask = torch.tril(tensor, diagonal=0).to(h.dtype)
      # make the mask banded to account for sliding window
      mask = torch.triu(mask, diagonal=-self.config.sliding_window)
      mask = torch.log(mask)
    for layer in self.layers: h = layer(h, freqs_cis, positions, mask)
    return self.output(self.norm(h)).float()
