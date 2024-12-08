from typing import Optional, Tuple
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  assert dim % 2 == 0, f"dim must be even, {dim=}"
  freqs = torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).neg() / dim) # 1/(theta ^ 2d) for each d < dim/2
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  freqs = torch.outer(t, freqs) # m/(theta ^ 2d) for each m < end, d < dim/2
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, (end, dim/2)
  return freqs_cis

# note: x{q,k} is (bsz, seqlen, n_head, head_dim)
def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(torch.unflatten(xq.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  xk_ = torch.view_as_complex(torch.unflatten(xk.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  assert 1 < xq_.ndim and freqs_cis.size() == (xq_.shape[1], xq_.shape[-1])
  freqs_cis = freqs_cis[:, None, :] # reshape_for_broadcast, (seqlen, 1, head_dim/2)
  xq_out = torch.view_as_real(xq_ * freqs_cis).reshape_as(xq) # (bsz, seqlen, n_head, head_dim)
  xk_out = torch.view_as_real(xk_ * freqs_cis).reshape_as(xk) # (bsz, seqlen, n_head, head_dim)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
  scores = (query @ key.transpose(2, 3)) * scale
  if mask is not None: scores = scores + mask
  scores = F.softmax(scores.float(), dim=-1).type_as(query)
  output = scores @ value
  return output

def _fused_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
  output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
  return output


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor):
    return self._norm(x.float()).type_as(x) * self.weight


class Attention(nn.Module):
  def __init__(self, dim: int, n_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
    super().__init__()
    self.n_heads, self.head_dim = n_heads, head_dim
    self.q_proj = nn.Linear(dim, dim, bias=False)
    self.k_proj = nn.Linear(dim, dim, bias=False)
    self.v_proj = nn.Linear(dim, dim, bias=False)
    self.o_proj = nn.Linear(dim, dim, bias=False)

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
    output = _fused_attention(xq, keys, values, mask, 1.0/math.sqrt(self.head_dim))

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
  def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int,
               max_batch_size: int, max_seq_len: int, norm_eps: float):
    super().__init__()
    self.input_layernorm = RMSNorm(dim, eps=norm_eps)
    self.self_attn = Attention(dim, n_heads, head_dim, max_batch_size, max_seq_len)
    self.post_attention_layernorm = RMSNorm(dim, eps=norm_eps)
    self.mlp = FeedForward(dim, hidden_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    x = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int, n_layers: int,
               max_batch_size: int, max_seq_len: int, vocab_size: int, norm_eps: float, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(vocab_size, dim),
      layers = nn.ModuleList(
        [Block(dim, n_heads, head_dim, hidden_dim, max_batch_size, max_seq_len, norm_eps) for _ in range(n_layers)]),
      norm = RMSNorm(dim, eps=norm_eps),
    ))
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, tokens: Tensor, start_pos: int):
    seqlen = tokens.size(1)
    assert seqlen <= self.max_seq_len
    device = tokens.device
    h = self.model.embed_tokens(tokens)
    self.freqs_cis = self.freqs_cis.to(device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
      mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=device)
      mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    for layer in self.model.layers: h = layer(h, start_pos, freqs_cis, mask)
    h = self.model.norm(h)
    output = self.lm_head(h[:,-1,:])
    return output

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params
