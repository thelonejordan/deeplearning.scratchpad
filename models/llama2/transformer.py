from typing import Optional
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.llama.transformer import precompute_freqs_cis, apply_rotary_emb
from models.llama.transformer import RMSNorm, FeedForward

# n_rep > 1 aka n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
def repeat_kv(x: torch.Tensor, n_rep: int) -> Tensor:
    if n_rep == 1: return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(-2).expand(-1, -1, -1, n_rep, -1)
    return x.reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
    self.n_rep  = n_heads // n_kv_heads
    self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

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
    keys = repeat_kv(keys, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    output = self._attention(xq, keys, values, mask, 1.0/math.sqrt(self.head_dim))

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output

  @staticmethod
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
    scores = (query @ key.transpose(2, 3)) * scale
    if mask is not None: scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(query)
    output = scores @ value
    return output

  @staticmethod
  def _fused_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float):
    output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
    return output

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

class Block(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int,
               max_batch_size: int, max_seq_len: int, norm_eps: float):
    super().__init__()
    self.input_layernorm = RMSNorm(dim, eps=norm_eps)
    self.self_attn = Attention(dim, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len)
    self.post_attention_layernorm = RMSNorm(dim, eps=norm_eps)
    self.mlp = FeedForward(dim, hidden_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int, n_layers: int,
               max_batch_size: int, max_seq_len: int, vocab_size: int, norm_eps: float, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(vocab_size, dim),
      layers = nn.ModuleList(
        [Block(dim, n_heads, n_kv_heads, head_dim, hidden_dim, max_batch_size, max_seq_len, norm_eps) for _ in range(n_layers)]),
      norm = RMSNorm(dim, eps=norm_eps)
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
    output = self.lm_head(h).float()
    return output

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params
