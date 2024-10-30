from typing import Optional
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.llama.transformer import precompute_freqs_cis, apply_rotary_emb
from models.llama.transformer import RMSNorm, FeedForward

@dataclass
class LlamaConfig:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = 32000
  max_seq_len: int = 2048
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[int] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  # for post init
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.hidden_dim is None
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of, self.ffn_dim_multiplier)
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

# https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py#L331
def compute_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
  hidden_dim = int(2 * (4 * dim) / 3)
  if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim) # custom dim factor multiplier
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

# n_rep > 1 aka n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
def repeat_kv(x: torch.Tensor, n_rep: int) -> Tensor:
    if n_rep == 1: return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(-2).expand(-1, -1, -1, n_rep, -1)
    return x.reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    self.n_rep  = config.n_heads // config.n_kv_heads
    self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    self.cache_k = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
    self.cache_v = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)

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
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float, use_fused: bool=True):
    if use_fused:
      output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
      return output
    scores = (query @ key.transpose(2, 3)) * scale
    if mask is not None: scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(query)
    output = scores @ value
    return output


class TransformerBlock(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    hidden_dim = compute_hidden_dim(config.dim, config.multiple_of, config.ffn_dim_multiplier)
    self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.self_attn = Attention(config)
    self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.mlp = FeedForward(config.dim, hidden_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.dim),
      layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
      norm = RMSNorm(config.dim, eps=config.norm_eps)
    ))
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, tokens: Tensor, start_pos: int):
    seqlen = tokens.size(1)
    assert seqlen <= self.config.max_seq_len
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
