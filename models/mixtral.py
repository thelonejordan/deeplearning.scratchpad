# python3 models/mixtral.py

# https://paperswithcode.com/paper/mixtral-of-experts
# https://arxiv.org/abs/2401.04088
# https://mistral.ai/news/mixtral-of-experts/
# https://mistral.ai/news/mixtral-8x22b/

# Mixtral is based on a transformer architecture and uses the same
# modifications as described in Mistral 7B, with the notable exceptions
# that Mixtral supports a fully dense context length of 32k tokens,
# and the feedforward blocks are replaced by Mixture-of-Expert layers

from typing import Optional
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llama import precompute_freqs_cis, apply_rotary_emb
from llama2 import repeat_kv
from llama import RMSNorm, Tokenizer


@dataclass
class MixtralConfig: # 8x7B
  dim: int = 4096
  n_layers: int = 32
  head_dim: int = 128
  hidden_dim: int = 14336
  n_heads: int = 32
  n_kv_heads: int = 8
  context_len: int = 32768
  vocab_size: int = 32000
  norm_eps: int = 1e-05
  num_experts: int = 8
  top_k_experts: int = 2
  # extras
  max_batch_size: int = 1


class Attention(nn.Module):
  def __init__(self, config: MixtralConfig):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    self.repeats = self.n_heads // self.n_kv_heads
    self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wo = nn.Linear(config.dim, config.dim, bias=False)

    self.cache_k = torch.zeros((config.max_batch_size, config.context_len, self.n_kv_heads, self.head_dim))
    self.cache_v = torch.zeros((config.max_batch_size, config.context_len, self.n_kv_heads, self.head_dim))

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]=None):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
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
    keys = repeat_kv(keys, self.repeats) # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.repeats) # (bs, seqlen, n_local_heads, head_dim)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

    scores = (xq @ keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None: scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ values
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.wo(output)
    return output


class FeedForward(nn.Module):
  def __init__(self, config: MixtralConfig):
    super().__init__()
    self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
    self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
    self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

  def forward(self, x: Tensor):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
  def __init__(self, config: MixtralConfig):
    super().__init__()
    self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.attention = Attention(config)
    self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.feed_forward = FeedForward(config)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    x = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: MixtralConfig):
    super().__init__()
    self.context_len = config.context_len
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(4096, eps=config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, config.context_len * 2)

  def forward(self, tokens: Tensor, start_pos: int):
    seqlen = tokens.size(1)
    assert seqlen <= self.context_len
    device = tokens.device
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
      mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=device)
      mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    logits = self.output(h).float()
    return logits


class MoE(nn.Module):
  def __init__(self, config: MixtralConfig):
    super().__init__()
    self.config = config
    self.top_k_experts = config.top_k_experts
    self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
    self.experts = [Transformer(config) for _ in range(config.num_experts)]
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, x: Tensor):
    assert x.size(0) == 1, "only BS=1"
    g = torch.exp(self.gate(x[0, -1, :])).float() # (bsz, seqlen, num_experts)
    topk, topk_indices = torch.topk(g, self.top_k_experts)
    scale = topk / topk.sum()
    out = torch.zeros_like(x)
    for i in range(self.top_k_experts):
      expert = self.experts[topk_indices[i]]
      out += expert(x) * scale[i]
    return out

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      for expert in self.experts:
        n_params -= expert.tok_embeddings.weight.numel()
    return n_params


class Mixtral:
  def __init__(self, model: MoE, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.args = self.model.config
    self.device = 'cpu'

  def to(self, device):
    self.device = device
    self.model = self.model.to(device)
    return self
