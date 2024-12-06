from typing import  Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.mistral_nonrolling.config import MistralConfig
from models.mistral_rolling.transformer import precompute_freqs_cis, apply_rotary_emb, repeat_kv
from models.mistral_rolling.transformer import RMSNorm, FeedForward


class Attention(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.head_dim, self.n_heads, self.n_kv_heads = config.head_dim, config.n_heads, config.n_kv_heads
    self.repeats = self.n_heads // self.n_kv_heads
    self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    cache_size = (config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
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
    output = self._attention(query, key, value, mask, self.head_dim**-0.5)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float) -> Tensor:
    scores = torch.matmul(query, key.transpose(2, 3)) * scale # (bsz, n_heads, seqlen | 1, seqlen)
    if mask is not None: scores += mask[None, None, ...]
    scores = scores.float()
    scores = F.softmax(scores, dim=-1).type_as(query)
    output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
    return output

  @staticmethod
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float) -> Tensor:
    output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
    return output


class Block(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.attention = Attention(config)
    self.feed_forward = FeedForward(config.dim, config.hidden_dim)
    self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.config = config
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    self.layers = nn.ModuleList([Block(config=config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_pos_embd, config.rope_theta)

  def forward(self, input_ids: Tensor, positions: Tensor):
    seqlen = input_ids.size(1)
    h: Tensor = self.tok_embeddings(input_ids)
    self.freqs_cis = self.freqs_cis.to(input_ids.device)
    freqs_cis = self.freqs_cis[positions]
    mask: Optional[Tensor] = None
    if seqlen > 1:
      base = torch.full((seqlen, seqlen), fill_value=1, dtype=h.dtype, device=h.device)
      mask = torch.tril(base, diagonal=0).type_as(h)
      mask = torch.triu(mask, diagonal=-self.config.max_seq_len)
      mask = torch.log(mask)
    for layer in self.layers: h = layer(h, freqs_cis, positions, mask)
    return self.output(self.norm(h)).float()
