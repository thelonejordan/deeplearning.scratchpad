from typing import Optional
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.qwen.config import QwQConfig
from models.llama.rope import precompute_freqs_cis, apply_rotary_emb
from models.llama2.attention import Attention, repeat_kv
from models.llama.transformer import RMSNorm, FeedForward


class Attention(nn.Module):
  def __init__(self, config: QwQConfig):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    self.n_rep  = config.n_heads // config.n_kv_heads
    self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim)
    self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim)
    self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim)
    self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim)

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
    # repeat k/v heads if n_kv_heads < n_heads
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
  def __init__(self, config: QwQConfig):
    super().__init__()
    self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.self_attn = Attention(config)
    self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.mlp = FeedForward(config.dim, config.hidden_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, config: QwQConfig):
    super().__init__()
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.dim),
      layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
      norm = RMSNorm(config.dim, eps=config.norm_eps)
    ))
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, config.rope_theta)
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
      mask = torch.full((seqlen, seqlen), float('-inf'), device=device)
      mask = torch.triu(mask, diagonal=1).type_as(h)
      # When performing key-value caching, we compute the attention scores
      # only for the new sequence. Thus, the matrix of scores is of size
      # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
      # j > cache_len + i, since row i corresponds to token cache_len + i.
      mask = torch.hstack(
        [torch.zeros((seqlen, start_pos), device=device), mask]).type_as(h)
    for layer in self.model.layers: h = layer(h, start_pos, freqs_cis, mask)
    h = self.model.norm(h)
    output = self.lm_head(h).float()
    return output

  def apply_weight_sharing(self):
    self.lm_head.weight = self.model.embed_tokens.weight
    return self

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params
