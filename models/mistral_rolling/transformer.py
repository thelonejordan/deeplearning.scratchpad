from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.mistral_nonrolling.transformer import precompute_freqs_cis, apply_rotary_emb, repeat_kv
from models.mistral_nonrolling.transformer import RMSNorm, FeedForward

class Attention(nn.Module):
  def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int, max_batch_size: int):
    super().__init__()
    self.n_heads, self.head_dim, self.n_kv_heads = n_heads, head_dim, n_kv_heads
    self.sliding_window = sliding_window
    self.repeats = self.n_heads // self.n_kv_heads
    self.scale = head_dim**-0.5

    self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    cache_size = (max_batch_size, sliding_window, self.n_kv_heads, head_dim)
    self.cache_k = torch.empty(cache_size, dtype=self.wq.weight.dtype)
    self.cache_v = torch.empty(cache_size, dtype=self.wq.weight.dtype)

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
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int,
               max_batch_size: int, norm_eps: float):
    super().__init__()
    self.attention = Attention(dim, head_dim, n_heads, n_kv_heads, sliding_window, max_batch_size)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, eps=norm_eps)
    self.ffn_norm = RMSNorm(dim, eps=norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward.forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int,
               n_layers: int, vocab_size: int, max_batch_size: int, max_pos_embd: int, norm_eps: float, rope_theta: float, **_):
    super().__init__()
    self.sliding_window = sliding_window
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList(
      [Block(dim, head_dim, hidden_dim, n_heads, n_kv_heads, sliding_window, max_batch_size, norm_eps) for _ in range(n_layers)])
    self.norm = RMSNorm(dim, eps=norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_pos_embd, rope_theta)

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
      mask = torch.triu(mask, diagonal=-self.sliding_window)
      mask = torch.log(mask)
    for layer in self.layers: h = layer(h, freqs_cis, positions, mask)
    return self.output(self.norm(h)).float()
