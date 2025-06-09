from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.helpers import set_device
from models.llama.transformer import RMSNorm
from models.mistral_nonrolling.rope import precompute_freqs_cis
from models.mistral_nonrolling.attention import Attention


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
               max_position_embeddings: int, max_seq_len: int, max_batch_size: int, norm_eps: float, rope_theta: float, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList(
      [Block(dim, head_dim, hidden_dim, n_heads, n_kv_heads, max_seq_len, max_batch_size, norm_eps) for _ in range(n_layers)])
    self.norm = RMSNorm(dim, eps=norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    with torch.device(set_device(quiet=True)):
      self.freqs_cis = precompute_freqs_cis(head_dim, max_position_embeddings, rope_theta)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, input_ids: Tensor, positions: Tensor) -> Tensor:
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

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.tok_embeddings.weight.numel()
    return n_params
