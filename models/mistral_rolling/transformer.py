from typing import Optional

import torch
from torch import Tensor, nn

from models.llama.transformer import RMSNorm
from models.mistral_nonrolling.rope import precompute_freqs_cis
from models.mistral_nonrolling.transformer import FeedForward
from models.mistral_rolling.attention import Attention


class Block(nn.Module):
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int,
               max_batch_size: int, norm_eps: float):
    super().__init__()
    self.attention = Attention(dim, head_dim, n_heads, n_kv_heads, sliding_window, max_batch_size)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, eps=norm_eps)
    self.ffn_norm = RMSNorm(dim, eps=norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, dim: int, head_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int,
               vocab_size: int, n_layers: int, max_position_embeddings: int, max_batch_size: int, norm_eps: float, rope_theta: float, **_):
    super().__init__()
    self.sliding_window = sliding_window
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList(
      [Block(dim, head_dim, hidden_dim, n_heads, n_kv_heads, sliding_window, max_batch_size, norm_eps) for _ in range(n_layers)])
    self.norm = RMSNorm(dim, eps=norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_position_embeddings, rope_theta)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, input_ids: Tensor, positions: Tensor) -> Tensor:
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

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.tok_embeddings.weight.numel()
    return n_params
