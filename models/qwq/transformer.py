# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

from typing import Optional

import torch
from torch import Tensor, nn

from models.qwq.rope import Qwen2RotaryEmbedding
from models.llama.transformer import RMSNorm, FeedForward
from models.qwq.attention import Attention


class Block(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int,
               max_batch_size: int, max_seq_len: int, norm_eps: float):
    super().__init__()
    self.input_layernorm = RMSNorm(dim, eps=norm_eps)
    self.self_attn = Attention(dim, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len)
    self.post_attention_layernorm = RMSNorm(dim, eps=norm_eps)
    self.mlp = FeedForward(dim, hidden_dim)

  def forward(self, x: Tensor, start_pos: int, position_embeddings: tuple[Tensor], mask: Optional[Tensor]):
    x = x + self.self_attn(self.input_layernorm(x), start_pos, position_embeddings, mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int, n_layers: int, max_batch_size: int,
               max_seq_len: int, max_position_embeddings: int, vocab_size: int, norm_eps: float, rope_theta: float, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(vocab_size, dim),
      layers = nn.ModuleList(
        [Block(dim, n_heads, n_kv_heads, head_dim, hidden_dim, max_batch_size, max_seq_len, norm_eps) for _ in range(n_layers)]),
      norm = RMSNorm(dim, eps=norm_eps),
    ))
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    self.rotary_emb = Qwen2RotaryEmbedding(dim, head_dim, max_position_embeddings, rope_theta)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, tokens: Tensor, start_pos: int) -> Tensor:
    seqlen = tokens.size(1)
    assert seqlen > 0 and seqlen <= self.max_seq_len
    device = tokens.device
    h = self.model.embed_tokens(tokens)
    # create position embeddings to be shared across the decoder layers
    position_ids = torch.arange(start_pos, start_pos + seqlen, device=device)
    position_embeddings = self.rotary_emb(h, position_ids.unsqueeze(0))

    mask = None
    if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float('-inf'), device=device)
      mask = torch.triu(mask, diagonal=1)

      # When performing key-value caching, we compute the attention scores
      # only for the new sequence. Thus, the matrix of scores is of size
      # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
      # j > cache_len + i, since row i corresponds to token cache_len + i.
      mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask]).type_as(h)
    for layer in self.model.layers: h = layer(h, start_pos, position_embeddings, mask)

    h = self.model.norm(h)
    output = self.lm_head(h).float()
    return output

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params
