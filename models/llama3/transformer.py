import torch
from torch import Tensor, nn

from models.llama.transformer import precompute_freqs_cis
from models.llama.transformer import RMSNorm
from models.llama2.transformer import Block


class Transformer(nn.Module):
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int, n_layers: int,
               max_batch_size: int, max_seq_len: int, vocab_size: int, norm_eps: float, rope_theta: int, **_):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(vocab_size, dim),
      layers = nn.ModuleList(
        [Block(dim, n_heads, n_kv_heads, head_dim, hidden_dim, max_batch_size, max_seq_len, norm_eps) for _ in range(n_layers)]),
      norm = RMSNorm(dim, eps=norm_eps)
    ))
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta)
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
