from typing import Optional
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class Config:
  dim: int = 512
  n_layers: int = 6
  n_heads: int = 8
  hidden_dim: int = 2048
  vocab_size: int = 30000
  sinusoid: bool = True
  # arguments
  max_seq_len: Optional[int] = None
  max_batch_size: Optional[int] = None
  # post init
  head_dim: Optional[int] = None

  def __post_init__(self):
    assert self.max_seq_len is not None and self.max_batch_size is not None
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.linear1 = nn.Linear(dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, dim)

  def forward(self, x):
    return self.linear2(F.relu(self.linear1(x)))

class EncoderBlock(nn.Module):
  def __init__(self, dim, n_heads, head_dim, hidden_dim):
    super().__init__()
    self.n_heads, self.head_dim = n_heads, head_dim
    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)
    self.attn = nn.MultiheadAttention(
      dim, n_heads, batch_first=True)
    self.ffn = FeedForward(dim, hidden_dim)

  def forward(self, x: Tensor):
    x = self.ln1(x + self.attn(x, x, x)[0])
    x = self.ln2(x + self.ffn(x))
    return x

class DecoderBlock(nn.Module):
  def __init__(self, dim, n_heads, head_dim, hidden_dim, max_seqlen, max_batch_size):
    super().__init__()
    self.n_heads, self.head_dim = n_heads, head_dim
    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)
    self.ln3 = nn.LayerNorm(dim)
    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
    self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    self.ffn = FeedForward(dim, hidden_dim)
    self.bias = torch.tril(torch.ones(max_batch_size*n_heads, max_seqlen, max_seqlen)) == 0.

  def forward(self, x: Tensor, enc: Tensor):
    bsz, seqlen, _ = x.shape
    # causal self attention
    mask = self.bias[:(bsz*self.n_heads), :seqlen, :seqlen]
    x = self.ln1(x + self.attn(x, x, x, attn_mask=mask, is_causal=True)[0])
    # cross attention
    x = self.ln2(x + self.cross_attn(x, enc, enc)[0])
    x = self.ln3(x + self.ffn(x))
    return x


class PosEmbedding(nn.Module):
  def __init__(self, max_seqlen, dim):
    super().__init__()
    self.weight = self.pos_embedding(max_seqlen, dim)

  def forward(self, positions: Tensor):
    return self.weight[positions]

  @staticmethod
  def pos_embedding(max_seqlen: int, dim: int, base: float=10000.0):
    pos = torch.arange(max_seqlen, dtype=torch.float32)[:, None]
    dims = torch.arange(dim, dtype=torch.float32)
    i = (dims / 2).to(torch.int32)
    deno = torch.pow(base, 2*i / dim)
    theta = pos / deno
    idxs = torch.arange(0, dim, 2)
    theta[idxs] = torch.sin(theta[idxs])
    idxs = torch.arange(1, dim, 2)
    theta[idxs] = torch.cos(theta[idxs])
    return theta


class Transformer(nn.Module):
  def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int, vocab_size: int,
               n_layers: int, max_seqlen: int, max_batch_size: int, sinusoid: bool = True):
    super().__init__()
    self.enc_tok_embedding = nn.Embedding(vocab_size, dim)
    self.dec_tok_embedding = nn.Embedding(vocab_size, dim)
    self.pos_embedding = (PosEmbedding if sinusoid else nn.Embedding)(max_seqlen, dim)
    self.encoder = nn.ModuleList([
      EncoderBlock(dim, n_heads, head_dim, hidden_dim) for _ in range(n_layers)])
    self.decoder = nn.ModuleList([
      DecoderBlock(dim, n_heads, head_dim, hidden_dim, max_seqlen, max_batch_size) for _ in range(n_layers)])
    self.lm_head = nn.Linear(dim, vocab_size)
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def forward(self, x: Tensor, y: Tensor):
    enc_seqlen, dec_seqlen = x.size(1), y.size(1)
    x = self.enc_tok_embedding(x) + self.pos_embedding(torch.arange(0, enc_seqlen))
    y = self.dec_tok_embedding(y) + self.pos_embedding(torch.arange(0, dec_seqlen))
    for layer in self.encoder:
      x = layer(x)
    for layer in self.decoder:
      y = layer(y, x)
    logits = self.lm_head(y)
    return logits

  @staticmethod
  def build(config: Config):
    model = Transformer(
      dim = config.dim,
      n_heads = config.n_heads,
      head_dim = config.head_dim,
      hidden_dim = config.hidden_dim,
      vocab_size = config.vocab_size,
      n_layers = config.n_layers,
      max_seqlen = config.max_seq_len,
      max_batch_size = config.max_batch_size,
      sinusoid = config.sinusoid
    )
    return model

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.enc_tok_embedding.weight.numel()
      n_params -= self.dec_tok_embedding.weight.numel()
    return n_params


if __name__ == "__main__":
  config = Config(max_seq_len=1024, max_batch_size=32)
  model = Transformer.build(config)
  x = torch.randint(0, config.vocab_size, (1, 16))
  y = torch.randint(0, config.vocab_size, (1, 16))
  model.eval()
  with torch.no_grad():
    out = model(x, y)
  print(out.shape)
