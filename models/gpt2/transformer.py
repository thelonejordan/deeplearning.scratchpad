import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.helpers import accept_extra_kwargs
from models.gpt2.attention import CausalSelfAttention


class MLP(nn.Module):
  def __init__(self, n_embd: int):
    super().__init__()
    self.c_fc = nn.Linear(n_embd, n_embd * 4)
    self.c_proj = nn.Linear(4 * n_embd, n_embd)

  def forward(self, x: Tensor):
    x = self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))
    return x


class Block(nn.Module):
  def __init__(self, n_embd: int, n_head: int, n_ctx: int, norm_eps: float):
    super().__init__()
    self.ln_1 = nn.LayerNorm(n_embd, eps=norm_eps)
    self.attn = CausalSelfAttention(n_embd, n_head, n_ctx)
    self.ln_2 = nn.LayerNorm(n_embd, eps=norm_eps)
    self.mlp = MLP(n_embd)

  def forward(self, x: Tensor):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class Transformer(nn.Module):
  @accept_extra_kwargs()
  def __init__(self, n_embd: int, n_head: int, n_ctx: int, norm_eps: float, vocab_size: int, n_layer: int):
    super().__init__()
    self.transformer = nn.ModuleDict(
      dict(
        wte=nn.Embedding(vocab_size, n_embd),
        wpe=nn.Embedding(n_ctx, n_embd),
        h=nn.ModuleList([Block(n_embd, n_head, n_ctx, norm_eps) for _ in range(n_layer)]),
        ln_f=nn.LayerNorm(n_embd, eps=norm_eps),
      )
    )
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    self.apply_weight_sharing()
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def forward(self, idx: Tensor) -> Tensor:
    pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device)
    tok_emb = self.transformer.wte(idx)  # (B, T, C)
    pos_emb = self.transformer.wpe(pos)  # (B, T, C)
    x = tok_emb + pos_emb  # (B, T, C)
    for block in self.transformer.h: x = block(x)  # (B, T, C)
    x = self.transformer.ln_f(x)  # (B, T, C)
    logits = self.lm_head(x) if self.training else self.lm_head(x[:, [-1], :])
    return logits

  def apply_weight_sharing(self):
    self.lm_head.weight = self.transformer.wte.weight
    return self

  @staticmethod
  def get_loss(x: Tensor, y: Tensor):
    return F.cross_entropy(x.view(-1, x.size(-1)), y.to(x.device).view(-1))

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.transformer.wpe.weight.numel()
    return n_params
