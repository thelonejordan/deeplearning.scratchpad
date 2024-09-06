# python3 models/gpt-2.py

# from paper: Language Models are Unsupervised Multitask Learners
# https://github.com/openai/gpt-2
# https://paperswithcode.com/method/gpt-2
# checkout nanoGPT by karpathy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py

from __future__ import annotations
import math, os
from dataclasses import dataclass

import tiktoken
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.register_buffer('bias', torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))
    self.n_head, self.n_embd = config.n_head, config.n_embd

  def forward(self, x: Tensor):
    B, T, C = x.size()
    head_size = C // self.n_head
    qkv = self.c_attn(x) # (B, T, 3C)
    q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C)
    q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, H, T, C')
    k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, H, T, C')
    v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, H, T, C')
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size)) # (B, H, T, T)
    att = att.masked_fill(self.bias[:,:,:T,:T] < 0.5, -float('inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, H, T, T) x (B, H, T, C') -> (B, H, T, C')
    # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y

class MLP(nn.Module):

  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x: Tensor):
    x = self.c_proj(self.gelu(self.c_fc(x)))
    return x

class Block(nn.Module):

  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
    self.mlp = MLP(config)

  def forward(self, x: Tensor):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):

  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.config = config
    self.tokenizer = tiktoken.get_encoding('gpt2')

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd, eps=config.norm_eps),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params

  def encode(self, input, device='cpu'):
    batch = [input] if isinstance(input, str) else input
    return torch.tensor(self.tokenizer.encode_batch(batch), dtype=torch.long, device=device)

  def decode(self, idx: Tensor):
    return self.tokenizer.decode_batch(idx.tolist())

  def forward(self, idx: Tensor):
    device = idx.device
    _, T = idx.size()
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = self.transformer.wte(idx) # (B, T, C)
    pos_emb = self.transformer.wpe(pos) # (B, T, C)
    x = tok_emb + pos_emb # (B, T, C)
    for block in self.transformer.h: x = block(x) # (B, T, C)
    x = self.transformer.ln_f(x) # (B, T, C)
    logits = self.lm_head(x) if self.training else self.lm_head(x[:,[-1], :])
    return logits

  @staticmethod
  def get_loss(x: Tensor, y: Tensor):
    return F.cross_entropy(x.view(-1, x.size(-1)), y.to(x.device).view(-1))

  @staticmethod
  def from_pretrained(model_type: str='gpt2'):
    config_args = {
      'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    # copy while ensuring all of the parameters are aligned and match in names and shapes
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model

  @torch.no_grad()
  def generate(self, idx: Tensor, max_new_tokens, num_return_sequences=1, temperature=1.0, top_k=None):
    assert idx.size(0) == 1 and num_return_sequences >= 1 and temperature > 0.0
    idx = idx.repeat(num_return_sequences, 1)
    self.eval()
    while idx.size(1) < max_new_tokens:
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits = self(idx_cond)
      logits = logits / temperature
      if top_k is not None:
        assert top_k > 0 and top_k <= self.config.vocab_size
        _, topk_indices = torch.topk(logits, self.config.vocab_size - top_k, largest=False)
        logits.scatter_(-1, topk_indices, -float('inf'))
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
      idx = torch.cat((idx, idx_next), dim=-1)
    return idx

@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  norm_eps: float = 1e-5

if __name__ == '__main__':
  seed = os.getenv("SEED", 420)
  device = 'cpu'
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = 'cuda'
  elif torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
    device = 'mps'
  print(f'Using device: {device}')

  config = GPTConfig()
  model = GPT.from_pretrained('gpt2').to(device)

  num_return_sequences = 5
  max_length = 30
  context = "Hello, I'm a language model,"
  idx = model.encode(context, device)

  out = model.generate(idx, 3000, num_return_sequences=1, top_k=50)
  for i, sentence in enumerate(model.decode(out)):
    print(f'sample {i+1}:', sentence)
    print()
