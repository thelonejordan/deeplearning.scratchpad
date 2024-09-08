# python3 models/gpt2.py

# from paper: Language Models are Unsupervised Multitask Learners
# https://github.com/openai/gpt-2
# https://paperswithcode.com/method/gpt-2
# checkout nanoGPT by karpathy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py

from typing import Optional, List
from dataclasses import dataclass
import math, os
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Embedding, Linear, LayerNorm
import tiktoken

@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  norm_eps: float = 1e-5


class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.n_head, self.n_embd, self.head_size = config.n_head, config.n_embd, config.n_embd // config.n_head
    self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = Linear(config.n_embd, config.n_embd)
    self.register_buffer('bias', torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

  def forward(self, x: Tensor):
    B, T, C= x.size()
    qkv = self.c_attn(x) # (B, T, 3C)
    q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C)
    q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, C')
    k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, C')
    v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, C')
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size)) # (B, H, T, T)
    att = att.masked_fill(self.bias[:,:,:T,:T] < 0.5, -float('inf')) # causal mask
    y = F.softmax(att, dim=-1) @ v # (B, H, T, T) x (B, H, T, C') -> (B, H, T, C')
    # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


class MLP(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = Linear(config.n_embd, config.n_embd * 4)
    self.c_proj = Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x: Tensor):
    x = self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))
    return x


class Block(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
    self.mlp = MLP(config)

  def forward(self, x: Tensor):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte = Embedding(config.vocab_size, config.n_embd),
        wpe = Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = LayerNorm(config.n_embd, eps=config.norm_eps),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.apply_weight_sharing()
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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

  def apply_weight_sharing(self):
    self.transformer.wte.weight = self.lm_head.weight
    return self

  @staticmethod
  def get_loss(x: Tensor, y: Tensor):
    return F.cross_entropy(x.view(-1, x.size(-1)), y.to(x.device).view(-1))

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.transformer.wpe.weight.numel()
    return n_params


class GPT2:
  def __init__(self, model: Transformer, tokenizer: tiktoken.Encoding):
    self.model = model
    self.tokenizer = tokenizer
    self.device = 'cpu'

  def to(self, device):
    self.device = device
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(model_type: str='gpt2', half: bool=False):
    config_args = {
      'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    from transformers import GPT2LMHeadModel
    model = Transformer(GPTConfig(**config_args))
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    if half: model, model_hf = model.half(), model_hf.half()
    sd, sd_hf = model.state_dict(), model_hf.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
    sd_keys_hf = [k for k in sd_hf.keys() if not any(k.endswith(w) for w in ('.attn.masked_bias', '.attn.bias'))]
    # copy while ensuring all of the parameters are aligned and match in names and shapes
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in tqdm(sd_keys_hf, desc=f'Loading pretrained weights'):
      if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad(): sd[k].copy_(sd_hf[k])
    return GPT2(model.apply_weight_sharing(), tiktoken.get_encoding('gpt2'))

  def _encode(self, input: List[str] | str):
    batch = [input] if isinstance(input, str) else input
    return torch.tensor(self.tokenizer.encode_batch(batch), dtype=torch.long, device=self.device)

  def _decode(self, idx: Tensor):
    return self.tokenizer.decode_batch(idx.tolist())

  @torch.no_grad()
  def generate(self, prompt: str, max_new_tokens: int, num_return_sequences: int=1,
               temperature: float=1.0, top_k: Optional[int]=None):
    config = self.model.config
    idx = model._encode(prompt)
    assert idx.size(0) == 1 and num_return_sequences >= 1 and temperature > 0.0
    idx = idx.repeat(num_return_sequences, 1)
    self.model.eval()
    while idx.size(1) < max_new_tokens:
      idx_cond = idx if idx.size(1)<=config.block_size else idx[:, -config.block_size:]
      logits = self.model(idx_cond) / temperature
      if top_k is not None and top_k < config.vocab_size:
        assert top_k > 0
        _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
        logits.scatter_(-1, topk_indices, -float('inf'))
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
      idx = torch.cat((idx, idx_next), dim=-1)
    return self._decode(idx)


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

  model = GPT2.from_pretrained('gpt2', half=False).to(device)

  num_return_sequences = 8
  max_length = 32
  context = "Hello, I'm a language model,"
  out = model.generate(context, max_length, num_return_sequences, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(f'sample {i+1}:', sentence)
    print('-'*50)