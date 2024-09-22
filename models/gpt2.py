# python3 models/gpt2.py

# from paper: Language Models are Unsupervised Multitask Learners
# https://github.com/openai/gpt-2
# https://paperswithcode.com/method/gpt-2
# checkout nanoGPT by karpathy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py

from typing import Optional, List, Set
from dataclasses import dataclass
import math
from tqdm import tqdm
from helpers import timeit

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
    # need to register as buffer so that to(device) works (consequently shows up in state_dict)
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
    # y = F.scaled_dot_product_attention(q, k, v, att_mask=mask, is_causal=True) # flash attention
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
    pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device)
    tok_emb = self.transformer.wte(idx) # (B, T, C)
    pos_emb = self.transformer.wpe(pos) # (B, T, C)
    x = tok_emb + pos_emb # (B, T, C)
    for block in self.transformer.h: x = block(x) # (B, T, C)
    x = self.transformer.ln_f(x) # (B, T, C)
    logits = self.lm_head(x) if self.training else self.lm_head(x[:,[-1], :])
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

class Tokenizer:
  def __init__(self):
    self.model = tiktoken.get_encoding('gpt2')
    self.eot_token = self.model.eot_token

  def encode_batch(self, input: List[str] | str, device: torch.device):
    batch = [input] if isinstance(input, str) else input
    return torch.tensor(self.model.encode_batch(batch), dtype=torch.long, device=device)

  def decode_batch(self, idx: Tensor):
    return self.model.decode_batch(idx.tolist())


class GPT2:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(model_type: str='gpt2', half: bool=False, assign: bool=False):
    config_args = {
      'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    model = Transformer(GPTConfig(**config_args))
    transposed = {'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'}
    if assign: model = GPT2._load_from_cache(model, model_type, transposed=transposed)
    else: model = GPT2._copy_from_hf(model, model_type, half=half, transposed=transposed)
    if half: model = model.half()
    return GPT2(model.apply_weight_sharing(), Tokenizer())

  @staticmethod
  def _copy_from_hf(model: Transformer, checkpoint: str, half: bool=False, transposed: Set[str]={}):
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(checkpoint)
    if half: model, model_hf = model.half(), model_hf.half()
    sd, sd_hf = model.state_dict(), model_hf.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
    sd_keys_hf = [k for k in sd_hf.keys() if not any(k.endswith(w) for w in ('.attn.masked_bias', '.attn.bias'))]
    # copy while ensuring all of the parameters are aligned and match in names and shapes
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in tqdm(sd_keys_hf, desc=f'Loading pretrained weights'):
      if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad(): sd[k].copy_(sd_hf[k])
    return model

  @staticmethod
  def _load_from_cache(model: Transformer, checkpoint: str, transposed: Set[str]={}):
    from transformers.utils import try_to_load_from_cache
    import safetensors.torch
    safetensors_model_file = try_to_load_from_cache(repo_id=checkpoint, filename="model.safetensors")
    loaded = safetensors.torch.load_file(str(safetensors_model_file))
    for k, v in loaded.items():
      if any(k.endswith(w) for w in transposed):
        loaded[k] = v.t()
    model.transformer.load_state_dict(loaded, assign=True, strict=True)
    return model

  @torch.inference_mode()
  def generate(self, prompt: str, max_new_tokens: int, num_return_sequences: int=1,
               temperature: float=1.0, top_k: Optional[int]=None):
    config = self.model.config
    idx = self.tokenizer.encode_batch(prompt, device=self.device)
    assert idx.size(0) == 1 and num_return_sequences >= 1 and temperature > 0.0
    idx = idx.repeat(num_return_sequences, 1)
    self.model.eval()
    while idx.size(1) < max_new_tokens:
      idx_cond = idx if idx.size(1)<=config.block_size else idx[:, -config.block_size:]
      with torch.no_grad():
        logits = self.model(idx_cond) / temperature
      if top_k is not None and top_k < config.vocab_size:
        assert top_k > 0
        _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
        logits.scatter_(-1, topk_indices, -float('inf'))
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
      idx = torch.cat((idx, idx_next), dim=-1)
    return self.tokenizer.decode_batch(idx)

  @torch.inference_mode()
  def completion(self, prompts: str | List[str], max_new_tokens: int,
                 temperature: float=1.0, top_k: Optional[int]=None):
    config, tokenizer = self.model.config, self.tokenizer
    idxs, masks = [], []
    start_pos = max_new_tokens
    for i in range(len(prompts)):
      idx = tokenizer.model.encode(prompts[i])
      mask = [1 for _ in range(len(idx))]
      start_pos = min(start_pos, len(idx))
      if len(idx) < max_new_tokens:
        rem = max_new_tokens - len(idx)
        idx.extend([tokenizer.eot_token for _ in range(rem)])
        mask.extend([0 for _ in range(rem)])
      idxs.append(idx)
      masks.append(mask)
    idx = torch.tensor(idxs, dtype=torch.long, device=self.device)
    mask = torch.tensor(masks, dtype=torch.long, device=self.device)
    self.model.eval()
    cur_pos = start_pos
    while cur_pos < max_new_tokens:
      idx_cond = idx[:,:cur_pos] if cur_pos<=config.block_size else idx[:, -config.block_size:]
      with torch.no_grad():
        logits = self.model(idx_cond) / temperature
      if top_k is not None and top_k < config.vocab_size:
        assert top_k > 0
        _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
        logits.scatter_(-1, topk_indices, -float('inf'))
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
      idx[:,[cur_pos]] = torch.where(mask[:, [cur_pos]]>0.5, idx[:,[cur_pos]], idx_next)
      cur_pos += 1
    return self.tokenizer.decode_batch(idx)


if __name__ == '__main__':
  from helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  model = GPT2.from_pretrained('gpt2', assign=True).to(device)

  print("Testing generation...")
  num_return_sequences = 8
  max_length = 32
  context = "Hello, I'm a language model,"
  out = model.generate(context, max_length, num_return_sequences, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)

  print("Testing completion...")
  max_length = 200
  context = [
    "Hello, I'm a language model,",
    "Quantum computing is",
    "SpaceX and NASA have collaborated to make commercial"
  ]
  out = model.completion(context, max_length, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)
