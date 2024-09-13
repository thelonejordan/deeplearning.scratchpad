# llama 2
# https://github.com/meta-llama/llama-models
# https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py

# https://arxiv.org/abs/2307.09288
# https://huggingface.co/blog/llama2
# https://llama.meta.com/llama2/
# https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/

from typing import Optional
from dataclasses import dataclass
import os, math
from tqdm import tqdm
from helpers import timeit

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llama import precompute_freqs_cis, apply_rotary_emb, sample_top_p
from llama import RMSNorm, Tokenizer

@dataclass
class LlamaConfig:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = 32000
  max_seq_len: int = 2048
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[int] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  # for post init
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.hidden_dim is None
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of, self.ffn_dim_multiplier)
    if self.n_kv_heads is None: self.n_kv_heads = self.n_heads

# https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py#L331
def compute_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
  hidden_dim = int(2 * (4 * dim) / 3)
  if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim) # custom dim factor multiplier
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

# n_rep > 1 aka n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
def repeat_kv(x: torch.Tensor, n_rep: int) -> Tensor:
    if n_rep == 1: return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(-2).expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
    return x.reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    self.n_rep  = config.n_heads // config.n_kv_heads
    self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    self.cache_k = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
    self.cache_v = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]
    keys = repeat_kv(keys, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

    scores = (xq @ keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None: scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ values
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output


class FeedForward(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    hidden_dim = compute_hidden_dim(config.dim, config.multiple_of, config.ffn_dim_multiplier)
    self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
    self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.self_attn = Attention(config)
    self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.mlp = FeedForward(config)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.dim),
      layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
      norm = RMSNorm(config.dim, eps=config.norm_eps)
    ))
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  def forward(self, tokens: Tensor, start_pos: int):
    seqlen = tokens.size(1)
    assert seqlen <= self.config.max_seq_len
    device = tokens.device
    h = self.model.embed_tokens(tokens)
    self.freqs_cis = self.freqs_cis.to(device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
      mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=device)
      mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    for layer in self.model.layers: h = layer(h, start_pos, freqs_cis, mask)
    h = self.model.norm(h)
    output = self.lm_head(h).float()
    return output

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params


class Llama:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.args = self.model.config
    self.device = 'cpu'

  def to(self, device):
    self.device = device
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(model_type: str='7B', chat: bool=False, half=False):
    assert model_type in ('7B', '13B', '70B'), f'invalid model_type: {model_type}'
    config_args = {
      '7B' : dict(dim=4096, n_heads=32, n_layers=32),
      '13B': dict(dim=5120, n_heads=40, n_layers=40),
      '70B': dict(dim=8192, n_heads=64, n_kv_heads=8, n_layers=80),
    }[model_type]
    config = LlamaConfig(**config_args)
    from transformers import LlamaTokenizer, LlamaForCausalLM
    checkpoint = f'meta-llama/Llama-2-{model_type.lower()}' + ('-chat-hf' if chat else '-hf')
    tokenizer = Tokenizer(LlamaTokenizer.from_pretrained(checkpoint).vocab_file)
    model_hf = LlamaForCausalLM.from_pretrained(checkpoint)
    model = Transformer(config)
    if half: model, model_hf = model.half(), model_hf.half()
    sd, sd_hf = model.state_dict(), model_hf.state_dict()
    sd_keys, sd_keys_hf = list(sd.keys()), list(sd_hf.keys())
    assert len(sd_keys_hf) == len(sd_keys),f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}, {set(sd_keys).difference(set(sd_keys_hf))}"
    itr = tqdm(sd_keys_hf)
    for k in itr:
      itr.set_description(f'Loading {k}')
      assert sd_hf[k].shape == sd[k].shape, f'{k} not found'
      with torch.no_grad(): sd[k].copy_(sd_hf[k])
      del sd_hf[k] # free memory after copying
    return Llama(model, tokenizer)

  @torch.inference_mode
  def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
    if max_gen_len is None:
      max_gen_len = self.args.max_seq_len - 1
    prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
    bsz = len(prompt_tokens)
    assert bsz <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
    min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
    total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
    prev_pos = 0
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
    eos_reached = torch.tensor([False] * bsz, device=self.device)
    tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
    self.model.eval()
    for cur_pos in tqdm(range(min_prompt_len, total_len), desc='Generating tokens'):
      with torch.no_grad():
        logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1) # greedy
      prev_pos = cur_pos
      next_token = next_token.reshape(-1)
      # Only replace token if it is a padding token
      next_token = torch.where(tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
      tokens[:, cur_pos] = next_token
      # EOS is reached only if we found an EOS token for a padding position
      eos_reached |= (~tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
      if all(eos_reached): break
    out_tokens, out_text = [], []
    for current_prompt_tokens in tokens.tolist():
      # Cut to the EOS token, if present
      if self.tokenizer.eos_id in current_prompt_tokens:
        eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
        current_prompt_tokens = current_prompt_tokens[:eos_idx]
      out_tokens.append(current_prompt_tokens)
      out_text.append(self.tokenizer.decode(current_prompt_tokens))
    return out_tokens, out_text


if __name__ == "__main__":
  from helpers import set_device, set_seed
  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained('7B').to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "If Google was an Italian company founded in Milan, it would",
  ]

  out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
  assert len(out_texts) == len(prompts)
  print('-' * 50)
  for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)
