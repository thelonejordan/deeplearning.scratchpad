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

from llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv
from llama import RMSNorm, Tokenizer

@dataclass
class LlamaConfig:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = 32000
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[int] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  max_seq_len: int = 2048

def compute_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
  hidden_dim = int(2 * (4 * dim) / 3)
  if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim) # custom dim factor multiplier
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim


class Attention(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.n_heads = config.n_heads
    self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
    self.n_rep = config.n_heads // self.n_kv_heads
    self.head_dim = config.dim // config.n_heads
    self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    self.cache_k = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
    self.cache_v = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor):
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
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ values
    # if mask is not None: scores = scores + mask
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output


class FeedForward(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    hidden_dim = compute_hidden_dim(config.dim, config.multiple_of, config.ffn_dim_multiplier)
    self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(config.dim, hidden_dim, bias=False)
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

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor):
    h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    assert config.dim % config.n_heads == 0, "n_heads must divide dim"
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.dim),
      layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
      norm = RMSNorm(config.dim, eps=config.norm_eps)
    ))
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len * 2)
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  @torch.inference_mode()
  def forward(self, tokens: Tensor, start_pos: int):
    _, seqlen = tokens.shape
    assert seqlen == 1
    h = self.model.embed_tokens(tokens)
    self.freqs_cis = self.freqs_cis.to(tokens.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    for layer in self.model.layers: h = layer(h, start_pos, freqs_cis)
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

  def to(self, device):
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
    for k in sd_keys_hf:
      assert sd_hf[k].shape == sd[k].shape, f'{k} not found'
      with torch.no_grad(): sd[k].copy_(sd_hf[k])
      print(f'loaded: {k}, {sd[k].shape}, {sd[k].dtype}')
      del sd_hf[k] # free memory after copying
    return Llama(model, tokenizer)

  def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None, device='cpu'):
    if max_gen_len is None:
      max_gen_len = self.args.max_seq_len - 1
    # Convert each prompt into tokens
    prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
    # Make sure the batch size is not too large
    bsz = len(prompt_tokens)
    assert bsz <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    # Make sure the prompt length is not larger than the maximum sequence length
    assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
    total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
    # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
      # Populate the initial tokens with the prompt tokens
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    eos_reached = torch.tensor([False] * bsz, device=device)
    prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
    for cur_pos in cur_iterator:
      with torch.no_grad():
        logits = self.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = self._sample_top_p(probs, top_p)
      else:
        # Greedily select the token with the max probability
        next_token = torch.argmax(logits[:, -1], dim=-1)
      next_token = next_token.reshape(-1)
      # Only replace token if it is a padding token
      next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
      tokens[:, cur_pos] = next_token
      # EOS is reached only if we found an EOS token for a padding position
      eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
      if all(eos_reached):
        break
    out_tokens = []
    out_text = []
    for current_prompt_tokens in tokens.tolist():
      # Cut to the EOS token, if present
      if self.tokenizer.eos_id in current_prompt_tokens:
        eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
        current_prompt_tokens = current_prompt_tokens[:eos_idx]
      out_tokens.append(current_prompt_tokens)
      out_text.append(self.tokenizer.decode(current_prompt_tokens))
    return out_tokens, out_text

  def _sample_top_p(self, probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1) # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p # (B, vocab_size)
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


if __name__ == "__main__":
  seed = os.getenv("SEED", 420)
  device = 'cpu'
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = 'cuda'
  elif torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
    device = 'mps'
  device = 'cpu' # hardcode, as MPS OOMs
  print(f'Using device: {device}')

  model = Llama.from_pretrained('7B', half=True).to(device)

  prompts = [
    "Simply put, the theory of relativity states that ",
    "If Google was an Italian company founded in Milan, it would",
  ]

  out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64, device=device)
  assert len(out_texts) == len(prompts)
  print('-' * 50)
  for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)