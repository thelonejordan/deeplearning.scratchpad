# python3 models/llama.py

# https://arxiv.org/abs/2302.13971
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py (57b0eb62de0636e75af471e49e2f1862d908d9d8)
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py

from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import math
from tqdm import tqdm
from helpers import timeit

from sentencepiece import SentencePieceProcessor
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
  n_heads: int = 32
  n_layers: int = 32
  dim: int = 4096
  vocab_size: int = 32000
  max_seq_len: int = 2048
  norm_eps: float = 1e-5
  multiple_of: int = 256
  max_batch_size: int = 32
  # for post init
  head_dim: Optional[int] = None
  hidden_dim: Optional[int] = None

  def __post_init__(self):
    assert self.head_dim is None and self.dim % self.n_heads == 0
    self.head_dim = self.dim // self.n_heads
    assert self.hidden_dim is None
    self.hidden_dim = compute_hidden_dim(self.dim, self.multiple_of)

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L161
def compute_hidden_dim(dim: int, multiple_of: int):
  hidden_dim = 4 * dim
  hidden_dim = int(2 * hidden_dim / 3)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  assert dim % 2 == 0, f"dim must be even, {dim=}"
  freqs = torch.pow(theta, torch.arange(0, dim, 2).neg().float() / dim) # 1/(theta ^ 2d) for each d < dim/2
  freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float() # m/(theta ^ 2d) for each m < end, d < dim/2
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, (end, dim/2)
  return freqs_cis

# note: x{q,k} is (bsz, seqlen, n_head, head_dim)
def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_complex = torch.view_as_complex(torch.unflatten(xq.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  xk_complex = torch.view_as_complex(torch.unflatten(xk.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  freqs_cis = freqs_cis[:, None, :] # reshape_for_broadcast, (seqlen, 1, head_dim/2)
  xq_out = torch.view_as_real(xq_complex * freqs_cis).reshape_as(xq) # (bsz, seqlen, n_head, head_dim)
  xk_out = torch.view_as_real(xk_complex * freqs_cis).reshape_as(xk) # (bsz, seqlen, n_head, head_dim)
  return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor):
    return self._norm(x.float()).type_as(x) * self.weight


class Attention(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.n_heads, self.head_dim = config.n_heads, config.head_dim
    self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    self.cache_k = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_heads, self.head_dim)
    self.cache_v = torch.zeros(config.max_batch_size, config.max_seq_len, self.n_heads, self.head_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]=None):
    bsz, seqlen, _ = x.size()
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    output = self._attention(xq, keys, values, mask, 1.0/math.sqrt(self.head_dim))

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output

  @staticmethod
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float, use_fused: bool=True):
    if use_fused:
      output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
      return output
    scores = (query @ key.transpose(2, 3)) * scale
    if mask is not None: scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(query)
    output = scores @ value
    return output


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.self_attn = Attention(config)
    self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
    self.mlp = FeedForward(config.dim, config.hidden_dim)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    x = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.dim),
      layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
      norm = RMSNorm(config.dim, eps=config.norm_eps),
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
    output = self.lm_head(h[:,-1,:])
    return output

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params


class Tokenizer:
  def __init__(self, model_path: str):
    assert Path(model_path).exists(), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)
    print(f"Reloaded SentencePiece model from {model_path}")
    self.n_words: int = self.sp_model.vocab_size()
    print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  @property
  def bos_id(self) -> int: return self.sp_model.bos_id()
  @property
  def eos_id(self) -> int: return self.sp_model.eos_id()
  @property
  def pad_id(self) -> int: return self.sp_model.pad_id()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert isinstance(s, str)
    t = self.sp_model.encode(s)
    if bos: t = [self.bos_id] + t
    if eos: t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)


class Llama:
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
  def from_pretrained(model_type: str='7B', half=False, assign: bool=False):
    assert model_type in ('7B', '13B', '30B', '65B'), f'invalid model_type: {model_type}'
    config_args = {
      '7B' : dict(dim=4096, n_heads=32, n_layers=32), # 6.7B
      '13B': dict(dim=5120, n_heads=40, n_layers=40), # 13.0B
      '30B': dict(dim=6656, n_heads=52, n_layers=60), # 32.5B
      '65B': dict(dim=8192, n_heads=64, n_layers=80), # 65.2B
    }[model_type]
    config = LlamaConfig(**config_args)
    model = Transformer(config)
    checkpoint = f'huggyllama/llama-{model_type.lower()}'
    if assign: model, tokenizer = Llama._load_from_cache(model, checkpoint)
    else: model, tokenizer = Llama._copy_from_hf(model, checkpoint, half)
    if half: model = model.half()
    return Llama(model, tokenizer)

  @staticmethod
  def _copy_from_hf(model, checkpoint: str, half=False):
    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = Tokenizer(LlamaTokenizer.from_pretrained(checkpoint).vocab_file)
    model_hf = LlamaForCausalLM.from_pretrained(checkpoint)
    if half: model, model_hf = model.half(), model_hf.half()
    sd, sd_hf = model.state_dict(), model_hf.state_dict()
    sd_keys, sd_keys_hf = list(sd.keys()), list(sd_hf.keys())
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    itr = tqdm(sd_keys_hf)
    for k in itr:
      itr.set_description(f'Loading {k}')
      assert sd_hf[k].shape == sd[k].shape, f'{k} not found'
      with torch.no_grad(): sd[k].copy_(sd_hf[k])
      del sd_hf[k] # free memory after copying
    return model, tokenizer

  @staticmethod
  def _load_from_cache(model: Transformer, checkpoint: str):
    from transformers.utils import try_to_load_from_cache
    import safetensors.torch
    filenames=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    model_files = [try_to_load_from_cache(repo_id=checkpoint, filename=filename) for filename in filenames]
    loaded = dict()
    for file in model_files: loaded.update(safetensors.torch.load_file(str(file)))
    loaded = {k:v for k, v in loaded.items() if not k.endswith("freq")}
    model.load_state_dict(loaded, assign=True, strict=True)
    tokenizer_path = try_to_load_from_cache(repo_id=checkpoint, filename="tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)
    return model, tokenizer

  @torch.inference_mode()
  def generate(self, prompts: List[str], max_gen_len: int, temperature: float=0.8, top_p: float=0.95) -> List[str]:
    bsz = len(prompts)
    params = self.model.config
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id, device=self.device).long()
    for k, t in enumerate(prompt_tokens): tokens[k, : len(t)] = torch.tensor(t, device=self.device).long()
    input_text_mask = tokens != self.tokenizer.pad_id
    prev_pos = 0
    self.model.eval()
    for cur_pos in tqdm(range(min_prompt_size, total_len), desc='Generating tokens'):
      with torch.no_grad():
        logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
      if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits, dim=-1)
      next_token = next_token.reshape(-1)
      # only replace token if prompt has already been generated
      next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
      tokens[:, cur_pos] = next_token
      prev_pos = cur_pos
    decoded = []
    for i, t in enumerate(tokens.tolist()):
      t = t[: len(prompt_tokens[i]) + max_gen_len] # cut to max gen len
      try: t = t[: t.index(self.tokenizer.eos_id)] # cut to eos tok if any
      except ValueError: pass
      decoded.append(self.tokenizer.decode(t))
    return decoded

def sample_top_p(probs, p):
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token


if __name__ == "__main__":
  from helpers import set_device, set_seed
  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained('7B', assign=True).to(device)

  num_return_sequences = 4
  max_gen_len = 32
  context = "Hello, I'm a language model,"

  prompts = [context] * num_return_sequences
  out = model.generate(prompts, max_gen_len)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence)
    print('-'*50)
