# python3 models/mistral_rolling.py

# https://github.com/mistralai/mistral-inference/blob/147c4e68279b90eb61b19bdea44e16f5539d5a5d/one_file_ref.py

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, Tuple, List
from helpers import timeit

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import safetensors.torch
from sentencepiece import SentencePieceProcessor


DEFAULT_FLOAT = torch.bfloat16
torch.set_default_dtype(DEFAULT_FLOAT)


@dataclass
class MistralConfig:
  dim: int
  n_layers: int
  head_dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  sliding_window: int
  norm_eps: float
  vocab_size: int

  max_batch_size: int = 0


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def _reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
  # freqs_cis: complex - (seq_len, head_dim / 2)
  # x: complex - (bsz, seq_len, head_dim / 2)
  assert 1 < x.ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, (x.shape[1], x.shape[-1]))
  shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int=2) -> Tuple[Tensor, Tensor]:
  keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
  values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
  return keys, values


class Attention(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.n_heads, self.head_dim, self.n_kv_heads = config.n_heads, config.head_dim, config.n_kv_heads
    self.sliding_window = config.sliding_window
    self.repeats = self.n_heads // self.n_kv_heads
    self.scale = config.head_dim**-0.5

    self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
    self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    cache_size = (config.max_batch_size, config.sliding_window, self.n_kv_heads, config.head_dim)
    self.cache_k = torch.empty(cache_size, dtype=DEFAULT_FLOAT)
    self.cache_v = torch.empty(cache_size, dtype=DEFAULT_FLOAT)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    bsz, seqlen, _ = x.shape

    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # The cache is a rotating buffer
    scatter_pos: Tensor = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
    scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.head_dim)
    self.cache_k, self.cache_v = self.cache_k.to(xq.device), self.cache_v.to(xq.device)
    self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
    self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

    if positions.shape[0] > 1:
      key, value = repeat_kv(xk, xv, self.repeats) # prefill
    else:
      cur_pos = positions[-1].item() + 1
      key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)

    query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    output = self._attention(query, key, value, mask, self.scale)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

  @staticmethod
  def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], scale: float, use_fused: bool=True) -> Tensor:
    if use_fused:
      output = F.scaled_dot_product_attention(query, key, value, mask, scale=scale)
      return output
    scores = torch.matmul(query, key.transpose(2, 3)) * scale # (bsz, n_heads, seqlen | 1, seqlen)
    if mask is not None: scores += mask[None, None, ...]
    scores = scores.float()
    scores = F.softmax(scores, dim=-1).type_as(query)
    output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
    return output


class FeedForward(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


class TransformerBlock(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    self.attention = Attention(config)
    self.feed_forward = FeedForward(config.dim, config.hidden_dim)
    self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

  def forward(self, x: Tensor, freqs_cis: Tensor, positions: Tensor, mask: Optional[Tensor]) -> Tensor:
    x = x + self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
    x = x + self.feed_forward.forward(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: MistralConfig):
    super().__init__()
    assert config.vocab_size > 0
    self.config = config
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    self.layers = nn.ModuleList([TransformerBlock(config=config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.dim, eps=config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.head_dim, 128_000)

  def forward(self, input_ids: Tensor, positions: Tensor):
    seqlen = input_ids.size(1)
    h: Tensor = self.tok_embeddings(input_ids)
    self.freqs_cis = self.freqs_cis.to(input_ids.device)
    freqs_cis = self.freqs_cis[positions]
    mask: Optional[Tensor] = None
    if seqlen > 1:
      tensor = torch.full((seqlen, seqlen),dtype=h.dtype, fill_value=1, device=h.device,)
      mask = torch.tril(tensor, diagonal=0).to(h.dtype)
      # make the mask banded to account for sliding window
      mask = torch.triu(mask, diagonal=-self.config.sliding_window)
      mask = torch.log(mask)
    for layer in self.layers: h = layer(h, freqs_cis, positions, mask)
    return self.output(self.norm(h)).float()


class Tokenizer:
  def __init__(self, model_path: str):
    assert Path(model_path).exists(), model_path
    self._model = SentencePieceProcessor(model_file=model_path)
    assert self._model.vocab_size() == self._model.get_piece_size()

  @property
  def bos_id(self) -> int: return self._model.bos_id()
  @property
  def eos_id(self) -> int: return self._model.eos_id()
  @property
  def pad_id(self) -> int: return self._model.pad_id()

  def encode(self, s: str, bos: bool=True, eos: bool=False) -> List[int]:
    assert type(s) is str
    t: List[int] = self._model.encode(s)
    if bos: t = [self.bos_id] + t
    if eos: t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self._model.decode(t)


class Mistral:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.config = model.config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device
  @property
  def dtype(self) -> torch.dtype: return next(self.model.parameters()).dtype

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(folder: str, max_batch_size: int=1, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
    device = torch.device('cpu') if device is None else device
    dtype = DEFAULT_FLOAT if dtype is None else dtype
    model = Mistral.load_model(Path(folder), device, dtype, max_batch_size)
    tokenizer = Mistral.load_tokenizer(Path(folder))
    return Mistral(model, tokenizer)

  @staticmethod
  def load_model(folder: Path, device: torch.device, dtype: torch.dtype, max_batch_size: int=1):
    with open(Path(folder) / "params.json", "r") as f:
      config = MistralConfig(**json.load(f))
    config.max_batch_size = max_batch_size
    model = Transformer(config)
    pt_model_file = Path(folder) / "consolidated.00.pth"
    safetensors_model_file = Path(folder) / "consolidated.safetensors"
    assert (
      pt_model_file.exists() or safetensors_model_file.exists()
    ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
    assert not (
      pt_model_file.exists() and safetensors_model_file.exists()
    ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"
    if pt_model_file.exists(): loaded = torch.load(str(pt_model_file), mmap=True, weights_only=True)
    else: loaded = safetensors.torch.load_file(str(safetensors_model_file))
    model.load_state_dict(loaded, assign=True, strict=True)
    return model.to(device=device, dtype=dtype)

  @staticmethod
  def load_tokenizer(model_path: Path) -> Tokenizer:
    path = model_path / "tokenizer.model"
    assert path.exists()
    tokenizer = Tokenizer(path.as_posix())
    return tokenizer

  @torch.no_grad()
  def generate(self, prompts: List[str], max_tokens: int):
    model, tokenizer = self.model, self.tokenizer
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len, max_prompt_len = min(prompt_lens), max(prompt_lens)
    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device=self.device)
    for i, encoded in enumerate(encoded_prompts):
      input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id
    # pre-fill
    positions = torch.arange(0, min_prompt_len).to(self.device)
    logits = model(input_tokens[:, :min_prompt_len], positions)
    logprobs = F.log_softmax(logits, dim=-1)
    # decode
    generated = []
    all_logprobs = [logprobs[:,:-1,:].gather(2, input_tokens[:,1:min_prompt_len,None]).squeeze(-1),]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
      next_token = torch.argmax(logprobs[:, -1,:], dim=-1)
      if cur_pos < input_mask.shape[1]:
        next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
      all_logprobs.append(logprobs[:,-1,:].gather(1, next_token[:, None]))
      generated.append(next_token[:, None])
      logits = model(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
      logprobs = F.log_softmax(logits, dim=-1)
      cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
      generated = torch.cat(generated, 1)
      for i, x in enumerate(encoded_prompts):
        res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res, all_logprobs


if __name__ == "__main__":
  from helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  model_path = "downloads/mistral-7B-v0.1"
  model = Mistral.from_pretrained(model_path, max_batch_size=3, device=device)

  max_tokens: int = 35
  context = [
    "Quantum computing is",
    "Simply put, the theory of relativity states that",
    "SpaceX and NASA have collaborated to make commercial",
  ]
  res, _logprobs = model.generate(context, max_tokens=max_tokens)
  print('-' * 50)
  for x in res:
    print(x)
    print('-' * 50)
