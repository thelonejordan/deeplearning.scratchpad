# python3 models/llama.py

# https://arxiv.org/abs/2302.13971
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py (57b0eb62de0636e75af471e49e2f1862d908d9d8)
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py

from typing import Optional, Tuple, List
from dataclasses import dataclass
import os, math, warnings

from sentencepiece import SentencePieceProcessor
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
  n_heads: int
  n_layers: int = 32
  n_embd: int = 4096
  hidden_dim: int = 11008
  vocab_size: int = 32000
  block_size: int = 2048
  norm_eps: float = 1e-5
  multiple_of: int = 256
  max_batch_size: int = 32
  # padding_idx: int = 0 # for nn.Embedding? see huggingface implementation

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L55
def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L63
def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L161
def compute_hidden_dim(n_embd: int, multiple_of: int):
  hidden_dim = 4 * n_embd
  hidden_dim = int(2 * hidden_dim / 3)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return hidden_dim

class LlamaAttention(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    assert config.n_embd % config.n_heads == 0
    self.n_heads, self.head_dim = config.n_heads, config.n_embd // config.n_heads
    self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]=None):
    bsz, seqlen, _ = x.size()
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    scores = (xq @ xk.transpose(2, 3)) / math.sqrt(self.head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, xv)
    if mask is not None: scores += mask
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = self.o_proj(output)
    return output

class LlamaFeedForward(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    hidden_dim = compute_hidden_dim(config.n_embd, config.multiple_of)
    self.act_fn = nn.SiLU()
    self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
    self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaBlock(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
    self.self_attn = LlamaAttention(config)
    self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
    self.mlp = LlamaFeedForward(config)

  def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor]):
    x = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x

class LlamaTransformer(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.config = config
    self.model = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.n_embd), #, padding_idx=config.padding_idx),
      layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)]),
      norm = nn.RMSNorm(4096, eps=config.norm_eps),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_heads, config.block_size * 2)
    self.register_buffer('mask', torch.full((1, 1, config.block_size, config.block_size), float("-inf")))
    print("number of parameters: %.2fB" % (self.get_num_params()/1e9,))

  @torch.inference_mode() # clamp to inference mode
  def forward(self, tokens: Tensor, start_pos: int):
    seqlen = tokens.size(1)
    assert tokens.size(1) <= self.config.block_size
    device = tokens.device
    h = self.model.embed_tokens(tokens)
    self.freqs_cis = self.freqs_cis.to(device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = torch.triu(self.mask.to(device), diagonal=start_pos + 1).type_as(h) if seqlen > 1 else None
    for layer in self.model.layers: h = layer(h, start_pos, freqs_cis, mask)
    h = self.model.norm(h)
    if not self.training: h = h[:, [-1], :]
    h = self.lm_head(h)
    return h

  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.model.embed_tokens.weight.numel()
    return n_params

class LlamaTokenizer:
  def __init__(self, model_path: str):
    # reload tokenizer
    assert os.path.isfile(model_path), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)
    print(f"Reloaded SentencePiece model from {model_path}")
    # BOS / EOS token IDs
    self.n_words: int = self.sp_model.vocab_size()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos: t = [self.bos_id] + t
    if eos: t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
      return self.sp_model.decode(t)

class Llama:
  def __init__(self, model: LlamaTransformer, tokenizer: LlamaTokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def to(self, device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(model_type: str='7B'):
    assert model_type in ('7B', '13B', '30B', '65B'), f'invalid model_type: {model_type}'
    config_args = {
      '7B' : dict(n_embd=4096, n_heads=32, n_layers=32, hidden_dim=11008), # 6.7B
      '13B': dict(n_embd=5120, n_heads=40, n_layers=40, hidden_dim=11008), # 13.0B
      '30B': dict(n_embd=6656, n_heads=52, n_layers=60, hidden_dim=11008), # 32.5B
      '65B': dict(n_embd=8192, n_heads=64, n_layers=80, hidden_dim=11008), # 65.2B
    }[model_type]
    config = LlamaConfig(**config_args)
    from transformers import LlamaTokenizer as Tokenizer, LlamaForCausalLM
    checkpoint = f'huggyllama/llama-{model_type.lower()}'
    with warnings.catch_warnings(action="ignore"):
      tokenizer = LlamaTokenizer(Tokenizer.from_pretrained(checkpoint).vocab_file)
    model_hf = LlamaForCausalLM.from_pretrained(checkpoint)
    model = LlamaTransformer(config)
    sd, sd_hf = model.state_dict(), model_hf.state_dict()
    sd_keys = [key for key in sd if key!='mask']
    assert len(sd_hf.keys()) == len(sd_keys), f"mismatched keys: {len(sd_hf.keys())} != {len(sd_keys)}"
    for k in sd_hf:
      assert sd_hf[k].shape == sd[k].shape, f'{k} not found'
      with torch.no_grad(): sd[k].copy_(sd_hf[k])
    return Llama(model, tokenizer)

  @torch.no_grad()
  def generate(self, prompts: List[str], max_gen_len: int, temperature: float=0.8, top_p: float=0.95, device='cpu') -> List[str]:
    bsz = len(prompts)
    params = self.model.config
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])
    total_len = min(params.block_size, max_gen_len + max_prompt_size)
    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device).long()
    for k, t in enumerate(prompt_tokens): tokens[k, : len(t)] = torch.tensor(t).long()
    input_text_mask = tokens != self.tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    for cur_pos in range(start_pos, total_len):
      logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
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
      # cut to max gen len
      t = t[: len(prompt_tokens[i]) + max_gen_len]
      # cut to eos tok if any
      try: t = t[: t.index(self.tokenizer.eos_id)]
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

  model = Llama.from_pretrained('7B').to(device)

  num_return_sequences = 4
  max_gen_len = 32
  context = "Hello, I'm a language model,"

  prompts = [context]*num_return_sequences
  out = model.generate(prompts, max_gen_len, device=device)
  for i, sentence in enumerate(out):
    print(f'sample {i+1}:', sentence)
    print()
