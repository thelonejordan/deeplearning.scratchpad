from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from models.helpers import Generator, timeit, SAFETENSORS
from models.gpt2.load import build, ModelOptions
from models.gpt2.config import GPTConfig
from models.gpt2.tokenizer import Tokenizer
from models.gpt2.transformer import Transformer


class GPT2(Transformer, Generator):
  def __init__(self, *args, **kwargs):
    assert "config" in kwargs and "tokenizer" in kwargs
    self.config: GPTConfig = kwargs.pop("config")
    self.tokenizer: Tokenizer = kwargs.pop("tokenizer")
    super().__init__(*args, **kwargs)

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(model_desc: ModelOptions='gpt2'):
    generator: GPT2 = build(model_desc, safetensors=bool(SAFETENSORS), model_class=GPT2)
    return generator

  @property
  def G(self): return GPT2Generator(self, self.tokenizer, self.config.n_ctx, self.tokenizer.pad_id)

  def text_completion(self, prompts: list[str], max_new_tokens: int,
                      temperature: float=1.0, top_k: int=0, top_p: float=1.0):
    return text_completion(self.G, prompts, max_new_tokens, temperature, top_k, top_p)


@dataclass
class GPT2Generator:
  model: GPT2
  tokenizer: Tokenizer
  context_len: int
  pad_id: int


@torch.inference_mode()
def generate(generator: GPT2Generator, prompt_tokens: list[list[int]], max_new_tokens: int,
             temperature: float=1.0, top_k: int=0, top_p: float=1.0):
  model, n_ctx, pad_id = generator.model, generator.context_len, generator.pad_id
  device = model.device
  min_prompt_size = min([len(t) for t in prompt_tokens])
  max_prompt_size = max([len(t) for t in prompt_tokens])
  total_len = min(n_ctx, max_prompt_size + max_new_tokens)
  batch = torch.full(
    (len(prompt_tokens), total_len), pad_id, dtype=torch.long, device=device)
  mask = torch.ones_like(batch, dtype=torch.long, device=device)
  for i, toks in enumerate(prompt_tokens):
    batch[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
    mask[i, len(toks):] = 0
  model.eval()
  cur_pos = min_prompt_size
  while cur_pos < total_len:
    context = batch[:,:cur_pos] if cur_pos<=n_ctx else batch[:, -n_ctx:]
    with torch.no_grad():
      logits = model.forward(context)
    if temperature > 0:
      logits.div_(temperature)
      logits = top_k_logits(logits, top_k)
      logits = top_p_logits(logits, top_p)
    else:
      indices = logits.argmax(-1, keepdim=True)
      logits = torch.full_like(
        logits, -1e10, dtype=logits.dtype, device=device).scatter(-1, indices, logits)
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs[:, -1, :], num_samples=1)
    batch[:,[cur_pos]] = torch.where(
      mask[:, [cur_pos]] > 0.5, batch[:,[cur_pos]], next_token)
    cur_pos += 1
  return batch.tolist()


@torch.inference_mode()
def text_completion(generator: GPT2Generator, prompts: list[str], max_new_tokens: int,
                    temperature: float=1.0, top_k: int=0, top_p: float=1.0):
  tokenizer = generator.tokenizer
  prompt_tokens = tokenizer.encode_batch(prompts)
  completions = generate(generator, prompt_tokens, max_new_tokens, temperature, top_k, top_p)
  return tokenizer.decode_batch(completions)


def top_p_logits(logits: Tensor, p: float):
  """Nucleus sampling"""
  sorted_logits = logits.sort(dim=-1, descending=True).values
  cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
  counts = (cumulative_probs <= p).long().sum(dim=-1, keepdim=True)
  indices = torch.maximum(counts - 1, torch.zeros_like(counts)).long()
  min_values = torch.gather(sorted_logits, -1, indices)
  return torch.where(
    logits < min_values,
    torch.full_like(logits, -1e10, dtype=logits.dtype, device=logits.device),
    logits)


def top_k_logits(logits: Tensor, k: int):
  if k == 0:
    # no truncation
    return logits
  values, _ = torch.topk(logits, k=k)
  min_values = values[:, :, [-1]]
  return torch.where(
    logits < min_values,
    torch.full_like(logits, -1e10, dtype=logits.dtype, device=logits.device),
    logits)
