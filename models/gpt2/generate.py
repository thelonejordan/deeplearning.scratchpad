from typing import Optional, List, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from models.gpt2.load import from_pretrained

from models.gpt2.tokenizer import Tokenizer
from models.gpt2.transformer import Transformer, GPTConfig


class GPT2:
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: GPTConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(model_desc: str='gpt2'):
    model, tokenizer, config = from_pretrained(model_desc)
    return GPT2(model, tokenizer, config)

  def text_completion(self, prompts: List[str], max_new_tokens: int,
                      temperature: float=1.0, top_k: int=0, top_p: float=1.0):
    return text_completion(self, prompts, max_new_tokens, temperature, top_k, top_p)


@torch.inference_mode()
def generate(generator: GPT2, prompt_tokens: List[List[int]], max_new_tokens: int,
             temperature: float=1.0, top_k: int=0, top_p: float=1.0):
  model, tokenizer = generator.model, generator.tokenizer
  n_ctx, device = generator.config.n_ctx, generator.device
  min_prompt_size = min([len(t) for t in prompt_tokens])
  max_prompt_size = max([len(t) for t in prompt_tokens])
  total_len = min(n_ctx, max_prompt_size + max_new_tokens)
  batch = torch.full(
    (len(prompt_tokens), total_len), tokenizer.eot_token, dtype=torch.long, device=device)
  mask = torch.ones_like(batch, dtype=torch.long, device=device)
  for i, toks in enumerate(prompt_tokens):
    batch[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
    mask[i, len(toks):] = 0
  model.eval()
  cur_pos = min_prompt_size
  while cur_pos < total_len:
    context = batch[:,:cur_pos] if cur_pos<=n_ctx else batch[:, -n_ctx:]
    with torch.no_grad():
      logits: Tensor = model(context)
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
def text_completion(generator: GPT2, prompts: List[str], max_new_tokens: int,
                    temperature: float=1.0, top_k: int=0, top_p: float=1.0):
  tokenizer = generator.tokenizer
  prompt_tokens = tokenizer.encode_batch(prompts)
  completions = generate(generator, prompt_tokens, max_new_tokens, temperature, top_k, top_p)
  return tokenizer.decode_batch(completions)


def top_p_logits(logits: Tensor, p: float):
  """Nucleus sampling"""
  sorted_logits = logits.sort(dim=-1, descending=True).values
  cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
  counts = cast(Tensor, cumulative_probs <= p).long().sum(dim=-1, keepdim=True)
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
