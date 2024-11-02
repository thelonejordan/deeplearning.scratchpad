from typing import Optional, List

import torch
from torch.nn import functional as F
from models.gpt2.load import from_pretrained

from models.gpt2.tokenizer import Tokenizer
from models.gpt2.transformer import Transformer


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
  def from_pretrained(model_desc: str='gpt2'):
    model, tokenizer = from_pretrained(model_desc)
    return GPT2(model, tokenizer)

  def text_completion(self, prompts: str | List[str], max_new_tokens: int,
                 temperature: float=1.0, top_k: Optional[int]=None):
    return text_completion(self, prompts, max_new_tokens, temperature, top_k)


@torch.inference_mode()
def generate(generator: GPT2, prompt_tokens: List[List[int]],
               max_new_tokens: int, temperature: float=1.0, top_k: Optional[int]=None):
  model, tokenizer = generator.model, generator.tokenizer
  config, device = model.config, generator.device
  batch, masks = [], []
  start_pos = max_new_tokens
  for toks in prompt_tokens:
    mask = [1 for _ in range(len(toks))]
    start_pos = min(start_pos, len(toks))
    if len(toks) < max_new_tokens:
      rem = max_new_tokens - len(toks)
      toks.extend([tokenizer.eot_token for _ in range(rem)])
      mask.extend([0 for _ in range(rem)])
    batch.append(toks)
    masks.append(mask)
  assert all(len(toks) < config.n_ctx for toks in batch)
  batch = torch.tensor(batch, dtype=torch.long, device=device)
  mask = torch.tensor(masks, dtype=torch.long, device=device)
  model.eval()
  cur_pos = start_pos
  while cur_pos < max_new_tokens:
    context = batch[:,:cur_pos] if cur_pos<=config.n_ctx else batch[:, -config.n_ctx:]
    with torch.no_grad():
      logits = model(context) / temperature
    if top_k is not None and top_k < config.vocab_size:
      assert top_k > 0
      _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
      logits.scatter_(-1, topk_indices, -float('inf'))
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs[:, -1, :], num_samples=1)
    batch[:,[cur_pos]] = torch.where(mask[:, [cur_pos]]>0.5, batch[:,[cur_pos]], next_token)
    cur_pos += 1
  return batch.tolist()


@torch.inference_mode()
def text_completion(generator: GPT2, prompts: str | List[str],
               max_new_tokens: int, temperature: float=1.0, top_k: Optional[int]=None):
  tokenizer = generator.tokenizer
  if isinstance(prompts, str): prompts = [prompts]
  prompt_tokens = tokenizer.encode_batch(prompts)
  completions = generate(generator, prompt_tokens, max_new_tokens, temperature, top_k)
  return tokenizer.decode_batch(completions)
