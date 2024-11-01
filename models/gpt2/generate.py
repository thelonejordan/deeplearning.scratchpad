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

  def generate(self, prompt: str, max_new_tokens: int, num_return_sequences: int=1,
               temperature: float=1.0, top_k: Optional[int]=None):
    return generate(self, prompt, max_new_tokens, num_return_sequences, temperature, top_k)

  def completion(self, prompts: str | List[str], max_new_tokens: int,
                 temperature: float=1.0, top_k: Optional[int]=None):
    return completion(self, prompts, max_new_tokens, temperature, top_k)


@torch.inference_mode()
def generate(generator: GPT2, prompt: str, max_new_tokens: int,
             num_return_sequences: int=1, temperature: float=1.0, top_k: Optional[int]=None):
  model, tokenizer = generator.model, generator.tokenizer
  config, device = model.config, generator.device
  idx = tokenizer.encode_batch(prompt, device=device)
  assert idx.size(0) == 1 and num_return_sequences >= 1 and temperature > 0.0
  idx = idx.repeat(num_return_sequences, 1)
  model.eval()
  while idx.size(1) < max_new_tokens:
    idx_cond = idx if idx.size(1)<=config.block_size else idx[:, -config.block_size:]
    with torch.no_grad():
      logits = model(idx_cond) / temperature
    if top_k is not None and top_k < config.vocab_size:
      assert top_k > 0
      _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
      logits.scatter_(-1, topk_indices, -float('inf'))
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
    idx = torch.cat((idx, idx_next), dim=-1)
  return tokenizer.decode_batch(idx.tolist())

@torch.inference_mode()
def completion(generator: GPT2, prompts: str | List[str],
               max_new_tokens: int, temperature: float=1.0, top_k: Optional[int]=None):
  model, tokenizer = generator.model, generator.tokenizer
  config, device = model.config, generator.device
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
  idx = torch.tensor(idxs, dtype=torch.long, device=device)
  mask = torch.tensor(masks, dtype=torch.long, device=device)
  model.eval()
  cur_pos = start_pos
  while cur_pos < max_new_tokens:
    idx_cond = idx[:,:cur_pos] if cur_pos<=config.block_size else idx[:, -config.block_size:]
    with torch.no_grad():
      logits = model(idx_cond) / temperature
    if top_k is not None and top_k < config.vocab_size:
      assert top_k > 0
      _, topk_indices = torch.topk(logits, config.vocab_size - top_k, largest=False)
      logits.scatter_(-1, topk_indices, -float('inf'))
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
    idx[:,[cur_pos]] = torch.where(mask[:, [cur_pos]]>0.5, idx[:,[cur_pos]], idx_next)
    cur_pos += 1
  return tokenizer.decode_batch(idx.tolist())
