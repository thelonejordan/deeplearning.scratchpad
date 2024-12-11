from typing import List

import torch
import torch.nn.functional as F

from models.helpers import timeit
from models.mistral_nonrolling.config import MistralConfig
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_nonrolling.transformer import Transformer
from models.mistral_nonrolling.load import build


class Mistral:
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: MistralConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device
  @property
  def dtype(self) -> torch.dtype: return next(self.model.parameters()).dtype

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(folder: str, max_seq_len: int, max_batch_size: int, device: torch.device):
    model, tokenizer, config = build(folder, max_seq_len, max_batch_size)
    return Mistral(model, tokenizer, config).to(device)

  @torch.no_grad()
  def generate(self, prompts: List[str], max_tokens: int):
    return generate(self.model, self.tokenizer, self.device, prompts, max_tokens)


@torch.no_grad()
def generate(model: Transformer, tokenizer: Tokenizer, device: torch.device, prompts: List[str], max_tokens: int):
  encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
  prompt_lens = [len(x) for x in encoded_prompts]
  min_prompt_len, max_prompt_len = min(prompt_lens), max(prompt_lens)
  input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device=device)
  for i, encoded in enumerate(encoded_prompts):
    input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
  input_mask = input_tokens != tokenizer.pad_id
  # pre-fill
  positions = torch.arange(0, min_prompt_len).to(device)
  logits = model(input_tokens[:, :min_prompt_len], positions)
  logprobs = F.log_softmax(logits, dim=-1)
  # decode
  generated = []
  all_logprobs = [logprobs[:, :-1, :].gather(2, input_tokens[:, 1:min_prompt_len, None]).squeeze(-1)]
  cur_pos = min_prompt_len
  for _ in range(max_tokens):
    next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
    if cur_pos < input_mask.shape[1]:
      next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
    all_logprobs.append(logprobs[:, -1, :].gather(1, next_token[:, None]))
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
