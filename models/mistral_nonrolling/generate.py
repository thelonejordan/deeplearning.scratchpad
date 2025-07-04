from __future__ import annotations
from tqdm import trange

import torch
import torch.nn.functional as F

from models.helpers import Generator, timeit, SAFETENSORS
from models.mistral_nonrolling.config import MistralConfig
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_nonrolling.transformer import Transformer
from models.mistral_nonrolling.load import build


class Mistral(Transformer, Generator):
  def __init__(self, *args, **kwargs):
    self.tokenizer: Tokenizer = kwargs.pop("tokenizer")
    self.config: MistralConfig = kwargs.pop("config")
    super().__init__(*args, **kwargs)

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(version: str, max_seq_len: int, max_batch_size: int, device: torch.device) -> Mistral:
    generator, _, __ = build(max_seq_len, max_batch_size, version=version, safetensors=bool(SAFETENSORS), model_class=Mistral)
    return generator.to(device)

  @property
  def args(self):
    return self, self.tokenizer, self.tokenizer.pad_id

  @torch.no_grad()
  def generate(self, prompts: list[str], max_tokens: int):
    return generate(*self.args, prompts, max_tokens)


@torch.no_grad()
def generate(model: Mistral, tokenizer: Tokenizer, pad_id: int, prompts: list[str], max_tokens: int):
  device = model.device
  encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
  prompt_lens = [len(x) for x in encoded_prompts]
  min_prompt_len, max_prompt_len = min(prompt_lens), max(prompt_lens)
  input_tokens = torch.full((len(prompts), max_prompt_len), pad_id, dtype=torch.long, device=device)
  for i, encoded in enumerate(encoded_prompts):
    input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
  input_mask = input_tokens != pad_id
  # pre-fill
  positions = torch.arange(0, min_prompt_len).to(device)
  logits = model.forward(input_tokens[:, :min_prompt_len], positions)
  logprobs = F.log_softmax(logits, dim=-1)
  # decode
  generated = []
  all_logprobs = [logprobs[:, :-1, :].gather(2, input_tokens[:, 1:min_prompt_len, None]).squeeze(-1)]
  cur_pos = min_prompt_len
  for _ in trange(max_tokens, desc="Generating tokens"):
    next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
    if cur_pos < input_mask.shape[1]:
      next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
    all_logprobs.append(logprobs[:, -1, :].gather(1, next_token[:, None]))
    generated.append(next_token[:, None])
    logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
    logprobs = F.log_softmax(logits, dim=-1)
    cur_pos += 1

  all_logprobs_ = torch.cat(all_logprobs, 1)
  res = []
  if max_tokens > 0:
    generated_ = torch.cat(generated, 1)
    for i, x in enumerate(encoded_prompts):
      res.append(tokenizer.decode(x[:min_prompt_len] + generated_[i].tolist()))
  return res, all_logprobs_
