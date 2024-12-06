from typing import List

import torch
import torch.nn.functional as F

from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_nonrolling.transformer import Transformer

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
