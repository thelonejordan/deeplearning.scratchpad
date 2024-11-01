from __future__ import annotations
from typing import Optional, List
from tqdm import tqdm

import torch
from models.llama.tokenizer import Tokenizer
from models.llama2.transformer import Transformer
from models.llama2.load import build
from models.llama.generate import sample_top_p


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
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, model_desc: str='7B', chat: bool=False) -> Llama:
    model, tokenizer = build(max_seq_len, max_batch_size, model_desc, chat)
    return Llama(model, tokenizer)

  def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9, max_gen_len: Optional[int]=None):
    return text_completion(self, prompts, temperature, top_p, max_gen_len)


@torch.inference_mode
def text_completion(generator: Llama, prompts: List[str], temperature: float=0.6,
                    top_p: float=0.9, max_gen_len: Optional[int]=None):
  model, tokenizer = generator.model, generator.tokenizer
  args, device = model.config, generator.device
  if max_gen_len is None:
    max_gen_len = args.max_seq_len - 1
  prompt_tokens = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
  bsz = len(prompt_tokens)
  assert bsz <= args.max_batch_size, f"batch size must be less than or equal to {args.max_batch_size}"
  min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
  max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
  assert max_prompt_len <= args.max_seq_len, f"prompt length must be less than or equal to {args.max_seq_len}"
  total_len = min(args.max_seq_len, max_gen_len + max_prompt_len)
  pad_id = tokenizer.pad_id
  tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
  prev_pos = 0
  for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
  eos_reached = torch.tensor([False] * bsz, device=device)
  tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
  model.eval()
  for cur_pos in tqdm(range(min_prompt_len, total_len), desc='Generating tokens'):
    with torch.no_grad():
      logits = model(tokens[:, prev_pos:cur_pos], prev_pos)
    if temperature > 0:
      probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
      next_token = sample_top_p(probs, top_p)
    else:
      next_token = torch.argmax(logits[:, -1], dim=-1) # greedy
    prev_pos = cur_pos
    next_token = next_token.reshape(-1)
    # Only replace token if it is a padding token
    next_token = torch.where(tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token
    # EOS is reached only if we found an EOS token for a padding position
    eos_reached |= (~tokens_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    if all(eos_reached): break
  out_tokens, out_text = [], []
  for current_prompt_tokens in tokens.tolist():
    # Cut to the EOS token, if present
    if tokenizer.eos_id in current_prompt_tokens:
      eos_idx = current_prompt_tokens.index(tokenizer.eos_id)
      current_prompt_tokens = current_prompt_tokens[:eos_idx]
    out_tokens.append(current_prompt_tokens)
    out_text.append(tokenizer.decode(current_prompt_tokens))
  return out_tokens, out_text
