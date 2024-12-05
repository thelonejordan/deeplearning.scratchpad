from __future__ import annotations
from typing import Optional, List, Tuple, TypedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from models.llama.tokenizer import Tokenizer
from models.llama2.transformer import Transformer
from models.llama.config import LlamaConfig
from models.llama2.load import build
from models.llama.generate import sample_top_p


class Llama:
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: LlamaConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, model_desc: str='7B', chat: bool=False) -> Llama:
    model, tokenizer, config = build(max_seq_len, max_batch_size, model_desc, chat)
    return Llama(model, tokenizer, config)

  def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool = False, echo: bool = False):
    return text_completion(self, prompts, temperature, top_p, max_gen_len, logprobs, echo)


@torch.inference_mode()
def generate(generator: Llama, prompt_tokens: List[List[int]], max_gen_len: int, temperature: float = 0.6,
    top_p: float = 0.9, logprobs: bool = False, echo: bool = False) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    """
    Generate text sequences based on provided prompts using the language generation model.

    Args:
      prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
      max_gen_len (int): Maximum length of the generated text sequence.
      temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
      top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
      logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
      echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
      Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    Note:
      This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
      If logprobs is True, token log probabilities are computed for each generated token.

    """
    model, tokenizer = generator.model, generator.tokenizer
    max_batch_size, max_seq_len = generator.config.max_batch_size, generator.config.max_seq_len
    device = generator.device
    bsz = len(prompt_tokens)
    assert bsz <= max_batch_size, (bsz, max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= max_seq_len
    total_len = min(max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    if logprobs:
      token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id
    for cur_pos in range(min_prompt_len, total_len):
      logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
      if logprobs:
        token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
          input=logits.transpose(1, 2),
          target=tokens[:, prev_pos + 1 : cur_pos + 1],
          reduction="none",
          ignore_index=pad_id,
        )
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1)

      next_token = next_token.reshape(-1)
      # only replace token if prompt has already been generated
      next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
      )
      tokens[:, cur_pos] = next_token
      eos_reached |= (~input_text_mask[:, cur_pos]) & (
        next_token == tokenizer.eos_id
      )
      prev_pos = cur_pos
      if all(eos_reached):
        break

    if logprobs:
      token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      start = 0 if echo else len(prompt_tokens[i])
      toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
      probs = None
      if logprobs:
        probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
      # cut to eos tok if any
      if tokenizer.eos_id in toks:
        eos_idx = toks.index(tokenizer.eos_id)
        toks = toks[:eos_idx]
        probs = probs[:eos_idx] if logprobs else None
      out_tokens.append(toks)
      out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)


class CompletionPrediction(TypedDict, total=False):
  generation: str
  tokens: List[str]  # not required
  logprobs: List[float]  # not required


def text_completion(generator: Llama, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9,
                    max_gen_len: Optional[int] = None, logprobs: bool = False, echo: bool = False) -> List[CompletionPrediction]:
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    tokenizer, max_seq_len = generator.tokenizer, generator.config.max_seq_len
    if max_gen_len is None:
        max_gen_len = max_seq_len - 1
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens, generation_logprobs = generate(
      generator=generator,
      prompt_tokens=prompt_tokens,
      max_gen_len=max_gen_len,
      temperature=temperature,
      top_p=top_p,
      logprobs=logprobs,
      echo=echo,
    )
    if logprobs:
      return [
        {
          "generation": tokenizer.decode(t),
          "tokens": [tokenizer.decode(x) for x in t],
          "logprobs": logprobs_i,
        }
        for t, logprobs_i in zip(generation_tokens, generation_logprobs)
      ]
    return [{"generation": tokenizer.decode(t)} for t in generation_tokens]


@torch.inference_mode
def text_completion2(generator: Llama, prompts: List[str], temperature: float=0.6,
                    top_p: float=0.9, max_gen_len: Optional[int]=None):
  model, tokenizer = generator.model, generator.tokenizer,
  max_batch_size, max_seq_len = generator.config.max_batch_size, generator.config.max_seq_len
  device = generator.device
  if max_gen_len is None:
    max_gen_len = max_seq_len - 1
  prompt_tokens = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
  bsz = len(prompt_tokens)
  assert bsz <= max_batch_size, f"batch size must be less than or equal to {max_batch_size}"
  min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
  max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
  assert max_prompt_len <= max_seq_len, f"prompt length must be less than or equal to {max_seq_len}"
  total_len = min(max_seq_len, max_gen_len + max_prompt_len)
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
