from __future__ import annotations
from typing import Optional, List, Tuple
import os

import torch
import torch.nn.functional as F

from models.helpers import Generator, timeit
from models.llama3.tokenizer import Tokenizer
from models.llama3.transformer import Transformer
from models.llama3.config import LlamaConfig
from models.llama3.load import build, ModelOptions, VersionOptions
from models.llama.generate import sample_top_p
from models.llama2.generate import CompletionPrediction

TORCH = os.getenv("TORCH", "0") == "1"

class Llama(Generator):
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: LlamaConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8,
                      model_desc: ModelOptions='8B', version: VersionOptions=0, instruct: bool=False) -> Llama:

    model, tokenizer, config = build(
      max_seq_len, max_batch_size, model_desc=model_desc, version=version, instruct=instruct, safetensors=not TORCH)
    return Llama(model, tokenizer, config)

  def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool=False, echo: bool=False):
    return text_completion(self, prompts, temperature, top_p, max_gen_len, logprobs, echo)


@torch.inference_mode()
def generate(generator: Llama, prompt_tokens: List[List[int]], max_gen_len: int, temperature: float=0.6, top_p: float=0.9,
             logprobs: bool=False, echo: bool=False) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
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
  model, tokenizer, device = generator.model, generator.tokenizer, generator.device
  max_batch_size, max_seq_len = generator.config.max_batch_size, generator.config.max_seq_len
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
  if min_prompt_len == total_len:
    logits = model.forward(tokens, prev_pos)
    token_logprobs = -F.cross_entropy(
      input=logits.transpose(1, 2),
      target=tokens,
      reduction="none",
      ignore_index=pad_id,
    )

  stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device=device)

  for cur_pos in range(min_prompt_len, total_len):
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
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
    if logprobs:
      token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
        input=logits.transpose(1, 2),
        target=tokens[:, prev_pos + 1 : cur_pos + 1],
        reduction="none",
        ignore_index=pad_id,
      )
    eos_reached |= (~input_text_mask[:, cur_pos]) & (
      torch.isin(next_token, stop_tokens)
    )
    prev_pos = cur_pos
    if all(eos_reached):
      break

  if logprobs:
    token_logprobs = token_logprobs.tolist()  # type: ignore
  out_tokens, out_logprobs = [], []
  for i, toks in enumerate(tokens.tolist()):
    # cut to max gen len
    start = 0 if echo else len(prompt_tokens[i])
    toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
    if logprobs:
      probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
    # cut to after eos tok if any
    for stop_token in tokenizer.stop_tokens:
      try:
        eos_idx = toks.index(stop_token)
        toks = toks[:eos_idx]
        if logprobs:
          probs = probs[:eos_idx]
      except ValueError:
        pass
    out_tokens.append(toks)
    out_logprobs.append(probs.tolist())
  return out_tokens, (out_logprobs if logprobs else None)


def text_completion(generator: Llama, prompts: List[str], temperature: float=0.6, top_p: float=0.9,
                    max_gen_len: Optional[int]=None, logprobs: bool=False, echo: bool=False) -> List[CompletionPrediction]:
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
        "tokens": [tokenizer.decode([x]) for x in t],
        "logprobs": logprobs_i,
      }
      for t, logprobs_i in zip(generation_tokens, generation_logprobs)  # type: ignore
    ]
  return [{"generation": tokenizer.decode(t)} for t in generation_tokens]
