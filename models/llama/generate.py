from __future__ import annotations
from tqdm import trange

import torch
from torch import Tensor

from models.helpers import Generator, timeit, SAFETENSORS
from models.llama.transformer import Transformer
from models.llama.tokenizer import Tokenizer
from models.llama.config import LlamaConfig
from models.llama.load import build, ModelOptions

class Llama(Transformer, Generator):
  def __init__(self, *args, **kwargs):
    assert "config" in kwargs and "tokenizer" in kwargs
    self.config: LlamaConfig = kwargs.pop("config")
    self.tokenizer: Tokenizer = kwargs.pop("tokenizer")
    super().__init__(*args, **kwargs)

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, model_desc: ModelOptions='7B') -> Llama:
    generator, _, __ = build(max_seq_len, max_batch_size, model_desc, safetensors=bool(SAFETENSORS), model_class=Llama)
    return generator

  @property
  def G(self):
    return self, self.tokenizer, self.config.max_seq_len, self.config.max_batch_size, self.tokenizer.pad_id

  def text_completion(self, prompts: list[str], max_gen_len: int,
                      temperature: float=0.8, top_p: float=0.95) -> list[str]:
    return text_completion(*self.G, prompts, max_gen_len, temperature, top_p)


@torch.inference_mode()
def generate(model: Llama, max_seq_len: int, max_batch_size: int, pad_id: int,
             prompt_tokens: list[list[int]], max_gen_len: int, temperature: float=0.8, top_p: float=0.95) -> list[list[int]]:
  device = model.device
  bsz = len(prompt_tokens)
  assert bsz <= max_batch_size, (bsz, max_batch_size)
  min_prompt_size = min([len(t) for t in prompt_tokens])
  max_prompt_size = max([len(t) for t in prompt_tokens])
  total_len = min(max_seq_len, max_gen_len + max_prompt_size)
  tokens = torch.full(
    (len(prompt_tokens), total_len), pad_id, device=device, dtype=torch.long)
  for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t, device=device, dtype=torch.long)
  input_text_mask = tokens != pad_id
  prev_pos = 0
  model.eval()
  for cur_pos in trange(min_prompt_size, total_len, desc='Generating tokens'):
    with torch.no_grad():
      logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    if temperature > 0:
      probs = torch.softmax(logits / temperature, dim=-1)
      next_token = sample_top_p(probs, top_p)
    else:
      next_token = torch.argmax(logits, dim=-1)
    next_token = next_token.reshape(-1)
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token
    prev_pos = cur_pos
  return tokens.tolist()


@torch.inference_mode()
def text_completion(model: Llama, tokenizer: Tokenizer, max_seq_len: int, max_batch_size: int, pad_id: int,
                    prompts: list[str], max_gen_len: int, temperature: float=0.8, top_p: float=0.95) -> list[str]:
  bsz = len(prompts)
  assert bsz <= max_batch_size, (bsz, max_batch_size)
  prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
  tokens = generate(model, max_seq_len, max_batch_size, pad_id, prompt_tokens, max_gen_len, temperature, top_p)
  decoded = []
  for i, t in enumerate(tokens):
    t = t[: len(prompt_tokens[i]) + max_gen_len]  # cut to max gen len
    try: t = t[: t.index(tokenizer.eos_id)]  # cut to eos tok if any
    except ValueError: pass
    decoded.append(tokenizer.decode(t))
  return decoded


def sample_top_p(probs: Tensor, p: float) -> Tensor:
  """
  Perform top-p (nucleus) sampling on a probability distribution.

  Args:
    probs (torch.Tensor): Probability distribution tensor.
    p (float): Probability threshold for top-p sampling.

  Returns:
    torch.Tensor: Sampled token indices.

  Note:
    Top-p sampling selects the smallest set of tokens whose cumulative probability mass
    exceeds the threshold p. The distribution is renormalized based on the selected tokens.
  """
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token
