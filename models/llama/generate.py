from typing import List
from tqdm import tqdm
import torch
from torch import Tensor
from models.llama.transformer import Transformer
from models.llama.tokenizer import Tokenizer
from models.llama.config import LlamaConfig
from models.llama.load import build

class Llama:
  def __init__(self, model: Transformer, tokenizer: Tokenizer, config: LlamaConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, model_desc: str='7B'):
    model, tokenizer, config = build(max_seq_len, max_batch_size, model_desc)
    return Llama(model, tokenizer, config)

  def text_completion(self, prompts: List[str], max_gen_len: int,
                      temperature: float=0.8, top_p: float=0.95) -> List[str]:
    return text_completion(self, prompts, max_gen_len, temperature, top_p)


@torch.inference_mode()
def generate(generator: Llama, prompt_tokens: List[List[int]],
             max_gen_len: int, temperature: float=0.8, top_p: float=0.95) -> List[List[int]]:
  model, tokenizer = generator.model, generator.tokenizer
  max_seq_len, device = generator.config.max_seq_len, generator.device
  min_prompt_size = min([len(t) for t in prompt_tokens])
  max_prompt_size = max([len(t) for t in prompt_tokens])
  total_len = min(max_seq_len, max_gen_len + max_prompt_size)
  tokens = torch.full(
    (len(prompt_tokens), total_len), tokenizer.pad_id, device=device, dtype=torch.long)
  for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t, device=device, dtype=torch.long)
  input_text_mask = tokens != tokenizer.pad_id
  prev_pos = 0
  model.eval()
  for cur_pos in tqdm(range(min_prompt_size, total_len), desc='Generating tokens'):
    with torch.no_grad():
      logits = model(tokens[:, prev_pos:cur_pos], prev_pos)
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
def text_completion(generator: Llama, prompts: List[str], max_gen_len: int,
                    temperature: float=0.8, top_p: float=0.95) -> List[str]:
  tokenizer, max_batch_size = generator.tokenizer, generator.config.max_batch_size
  bsz = len(prompts)
  assert bsz <= max_batch_size, (bsz, max_batch_size)
  prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
  tokens = generate(generator, prompt_tokens, max_gen_len, temperature, top_p)
  decoded = []
  for i, t in enumerate(tokens):
    t = t[: len(prompt_tokens[i]) + max_gen_len] # cut to max gen len
    try: t = t[: t.index(tokenizer.eos_id)] # cut to eos tok if any
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
