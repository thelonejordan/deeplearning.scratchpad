# PYTHONPATH=. python3 models/llama2/main.py

# https://github.com/meta-llama/llama-models
# https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py

# https://arxiv.org/abs/2307.09288
# https://huggingface.co/blog/llama2
# https://llama.meta.com/llama2/
# https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/

from typing import Optional
from tqdm import tqdm

import torch
from models.llama.tokenizer import Tokenizer
from models.llama.generate import sample_top_p
from models.llama2.transformer import Transformer
from models.llama2.load import from_pretrained
from models.llama2.generate import text_completion

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
  def from_pretrained(model_type: str='7B', chat: bool=False, half=False, assign: bool=False):
    model, tokenizer = from_pretrained(model_type, chat, half, assign)
    return Llama(model, tokenizer)

  @torch.inference_mode
  def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
    return text_completion(self.model, self.tokenizer, self.device, prompts, temperature, top_p, max_gen_len)


if __name__ == "__main__":
  from models.helpers import set_device, set_seed
  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained('7B', assign=True).to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "If Google was an Italian company founded in Milan, it would",
  ]

  out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
  assert len(out_texts) == len(prompts)
  print('-' * 50)
  for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)
