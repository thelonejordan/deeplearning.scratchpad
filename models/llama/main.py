# PYTHONPATH=. python3 models/llama/main.py

# https://arxiv.org/abs/2302.13971
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py (57b0eb62de0636e75af471e49e2f1862d908d9d8)
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py

from typing import List
from tqdm import tqdm
from models.helpers import timeit

import torch
from models.llama.tokenizer import Tokenizer
from models.llama.transformer import Transformer
from models.llama.load import from_pretrained
from models.llama.generate import generate

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
  def from_pretrained(model_type: str='7B', half=False, assign: bool=False):
    model, tokenizer = from_pretrained(model_type, half, assign)
    return Llama(model, tokenizer)

  def generate(self, prompts: List[str], max_gen_len: int, temperature: float=0.8, top_p: float=0.95) -> List[str]:
    return generate(self.model, self.tokenizer, self.device, prompts, max_gen_len, temperature, top_p)


if __name__ == "__main__":
  from models.helpers import set_device, set_seed
  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained('7B', assign=True).to(device)

  num_return_sequences = 4
  max_gen_len = 32
  context = "Hello, I'm a language model,"

  prompts = [context] * num_return_sequences
  out = model.generate(prompts, max_gen_len)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence)
    print('-'*50)
