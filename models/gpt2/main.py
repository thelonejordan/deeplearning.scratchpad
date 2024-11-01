# PYTHONPATH=. python3 models/gpt2/main.py

# from paper: Language Models are Unsupervised Multitask Learners
# https://github.com/openai/gpt-2
# https://paperswithcode.com/method/gpt-2
# checkout nanoGPT by karpathy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py

from typing import Optional, List

import torch
from torch.nn import functional as F
from models.gpt2.load import from_pretrained

from models.gpt2.tokenizer import Tokenizer
from models.gpt2.transformer import Transformer
from models.gpt2.generate import generate, completion

class GPT2:
  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  def from_pretrained(model_desc: str='gpt2'):
    model, tokenizer = from_pretrained(model_desc)
    return GPT2(model, tokenizer)

  def generate(self, prompt: str, max_new_tokens: int, num_return_sequences: int=1,
               temperature: float=1.0, top_k: Optional[int]=None):
    return generate(
      self.model, self.tokenizer, self.device, prompt,
      max_new_tokens, num_return_sequences, temperature, top_k)

  def completion(self, prompts: str | List[str], max_new_tokens: int,
                 temperature: float=1.0, top_k: Optional[int]=None):
    return completion(
      self.model, self.tokenizer, self.device,prompts, max_new_tokens, temperature, top_k)


if __name__ == '__main__':
  from models.helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  model = GPT2.from_pretrained().to(device)

  print("Testing generation...")
  num_return_sequences = 8
  max_length = 32
  context = "Hello, I'm a language model,"
  out = model.generate(context, max_length, num_return_sequences, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)

  print("Testing completion...")
  max_length = 200
  context = [
    "Hello, I'm a language model,",
    "Quantum computing is",
    "SpaceX and NASA have collaborated to make commercial"
  ]
  out = model.completion(context, max_length, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)
