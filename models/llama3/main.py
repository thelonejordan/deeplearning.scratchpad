# PYTHONPATH=. python3 models/llama3/main.py

# https://github.com/meta-llama/llama3/blob/main/llama/model.py

from __future__ import annotations

from models.llama3.generate import Llama
from models.helpers import set_device, set_seed

def main():

  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  # TODO: Llama-3.1-8B generation looks sketchy!
  model = Llama.from_pretrained(max_batch_size=2, model_desc='8B', version=1).to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "If Google was an Italian company founded in Milan, it would",
  ]

  out = model.text_completion(prompts, max_gen_len=64, echo=True)
  assert len(out) == len(prompts)
  print('-' * 50)
  for i in range(len(out)):
    text = out[i]['generation']
    print(f'{text}')
    print('-' * 50)


if __name__ == "__main__":
  main()
