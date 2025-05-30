# PYTHONPATH=. python3 models/llama3/main.py

# https://github.com/meta-llama/llama3/blob/main/llama/model.py (for Llama3.1)
# https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py (for Llama3.2)

from models.llama3.generate import Llama
from models.helpers import set_device, set_seed

def main():

  device = set_device()
  set_seed(device)

  generator: Llama = Llama.from_pretrained(max_batch_size=2, model_desc='3B', version='2')
  generator = generator.to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "The phenomenon of global warming refers to the",
  ]

  out = generator.text_completion(prompts, max_gen_len=64, temperature=0.9, echo=True)
  assert len(out) == len(prompts)
  print('-' * 50)
  for item in out:
    text = item['generation']
    print(text)
    print('-' * 50)


if __name__ == "__main__":
  main()
