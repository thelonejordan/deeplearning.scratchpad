# PYTHONPATH=. python3 models/mistral_rolling/main.py

# https://github.com/mistralai/mistral-inference/blob/147c4e68279b90eb61b19bdea44e16f5539d5a5d/one_file_ref.py

from models.helpers import set_device, set_seed
from models.mistral_rolling.generate import Mistral


def main():

  device = set_device()
  set_seed(device)

  version = '1'
  generator = Mistral.from_pretrained(version, max_seq_len=36, max_batch_size=4, device=device)

  max_tokens: int = 36
  context = [
    "Quantum computing is",
    "Simply put, the theory of relativity states that",
    "SpaceX and NASA have collaborated to make commercial",
  ]
  res, _logprobs = generator.generate(context, max_tokens=max_tokens)
  print('-' * 50)
  for x in res:
    print(x)
    print('-' * 50)


if __name__ == "__main__":
  main()
