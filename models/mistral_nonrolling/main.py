# PYTHONPATH=. python3 models/mistral_nonrolling/main.py

# https://mistral.ai/news/announcing-mistral-7b/
# https://github.com/mistralai/mistral-inference/tree/v1.0.4

# NOTE: This implementation lacks sliding window attention & rolling KV cache, see cache_size

from models.helpers import set_device, set_seed
from models.mistral_nonrolling.generate import Mistral


def main():

  device = set_device()
  set_seed(device)

  model_path = "downloads/mistral-7B-v0.1"
  version = '1'
  generator = Mistral.from_pretrained(model_path, version, max_seq_len=36, max_batch_size=4, device=device)

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
