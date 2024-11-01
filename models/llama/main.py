# PYTHONPATH=. python3 models/llama/main.py

# https://arxiv.org/abs/2302.13971
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py (57b0eb62de0636e75af471e49e2f1862d908d9d8)
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py

from models.helpers import set_device, set_seed
from models.llama.generate import Llama


def main():

  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained(max_batch_size=4).to(device)

  num_return_sequences = 4
  max_gen_len = 32
  context = "Hello, I'm a language model,"

  prompts = [context] * num_return_sequences
  out = model.generate(prompts, max_gen_len)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence)
    print('-'*50)


if __name__ == "__main__":
  main()
