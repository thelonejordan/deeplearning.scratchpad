# PYTHONPATH=. python3 models/llama2/main.py

# https://github.com/meta-llama/llama-models
# https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py

# https://arxiv.org/abs/2307.09288
# https://huggingface.co/blog/llama2
# https://llama.meta.com/llama2/
# https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/

from models.helpers import set_device, set_seed
from models.llama2.generate import Llama

def main():

  device = set_device('cpu') # hardcode, as MPS OOMs
  set_seed(device)

  model = Llama.from_pretrained(max_batch_size=2).to(device)

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
