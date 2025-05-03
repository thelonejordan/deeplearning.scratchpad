# CPU=1 PYTHONPATH=. python3 models/llama/main.py

# https://arxiv.org/abs/2302.13971
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py (57b0eb62de0636e75af471e49e2f1862d908d9d8)
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py

from models.helpers import set_device, set_seed
from models.llama.generate import Llama


def main():

  device = set_device()
  set_seed(device)

  model = Llama.from_pretrained(max_batch_size=4).to(device)

  num_return_sequences = 4
  context = "Hello, I'm a language model,"
  prompts = [context] * num_return_sequences

  out = model.text_completion(prompts, 32)
  print('-' * 50)
  for sentence in out:
    print(sentence)
    print('-' * 50)

def hf_main():
  from transformers import LlamaTokenizer, LlamaForCausalLM
  ckpt_path = "huggyllama/llama-7b"
  # tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
  model = LlamaForCausalLM.from_pretrained(ckpt_path, device_map="cpu")
  print(model.model.rotary_emb.rope_type)

  # generator = pipeline(model=model, tokenizer=tokenizer, device=0)
  # context = "Hello, I'm a language model,"
  # prompts = [context] * 4

  # out = generator(prompts, max_length=32)
  # print('-' * 50)
  # for sentence in out:
  #   print(sentence['generated_text'])
  #   print('-' * 50)

if __name__ == "__main__":
  main()
  # hf_main()
