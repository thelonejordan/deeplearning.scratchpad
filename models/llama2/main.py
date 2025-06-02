# PYTHONPATH=. python3 models/llama2/main.py

# https://github.com/meta-llama/llama-models
# https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py

# https://arxiv.org/abs/2307.09288
# https://huggingface.co/blog/llama2
# https://llama.meta.com/llama2/
# https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
# https://ai.meta.com/blog/large-language-model-llama-meta-ai/
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/

from models.helpers import set_device, set_seed, BASE
from models.llama2.generate import Llama

def base_example():

  device = set_device()
  set_seed(device)

  generator = Llama.from_pretrained(max_batch_size=2).to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "If Google was an Italian company founded in Milan, it would",
  ]

  out = generator.text_completion(prompts, max_gen_len=64, echo=True)
  assert len(out) == len(prompts)
  print('-' * 50)
  for item in out:
    text = item['generation']
    print(text)
    print('-' * 50)


def instruct_example():

  device = set_device()
  set_seed(device)

  generator = Llama.from_pretrained(max_batch_size=2, chat=True).to(device)

  system_prompt = "Answer concisely in not more than 3 lines."
  dialogs = [
    [dict(role="system", content=system_prompt), dict(role="user", content="What is theory of relativity?")],
    [dict(role="system", content=system_prompt), dict(role="user", content="Tell me about the phenomenon of global warming.")],
  ]

  out = generator.chat_completion(dialogs, max_gen_len=128)
  assert len(out) == len(dialogs)
  print('-' * 50)
  for item in out:
    text = item['generation']["content"]
    print(text)
    print('-' * 50)


if __name__ == "__main__":
  if bool(BASE): base_example()
  else: instruct_example()
