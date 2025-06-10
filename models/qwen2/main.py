# TRANSFORMERS_VERBOSITY=info PYTHONPATH=. python models/qwen2/main.py

# https://huggingface.co/Qwen/QwQ-32B-Preview
# https://qwenlm.github.io/blog/qwq-32b-preview/
# https://huggingface.co/Qwen/QwQ-32B
# https://qwenlm.github.io/blog/qwq-32b/

from models.helpers import set_device, set_seed, CHAT
from models.qwen2.generate import Qwen

def base_example():

  device = set_device()
  set_seed(device)

  generator = Qwen.from_pretrained(max_batch_size=2, repo_id="Qwen/Qwen2.5-0.5B")
  generator = generator.to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "The phenomenon of global warming refers to the",
  ]

  out = generator.text_completion(prompts, max_gen_len=128, temperature=0.9, echo=True)
  assert len(out) == len(prompts)
  print('-' * 50)
  for item in out:
    text = item['generation']
    print(text)
    print('-' * 50)


def instruct_example():

  device = set_device()
  set_seed(device)

  generator = Qwen.from_pretrained(max_batch_size=1, repo_id="Qwen/Qwen2.5-0.5B-Instruct")
  generator = generator.to(device)

  dialogs = [
    [
      dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
      dict(role="user", content="How many r in strawberry.")
    ],
  ]

  out = generator.chat_completion(dialogs)
  assert len(out) == len(dialogs)
  print('-' * 50)
  for item in out:
    text = item['generation']["content"]
    print(text)
    print('-' * 50)


if __name__ == "__main__":
  match bool(CHAT):
    case False: base_example()
    case _: instruct_example()
