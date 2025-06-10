# TRANSFORMERS_VERBOSITY=info PYTHONPATH=. python models/qwen2/main.py

# https://huggingface.co/Qwen/QwQ-32B-Preview
# https://qwenlm.github.io/blog/qwq-32b-preview/
# https://huggingface.co/Qwen/QwQ-32B
# https://qwenlm.github.io/blog/qwq-32b/

from models.helpers import set_device, set_seed, CHAT
from models.qwen2.generate import Qwen

def fixup_tokenizer(tokenizer):
  # HACK: AttributeError: Qwen2TokenizerFast has no attribute pad_id (File "/workspace/deeplearning.scratchpad/models/llama2/generate.py")
  tokenizer.pad_id = tokenizer.encode(tokenizer.special_tokens_map['pad_token'])[0]
  tokenizer.eos_id = tokenizer.encode(tokenizer.special_tokens_map['eos_token'])[0]
  return tokenizer


def base_example():

  device = set_device()
  set_seed(device)

  generator = Qwen.from_pretrained(max_batch_size=2, model_desc='2.5', model_size='0.5B', instruct=False)
  generator = generator.to(device)
  generator.tokenizer = fixup_tokenizer(generator.tokenizer)

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

  generator = Qwen.from_pretrained(
    max_seq_len=512, max_batch_size=1, model_desc="qwq", model_size="32B", preview=True, instruct=True,
  ).to(device)
  generator.tokenizer = fixup_tokenizer(generator.tokenizer)

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
