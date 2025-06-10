# TRANSFORMERS_VERBOSITY=info PYTHONPATH=. python models/qwen2/main.py

# https://huggingface.co/Qwen/QwQ-32B-Preview
# https://qwenlm.github.io/blog/qwq-32b-preview/
# https://huggingface.co/Qwen/QwQ-32B
# https://qwenlm.github.io/blog/qwq-32b/

from models.helpers import set_device, set_seed
from models.qwen2.generate import Qwen

def instruct_example():

  device = set_device()
  set_seed(device)

  generator = Qwen.from_pretrained(
    max_seq_len=512, max_batch_size=1, model_desc="qwq", model_size="32B", preview=True, instruct=True,
  ).to(device)

  dialogs = [
    [
      dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
      dict(role="user", content="How many r in strawberry.")
    ],
  ]

  # HACK: AttributeError: Qwen2TokenizerFast has no attribute pad_id (File "/workspace/deeplearning.scratchpad/models/llama2/generate.py")
  generator.tokenizer.pad_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['pad_token'])[0]
  generator.tokenizer.eos_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['eos_token'])[0]
  out = generator.chat_completion(dialogs)
  assert len(out) == len(dialogs)
  print('-' * 50)
  for item in out:
    text = item['generation']["content"]
    print(text)
    print('-' * 50)


if __name__ == "__main__":
  instruct_example()
