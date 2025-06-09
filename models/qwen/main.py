# TRANSFORMERS_VERBOSITY=info PYTHONPATH=. python models/qwen/main.py

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

from models.helpers import set_device, set_seed
from models.qwen.generate import QwQ

def instruct_example():

  device = set_device()
  set_seed(device)

  generator = QwQ.from_pretrained(max_seq_len=256, max_batch_size=2).to(device)

  # system_prompt = "Answer concisely in not more than 3 lines."
  dialogs = [
    # [
    #   dict(role="system", content=system_prompt),
    #   dict(role="user", content="What is theory of relativity?"),
    # ],
    # [
    #   dict(role="system", content=system_prompt),
    #   dict(role="user", content="Tell me about the phenomenon of global warming."),
    # ],
    [
      dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
      dict(role="user", content="How many r in strawberry.")
    ],
  ]

  # HACK:
  # File "/workspace/deeplearning.scratchpad/models/llama2/generate.py", line 74, in generate
  # AttributeError: Qwen2TokenizerFast has no attribute pad_id
  generator.tokenizer.pad_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['pad_token'])[0]
  generator.tokenizer.eos_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['eos_token'])[0]
  out = generator.chat_completion(dialogs, temperature=0.)
  assert len(out) == len(dialogs)
  print('-' * 50)
  for item in out:
    text = item['generation']["content"]
    print(text)
    print('-' * 50)


def huggingface_run():

  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_name = "Qwen/QwQ-32B-Preview"

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
  )

  # print(model.dtype)
  # print(model.device)

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  prompt = "How many r in strawberry."
  messages = [
    [
      {"role": "system", "content": "You are a helpful and truthful assistant. You should think step-by-step."},
      {"role": "user", "content": prompt},
    ]
  ]
  # print(tokenizer.chat_template)
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )

  # print("="*100)
  # print(text)
  # print("="*100)
  inputs_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\n"]
  assert text == inputs_target

  model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

  input_tokens = model_inputs["input_ids"].tolist()
  # print("="*100)
  # print(input_tokens)
  # print("="*100)
  input_tokens_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198]]
  assert input_tokens == input_tokens_target, "input tokens do not match"

  MAX_SEQ_LEN=256

  generated_ids = model.generate(
    **model_inputs,
    max_length=MAX_SEQ_LEN,
    # max_new_tokens=512,
    do_sample=False,
  )

  output_tokens = generated_ids.tolist()
  output_tokens_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 4416, 358, 614, 419, 3405, 25, 330, 4340, 1657, 435, 594, 525, 304, 364, 495, 672, 15357, 6, 7521, 1084, 4977, 5020, 30339, 11, 714, 358, 1366, 311, 1281, 2704, 358, 3535, 432, 12440, 13, 576, 3409, 374, 330, 495, 672, 15357, 1335, 323, 358, 1184, 311, 1760, 1246, 1657, 3039, 279, 6524, 330, 81, 1, 7952, 304, 432, 382, 5338, 11, 358, 3278, 3270, 700, 279, 3409, 311, 1490, 432, 9355, 25, 274, 2385, 3795, 7409, 2630, 1455, 5655, 3795, 3795, 12034, 382, 7039, 11, 358, 3278, 728, 1526, 1817, 6524, 825, 553, 825, 323, 1760, 279, 330, 81, 40787, 382, 24617, 448, 279, 1156, 6524, 25, 274, 1365, 429, 594, 537, 458, 435, 382, 5847, 374, 259, 1365, 537, 458, 435, 382, 12209, 435, 1365, 429, 594, 825, 435, 382, 5847, 374, 264, 1365, 537, 458, 435, 382, 54, 1365, 537, 458, 435, 382, 33, 1365, 537, 458, 435, 382, 36, 1365, 537, 458, 435, 382, 14037, 435, 1365, 429, 594, 279, 2086, 435, 382, 14037, 435, 1365, 429, 594, 279, 4843, 435, 382, 3036, 5499, 11, 379, 1365, 537, 458, 435, 382, 4416, 11, 1077, 594, 1490, 11, 358, 3003, 29994, 2326, 435, 594, 304, 330, 495, 672, 15357, 2217, 14190, 264, 9383, 11, 374, 429, 1290, 30, 6771, 752, 1990, 15934, 382, 50, 2385, 3795, 7409, 2630, 1455]]
  # print("="*100)
  # print(output_tokens)
  # print("="*100)
  assert output_tokens == output_tokens_target, "input tokens do not match"

  echo = True
  if echo:
    generated_ids = [
      output_ids for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
  else:
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
  completion_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\nSo I have this question: \"How many r\'s are in \'strawberry\'?\" It seems pretty straightforward, but I want to make sure I understand it correctly. The word is \"strawberry,\" and I need to count how many times the letter \"r\" appears in it.\n\nFirst, I\'ll write out the word to see it clearly: s-t-r-a-w-b-e-r-r-y.\n\nNow, I\'ll go through each letter one by one and count the \"r\"s.\n\nStarting with the first letter: s – that\'s not an r.\n\nNext is t – not an r.\n\nThen r – that\'s one r.\n\nNext is a – not an r.\n\nW – not an r.\n\nB – not an r.\n\nE – not an r.\n\nAnother r – that\'s the second r.\n\nAnother r – that\'s the third r.\n\nAnd finally, y – not an r.\n\nSo, let\'s see, I\'ve counted three r\'s in \"strawberry.\"\n\nWait a minute, is that right? Let me double-check.\n\nS-t-r-a-w-b"]
  # print("="*100)
  # print(response)
  # print("="*100)
  assert response == completion_target


if __name__ == "__main__":
  instruct_example()
  # huggingface_run()
