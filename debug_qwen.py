# PYTHONPATH=. python debug_qwen.py

import os
import unittest

from transformers import AutoTokenizer, Qwen2ForCausalLM
from models.qwen2.generate import Qwen
from models.llama2.generate import generate
from models.helpers import set_device


DEVICE = set_device()
MAX_SEQ_LEN = 256 - 218

dialogs = [
  [
    dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
    dict(role="user", content="How many r in strawberry.")
  ],
]

repo_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype="auto", device_map=DEVICE)
inputs_text = tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
input_ids = model_inputs["input_ids"].tolist()
generated_ids = model.generate(
  **model_inputs, max_length=MAX_SEQ_LEN, do_sample=False, temperature=None, top_p=None, top_k=None,
)
output_ids = generated_ids.tolist()
completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
generated_ids_trunc = [g[len(i):].tolist() for g, i in zip(generated_ids, input_ids)]
completion_trunc = tokenizer.batch_decode(generated_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)

# print(generated_ids_trunc[0])

generator = Qwen.from_pretrained(
  max_seq_len=MAX_SEQ_LEN, max_batch_size=len(dialogs), repo_id="Qwen/Qwen2-0.5B-Instruct",
).to(DEVICE)
"""
out = generator.chat_completion(dialogs, temperature=0., logprobs=True)
"""
out = generator.text_completion(inputs_text, temperature=0., logprobs=True)
output_ids_trunc = []
for item in out:
  output_ids_trunc_item = []
  for token in item['tokens']:
    output_ids_trunc_item.extend(generator.tokenizer.encode(token))
  output_ids_trunc.append(output_ids_trunc_item)
# completion_trunc = [item['generation']["content"] for item in out]
completion_trunc = [item['tokens'] for item in out]


print(out)
# print(output_ids_trunc[0][34:])

cmp1 = generated_ids_trunc[0]
cmp2 = output_ids_trunc[0]

assert cmp1 == cmp2, f"\n{cmp1=}\n{cmp2=}"
