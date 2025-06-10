# PYTHONPATH=. python -m unittest tests/test_qwen2.py

import os
import unittest

from transformers import AutoTokenizer, Qwen2ForCausalLM
from models.qwen2.generate import Qwen
from models.llama2.generate import generate
from models.helpers import set_device

DEVICE = set_device()
MAX_SEQ_LEN = 256

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# TODO: fails with SDPA=0

class TestQwen2InstructGreedy(unittest.TestCase):
  def setUp(self):
    self.dialogs = [
      [
        dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
        dict(role="user", content="How many r in strawberry.")
      ],
    ]
    self.inputs_text_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\n"]
    self.inputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198]]
    self.outputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 785, 1372, 315, 330, 81, 1, 304, 279, 3409, 330, 495, 672, 15357, 1, 374, 220, 17, 13, 151645]]
    self.completion_target = ['system\nYou are a helpful and truthful assistant. You should think step-by-step.\nuser\nHow many r in strawberry.\nassistant\nThe number of "r" in the word "strawberry" is 2.']

    self.outputs_target_trunc = [[785, 1372, 315, 330, 81, 1, 304, 279, 3409, 330, 495, 672, 15357, 1, 374, 220, 17, 13, 151645]]
    self.completion_target_trunc = ['The number of "r" in the word "strawberry" is 2.']

  def test_qwen2_inputs_text(self):
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    assert inputs_text == self.inputs_text_target, f"{inputs_text=}"
    input_ids = tokenizer(inputs_text)["input_ids"]
    assert input_ids == self.inputs_target, f"{input_ids=}\n\n{self.inputs_target=}"
    completion = tokenizer.batch_decode(self.outputs_target, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == self.completion_target, f"{completion=}\n\n{self.completion_target=}"
    output_ids_trunc = [g[len(i):] for g, i in zip(self.outputs_target, input_ids)]
    assert output_ids_trunc == self.outputs_target_trunc, f"{output_ids_trunc=}\n\n{self.outputs_target_trunc=}"
    completion_trunc = tokenizer.batch_decode(output_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion_trunc == self.completion_target_trunc, f"{completion_trunc=}\n\n{self.completion_target_trunc=}"

  def test_qwen2_smallest_hugginface(self):
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype="auto", device_map=DEVICE)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"].tolist()
    assert input_ids == self.inputs_target, f"{input_ids=}\n\n{self.inputs_target=}"
    # do not set max_new_tokens as it takes precedence over max_length
    # UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
    generated_ids = model.generate(
      **model_inputs, max_length=MAX_SEQ_LEN, do_sample=False, temperature=None, top_p=None, top_k=None,
    )
    output_ids = generated_ids.tolist()
    assert output_ids == self.outputs_target, f"{output_ids=}"
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == self.completion_target, f"{completion=}\n\n{self.completion_target=}"
    generated_ids_trunc = [g[len(i):] for g, i in zip(generated_ids, input_ids)]
    completion_trunc = tokenizer.batch_decode(generated_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion_trunc == self.completion_target_trunc, f"{completion_trunc=}\n\n{self.completion_target_trunc=}"

  @unittest.skip("non instruct")
  def test_qwen2_smallest_self_text_completion(self):
    generator = Qwen.from_pretrained(
      max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs), repo_id="Qwen/Qwen2-0.5B",
    ).to(DEVICE)
    output_ids, _ = generate(
      generator, self.inputs_target, max_gen_len=MAX_SEQ_LEN, temperature=0,
    )
    assert output_ids == self.outputs_target_trunc, f"{output_ids=}\n\n{self.outputs_target_trunc=}"
    completion = generator.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == self.completion_target_trunc, f"{completion=}\n\n{self.completion_target_trunc=}"

  def test_qwen2_smallest_self_chat_completion(self):
    generator = Qwen.from_pretrained(
      max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs), repo_id="Qwen/Qwen2-0.5B-Instruct",
    ).to(DEVICE)
    out = generator.chat_completion(self.dialogs, temperature=0.)
    completion = [item['generation']["content"] for item in out]
    assert completion == self.completion_target_trunc, f"{completion=}\n\n{self.completion_target_trunc=}"


if __name__ == "__main__":
  unittest.main()
