# PYTHONPATH=. python -m unittest tests/test_qwq.py

import unittest

from transformers import AutoTokenizer, Qwen2ForCausalLM
from models.qwen2.generate import Qwen
from models.llama2.generate import generate
from models.helpers import set_device, TESTING_MINIMAL

DEVICE = set_device()
MAX_SEQ_LEN = 256

# TODO: fails with SDPA=0

class TestQwQChat(unittest.TestCase):
  def setUp(self):
    self.dialogs = [
      [
        dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
        dict(role="user", content="How many r in strawberry.")
      ],
    ]
    self.inputs_text_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\n"]
    self.inputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198]]
    self.outputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 4416, 358, 614, 419, 3405, 25, 330, 4340, 1657, 435, 594, 525, 304, 364, 495, 672, 15357, 6, 7521, 1084, 4977, 5020, 30339, 11, 714, 358, 1366, 311, 1281, 2704, 358, 3535, 432, 12440, 13, 576, 3409, 374, 330, 495, 672, 15357, 1335, 323, 358, 1184, 311, 1760, 1246, 1657, 3039, 279, 6524, 330, 81, 1, 7952, 304, 432, 382, 5338, 11, 358, 3278, 3270, 700, 279, 3409, 311, 1490, 432, 9355, 25, 274, 2385, 3795, 7409, 2630, 1455, 5655, 3795, 3795, 12034, 382, 7039, 11, 358, 3278, 728, 1526, 1817, 6524, 825, 553, 825, 323, 1760, 279, 330, 81, 40787, 382, 24617, 448, 279, 1156, 6524, 25, 274, 1365, 429, 594, 537, 458, 435, 382, 5847, 374, 259, 1365, 537, 458, 435, 382, 12209, 435, 1365, 429, 594, 825, 435, 382, 5847, 374, 264, 1365, 537, 458, 435, 382, 54, 1365, 537, 458, 435, 382, 33, 1365, 537, 458, 435, 382, 36, 1365, 537, 458, 435, 382, 14037, 435, 1365, 429, 594, 279, 2086, 435, 382, 14037, 435, 1365, 429, 594, 279, 4843, 435, 382, 3036, 5499, 11, 379, 1365, 537, 458, 435, 382, 4416, 11, 1077, 594, 1490, 11, 358, 3003, 29994, 2326, 435, 594, 304, 330, 495, 672, 15357, 2217, 14190, 264, 9383, 11, 374, 429, 1290, 30, 6771, 752, 1990, 15934, 382, 50, 2385, 3795, 7409, 2630, 1455]]
    self.completion_target = ['system\nYou are a helpful and truthful assistant. You should think step-by-step.\nuser\nHow many r in strawberry.\nassistant\nSo I have this question: "How many r\'s are in\'strawberry\'?" It seems pretty straightforward, but I want to make sure I understand it correctly. The word is "strawberry," and I need to count how many times the letter "r" appears in it.\n\nFirst, I\'ll write out the word to see it clearly: s-t-r-a-w-b-e-r-r-y.\n\nNow, I\'ll go through each letter one by one and count the "r"s.\n\nStarting with the first letter: s – that\'s not an r.\n\nNext is t – not an r.\n\nThen r – that\'s one r.\n\nNext is a – not an r.\n\nW – not an r.\n\nB – not an r.\n\nE – not an r.\n\nAnother r – that\'s the second r.\n\nAnother r – that\'s the third r.\n\nAnd finally, y – not an r.\n\nSo, let\'s see, I\'ve counted three r\'s in "strawberry."\n\nWait a minute, is that right? Let me double-check.\n\nS-t-r-a-w-b']

    self.outputs_target_trunc = [[4416, 358, 614, 419, 3405, 25, 330, 4340, 1657, 435, 594, 525, 304, 364, 495, 672, 15357, 6, 7521, 1084, 4977, 5020, 30339, 11, 714, 358, 1366, 311, 1281, 2704, 358, 3535, 432, 12440, 13, 576, 3409, 374, 330, 495, 672, 15357, 1335, 323, 358, 1184, 311, 1760, 1246, 1657, 3039, 279, 6524, 330, 81, 1, 7952, 304, 432, 382, 5338, 11, 358, 3278, 3270, 700, 279, 3409, 311, 1490, 432, 9355, 25, 274, 2385, 3795, 7409, 2630, 1455, 5655, 3795, 3795, 12034, 382, 7039, 11, 358, 3278, 728, 1526, 1817, 6524, 825, 553, 825, 323, 1760, 279, 330, 81, 40787, 382, 24617, 448, 279, 1156, 6524, 25, 274, 1365, 429, 594, 537, 458, 435, 382, 5847, 374, 259, 1365, 537, 458, 435, 382, 12209, 435, 1365, 429, 594, 825, 435, 382, 5847, 374, 264, 1365, 537, 458, 435, 382, 54, 1365, 537, 458, 435, 382, 33, 1365, 537, 458, 435, 382, 36, 1365, 537, 458, 435, 382, 14037, 435, 1365, 429, 594, 279, 2086, 435, 382, 14037, 435, 1365, 429, 594, 279, 4843, 435, 382, 3036, 5499, 11, 379, 1365, 537, 458, 435, 382, 4416, 11, 1077, 594, 1490, 11, 358, 3003, 29994, 2326, 435, 594, 304, 330, 495, 672, 15357, 2217, 14190, 264, 9383, 11, 374, 429, 1290, 30, 6771, 752, 1990, 15934, 382, 50, 2385, 3795, 7409, 2630, 1455]]
    self.completion_target_trunc = ['So I have this question: "How many r\'s are in\'strawberry\'?" It seems pretty straightforward, but I want to make sure I understand it correctly. The word is "strawberry," and I need to count how many times the letter "r" appears in it.\n\nFirst, I\'ll write out the word to see it clearly: s-t-r-a-w-b-e-r-r-y.\n\nNow, I\'ll go through each letter one by one and count the "r"s.\n\nStarting with the first letter: s – that\'s not an r.\n\nNext is t – not an r.\n\nThen r – that\'s one r.\n\nNext is a – not an r.\n\nW – not an r.\n\nB – not an r.\n\nE – not an r.\n\nAnother r – that\'s the second r.\n\nAnother r – that\'s the third r.\n\nAnd finally, y – not an r.\n\nSo, let\'s see, I\'ve counted three r\'s in "strawberry."\n\nWait a minute, is that right? Let me double-check.\n\nS-t-r-a-w-b']

  def test_qwq_inputs_text(self):
    repo_id = "Qwen/QwQ-32B-Preview"
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

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_qwq_hugginface_chat_completion(self):
    repo_id = "Qwen/QwQ-32B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype="auto", device_map=DEVICE)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"].tolist()
    assert input_ids == self.inputs_target, f"{input_ids=}\n\n{self.inputs_target=}"
    # do not set max_new_tokens as it takes precedence over max_length
    generated_ids = model.generate(
      **model_inputs, max_length=MAX_SEQ_LEN, do_sample=False,
    )
    output_ids = generated_ids.tolist()
    assert output_ids == self.outputs_target, f"{output_ids=}"
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == self.completion_target, f"{completion=}\n\n{self.completion_target=}"
    generated_ids_trunc = [g[len(i):] for g, i in zip(generated_ids, input_ids)]
    completion_trunc = tokenizer.batch_decode(generated_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion_trunc == self.completion_target_trunc, f"{completion_trunc=}\n\n{self.completion_target_trunc=}"

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_qwq_self_text_completion(self):
    generator = Qwen.from_pretrained(
      max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs), repo_id="Qwen/QwQ-32B-Preview",
    ).to(DEVICE)
    output_ids, _ = generate(
      generator, self.inputs_target, max_gen_len=MAX_SEQ_LEN, temperature=0,
    )
    assert output_ids == self.outputs_target_trunc, f"{output_ids=}\n\n{self.outputs_target_trunc=}"
    completion = generator.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == self.completion_target_trunc, f"{completion=}\n\n{self.completion_target_trunc=}"

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_qwq_self_chat_completion(self):
    generator = Qwen.from_pretrained(
      max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs), repo_id="Qwen/QwQ-32B-Preview",
    ).to(DEVICE)
    out = generator.chat_completion(self.dialogs, temperature=0.)
    completion = [item['generation']["content"] for item in out]
    assert completion == self.completion_target_trunc, f"{completion=}\n\n{self.completion_target_trunc=}"


if __name__ == "__main__":
  unittest.main()
