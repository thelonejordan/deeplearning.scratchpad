# PYTHONPATH=. python -m unittest tests/test_qwq.py

import unittest

from transformers import AutoTokenizer, Qwen2ForCausalLM
from models.qwen.generate import QwQ
from models.qwen.load import huggingface_repo_id
from models.llama2.generate import generate
from models.helpers import set_device

DEVICE = set_device()
MAX_SEQ_LEN = 256


class TestQwQChat(unittest.TestCase):
  def setUp(self):
    # system_prompt = "You are a truthful and helpful assistant."
    self.dialogs = [
      # [
      #   dict(role="system", content=system_prompt),
      #   dict(role="user", content="What is theory of relativity?")
      # ],
      # [
      #   dict(role="system", content=system_prompt),
      #   dict(role="user", content="Hi"),
      #   dict(role="assistant", content="Hello, how may I assist you?"),
      #   dict(role="user", content="Tell me about the phenomenon of global warming.")
      # ],
      [
        dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
        dict(role="user", content="How many r in strawberry.")
      ],
    ]
    self.inputs_text_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\n"]
    self.inputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198]]
    self.outputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 4416, 358, 614, 419, 3405, 25, 330, 4340, 1657, 435, 594, 525, 304, 364, 495, 672, 15357, 6, 7521, 1084, 4977, 5020, 30339, 11, 714, 358, 1366, 311, 1281, 2704, 358, 3535, 432, 12440, 13, 576, 3409, 374, 330, 495, 672, 15357, 1335, 323, 358, 1184, 311, 1760, 1246, 1657, 3039, 279, 6524, 330, 81, 1, 7952, 304, 432, 382, 5338, 11, 358, 3278, 3270, 700, 279, 3409, 311, 1490, 432, 9355, 25, 274, 2385, 3795, 7409, 2630, 1455, 5655, 3795, 3795, 12034, 382, 7039, 11, 358, 3278, 728, 1526, 1817, 6524, 825, 553, 825, 323, 1760, 279, 330, 81, 40787, 382, 24617, 448, 279, 1156, 6524, 25, 274, 1365, 429, 594, 537, 458, 435, 382, 5847, 374, 259, 1365, 537, 458, 435, 382, 12209, 435, 1365, 429, 594, 825, 435, 382, 5847, 374, 264, 1365, 537, 458, 435, 382, 54, 1365, 537, 458, 435, 382, 33, 1365, 537, 458, 435, 382, 36, 1365, 537, 458, 435, 382, 14037, 435, 1365, 429, 594, 279, 2086, 435, 382, 14037, 435, 1365, 429, 594, 279, 4843, 435, 382, 3036, 5499, 11, 379, 1365, 537, 458, 435, 382, 4416, 11, 1077, 594, 1490, 11, 358, 3003, 29994, 2326, 435, 594, 304, 330, 495, 672, 15357, 2217, 14190, 264, 9383, 11, 374, 429, 1290, 30, 6771, 752, 1990, 15934, 382, 50, 2385, 3795, 7409, 2630, 1455]]
    self.completion_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\nSo I have this question: \"How many r\'s are in \'strawberry\'?\" It seems pretty straightforward, but I want to make sure I understand it correctly. The word is \"strawberry,\" and I need to count how many times the letter \"r\" appears in it.\n\nFirst, I\'ll write out the word to see it clearly: s-t-r-a-w-b-e-r-r-y.\n\nNow, I\'ll go through each letter one by one and count the \"r\"s.\n\nStarting with the first letter: s – that\'s not an r.\n\nNext is t – not an r.\n\nThen r – that\'s one r.\n\nNext is a – not an r.\n\nW – not an r.\n\nB – not an r.\n\nE – not an r.\n\nAnother r – that\'s the second r.\n\nAnother r – that\'s the third r.\n\nAnd finally, y – not an r.\n\nSo, let\'s see, I\'ve counted three r\'s in \"strawberry.\"\n\nWait a minute, is that right? Let me double-check.\n\nS-t-r-a-w-b"]

    self.outputs_target_trunc = [out[len(inp):] for inp, out in zip(self.inputs_target, self.outputs_target)]
    self.completion_target_trunc = [out[len(inp):] for inp, out in zip(self.inputs_text_target, self.completion_target)]

  def test_qwq_inputs_text(self):
    repo_id = huggingface_repo_id()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    assert inputs_text == self.inputs_text_target, f"{inputs_text=}"
    completion = tokenizer.batch_decode(self.outputs_target, skip_special_tokens=False)
    assert completion == self.completion_target, f"{completion=}"

  def test_qwq_hugginface(self):
    repo_id = huggingface_repo_id()
    tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id())
    model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype="auto", device_map=DEVICE)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
      **model_inputs, max_length=MAX_SEQ_LEN, do_sample=False, #max_new_tokens=MAX_SEQ_LEN,
    )
    output_ids = generated_ids.tolist()
    assert output_ids == self.outputs_target, f"{output_ids=}"
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    assert completion == self.completion_target, f"{completion=}"

  def test_qwq_self_text_completion(self):
    generator = QwQ.from_pretrained(max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs)).to(DEVICE)
    generator.tokenizer.pad_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['pad_token'])[0]
    generator.tokenizer.eos_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['eos_token'])[0]
    output_ids, _ = generate(
      generator, self.inputs_target, max_gen_len=MAX_SEQ_LEN, temperature=0,
    )
    assert output_ids == self.outputs_target_trunc, f"{output_ids=}"
    # [[40, 2776, 537, 2704, 1128, 498, 2299, 10161, 1588, 13, 1446, 1053, 330, 4340, 1657, 435, 304, 72600, 1189, 358, 1744, 7196, 1052, 594, 264, 85105, 476, 2494, 13, 10696, 498, 8791, 330, 4340, 1657, 525, 304, 72600, 7521, 358, 1513, 944, 1414, 13, 358, 1513, 944, 1414, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 1492, 448, 429, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 1492, 448, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 1492, 448, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429, 13, 358, 1513, 944, 1414, 13, 358, 646, 944, 4226, 429]]
    completion = generator.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    assert completion == self.completion_target_trunc, f"{completion=}"
    # ['I\'m not sure what you\'re asking here. You said "How many r in strawberry." I think maybe there\'s a typo or something. Maybe you meant "How many are in strawberry?" I don\'t know. I don\'t know. I don\'t know. I can\'t help with that. I can\'t answer that. I don\'t know. I can\'t help with that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t help with that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that']

  def test_qwq_self_chat_completion(self):
    generator = QwQ.from_pretrained(max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs)).to(DEVICE)
    generator.tokenizer.pad_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['pad_token'])[0]
    generator.tokenizer.eos_id = generator.tokenizer.encode(generator.tokenizer.special_tokens_map['eos_token'])[0]
    out = generator.chat_completion(self.dialogs, temperature=0.)
    completion = [item['generation']["content"] for item in out]
    assert completion == self.completion_target_trunc, f"{completion=}"
    # ['I\'m not sure what you\'re asking here. You said "How many r in strawberry." I think maybe there\'s a typo or something. Maybe you meant "How many are in strawberry?" I don\'t know. I don\'t know. I don\'t know. I can\'t help with that. I can\'t answer that. I don\'t know. I can\'t help with that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t help with that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that. I don\'t know. I can\'t answer that']


if __name__ == "__main__":
  unittest.main()
