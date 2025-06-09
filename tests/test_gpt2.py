# PYTHONPATH=. python -m unittest tests/test_gpt2.py

import os
import unittest

from transformers import AutoTokenizer, GPT2LMHeadModel
from models.helpers import set_device, Context, TESTING_MINIMAL
from models.gpt2.generate import GPT2, generate

DEVICE = set_device()

def huggingface_run(prompts: list[str], model_desc: str="gpt2"):
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(model_desc)
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  inputs = tokenizer(prompts, return_tensors="pt")
  input_tokens = inputs["input_ids"].tolist()
  inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
  model = GPT2LMHeadModel.from_pretrained(model_desc).to(DEVICE)
  # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
  model.generation_config.pad_token_id = model.config.eos_token_id
  outputs = model.generate(**inputs, max_length=30, do_sample=False, temperature=None, top_p=None)
  output_tokens = outputs.tolist()
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return input_tokens, output_tokens, texts

def self_run(prompts: list[str], model_desc: str="gpt2"):
  max_seq_len = 30
  generator = GPT2.from_pretrained(model_desc).to(DEVICE)
  tokenizer = generator.tokenizer
  tokenizer.pad_id = tokenizer.eos_id
  inputs = tokenizer.encode_batch(prompts)
  max_new_tokens = max_seq_len - max([len(i) for i in inputs])
  outputs = generate(generator, inputs, max_new_tokens, temperature=0.0)
  texts = tokenizer.decode_batch(outputs)
  return inputs, outputs, texts


class TestGPT2Greedy(unittest.TestCase):
  def setUp(self):
    self.prompts = ["The theory of relativity states that"]
    self.inputs_target = [[464, 4583, 286, 44449, 2585, 326]]

    self.target = {
      "gpt2": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[464, 4583, 286, 44449, 2585, 326, 262, 2866, 286, 1657, 318, 27111, 284, 262, 5253, 1022, 734, 2173, 13, 383, 2866, 286, 1657, 318, 27111, 284, 262, 5253, 1022, 734]],
        "completion_target": ["The theory of relativity states that the speed of light is proportional to the distance between two points. The speed of light is proportional to the distance between two"]
      },
      "gpt2-medium": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[464, 4583, 286, 44449, 2585, 326, 262, 2866, 286, 1657, 318, 6937, 11, 290, 326, 262, 2866, 286, 1657, 318, 262, 2866, 286, 1657, 13, 383, 2866, 286, 1657, 318]],
        "completion_target": ["The theory of relativity states that the speed of light is constant, and that the speed of light is the speed of light. The speed of light is"]
      },
    }

  def _check_output(self, inputs, outputs, completion, inputs_target, outputs_target, completion_target):
    self.assertEqual(inputs_target, inputs, "input tokens do not match")
    self.assertEqual(outputs_target, outputs, "output tokens do not match")
    self.assertEqual(completion_target, completion, "completion does not match")

  def test_gpt2_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="gpt2")
    self._check_output(inputs, outputs, completion, **self.target["gpt2"])

  def test_gpt2_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="gpt2")
    self._check_output(inputs, outputs, completion, **self.target["gpt2"])

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_gpt2_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="gpt2")
    self._check_output(inputs, outputs, completion, **self.target["gpt2"])

  def test_gpt2_medium_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="gpt2-medium")
    self._check_output(inputs, outputs, completion, **self.target["gpt2-medium"])

  def test_gpt2_medium_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="gpt2-medium")
    self._check_output(inputs, outputs, completion, **self.target["gpt2-medium"])

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_gpt2_medium_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="gpt2-medium")
    self._check_output(inputs, outputs, completion, **self.target["gpt2-medium"])


if __name__ == "__main__":
  unittest.main()
