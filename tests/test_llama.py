# PYTHONPATH=. python -m unittest tests/test_llama.py

# https://huggingface.co/docs/transformers/en/model_doc/llama
#
# You can find all the original Llama checkpoints under the Huggy Llama organization.

import os
import unittest

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from models.helpers import set_device, Context, TESTING_MINIMAL
from models.llama.generate import generate, Llama

DEVICE = set_device()
MAX_SEQ_LEN = 48


def huggingface_run(prompts: list[str], model_desc: str="7B"):
  model_desc = {'33B': '30B','70B': '65B'}.get(model_desc, model_desc)
  repo_id = f'huggyllama/llama-{model_desc.lower()}'
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  inputs = tokenizer(prompts, return_tensors="pt")
  input_tokens = inputs["input_ids"].tolist()
  inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
  model = LlamaForCausalLM.from_pretrained(repo_id).to(DEVICE, dtype=torch.float16)
  # model.generation_config.pad_token_id = model.config.eos_token_id
  outputs = model.generate(**inputs, max_length=MAX_SEQ_LEN, do_sample=False, temperature=None, top_p=None)
  output_tokens = outputs.tolist()
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return input_tokens, output_tokens, texts


def self_run(prompts: list[str], model_desc: str="7B"):
  max_seq_len = MAX_SEQ_LEN
  max_batch_size = 1
  generator = Llama.from_pretrained(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, model_desc=model_desc
  ).to(DEVICE)
  model, tokenizer, max_seq_len, max_batch_size, pad_id = generator.args
  inputs = [tokenizer.encode(s, bos=True, eos=False) for s in prompts]
  outputs = generate(model, max_seq_len, max_batch_size, pad_id, inputs, max_gen_len=max_seq_len, temperature=0.0)
  outputs_ = [i[i.index(tokenizer.bos_id)+1:]for i in outputs]
  texts = [tokenizer.decode(toks) for toks in outputs_]
  return inputs, outputs, texts


class TestLlamaGreedy(unittest.TestCase):
  def setUp(self):
    self.prompts = ["The theory of relativity states that"]
    self.inputs_target = [[1, 450, 6368, 310, 14215, 537, 5922, 393]]

    self.target = {
      "7B": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[1, 450, 6368, 310, 14215, 537, 5922, 393, 278, 6210, 310, 3578, 338, 278, 1021, 363, 599, 5366, 874, 29892, 17126, 310, 1009, 6198, 10884, 470, 310, 278, 10884, 310, 278, 2752, 29889, 910, 2794, 393, 278, 6210, 310, 3578, 338, 4868, 297, 599, 297, 814, 616, 16608]],
        "completion_target": ["The theory of relativity states that the speed of light is the same for all observers, regardless of their relative motion or of the motion of the source. This means that the speed of light is constant in all inertial frames"]
      },
    }

  def _check_output(self, inputs, outputs, completion, inputs_target, outputs_target, completion_target):
    self.assertEqual(inputs_target, inputs, "input tokens do not match")
    self.assertEqual(outputs_target, outputs, "output tokens do not match")
    self.assertEqual(completion_target, completion, "completion does not match")

  def test_llama_7B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])

  def test_llama_7B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])

  @unittest.skipIf(bool(TESTING_MINIMAL), "testing minimal")
  def test_llama_7B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])


if __name__ == "__main__":
  unittest.main()
