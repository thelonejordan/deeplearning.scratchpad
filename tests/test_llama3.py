# PYTHONPATH=. python -m unittest tests/test_llama3.py

# https://huggingface.co/docs/transformers/en/model_doc/llama3#usage-tips
#
# The Llama3 models were trained using bfloat16, but the original inference uses float16.
#
# The original model uses pad_id = -1 which means that there is no padding token.
# We can’t have the same logic, make sure to add a padding token using tokenizer.add_special_tokens({"pad_token":"<pad>"})
# and resize the token embedding accordingly. You should also set the model.config.pad_token_id. The embed_tokens layer of
# the model is initialized with self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx),
# which makes sure that encoding the padding token will output zeros, so passing it when initializing is recommended.

import os
import unittest

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama3.generate import generate, Llama
from models.llama3.tokenizer import Tokenizer, ChatFormat
from models.llama3.load import _tokenizer_path, huggingface_repo_id
from models.helpers import set_device, Context

DEVICE = set_device()
MAX_SEQ_LEN = 48


def huggingface_run(prompts: list[str], model_desc: str="3B", version: str="2"):
  repo_id = huggingface_repo_id(model_desc, version)
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  inputs = tokenizer(prompts, return_tensors="pt")
  input_tokens = inputs["input_ids"].tolist()
  inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
  model = LlamaForCausalLM.from_pretrained(repo_id, torch_dtype="float16").to(DEVICE)
  # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
  model.generation_config.pad_token_id = model.config.eos_token_id
  outputs = model.generate(**inputs, max_length=MAX_SEQ_LEN, do_sample=False, temperature=None, top_p=None)
  output_tokens = outputs.tolist()
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return input_tokens, output_tokens, texts


def self_run(prompts: list[str], model_desc: str="3B", version: str="2"):
  max_seq_len = MAX_SEQ_LEN
  max_batch_size = 1
  generator = Llama.from_pretrained(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, model_desc=model_desc, version=version, force_dtype="float16"
  ).to(DEVICE)
  tokenizer = generator.tokenizer
  tokenizer.pad_id = tokenizer.eos_id
  inputs = [tokenizer.encode(s, bos=True, eos=False) for s in prompts]
  outputs, _ = generate(generator, inputs, max_gen_len=max_seq_len, temperature=0.0, echo=True)
  outputs_ = [i[i.index(tokenizer.bos_id)+1:]for i in outputs]
  texts = [tokenizer.decode(toks) for toks in outputs_]
  return inputs, outputs, texts


class TestLlama3Greedy(unittest.TestCase):
  def setUp(self):
    self.prompts = ["The theory of relativity states that"]
    self.inputs_target = [[128000, 791, 10334, 315, 1375, 44515, 5415, 430]]

    self.target = {
      "2": {
        "3B": {
          "inputs_target": self.inputs_target,
          "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 304, 682, 5905, 14418, 13, 1115, 3445, 430, 422, 499, 527, 7366, 520, 264, 6926, 4732, 11, 279, 4732, 315, 3177, 690, 2744, 387, 279, 1890, 369, 499, 13, 4452, 11, 422, 499, 527]],
          "completion_target": ["The theory of relativity states that the speed of light is constant in all reference frames. This means that if you are moving at a constant speed, the speed of light will always be the same for you. However, if you are"]
        },
        "1B": {
          "inputs_target": self.inputs_target,
          "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 279, 1890, 369, 682, 37643, 13, 1115, 374, 264, 1633, 3062, 7434, 304, 22027, 11, 1606, 433, 3445, 430, 279, 4732, 315, 3177, 374, 279, 1890, 369, 682, 37643, 11, 15851, 315, 872, 3813, 304]],
          "completion_target": ["The theory of relativity states that the speed of light is the same for all observers. This is a very important concept in physics, because it means that the speed of light is the same for all observers, regardless of their location in"]
        },
      },
      "1": {
        "8B": {
          "inputs_target": self.inputs_target,
          "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 304, 264, 29302, 13, 1115, 3445, 430, 279, 4732, 315, 3177, 374, 279, 1890, 369, 682, 37643, 11, 15851, 315, 872, 8844, 11633, 477, 279, 11633, 315, 279, 2592, 315, 279, 3177, 13, 1115]],
          "completion_target": ["The theory of relativity states that the speed of light is constant in a vacuum. This means that the speed of light is the same for all observers, regardless of their relative motion or the motion of the source of the light. This"]
        },
      },
      "0": {
        "8B": {
          "inputs_target": self.inputs_target,
          "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 11, 323, 430, 892, 323, 3634, 527, 8844, 13, 1115, 3445, 430, 279, 4732, 315, 3177, 374, 279, 1890, 369, 682, 37643, 11, 15851, 315, 872, 4124, 315, 5905, 13, 4212, 323, 3634, 527]],
          "completion_target": ["The theory of relativity states that the speed of light is constant, and that time and space are relative. This means that the speed of light is the same for all observers, regardless of their frame of reference. Time and space are"]
        },
      },
    }

  def _check_output(self, inputs, outputs, completion, inputs_target, outputs_target, completion_target):
    self.assertEqual(inputs_target, inputs, "input tokens do not match")
    self.assertEqual(outputs_target, outputs, "output tokens do not match")
    self.assertEqual(completion_target, completion, "completion does not match")

  def test_llama_3_dot_2_3B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="3B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["3B"])

  def test_llama_3_dot_2_3B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="3B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["3B"])

  def test_llama_3_dot_2_3B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="3B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["3B"])

  def test_llama_3_dot_2_1B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="1B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["1B"])

  def test_llama_3_dot_2_1B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="1B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["1B"])

  def test_llama_3_dot_2_1B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="1B", version="2")
    self._check_output(inputs, outputs, completion, **self.target["2"]["1B"])

  def test_llama_3_dot_1_8B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="8B", version="1")
    self._check_output(inputs, outputs, completion, **self.target["1"]["8B"])

  def test_llama_3_dot_1_8B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="8B", version="1")
    self._check_output(inputs, outputs, completion, **self.target["1"]["8B"])

  def test_llama_3_dot_1_8B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="8B", version="1")
    self._check_output(inputs, outputs, completion, **self.target["1"]["8B"])

  def test_llama_3_8B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="8B", version="0")
    self._check_output(inputs, outputs, completion, **self.target["0"]["8B"])

  def test_llama_3_8B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="8B", version="0")
    self._check_output(inputs, outputs, completion, **self.target["0"]["8B"])

  def test_llama_3_8B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="8B", version="0")
    self._check_output(inputs, outputs, completion, **self.target["0"]["8B"])


# https://huggingface.co/docs/transformers/main/en/chat_templating_writing

LLAMA_CHAT_TEMPLATE = r"""
{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n'}}
{%- endif %}
"""

class TestLlama3ChatFormat(unittest.TestCase):
  def setUp(self):
    system_prompt = "You are a truthful and helpful assistant."
    self.dialogs = [
      [
        dict(role="system", content=system_prompt),
        dict(role="user", content="What is theory of relativity?")
      ],
      [
        dict(role="system", content=system_prompt),
        dict(role="user", content="Hi"),
        dict(role="assistant", content="Hello, how may I assist you?"),
        dict(role="user", content="Tell me about the phenomenon of global warming.")
      ],
    ]

  def helper_test_llama3_chat_format(self, model_desc: str, version: str):
    repo_id = huggingface_repo_id(model_desc, version, instruct=True)
    tokenizer = Tokenizer(_tokenizer_path(repo_id))
    formatter = ChatFormat(tokenizer)
    tokenizer_hf = AutoTokenizer.from_pretrained(repo_id)
    tokenizer_hf.chat_template = LLAMA_CHAT_TEMPLATE
    formatted_dialogs_self = [tokenizer.decode(formatter.encode_dialog_prompt(d)) for d in self.dialogs]
    formatted_dialogs_hf = tokenizer_hf.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    assert formatted_dialogs_self == formatted_dialogs_hf, "chat format mismatch"

  def test_llama3_dot_2_chat_format(self):
    self.helper_test_llama3_chat_format(model_desc="1B", version="2")

  def test_llama3_dot_1_chat_format(self):
    self.helper_test_llama3_chat_format(model_desc="8B", version="1")

  def test_llama3_dot_1_chat_format(self):
    self.helper_test_llama3_chat_format(model_desc="8B", version="0")


if __name__ == "__main__":
  unittest.main()
