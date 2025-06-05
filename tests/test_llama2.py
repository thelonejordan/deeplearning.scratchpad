# PYTHONPATH=. python -m unittest tests/test_llama2.py

# https://huggingface.co/docs/transformers/en/model_doc/llama2#notes
#
# The original model uses pad_id = -1 to indicate a padding token.
# The Transformers implementation requires adding a padding token and resizing the token embedding accordingly.
# tokenizer.add_special_tokens({"pad_token":"<pad>"})
# # update model config with padding token
# model.config.pad_token_id
#
# It is recommended to initialize the embed_tokens layer with the following code to ensure encoding the padding token outputs zeros.
# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)
#
# The tokenizer is a byte-pair encoding model based on SentencePiece. During decoding, if the first token is the start of the
# word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.
#
# Don’t use the torch_dtype parameter in from_pretrained() if you’re using FlashAttention-2 because it only supports fp16 or bf16.
# You should use Automatic Mixed Precision, set fp16 or bf16 to True if using Trainer, or use torch.autocast.

import unittest

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama2.generate import generate, Llama
from models.llama2.tokenizer import encode_dialog_prompt, preprocess_dialog
from models.llama.tokenizer import Tokenizer
from models.llama.load import _tokenizer_path
from models.helpers import set_device, Context

DEVICE = set_device()


def huggingface_repo_id(model_desc: str="7B"):
  return f'meta-llama/Llama-2-{model_desc.lower()}-hf'

def huggingface_run(prompts: list[str], model_desc: str="7B"):
  repo_id = huggingface_repo_id(model_desc)
  # os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  inputs = tokenizer(prompts, return_tensors="pt")
  input_tokens = inputs["input_ids"].tolist()
  inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
  model = LlamaForCausalLM.from_pretrained(repo_id, torch_dtype="float16").to(DEVICE)
  # model.generation_config.pad_token_id = model.config.eos_token_id
  outputs = model.generate(**inputs, max_length=30, do_sample=False, temperature=None, top_p=None)
  output_tokens = outputs.tolist()
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  return input_tokens, output_tokens, texts


def self_run(prompts: list[str], model_desc: str="7B"):
  max_seq_len = 30
  max_batch_size = 1
  generator = Llama.from_pretrained(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, model_desc=model_desc
  ).to(DEVICE)
  tokenizer = generator.tokenizer
  inputs = [tokenizer.encode(s, bos=True, eos=False) for s in prompts]
  outputs, _ = generate(generator, inputs, max_gen_len=max_seq_len, temperature=0.0, echo=True)
  outputs_ = [i[i.index(tokenizer.bos_id)+1:]for i in outputs]
  texts = [tokenizer.decode(toks) for toks in outputs_]
  return inputs, outputs, texts


class TestLlama2Greedy(unittest.TestCase):
  def setUp(self):
    self.prompts = ["The theory of relativity states that"]
    self.inputs_target = [[1, 450, 6368, 310, 14215, 537, 5922, 393]]

    self.target = {
      "7B": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[1, 450, 6368, 310, 14215, 537, 5922, 393, 278, 6210, 310, 3578, 338, 278, 1021, 363, 599, 5366, 874, 29892, 17126, 310, 1009, 6198, 10884, 470, 1009, 3515, 310, 3407]],
        "completion_target": ["The theory of relativity states that the speed of light is the same for all observers, regardless of their relative motion or their frame of reference"]
      },
    }

  def _check_output(self, inputs, outputs, completion, inputs_target, outputs_target, completion_target):
    self.assertEqual(inputs_target, inputs, "input tokens do not match")
    self.assertEqual(outputs_target, outputs, "output tokens do not match")
    self.assertEqual(completion_target, completion, "completion does not match")

  def test_llama_2_7B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])

  def test_llama_2_7B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])

  def test_llama_2_7B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="7B")
    self._check_output(inputs, outputs, completion, **self.target["7B"])


# https://huggingface.co/docs/transformers/main/en/chat_templating_writing

LLAMA_CHAT_TEMPLATE = r"""
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST]' + ' ' + message['content'] + ' ' + '[/INST]'}}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + '  '}}
    {%- endif %}
{%- endfor %}
"""

class TestLlama2ChatFormat(unittest.TestCase):
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

  def helper_test_llama2_chat_format(self, model_desc: str):
    repo_id = huggingface_repo_id(model_desc)
    tokenizer = Tokenizer(_tokenizer_path(repo_id))
    tokenizer_hf = AutoTokenizer.from_pretrained(repo_id)
    tokenizer_hf.chat_template = LLAMA_CHAT_TEMPLATE
    formatted_dialogs_self = [tokenizer.decode(encode_dialog_prompt(tokenizer, d)) for d in self.dialogs]
    dialogs = [preprocess_dialog(d) for d in self.dialogs]
    formatted_dialogs_hf = tokenizer_hf.apply_chat_template(dialogs, tokenize=False)
    assert formatted_dialogs_self == formatted_dialogs_hf, "chat format mismatch"

  def test_llama2_chat_format(self):
    self.helper_test_llama2_chat_format(model_desc="7B")


if __name__ == "__main__":
  unittest.main()
