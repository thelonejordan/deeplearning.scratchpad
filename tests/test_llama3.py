import os
import unittest

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama3.generate import generate, Llama
from models.helpers import set_device, Context

DEVICE = set_device()

def huggingface_run(prompts: list[str], model_desc="3B"):
  model_id = f"meta-llama/Llama-3.2-{model_desc}"
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  inputs = tokenizer(prompts, return_tensors="pt")
  input_tokens = inputs["input_ids"].tolist()
  inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
  model = LlamaForCausalLM.from_pretrained(model_id).to(DEVICE)
  model.generation_config.pad_token_id = model.config.eos_token_id
  outputs = model.generate(**inputs, max_length=30, do_sample=False, temperature=None, top_p=None)
  output_tokens = outputs.cpu().tolist()
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  return input_tokens, output_tokens, texts


def self_run(prompts: list[str], model_desc="3B"):
  max_seq_len = 30
  max_batch_size = 1
  generator = Llama.from_pretrained(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, model_desc=model_desc, version="2",
  ).to(DEVICE)
  tokenizer = generator.tokenizer
  tokenizer.pad_id = tokenizer.eos_id
  inputs = [tokenizer.encode(s, bos=True, eos=False) for s in prompts]
  outputs, _ = generate(generator, inputs, max_gen_len=max_seq_len, temperature=0.0, echo=True)
  outputs_ = [i[i.index(tokenizer.bos_id)+1:]for i in outputs]
  texts = [tokenizer.decode(toks) for toks in outputs_]
  return inputs, outputs, texts


class TestLlama3dot2Greedy(unittest.TestCase):
  def setUp(self):
    self.prompts = ["The theory of relativity states that"]
    self.inputs_target = [[128000, 791, 10334, 315, 1375, 44515, 5415, 430]]

    self.target = {
      "3B": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 304, 682, 5905, 14418, 13, 1115, 3445, 430, 422, 499, 527, 7366, 520, 264, 6926, 4732]],
        "completion_target": ["The theory of relativity states that the speed of light is constant in all reference frames. This means that if you are moving at a constant speed"]
      },
      "1B": {
        "inputs_target": self.inputs_target,
        "outputs_target": [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 279, 1890, 369, 682, 37643, 13, 1115, 374, 264, 1633, 3062, 7434, 304, 22027, 11, 1606, 433]],
        "completion_target": ["The theory of relativity states that the speed of light is the same for all observers. This is a very important concept in physics, because it"]
      },
    }

  def _check_output(self, inputs, outputs, completion, inputs_target, outputs_target, completion_target):
    self.assertEqual(inputs_target, inputs, "input tokens do not match")
    self.assertEqual(outputs_target, outputs, "output tokens do not match")
    self.assertEqual(completion_target, completion, "completion does not match")

  def test_llama_3_dot_2_3B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="3B")
    self._check_output(inputs, outputs, completion, **self.target["3B"])

  def test_llama_3_dot_2_3B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="3B")
    self._check_output(inputs, outputs, completion, **self.target["3B"])

  def test_llama_3_dot_2_3B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="3B")
    self._check_output(inputs, outputs, completion, **self.target["3B"])

  def test_llama_3_dot_2_1B_huggingface(self):
    inputs, outputs, completion = huggingface_run(self.prompts, model_desc="1B")
    self._check_output(inputs, outputs, completion, **self.target["1B"])

  @unittest.expectedFailure
  def test_llama_3_dot_2_1B_self_safetensors(self):
    with Context(SAFETENSORS=1):
      inputs, outputs, completion = self_run(self.prompts, model_desc="1B")
    self._check_output(inputs, outputs, completion, **self.target["1B"])

  @unittest.expectedFailure
  def test_llama_3_dot_2_1B_self_no_safetensors(self):
    with Context(SAFETENSORS=0):
      inputs, outputs, completion = self_run(self.prompts, model_desc="1B")
    self._check_output(inputs, outputs, completion, **self.target["1B"])


if __name__ == "__main__":
  unittest.main()
