# PYTHONPATH=. python -m unittest tests/test_qwen2.py

import os
import unittest
import pathlib
import json

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from huggingface_hub import snapshot_download
from models.qwen2.config import CONFIGS
from models.qwen2.generate import Qwen
from models.helpers import set_device

DEVICE = set_device()
MAX_SEQ_LEN = 128

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# TODO: numerical errors?

class TestQwen2InstructGreedy(unittest.TestCase):
  def setUp(self):
    self.model_dtype = "bfloat16"
    self.dialogs = [
      [
        dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
        dict(role="user", content="How many r in strawberry.")
      ],
    ]
    inputs_text_target = ["<|im_start|>system\nYou are a helpful and truthful assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\nHow many r in strawberry.<|im_end|>\n<|im_start|>assistant\n"]
    inputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198]]

    self.targets = {
      "Qwen/Qwen2-0.5B-Instruct": dict(
        inputs_text_target = inputs_text_target,
        inputs_target = inputs_target,
        outputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 785, 1372, 315, 330, 81, 1, 304, 279, 3409, 330, 495, 672, 15357, 1, 374, 220, 17, 13, 151645]],
        completion_target = ['system\nYou are a helpful and truthful assistant. You should think step-by-step.\nuser\nHow many r in strawberry.\nassistant\nThe number of "r" in the word "strawberry" is 2.'],
        outputs_target_trunc = [[785, 1372, 315, 330, 81, 1, 304, 279, 3409, 330, 495, 672, 15357, 1, 374, 220, 17, 13, 151645]],
        completion_target_trunc = ['The number of "r" in the word "strawberry" is 2.'],
      ),
      "Qwen/Qwen2.5-0.5B-Instruct": dict(
        inputs_text_target = inputs_text_target,
        inputs_target = inputs_target,
        outputs_target = [[151644, 8948, 198, 2610, 525, 264, 10950, 323, 89867, 17847, 13, 1446, 1265, 1744, 3019, 14319, 29208, 13, 151645, 198, 151644, 872, 198, 4340, 1657, 435, 304, 72600, 13, 151645, 198, 151644, 77091, 198, 1249, 8253, 1246, 1657, 330, 81, 40787, 525, 304, 279, 3409, 330, 495, 672, 15357, 1335, 582, 1184, 311, 1760, 1817, 3842, 330, 81, 1, 304, 279, 2661, 3409, 382, 10061, 594, 1438, 432, 1495, 1447, 16, 13, 3070, 5338, 364, 81, 1210, 334, 576, 1156, 6524, 315, 330, 495, 672, 15357, 1, 374, 330, 82, 10040, 256, 481, 4504, 25, 220, 16, 271, 17, 13, 3070, 15666, 364, 81, 1210, 1019, 256, 481, 576, 2086, 6524, 315, 330, 495, 672, 15357, 1, 374, 330, 86, 10040, 256, 481, 4504, 25, 220, 16, 271]],
        completion_target = ['system\nYou are a helpful and truthful assistant. You should think step-by-step.\nuser\nHow many r in strawberry.\nassistant\nTo determine how many "r"s are in the word "strawberry," we need to count each individual "r" in the given word.\n\nLet\'s break it down:\n\n1. **First \'r\':** The first letter of "strawberry" is "s."\n   - Count: 1\n\n2. **Second \'r\':**\n   - The second letter of "strawberry" is "w."\n   - Count: 1\n\n'],
        outputs_target_trunc = [[1249, 8253, 1246, 1657, 330, 81, 40787, 525, 304, 279, 3409, 330, 495, 672, 15357, 1335, 582, 1184, 311, 1760, 1817, 3842, 330, 81, 1, 304, 279, 2661, 3409, 382, 10061, 594, 1438, 432, 1495, 1447, 16, 13, 3070, 5338, 364, 81, 1210, 334, 576, 1156, 6524, 315, 330, 495, 672, 15357, 1, 374, 330, 82, 10040, 256, 481, 4504, 25, 220, 16, 271, 17, 13, 3070, 15666, 364, 81, 1210, 1019, 256, 481, 576, 2086, 6524, 315, 330, 495, 672, 15357, 1, 374, 330, 86, 10040, 256, 481, 4504, 25, 220, 16, 271]],
        completion_target_trunc = ['To determine how many "r"s are in the word "strawberry," we need to count each individual "r" in the given word.\n\nLet\'s break it down:\n\n1. **First \'r\':** The first letter of "strawberry" is "s."\n   - Count: 1\n\n2. **Second \'r\':**\n   - The second letter of "strawberry" is "w."\n   - Count: 1\n\n'],
      ),
    }

  def _get_targets(self, repo_id):
    targets = self.targets[repo_id]
    return targets["inputs_text_target"], targets["inputs_target"], targets["outputs_target"], targets["completion_target"], targets["outputs_target_trunc"], targets["completion_target_trunc"]

  def _helper_test_qwen2_inputs(self, repo_id):
    inputs_text_target, inputs_target, outputs_target, completion_target, outputs_target_trunc, completion_target_trunc = self._get_targets(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    assert inputs_text == inputs_text_target, f"\n{inputs_text=}\n\n{inputs_text_target=}"
    input_ids = tokenizer(inputs_text)["input_ids"]
    assert input_ids == inputs_target, f"\n{input_ids=}\n\n{inputs_target=}"
    completion = tokenizer.batch_decode(outputs_target, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == completion_target, f"\n{completion=}\n\n{completion_target=}"
    output_ids_trunc = [g[len(i):] for g, i in zip(outputs_target, input_ids)]
    assert output_ids_trunc == outputs_target_trunc, f"\n{output_ids_trunc=}\n\n{outputs_target_trunc=}"
    completion_trunc = tokenizer.batch_decode(output_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion_trunc == completion_target_trunc, f"\n{completion_trunc=}\n\n{completion_target_trunc=}"

  def _helper_test_huggingface_chat_completion(self, repo_id):
    inputs_text_target, inputs_target, outputs_target, completion_target, outputs_target_trunc, completion_target_trunc = self._get_targets(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype=self.model_dtype, device_map=DEVICE)
    inputs_text = tokenizer.apply_chat_template(self.dialogs, tokenize=False, add_generation_prompt=True)
    assert model.dtype == getattr(torch, self.model_dtype)
    model_inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"].tolist()
    assert input_ids == inputs_target, f"\n{input_ids=}\n\n{inputs_target=}"
    # NOTE: do not set max_new_tokens as it takes precedence over max_length
    # UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
    generated_ids = model.generate(
      **model_inputs, max_length=MAX_SEQ_LEN, do_sample=False, temperature=None, top_p=None, top_k=None,
    )
    output_ids = generated_ids.tolist()
    assert output_ids == outputs_target, f"\n{output_ids=}\n\n{outputs_target=}"
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion == completion_target, f"\n{completion=}\n\n{completion_target=}"
    generated_ids_trunc = [g[len(i):].tolist() for g, i in zip(generated_ids, input_ids)]
    completion_trunc = tokenizer.batch_decode(generated_ids_trunc, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert completion_trunc == completion_target_trunc, f"\n{completion_trunc=}\n\n{completion_target_trunc=}"

  def _helper_test_self_chat_completion(self, repo_id):
    inputs_text_target, inputs_target, outputs_target, completion_target, outputs_target_trunc, completion_target_trunc = self._get_targets(repo_id)
    generator = Qwen.from_pretrained(
      max_seq_len=MAX_SEQ_LEN, max_batch_size=len(self.dialogs), repo_id=repo_id, force_dtype=self.model_dtype
    ).to(DEVICE)
    model_dtype = getattr(torch, self.model_dtype)
    assert generator.dtype is model_dtype, f"{generator.dtype=}, {model_dtype=}"
    out = generator.chat_completion(self.dialogs, temperature=0., logprobs=True)
    output_ids_trunc = []
    for item in out:
      output_ids_trunc_item = []
      for token in item['tokens']:
        output_ids_trunc_item.extend(generator.tokenizer.encode(token))
      output_ids_trunc.append(output_ids_trunc_item)
    output_ids_trunc = [item for item in output_ids_trunc]  # [:21]
    outputs_target_trunc = [item for item in outputs_target_trunc]  # [:21]
    assert output_ids_trunc == outputs_target_trunc, f"\n{output_ids_trunc=}\n\n{outputs_target_trunc=}"
    # [Qwen/Qwen2-0.5B-Instruct] AssertionError: output_ids_trunc=[[785, 1372, 315, 435, 304, 72600, 374, 220, 16, 13]]
    # [Qwen/Qwen2.5-0.5B-Instruct] AssertionError: output_ids_trunc=[[1249, 8253, 1246, 1657, 330, 81, 40787, 525, 304, 279, 3409, 330, 495, 672, 15357, 1335, 582, 1184, 311, 1760, 1817, 330, 81, 1, 31299, 13, 6771, 594, 1438, 432, 1495, 3019, 14319, 29208, 1447, 16, 13, 3070, 28301, 1437, 279, 3409, 95518, 576, 3409, 582, 2299, 3330, 518, 374, 330, 495, 672, 15357, 2217, 17, 13, 3070, 2507, 1817, 330, 81, 1, 304, 279, 3409, 334, 510, 256, 481, 330, 82, 1, 374, 264, 330, 81, 1, 304, 330, 495, 672, 15357, 10040, 256, 481, 330, 86, 1, 374, 264, 330, 81, 1]]
    completion_trunc = [item['generation']['content'] for item in out]
    completion_trunc = [item for item in completion_trunc]  # [:78]
    completion_target_trunc = [item for item in completion_target_trunc]  # [:78]
    assert completion_trunc == completion_target_trunc, f"\n{completion_trunc=}\n\n{completion_target_trunc=}"
    # [Qwen/Qwen2-0.5B-Instruct] AssertionError: completion_trunc=['The number of r in strawberry is 1.']
    # [Qwen/Qwen2.5-0.5B-Instruct] AssertionError: completion_trunc=['To determine how many "r"s are in the word "strawberry," we need to count each "r" individually. Let\'s break it down step-by-step:\n\n1. **Identify the word**: The word we\'re looking at is "strawberry."\n\n2. **Count each "r" in the word**:\n   - "s" is a "r" in "strawberry."\n   - "w" is a "r"']

  def test_qwen2_smallest_inputs(self):
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    self._helper_test_qwen2_inputs(repo_id)

  def test_qwen2_smallest_huggingface_chat_completion(self):
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    self._helper_test_huggingface_chat_completion(repo_id)

  @unittest.expectedFailure
  def test_qwen2_smallest_self_chat_completion(self):
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    self._helper_test_self_chat_completion(repo_id)

  def test_qwen2_5_smallest_inputs(self):
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    self._helper_test_qwen2_inputs(repo_id)

  def test_qwen2_5_smallest_huggingface_chat_completion(self):
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    self._helper_test_huggingface_chat_completion(repo_id)

  @unittest.expectedFailure
  def test_qwen2_5_smallest_self_chat_completion(self):
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    self._helper_test_self_chat_completion(repo_id)


class TestQwen2Configs(unittest.TestCase):
  def test_qwen2_configs(self):
    for repo_id, conf in CONFIGS.items():
      hf_config_dir = snapshot_download(repo_id, allow_patterns="config.json")
      pth = pathlib.Path(hf_config_dir) / "config.json"
      with pth.open() as f:
        conf_hf = json.load(f)
      for key in conf:
        if key == "dim": key_hf = "hidden_size"
        elif key == "hidden_dim": key_hf = "intermediate_size"
        elif key == "n_heads": key_hf = "num_attention_heads"
        elif key == "n_layers": key_hf = "num_hidden_layers"
        elif key == "n_kv_heads": key_hf = "num_key_value_heads"
        elif key == "norm_eps": key_hf = "rms_norm_eps"
        else: key_hf = key
        assert conf[key] == conf_hf[key_hf], (key, conf[key], key_hf, conf_hf[key_hf])


if __name__ == "__main__":
  unittest.main()
