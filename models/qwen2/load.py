from typing import Optional, Literal, get_args
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from models.qwen2.config import QwenConfig, CONFIGS
from models.qwen2.transformer import Transformer
from models.llama2.load import _safetensors_load

ModelOptions = Literal["2", "2.5", "1M", "qwq"]
ModelSizes = Literal["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]

def huggingface_repo_id(model_desc: ModelOptions="qwq", model_size: ModelSizes="32B", preview: bool=True, instruct: bool=False):
  _instruct = "-Instruct" if instruct else ""
  assert model_desc in get_args(ModelOptions), f"invalid model desc, got {model_desc}"
  if model_desc == "2":
    assert model_size in ("0.5B", "1.5B" "7B", "72B"), f"invalid model size for {model_desc}, got {model_size}"
    return f"Qwen/Qwen2-{model_size}{_instruct}"
  if model_desc == "2.5":
    assert model_size in get_args(ModelSizes), f"invalid model size for {model_desc}, got {model_size}"
    return f"Qwen/Qwen2.5-{model_size}{_instruct}"
  if model_desc == "1M":
    assert model_size in ("7B", "14B"), f"invalid model size for {model_desc}, got {model_size}"
    return f"Qwen/Qwen2.5-{model_size}-Instruct-1M"
  if model_desc == "qwq":
    assert model_size == "32B", f"invalid model size for {model_desc}, got {model_size}"
    suffix = "-Preview" if preview else ""
    return f"Qwen/QwQ-32B{suffix}"
  return ""

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions="qwq", model_size: ModelSizes="32B",
          preview: bool=True, instruct: bool=False, force_dtype: Optional[str]=None):
  repo_id = huggingface_repo_id(model_desc, model_size, preview, instruct)
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  params = CONFIGS.get(repo_id, {})
  if force_dtype is not None:
    params['torch_dtype'] = force_dtype
  for key in ['bos_token_id', 'eos_token_id', 'sliding_window', 'max_window_layers']:
    params.pop(key)
  config = QwenConfig.build(max_seq_len, max_batch_size, **params)
  # AssertionError: config.vocab_size=152064 != len(tokenizer.get_vocab())=151665
  # AssertionError: config.vocab_size=152064 != len(tokenizer)=151665
  # AssertionError: config.vocab_size=152064 != tokenizer.vocab_size=151643
  # assert config.vocab_size == tokenizer.vocab_size, f"{config.vocab_size=} != {tokenizer.vocab_size=}"
  state_dict = _safetensors_load(repo_id)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  with torch.device("meta"):
    model = Transformer(**asdict(config))
  _model = model
  if config.tie_word_embeddings:
    _model = model.model
    fixup = lambda k: k[len('model.'):] if k.startswith('model.') else k
    state_dict = {fixup(k): v for k, v in state_dict.items() if k != "lm_head.weight"}
  _model.load_state_dict(state_dict, assign=True, strict=True)
  if config.tie_word_embeddings:
    model = model.tie_word_embeddings()
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
