from typing import Optional
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from models.qwq.config import QwQConfig
from models.qwq.transformer import Transformer
from models.llama2.load import _safetensors_load

def huggingface_repo_id(preview: bool=True):
  suffix = "-Preview" if preview else ""
  return f"Qwen/QwQ-32B{suffix}"

def build(max_seq_len: int, max_batch_size: int, preview: bool=True, force_dtype: Optional[str]=None):
  repo_id = huggingface_repo_id(preview)
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  params = {}
  if force_dtype is not None:
    params['torch_dtype'] = force_dtype
  config = QwQConfig.build(max_seq_len, max_batch_size, **params)
  # AssertionError: config.vocab_size=152064 != len(tokenizer.get_vocab())=151665
  # AssertionError: config.vocab_size=152064 != len(tokenizer)=151665
  # AssertionError: config.vocab_size=152064 != tokenizer.vocab_size=151643
  # assert config.vocab_size == tokenizer.vocab_size, f"{config.vocab_size=} != {tokenizer.vocab_size=}"
  state_dict = _safetensors_load(repo_id)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  with torch.device("meta"):
    model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
