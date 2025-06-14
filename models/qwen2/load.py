from typing import Optional
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from models.qwen2.config import QwenConfig, CONFIGS
from models.qwen2.transformer import Transformer
from models.llama2.load import _safetensors_load

def build(max_seq_len: int, max_batch_size: int, repo_id: str, force_dtype: Optional[str]=None):
  assert repo_id in CONFIGS.keys(), f"invalid repo id, must be one of {tuple(CONFIGS.keys())}, got {repo_id}"
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
    state_dict = {fixup(k): v for k, v in state_dict.items()} # if k != "lm_head.weight"}
  state_dict = {k:v.to(dtype=getattr(torch, config.torch_dtype)) for k,v in state_dict.items()}
  _model.load_state_dict(state_dict, assign=True, strict=True)
  if config.tie_word_embeddings:
    model = model.tie_word_embeddings()
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
