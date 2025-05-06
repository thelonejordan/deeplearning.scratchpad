from typing import get_args
from dataclasses import asdict

import torch

from models.mistral_nonrolling.load import _torch_load, _safetensors_load, _tokenizer_path, VersionOptions
from models.mistral_nonrolling.config import MistralConfig, CONFIGS
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_rolling.transformer import Transformer

def build(max_seq_len: int, max_batch_size: int, version: VersionOptions='1', safetensors: bool=True):
  model_desc = '7B'
  assert version in get_args(VersionOptions), f'invalid version: {version}'
  model_desc_full = f'Mistral-{model_desc}-v0.{version}'
  prefix = "mistralai" if safetensors else "downloads"
  repo_id_or_folder = f"{prefix}/{model_desc_full}"
  params = CONFIGS[f"v0.{version}"]["7B"]
  config = MistralConfig.build(max_seq_len, max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(repo_id_or_folder, safetensors))
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(repo_id_or_folder, **asdict(config))
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
