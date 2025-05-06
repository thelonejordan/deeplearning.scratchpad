from typing import Literal
import pathlib, json, torch
from dataclasses import asdict

from models.mistral_nonrolling.config import MistralConfig, CONFIGS
from models.mistral_nonrolling.transformer import Transformer
from models.mistral_nonrolling.tokenizer import Tokenizer

def _torch_load(folder: str):
  state_dict = {}
  for pt_model_file in sorted(pathlib.Path(folder).glob("consolidated*.pth")):
    assert pt_model_file.exists(), f"Make sure either {pt_model_file} exists"
    state_dict.update(torch.load(str(pt_model_file), map_location='cpu', weights_only=True, mmap=True))
  return state_dict

def _tokenizer_path(folder: str):
  tok_path = pathlib.Path(folder) / "tokenizer.model"
  assert tok_path.exists(), tok_path
  return tok_path.as_posix()

VersionOptions = Literal['1', '3']

def build(folder: str, max_seq_len: int, max_batch_size: int, version: VersionOptions='1'):
  # with open(pathlib.Path(folder) / "params.json", "r") as f:
  #   params = json.load(f)
  params = CONFIGS[f"v0.{version}"]["7B"]
  config = MistralConfig.build(max_seq_len, max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(folder))
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  state_dict = _torch_load(folder)
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
