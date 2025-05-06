import pathlib, json, torch
from dataclasses import asdict

from models.mistral_nonrolling.load import _torch_load, _tokenizer_path, VersionOptions
from models.mistral_nonrolling.config import MistralConfig, CONFIGS
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_rolling.transformer import Transformer

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
