import pathlib, json
from models.helpers import timeit

import torch
import safetensors.torch

from models.mistral_rolling.transformer import Transformer
from models.mistral_rolling.config import MistralConfig
from models.mistral_rolling.tokenizer import Tokenizer

def _load_state_dict(folder: str):
  pt_model_file = pathlib.Path(folder) / "consolidated.00.pth"
  safetensors_model_file = pathlib.Path(folder) / "consolidated.safetensors"
  assert (
    pt_model_file.exists() or safetensors_model_file.exists()
  ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
  assert not (
    pt_model_file.exists() and safetensors_model_file.exists()
  ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"
  if pt_model_file.exists(): state_dict = torch.load(str(pt_model_file), mmap=True, weights_only=True)
  else: state_dict = safetensors.torch.load_file(str(safetensors_model_file))
  return state_dict

def _load_tokenizer(folder: str) -> Tokenizer:
  path = pathlib.Path(folder) / "tokenizer.model"
  assert path.exists()
  tokenizer = Tokenizer(path.as_posix())
  return tokenizer

@timeit(desc="Load time", ms=False)
def build(folder: str, max_batch_size: int=2):
  with open(pathlib.Path(folder) / "params.json", "r") as f:
    config = MistralConfig(**json.load(f))
  config.max_batch_size = max_batch_size
  torch.set_default_dtype(config.torch_dtype)
  tokenizer = _load_tokenizer(folder)
  state_dict = _load_state_dict(folder)
  model = Transformer(config)
  model.load_state_dict(state_dict, assign=True, strict=True)
  return model, tokenizer
