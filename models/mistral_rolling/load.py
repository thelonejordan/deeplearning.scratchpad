from typing import Optional
from pathlib import Path
import json
from models.helpers import timeit

import torch
import safetensors.torch

from models.mistral_rolling.transformer import Transformer, DEFAULT_FLOAT, MistralConfig
from models.mistral_rolling.tokenizer import Tokenizer

def _load_model(folder: Path, device: torch.device, dtype: torch.dtype, max_batch_size: int=1):
  with open(Path(folder) / "params.json", "r") as f:
    config = MistralConfig(**json.load(f))
  config.max_batch_size = max_batch_size
  model = Transformer(config)
  pt_model_file = Path(folder) / "consolidated.00.pth"
  safetensors_model_file = Path(folder) / "consolidated.safetensors"
  assert (
    pt_model_file.exists() or safetensors_model_file.exists()
  ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
  assert not (
    pt_model_file.exists() and safetensors_model_file.exists()
  ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"
  if pt_model_file.exists(): loaded = torch.load(str(pt_model_file), mmap=True, weights_only=True)
  else: loaded = safetensors.torch.load_file(str(safetensors_model_file))
  model.load_state_dict(loaded, assign=True, strict=True)
  return model.to(device=device, dtype=dtype)

def _load_tokenizer(model_path: Path) -> Tokenizer:
  path = model_path / "tokenizer.model"
  assert path.exists()
  tokenizer = Tokenizer(path.as_posix())
  return tokenizer

@timeit(desc="Load time", ms=False)
def from_pretrained(folder: str, max_batch_size: int=1, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
  device = torch.device('cpu') if device is None else device
  dtype = DEFAULT_FLOAT if dtype is None else dtype
  model = _load_model(Path(folder), device, dtype, max_batch_size)
  tokenizer = _load_tokenizer(Path(folder))
  return model, tokenizer
