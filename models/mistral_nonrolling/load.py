import pathlib, json, torch
import safetensors.torch

from models.mistral_nonrolling.transformer import Transformer, MistralConfig
from models.mistral_nonrolling.tokenizer import Tokenizer

def _torch_load(folder: str):
  pt_model_file = pathlib.Path(folder) / "consolidated.00.pth"
  assert pt_model_file.exists(), f"Make sure either {pt_model_file} exists"
  state_dict = torch.load(str(pt_model_file), mmap=True, weights_only=True)
  return state_dict

def build(folder: str, max_seq_len: int, max_batch_size: int):
  state_dict = _torch_load(folder)
  with open(pathlib.Path(folder) / "params.json", "r") as f:
    config = MistralConfig.build(max_seq_len, max_batch_size, **json.load(f))
  torch.set_default_dtype(config.torch_dtype)
  model = Transformer(config)
  model.load_state_dict(state_dict, assign=True, strict=True)
  tok_path = pathlib.Path(folder) / "tokenizer.model"
  assert tok_path.exists(), tok_path
  tokenizer = Tokenizer(tok_path.as_posix())
  return model, tokenizer
