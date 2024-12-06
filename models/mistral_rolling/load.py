import pathlib, json, torch

from models.mistral_nonrolling.load import _torch_load
from models.mistral_nonrolling.config import MistralConfig
from models.mistral_nonrolling.tokenizer import Tokenizer
from models.mistral_rolling.transformer import Transformer

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
