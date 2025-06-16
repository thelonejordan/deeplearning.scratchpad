from models.mistral_nonrolling.load import VersionOptions
from models.mistral_nonrolling.load import build as _build
from models.mistral_rolling.transformer import Transformer

def build(max_seq_len: int, max_batch_size: int, version: VersionOptions='1', safetensors: bool=True,
          model_class: type[Transformer]=Transformer):
  return _build(max_seq_len, max_batch_size, version, safetensors, model_class)
