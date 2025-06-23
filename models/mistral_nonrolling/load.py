from typing import Literal, get_args
import pathlib
from dataclasses import asdict

import torch
import safetensors.torch
from huggingface_hub import snapshot_download

from models.llama.load import convert_from_huggingface
from models.mistral_nonrolling.config import MistralConfig, CONFIGS
from models.mistral_nonrolling.transformer import Transformer
from models.mistral_nonrolling.tokenizer import Tokenizer

def _torch_load(folder: str, **_):
  state_dict = {}
  for pt_model_file in sorted(pathlib.Path(folder).glob("consolidated*.pth")):
    assert pt_model_file.exists(), f"Make sure either {pt_model_file} exists"
    state_dict.update(torch.load(str(pt_model_file), map_location='cpu', weights_only=True, mmap=True))
  return state_dict

def _safetensors_load(repo_id: str, dim: int, n_layers: int, n_heads: int, n_kv_heads: int, **_):
  state_dict = {}
  folder = snapshot_download(repo_id, allow_patterns="*.safetensors")
  for pt_model_file in sorted(pathlib.Path(folder).glob("*.safetensors")):
    assert pt_model_file.exists(), f"Make sure either {pt_model_file} exists"
    state_dict.update(safetensors.torch.load_file(pt_model_file))
  if safetensors:
    state_dict = convert_from_huggingface(state_dict, dim, n_heads, n_kv_heads=n_kv_heads)
  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f'model.layers.{l}.self_attn.{var}_proj.weight': f'layers.{l}.attention.w{var}.weight' for var in "qkvo" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.{y}_proj.weight": f"layers.{l}.feed_forward.w{x}.weight" for x, y in {"1": "gate", "2": "down", "3": "up"}.items() \
        for l in range(n_layers)},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  state_dict = {keymap.get(k, k): v for k,v in state_dict.items() if "freq" not in k}
  return state_dict

def _tokenizer_path(repo_id_or_folder: str, safetensors: bool = True):
  if safetensors:
    repo_id_or_folder = snapshot_download(repo_id_or_folder, allow_patterns="tokenizer.model")
  tok_path = pathlib.Path(repo_id_or_folder) / "tokenizer.model"
  assert tok_path.exists(), tok_path
  return tok_path.as_posix()

VersionOptions = Literal['1', '3']

def build(max_seq_len: int, max_batch_size: int, version: VersionOptions='1', safetensors: bool=True,
          model_class: type[Transformer]=Transformer):
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
  with torch.device("meta"):
    model = model_class(**asdict(config), **{"tokenizer": tokenizer, "config": config})
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
