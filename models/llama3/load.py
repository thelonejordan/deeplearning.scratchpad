from typing import Literal, get_args
import json
from pathlib import Path
from dataclasses import asdict

from models.llama3.tokenizer import Tokenizer
from models.llama3.transformer import Transformer
from models.llama3.config import LlamaConfig
from huggingface_hub import snapshot_download
import safetensors.torch
import torch

def _safetensors_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {k:v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _torch_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/consolidated*.pth")
  checkpoints = sorted(Path(ckpt_dir).glob("original/consolidated*.pth"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  state_dict = {k:v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _tokenizer_path(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/tokenizer.model")
  return f"{ckpt_dir}/original/tokenizer.model"

def _load_params(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/params.json")
  # TODO: don't use params.json for generating config
  with open(f"{ckpt_dir}/original/params.json") as f:
    params = json.loads(f.read())
  # TODO: understand use_scaled_rope
  if 'use_scaled_rope' in params: params.pop('use_scaled_rope')
  return params

ModelOptions = Literal['1B','3B','8B','70B']
VersionOptions = Literal[0,1,2]

def build(max_seq_len: int, max_batch_size: int, seed: int=1,
          model_desc: ModelOptions='8B', version: VersionOptions=0, instruct: bool=False, safetensors: bool=True):
  assert model_desc in get_args(ModelOptions), f'invalid model_type: {model_desc}'
  assert version in get_args(VersionOptions), f'invalid version: {version}'
  if model_desc in ('8B', '70B'): assert version in (0, 1)
  if model_desc in ('1B', '3B'): assert version == 2
  v = '' if version == 0 else f'.{version}'
  prefix = 'Meta-'if version == 0 else ''
  repo_id = f'meta-llama/{prefix}Llama-3{v}-{model_desc}' + ('-Instruct' if instruct else '')
  params = _load_params(repo_id)
  config = LlamaConfig.build(max_seq_len, max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(repo_id))
  assert config.vocab_size == tokenizer.n_words, f"{config.vocab_size=} != {tokenizer.n_words=}"
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(repo_id)

  if safetensors:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

    def permute_back(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
      return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    dims_per_head = config.dim // config.n_heads
    num_key_value_heads = config.n_kv_heads  # for GQA / MQA
    key_value_dim = dims_per_head * num_key_value_heads

    for k, v in state_dict.items():
      if k.endswith("q_proj.weight"):
        state_dict[k] = permute_back(v)
      if k.endswith("k_proj.weight"):
        state_dict[k] = permute_back(v, n_heads=num_key_value_heads, dim1=key_value_dim)

  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=False)
  torch.set_default_dtype(default_dtype)
  # TODO: add this as a config parameter
  if version == 2: model = model.apply_weight_sharing()
  return model, tokenizer, config
