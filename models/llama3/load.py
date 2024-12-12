import json
from pathlib import Path
from dataclasses import asdict
from models.helpers import timeit
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
  state_dict = {k:v for k, v in state_dict.items() if not k.endswith("freq")}
  return state_dict

def _torch_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/*.pth")
  checkpoints = sorted(Path(ckpt_dir).glob("original/*.pth"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  state_dict = {k:v for k, v in state_dict.items() if not k.endswith("freq")}
  return state_dict

def build(max_seq_len: int, max_batch_size: int, seed: int=1,
          model_desc: str='8B', version: int=0, instruct: bool=False, safetensors: bool=True):
  assert model_desc in ('1B', '3B', '8B', '70B'), f'invalid model_type: {model_desc}'
  if model_desc in ('8B', '70B'): assert version in (0, 1)
  if model_desc in ('1B', '3B'): assert version == 2
  v = '' if version == 0 else f'.{version}'
  prefix = 'Meta-'if version == 0 else ''
  repo_id = f'meta-llama/{prefix}Llama-3{v}-{model_desc}' + ('-Instruct' if instruct else '')
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/params.json")
  with open(f"{ckpt_dir}/original/params.json") as f:
    params = json.loads(f.read())
    if 'use_scaled_rope' in params: params.pop('use_scaled_rope')
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/tokenizer.model")
  tokenizer = Tokenizer(f"{ckpt_dir}/original/tokenizer.model")
  config = LlamaConfig.build(max_seq_len, max_batch_size, **params)
  assert config.vocab_size == tokenizer.n_words, f"{config.vocab_size=} != {tokenizer.n_words=}"
  state_dict = (_safetensors_load if safetensors else _torch_load)(repo_id)
  torch.set_default_dtype(config.torch_dtype)
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=False)
  if version == 2: model = model.apply_weight_sharing()
  return model, tokenizer, config
