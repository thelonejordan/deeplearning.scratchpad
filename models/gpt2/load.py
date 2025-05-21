from typing import Literal, get_args
from pathlib import Path
from dataclasses import asdict

import torch
import safetensors.torch
from huggingface_hub import snapshot_download

from models.gpt2.config import GPTConfig, CONFIGS
from models.gpt2.transformer import Transformer
from models.gpt2.tokenizer import Tokenizer

def _safetensors_load(repo_id: str, transposed: set[str]=set(), skip: set[str]=set()):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  state_dict, filtered_state_dict  = {}, {}
  for ckpt in checkpoints: state_dict.update(safetensors.torch.load_file(ckpt))
  for k, v in state_dict.items():
    if any(k.endswith(w) for w in skip): continue
    filtered_state_dict[k] = v.t() if any(k.endswith(w) for w in transposed) else v
  return filtered_state_dict

def _torch_load(repo_id: str, transposed: set[str]=set(), skip: set[str]=set()):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="pytorch_model*.bin")
  checkpoints = sorted(Path(ckpt_dir).glob("pytorch_model*.bin"))
  state_dict, filtered_state_dict  = {}, {}
  for ckpt in checkpoints:
    # RuntimeError: mmap can only be used with files saved with `torch.save(_use_new_zipfile_serialization=True), please torch.save your checkpoint with this option in order to use mmap.
    state_dict.update(torch.load(str(ckpt), map_location='cpu', weights_only=True))
  for k, v in state_dict.items():
    if any(k.endswith(w) for w in skip): continue
    filtered_state_dict[k] = v.t() if any(k.endswith(w) for w in transposed) else v
  return filtered_state_dict

ModelOptions = Literal['gpt2','gpt2-medium','gpt2-large','gpt2-xl']

def build(model_desc: ModelOptions='gpt2', safetensors: bool=True):
  assert model_desc in get_args(ModelOptions), f'invalid model_desc: {model_desc}'
  params = CONFIGS[model_desc]
  transposed = {'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'}
  skip = {'.attn.masked_bias', '.attn.bias'}
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(model_desc, transposed, skip)
  tokenizer = Tokenizer()
  config = GPTConfig(**params)
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)
  # https://github.com/huggingface/transformers/issues/21681#issuecomment-1436552397
  # According to this comment, dtype of a model in PyTorch is always float32, regardless of the dtype of the checkpoint you saved.
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.transformer.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model.apply_weight_sharing(), tokenizer, config
