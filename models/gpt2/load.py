from typing import Set, Literal
from pathlib import Path
from dataclasses import asdict

import torch
import safetensors.torch
from huggingface_hub import snapshot_download  # type: ignore

from models.gpt2.config import GPTConfig, CONFIGS
from models.gpt2.transformer import Transformer
from models.gpt2.tokenizer import Tokenizer

def _safetensors_load(repo_id: str, transposed: Set[str]=set(), skip: Set[str]=set()):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  state_dict, filtered_state_dict  = {}, {}
  for ckpt in checkpoints: state_dict.update(safetensors.torch.load_file(ckpt))
  for k, v in state_dict.items():
    if any(k.endswith(w) for w in skip): continue
    filtered_state_dict[k] = v.t() if any(k.endswith(w) for w in transposed) else v
  return filtered_state_dict

def _torch_load(checkpoint: str, transposed: Set[str]=set(), skip: Set[str]=set()):
  ckpt_dir = snapshot_download(checkpoint, allow_patterns="*.bin")
  checkpoints = sorted(Path(ckpt_dir).glob("*.bin"))
  state_dict, filtered_state_dict  = {}, {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  for k, v in state_dict.items():
    if any(k.endswith(w) for w in skip): continue
    filtered_state_dict[k] = v.t() if any(k.endswith(w) for w in transposed) else v
  return filtered_state_dict

ModelOptions = Literal['gpt2','gpt2-medium','gpt2-large','gpt2-xl']

def build(model_desc: ModelOptions='gpt2', safetensors: bool=True):
  params = CONFIGS[model_desc]
  transposed = {'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'}
  skip = {'.attn.masked_bias', '.attn.bias'}
  loader = _safetensors_load if safetensors else _torch_load
  state_dict = loader(model_desc, transposed, skip)
  tokenizer = Tokenizer()
  params['vocab_size'] = tokenizer.model.n_vocab
  config = GPTConfig(**params)
  model = Transformer(**asdict(config))
  model.transformer.load_state_dict(state_dict, assign=True, strict=True)
  return model.apply_weight_sharing(), tokenizer, config
