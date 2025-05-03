from typing import Literal, get_args
import re
from pathlib import Path
from dataclasses import asdict
from models.llama.tokenizer import Tokenizer
from models.llama.transformer import Transformer
from models.llama.config import LlamaConfig, CONFIGS
from huggingface_hub import snapshot_download
import safetensors.torch
import torch

def _torch_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="pytorch_model*.bin")
  pattern = re.compile(r"^pytorch_model-\d{5}-of-\d{5}\.bin$")
  checkpoints = sorted([
    f for f in Path(ckpt_dir).glob("pytorch_model*.bin") if pattern.match(f.name)
  ])
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  state_dict = {k:v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _safetensors_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints: state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {k:v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _tokenizer_path(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="tokenizer.model")
  tokenizer_path = f"{ckpt_dir}/tokenizer.model"
  return tokenizer_path

ModelOptions = Literal['7B','13B','33B','70B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B', safetensors: bool=False):
  assert model_desc in get_args(ModelOptions), f'invalid model_desc: {model_desc}'
  model_desc = {'33B': '30B','70B': '65B'}.get(model_desc, model_desc)
  repo_id = f'huggyllama/llama-{model_desc.lower()}'
  params = CONFIGS[model_desc]
  config = LlamaConfig.build(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(repo_id))
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(repo_id)

  # https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama/convert_llama_weights_to_hf.py

  def permute_back(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

  for k, v in state_dict.items():
    if k.endswith(("q_proj.weight", "k_proj.weight")):
      state_dict[k] = permute_back(v)

  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
