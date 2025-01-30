from typing import Literal
from pathlib import Path
from dataclasses import asdict
from models.llama.tokenizer import Tokenizer
from models.llama.transformer import Transformer
from models.llama.config import LlamaConfig, CONFIGS
from huggingface_hub import snapshot_download
import safetensors.torch
import torch

def _safetensors_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  state_dict = {}
  for ckpt in checkpoints: state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {k:v for k, v in state_dict.items() if not k.endswith("freq")}
  ckpt_dir = snapshot_download(repo_id, allow_patterns="tokenizer.model")
  tokenizer_path = f"{ckpt_dir}/tokenizer.model"
  return state_dict, tokenizer_path

ModelOptions = Literal['7B','13B','30B','65B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B'):
  assert model_desc in ('7B', '13B', '30B', '65B'), f'invalid model_desc: {model_desc}'
  params = CONFIGS[model_desc]
  repo_id = f'huggyllama/llama-{model_desc.lower()}'
  state_dict, tokenizer_path = _safetensors_load(repo_id)
  tokenizer = Tokenizer(tokenizer_path)
  params['vocab_size'] = tokenizer.n_words
  config = LlamaConfig.build(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
  torch.set_default_dtype(config.torch_dtype)
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  return model, tokenizer, config
