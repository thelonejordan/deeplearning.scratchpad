from typing import Literal, get_args
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

ModelOptions = Literal['7B','13B','33B','70B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B'):
  assert model_desc in get_args(ModelOptions), f'invalid model_desc: {model_desc}'
  params = CONFIGS[model_desc]
  model_desc = {'33B': '30B','70B': '65B'}.get(model_desc, model_desc)
  repo_id = f'huggyllama/llama-{model_desc.lower()}'
  state_dict, tokenizer_path = _safetensors_load(repo_id)
  tokenizer = Tokenizer(tokenizer_path)
  config = LlamaConfig.build(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)

  # https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama/convert_llama_weights_to_hf.py

  def permute_back(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

  for k, v in state_dict.items():
    if k.endswith(("q_proj.weight", "k_proj.weight")):
      state_dict[k] = permute_back(v)

  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  return model, tokenizer, config
