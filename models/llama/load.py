from typing import Optional, Literal, get_args
import re
from pathlib import Path
from dataclasses import asdict
from models.llama.tokenizer import Tokenizer
from models.llama.transformer import Transformer
from models.llama.config import LlamaConfig, CONFIGS
from huggingface_hub import snapshot_download
import safetensors.torch
import torch
from torch import Tensor

def _torch_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="pytorch_model*.bin")
  pattern = re.compile(r"^pytorch_model-\d{5}-of-\d{5}\.bin$")
  checkpoints = sorted([
    f for f in Path(ckpt_dir).glob("pytorch_model*.bin") if pattern.match(f.name)
  ])
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(str(ckpt), map_location='cpu', weights_only=True, mmap=True))
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

def convert_from_huggingface(state_dict: dict[str, Tensor], dim: int, n_heads: int, n_kv_heads: Optional[int]=None):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  # https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama/convert_llama_weights_to_hf.py
  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

  def permute_back(w, n_heads=n_heads, dim1=dim, dim2=dim):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

  dims_per_head = dim // n_heads  # same as head_dim
  if n_kv_heads is None: n_kv_heads = n_heads
  num_key_value_heads = n_kv_heads  # for GQA / MQA
  key_value_dim = dims_per_head * num_key_value_heads

  for k, v in state_dict.items():
    if k.endswith("q_proj.weight"):
      state_dict[k] = permute_back(v)
    if k.endswith("k_proj.weight"):
      state_dict[k] = permute_back(v, n_heads=num_key_value_heads, dim1=key_value_dim)

  return state_dict

ModelOptions = Literal['7B','13B','33B','70B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B', safetensors: bool=True,
          model_class: type[Transformer]=Transformer):
  assert model_desc in get_args(ModelOptions), f'invalid model_desc: {model_desc}'
  model_desc = {'33B': '30B','70B': '65B'}.get(model_desc, model_desc)
  repo_id = f'huggyllama/llama-{model_desc.lower()}'
  params = CONFIGS[model_desc]
  config = LlamaConfig.build(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(repo_id))
  assert config.vocab_size == tokenizer.n_words, (config.vocab_size, tokenizer.n_words)
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(repo_id)

  # NOTE: torch weights were also permuted for rope in huggingface style
  state_dict = convert_from_huggingface(state_dict, config.dim, config.n_heads)

  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  with torch.device("meta"):
    model = model_class(**asdict(config), **{'tokenizer': tokenizer, 'config': config})
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
