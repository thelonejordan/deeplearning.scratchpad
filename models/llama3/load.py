from typing import Literal, get_args
from pathlib import Path
from dataclasses import asdict

from models.llama3.tokenizer import Tokenizer
from models.llama3.transformer import Transformer
from models.llama3.config import LlamaConfig, CONFIGS
from huggingface_hub import snapshot_download
import safetensors.torch
import torch

def _safetensors_load(repo_id: str, keymap: dict[str, str] = {}):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {keymap.get(k, k):v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _torch_load(repo_id: str, keymap: dict[str, str] = {}):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/consolidated*.pth")
  checkpoints = sorted(Path(ckpt_dir).glob("original/consolidated*.pth"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  state_dict = {keymap.get(k, k):v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _tokenizer_path(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="original/tokenizer.model")
  return f"{ckpt_dir}/original/tokenizer.model"

ModelOptions = Literal['1B','3B','8B','70B']
VersionOptions = Literal['0','1','2']

def build(max_seq_len: int, max_batch_size: int, seed: int=1,
          model_desc: ModelOptions='8B', version: VersionOptions='0', instruct: bool=False, safetensors: bool=True):
  assert model_desc in get_args(ModelOptions), f'invalid model_type: {model_desc}'
  assert version in get_args(VersionOptions), f'invalid version: {version}'
  if model_desc in ('8B', '70B'): assert version in ('0', '1')
  if model_desc in ('1B', '3B'): assert version == '2'
  v = '' if version == '0' else f'.{version}'
  prefix = 'Meta-'if version == '0' else ''
  repo_id = f'meta-llama/{prefix}Llama-3{v}-{model_desc}' + ('-Instruct' if instruct else '')
  params = CONFIGS[f"3{v}"][model_desc]
  config = LlamaConfig.build(max_seq_len, max_batch_size, **params)
  tokenizer = Tokenizer(_tokenizer_path(repo_id))
  assert config.vocab_size == tokenizer.n_words, f"{config.vocab_size=} != {tokenizer.n_words=}"
  keymap = {}
  if not safetensors:
    keymap.update({
      'tok_embeddings.weight': 'model.embed_tokens.weight',
      **{f'layers.{l}.attention.w{var}.weight': f'model.layers.{l}.self_attn.{var}_proj.weight' for var in "qkvo" for l in range(config.n_layers)},
      **{f'layers.{l}.attention_norm.weight': f'model.layers.{l}.input_layernorm.weight' for l in range(config.n_layers)},
      **{f"layers.{l}.feed_forward.w{x}.weight": f"model.layers.{l}.mlp.{y}_proj.weight" for x, y in {"1": "gate", "2": "down", "3": "up"}.items() for l in range(config.n_layers)},
      **{f"layers.{l}.ffn_norm.weight": f"model.layers.{l}.post_attention_layernorm.weight" for l in range(config.n_layers)},
      "output.weight": "lm_head.weight",
      "norm.weight": "model.norm.weight",
    })
  load_state_dict = _safetensors_load if safetensors else _torch_load
  state_dict = load_state_dict(repo_id, keymap)

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
  # TODO: add this as a config parameter?
  if version == '2': model = model.apply_weight_sharing()
  return model, tokenizer, config
