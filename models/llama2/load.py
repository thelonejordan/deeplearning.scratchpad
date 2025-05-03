from typing import Literal, get_args
from pathlib import Path
from dataclasses import asdict
from huggingface_hub import snapshot_download
import safetensors.torch
import torch
from models.llama.tokenizer import Tokenizer
from models.llama2.transformer import Transformer
from models.llama2.config import LlamaConfig, CONFIGS

def _torch_load(repo_id: str, keymap: dict[str, str] = {}):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="consolidated*.pth")
  checkpoints = sorted(Path(ckpt_dir).glob("consolidated*.pth"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(torch.load(ckpt, map_location='cpu', weights_only=True))
  state_dict = {keymap.get(k, k):v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _safetensors_load(repo_id: str, keymap: dict[str, str] = {}):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {keymap.get(k, k):v for k, v in state_dict.items() if "freq" not in k}
  return state_dict

def _tokenizer_path(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="tokenizer.model")
  return f"{ckpt_dir}/tokenizer.model"

ModelOptions = Literal['7B','13B','70B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B', chat: bool=False, safetensors: bool=True):
  assert model_desc in get_args(ModelOptions), f'invalid model_type: {model_desc}'
  repo_id = f'meta-llama/Llama-2-{model_desc.lower()}'
  repo_id += ('-chat' if chat else '') + ('-hf' if safetensors else '')
  params = CONFIGS[model_desc]
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

    dims_per_head = config.dim // config.n_heads  # same as head_dim
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
  model.load_state_dict(state_dict, assign=True, strict=True)
  torch.set_default_dtype(default_dtype)
  return model, tokenizer, config
