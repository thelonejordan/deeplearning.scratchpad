from typing import Literal, get_args
from pathlib import Path
from dataclasses import asdict
from huggingface_hub import snapshot_download
import safetensors.torch
import torch
from models.llama.tokenizer import Tokenizer
from models.llama2.transformer import Transformer
from models.llama2.config import LlamaConfig, CONFIGS

def _safetensors_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(safetensors.torch.load_file(ckpt))
  state_dict = {k:v for k, v in state_dict.items() if not k.endswith("freq")}
  ckpt_dir = snapshot_download(repo_id, allow_patterns="tokenizer.model")
  tokenizer_path = f"{ckpt_dir}/tokenizer.model"
  return state_dict, tokenizer_path

ModelOptions = Literal['7B','13B','70B']

def build(max_seq_len: int, max_batch_size: int, model_desc: ModelOptions='7B', chat: bool=False, safetensors: bool=True):
  # TODO: support safetensors=False
  assert model_desc in get_args(ModelOptions), f'invalid model_type: {model_desc}'
  params = CONFIGS[model_desc]
  repo_id = f'meta-llama/Llama-2-{model_desc.lower()}'
  repo_id += ('-chat' if chat else '') + ('-hf' if safetensors else '')
  state_dict, tokenizer_path = _safetensors_load(repo_id)
  tokenizer = Tokenizer(tokenizer_path)
  config = LlamaConfig.build(max_seq_len, max_batch_size, **params)
  assert config.vocab_size == tokenizer.n_words, f"{config.vocab_size=} != {tokenizer.n_words=}"

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

  torch.set_default_dtype(getattr(torch, config.torch_dtype))
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  return model, tokenizer, config
