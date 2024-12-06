from pathlib import Path
from dataclasses import asdict
from huggingface_hub import snapshot_download
import safetensors.torch
import torch
from models.helpers import timeit
from models.llama.tokenizer import Tokenizer
from models.llama2.transformer import Transformer
from models.llama2.config import LlamaConfig

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

@timeit(desc="Load time", ms=False)
def build(max_seq_len: int, max_batch_size: int, model_desc: str='7B', chat: bool=False, safetensors: bool=True):
  # TODO: support safetensors=False
  assert model_desc in ('7B', '13B', '70B'), f'invalid model_type: {model_desc}'
  params = {
    '7B' : dict(dim=4096, n_heads=32, n_layers=32, multiple_of=256, norm_eps=1e-05),
    '13B': dict(dim=5120, n_heads=40, n_layers=40, multiple_of=256, norm_eps=1e-05),
    '70B': dict(dim=8192, n_heads=64, n_kv_heads=8, n_layers=80, multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-05),
  }[model_desc]
  repo_id = f'meta-llama/Llama-2-{model_desc.lower()}'
  repo_id += ('-chat' if chat else '') + ('-hf' if safetensors else '')
  state_dict, tokenizer_path = _safetensors_load(repo_id)
  tokenizer = Tokenizer(tokenizer_path)
  params['vocab_size'] = tokenizer.n_words
  config = LlamaConfig.build(max_seq_len, max_batch_size, **params)
  torch.set_default_dtype(config.torch_dtype)
  model = Transformer(**asdict(config))
  model.load_state_dict(state_dict, assign=True, strict=True)
  return model, tokenizer, config
