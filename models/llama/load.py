from tqdm import tqdm
import torch
from models.helpers import timeit
from models.llama.tokenizer import Tokenizer
from models.llama.transformer import Transformer, LlamaConfig

def _copy_from_hf(model, checkpoint: str, half=False):
  from transformers import LlamaTokenizer, LlamaForCausalLM
  tokenizer = Tokenizer(LlamaTokenizer.from_pretrained(checkpoint).vocab_file)
  model_hf = LlamaForCausalLM.from_pretrained(checkpoint)
  if half: model, model_hf = model.half(), model_hf.half()
  sd, sd_hf = model.state_dict(), model_hf.state_dict()
  sd_keys, sd_keys_hf = list(sd.keys()), list(sd_hf.keys())
  assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
  itr = tqdm(sd_keys_hf)
  for k in itr:
    itr.set_description(f'Loading {k}')
    assert sd_hf[k].shape == sd[k].shape, f'{k} not found'
    with torch.no_grad(): sd[k].copy_(sd_hf[k])
    del sd_hf[k] # free memory after copying
  return model, tokenizer

def _load_from_cache(model: Transformer, checkpoint: str):
  from transformers.utils import try_to_load_from_cache
  import safetensors.torch
  filenames=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
  model_files = [try_to_load_from_cache(repo_id=checkpoint, filename=filename) for filename in filenames]
  loaded = dict()
  for file in model_files: loaded.update(safetensors.torch.load_file(str(file)))
  loaded = {k:v for k, v in loaded.items() if not k.endswith("freq")}
  model.load_state_dict(loaded, assign=True, strict=True)
  tokenizer_path = try_to_load_from_cache(repo_id=checkpoint, filename="tokenizer.model")
  tokenizer = Tokenizer(tokenizer_path)
  return model, tokenizer

def _safetensor_load(model: Transformer, checkpoint: str):
  from huggingface_hub import snapshot_download
  import safetensors.torch
  filenames=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
  base = snapshot_download(checkpoint, allow_patterns=filenames)
  loaded = {}
  for file in filenames: loaded.update(safetensors.torch.load_file(f"{base}/{file}"))
  loaded = {k:v for k, v in loaded.items() if not k.endswith("freq")}
  model.load_state_dict(loaded, assign=True, strict=True)
  base = snapshot_download(checkpoint, allow_patterns="tokenizer.model")
  tokenizer = Tokenizer(f"{base}/tokenizer.model")
  return model, tokenizer

@timeit(desc="Load time", ms=False)
def from_pretrained(model_type: str='7B', half=False, assign: bool=False):
  assert model_type in ('7B', '13B', '30B', '65B'), f'invalid model_type: {model_type}'
  config_args = {
    '7B' : dict(dim=4096, n_heads=32, n_layers=32), # 6.7B
    '13B': dict(dim=5120, n_heads=40, n_layers=40), # 13.0B
    '30B': dict(dim=6656, n_heads=52, n_layers=60), # 32.5B
    '65B': dict(dim=8192, n_heads=64, n_layers=80), # 65.2B
  }[model_type]
  config = LlamaConfig(**config_args)
  model = Transformer(config)
  checkpoint = f'huggyllama/llama-{model_type.lower()}'
  if assign: model, tokenizer = _safetensor_load(model, checkpoint)
  else: model, tokenizer = _copy_from_hf(model, checkpoint, half)
  if half: model = model.half()
  return model, tokenizer
