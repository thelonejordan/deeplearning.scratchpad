from typing import Set
from tqdm import tqdm
from models.helpers import timeit

import torch
from models.gpt2.transformer import Transformer, GPTConfig
from models.gpt2.tokenizer import Tokenizer

def _copy_from_hf(model: Transformer, checkpoint: str, half: bool=False, transposed: Set[str]=set(), skip: Set[str]=set()):
  from transformers import GPT2LMHeadModel
  model_hf = GPT2LMHeadModel.from_pretrained(checkpoint)
  if half: model, model_hf = model.half(), model_hf.half()
  sd, sd_hf = model.state_dict(), model_hf.state_dict()
  sd_keys = [k for k in sd.keys() if not any(k.endswith(w) for w in skip)]
  sd_keys_hf = [k for k in sd_hf.keys() if not any(k.endswith(w) for w in skip)]
  # copy while ensuring all of the parameters are aligned and match in names and shapes
  assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
  for k in tqdm(sd_keys_hf, desc=f'Loading pretrained weights'):
    if any(k.endswith(w) for w in transposed):
      assert sd_hf[k].shape[::-1] == sd[k].shape
      with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
    else:
      assert sd_hf[k].shape == sd[k].shape
      with torch.no_grad(): sd[k].copy_(sd_hf[k])
  return model

def _load_from_cache(model: Transformer, checkpoint: str, transposed: Set[str]=set(), skip: Set[str]=set()):
  from transformers.utils import try_to_load_from_cache
  import safetensors.torch
  safetensors_model_file = try_to_load_from_cache(repo_id=checkpoint, filename="model.safetensors")
  loaded = dict()
  for k, v in safetensors.torch.load_file(str(safetensors_model_file)).items():
    if any(k.endswith(w) for w in skip): continue
    loaded[k] = v.t() if any(k.endswith(w) for w in transposed) else v
  model.transformer.load_state_dict(loaded, assign=True, strict=True)
  return model

@timeit(desc="Load time", ms=False)
def from_pretrained(model_type: str='gpt2', half: bool=False, assign: bool=False):
  config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
  }[model_type]
  model = Transformer(GPTConfig(**config_args))
  transposed = {'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'}
  skip = {'.attn.masked_bias', '.attn.bias'}
  if assign: model = _load_from_cache(model, model_type, transposed, skip)
  else: model = _copy_from_hf(model, model_type, half, transposed, skip)
  if half: model = model.half()
  return model.apply_weight_sharing(), Tokenizer()
