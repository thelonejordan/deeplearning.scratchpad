import pathlib
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import safetensors.torch
import torch
from models.helpers import timeit
from models.qwen.config import QwenConfig
from models.qwen.transformer import Transformer

def _safetensors_load(repo_id: str):
  ckpt_dir = snapshot_download(repo_id, allow_patterns="*.safetensors")
  checkpoints = sorted(pathlib.Path(ckpt_dir).glob("*.safetensors"))
  assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
  state_dict = {}
  for ckpt in checkpoints:
    state_dict.update(safetensors.torch.load_file(ckpt))
  return state_dict


def huggingface_repo_id(preview: bool=True):
  p = "-Preview" if preview else ""
  return f"Qwen/QwQ-32B{p}"

@timeit(desc="Load time", ms=False)
def build(max_seq_len: int, max_batch_size: int, preview: bool=True, seed: int=1):
  repo_id = huggingface_repo_id(preview)
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  config = QwenConfig.build(max_seq_len, max_batch_size)
  # TODO: assert config and tokenizer have the same vocab size
  state_dict = _safetensors_load(repo_id)
  torch.set_default_dtype(config.torch_dtype)
  model = Transformer(config)
  model.load_state_dict(state_dict, assign=True, strict=False)
  return model, tokenizer


if __name__ == "__main__":
  # https://huggingface.co/Qwen/QwQ-32B-Preview
  # https://qwenlm.github.io/blog/qwq-32b-preview/
  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py
  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
  repo_id = "Qwen/QwQ-32B-Preview"
  state_dict = _safetensors_load(repo_id)
  for k, v in state_dict.items():
    print(k, ":", v.shape)

  ###

  build(8, 8)
