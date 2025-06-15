# PYTHONPATH=. python debug_qwen.py

import os
import unittest

from transformers import AutoTokenizer, Qwen2ForCausalLM
from models.qwen2.generate import Qwen
from models.llama2.generate import generate
from models.helpers import set_device
import torch

DEVICE = set_device()
MAX_SEQ_LEN = 256 - 218 -1

dialogs = [
  [
    dict(role="system", content="You are a helpful and truthful assistant. You should think step-by-step."),
    dict(role="user", content="How many r in strawberry.")
  ],
]

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(repo_id, torch_dtype="auto", device_map=DEVICE)

generator = Qwen.from_pretrained(
  max_seq_len=MAX_SEQ_LEN, max_batch_size=len(dialogs), repo_id="Qwen/Qwen2-0.5B-Instruct",
).to(DEVICE)

lm_head_hf = model.lm_head
lm_head = generator.model.lm_head
print(lm_head.weight.dtype)
print(lm_head_hf.weight.dtype)

assert torch.allclose(lm_head_hf.weight, lm_head.weight)

inp = torch.load(f"assets/post_norm.pt")[:, [-1], :]
inp_hf = torch.load(f"assets/post_norm_target.pt")[:, [-1], :]


out_hf_self = lm_head_hf(inp.to(model.device))
out_self = lm_head(inp.to(generator.device))

out_hf_hf = lm_head_hf(inp_hf.to(model.device))
out_hf = lm_head(inp_hf.to(generator.device))

assert torch.allclose(out_hf_self, out_hf_hf)
assert torch.allclose(out_self, out_hf)
assert torch.allclose(out_hf_hf, out_self)

# assert out.shape == out_hf.shape

# assert torch.allclose(out.cpu(), out_hf.cpu())
