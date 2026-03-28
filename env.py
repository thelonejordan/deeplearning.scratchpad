#!/usr/bin/env python3

import os
import sys
import subprocess
import torch
import torchvision

import huggingface_hub
import transformers
import sentencepiece
import tiktoken

from models.helpers import set_device
from models.helpers import SAFETENSORS, CHAT, SDPA, TESTING_MINIMAL

def get_git_info():
  try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    return f"thelonejordan/deeplearning.scratchpad {commit}"
  except subprocess.CalledProcessError:
    return "thelonejordan/deeplearning.scratchpad"

print(get_git_info())
print("Python:", sys.version)
print("Packages:")
print(f"- {torch.__name__}: {torch.__version__}")
print(f"- {torchvision.__name__}: {torchvision.__version__}")
print(f"- {huggingface_hub.__name__}: {huggingface_hub.__version__}")
print(f"- {transformers.__name__}: {transformers.__version__}")
print(f"- {tiktoken.__name__}: {tiktoken.__version__}")
print(f"- {sentencepiece.__name__}: {sentencepiece.__version__}")
print("Flags:")
print(f"- {SAFETENSORS.key}: {SAFETENSORS.value}")
print(f"- {SDPA.key}: {SDPA.value}")
print(f"- {CHAT.key}: {CHAT.value}")
print(f"- {TESTING_MINIMAL.key}: {TESTING_MINIMAL.value}")
print("Device Flags:")
print(f"- CUDA: {os.getenv('CUDA')}")
print(f"- MPS: {os.getenv('MPS')}")
print(f"- CPU: {os.getenv('CPU')}")
print("Device:")
print(f"- Defaut device: {set_device(quiet=True).type}")
