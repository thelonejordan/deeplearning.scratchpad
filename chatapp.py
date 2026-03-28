import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
from models.helpers import set_device, set_seed
from models.qwen2.generate import Qwen

device = set_device()
set_seed(device)

generator = Qwen.from_pretrained(max_batch_size=1, repo_id="Qwen/Qwen2.5-0.5B-Instruct")
generator = generator.to(device)

def respond(message, history):
  dialog = [{"role": m["role"], "content": m["content"]} for m in history if isinstance(m.get("content"), str)]
  dialog.append({"role": "user", "content": message})
  out = generator.chat_completion([dialog])
  return out[0]["generation"]["content"]

gr.ChatInterface(fn=respond).launch()
