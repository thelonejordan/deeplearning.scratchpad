# CPU=1 PYTHONPATH=. python3 models/llama3/main.py

# https://github.com/meta-llama/llama3/blob/main/llama/model.py

import os
from models.llama3.generate import Llama
from models.helpers import set_device, set_seed

def main():

  device = set_device()
  set_seed(device)

  model = Llama.from_pretrained(max_batch_size=2, model_desc='3B', version=2).to(device)

  prompts = [
    "Simply put, the theory of relativity states that",
    "The phenomenon of global warming refers to the",
  ]

  out = model.text_completion(prompts, max_gen_len=64, temperature=0.9, echo=True)
  assert len(out) == len(prompts)
  print('-' * 50)
  for item in out:
    text = item['generation']
    print(text)
    print('-' * 50)

import torch
import torch.nn.functional as F
from models.llama3.transformer import Transformer
from models.llama3.tokenizer import Tokenizer
from models.llama3.config import LlamaConfig

@torch.inference_mode()
def generate(prompt_tokens: list[str],model: Transformer, tokenizer: Tokenizer, config: LlamaConfig, device, logprobs: bool=False):
  max_batch_size, max_seq_len = config.max_batch_size, config.max_seq_len
  bsz = len(prompt_tokens)
  assert bsz <= max_batch_size, (bsz, max_batch_size)
  max_gen_len = config.max_seq_len
  min_prompt_len = min(len(t) for t in prompt_tokens)
  max_prompt_len = max(len(t) for t in prompt_tokens)
  assert max_prompt_len <= max_seq_len
  total_len = min(max_seq_len, max_gen_len + max_prompt_len)
  pad_id = tokenizer.pad_id
  tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
  for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
  if logprobs:
    token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
  prev_pos = 0
  eos_reached = torch.tensor([False] * bsz, device=device)
  input_text_mask = tokens != pad_id
  if min_prompt_len == total_len:
    logits = model.forward(tokens, prev_pos)
    token_logprobs = -F.cross_entropy(
      input=logits.transpose(1, 2),
      target=tokens,
      reduction="none",
      ignore_index=pad_id,
    )
  stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device=device)

  for cur_pos in range(min_prompt_len, total_len):
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    probs = torch.softmax(logits[:, -1], dim=-1)
    next_token = torch.argmax(logits[:, -1], dim=-1)
    next_token = next_token.reshape(-1)
    # only replace token if prompt has already been generated
    next_token = torch.where(
      input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token
    if logprobs:
      token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
        input=logits.transpose(1, 2),
        target=tokens[:, prev_pos + 1 : cur_pos + 1],
        reduction="none",
        ignore_index=pad_id,
      )
    eos_reached |= (~input_text_mask[:, cur_pos]) & (
      torch.isin(next_token, stop_tokens)
    )
    prev_pos = cur_pos
    if all(eos_reached):
      break
  if logprobs:
    token_logprobs = token_logprobs.tolist()  # type: ignore
  out_tokens, out_logprobs = [], []
  for i, toks in enumerate(tokens.tolist()):
    # cut to max gen len
    start = 0
    toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
    if logprobs:
      probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
    # cut to after eos tok if any
    for stop_token in tokenizer.stop_tokens:
      try:
        eos_idx = toks.index(stop_token)
        toks = toks[:eos_idx]
        if logprobs:
          probs = probs[:eos_idx]
      except ValueError:
        pass
    out_tokens.append(toks)
    out_logprobs.append(probs)
  return out_tokens, (out_logprobs if logprobs else None)


def my_main(prompts):
  import numpy as np
  from models.llama3.load import build
  device = "mps"
  model, tokenizer, config = build(
    max_seq_len=30,
    max_batch_size=1,
    model_desc="3B", version=2
  )
  model = model.to(device)
  # prompts = ["The theory of relativity states that"]
  print(prompts[0])
  tokenizer.pad_id = tokenizer.eos_id
  inputs = [tokenizer.encode(s, bos=True, eos=False) for s in prompts]
  print(inputs)
  # [[128000, 791, 10334, 315, 1375, 44515, 5415, 430]]
  outputs, logprobs = generate(inputs, model, tokenizer, config, device, logprobs=False)
  print(outputs)
  # [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 304, 682, 5905, 14418, 315, 5905, 13, 578, 10334, 315, 1375, 44515, 374, 264, 10334, 315]]
  outputs = [i[i.index(tokenizer.bos_id)+1:]for i in outputs]
  texts = [tokenizer.decode(toks) for toks in outputs]
  print(texts[0])
  # print(np.exp(np.array(logprobs, dtype=np.float32)).tolist())
  return model, tokenizer, config, None

def hf_main(prompts):
  from transformers import AutoTokenizer, LlamaForCausalLM
  device = "mps"
  model_id = "meta-llama/Llama-3.2-3B"
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
  # prompts = ["The theory of relativity states that"]
  print(prompts[0])
  inputs = tokenizer(prompts, return_tensors="pt")
  print(inputs["input_ids"].numpy().tolist())
  # [[128000, 791, 10334, 315, 1375, 44515, 5415, 430]]
  inputs = {k:v.to(device) for k,v in inputs.items()}
  model = LlamaForCausalLM.from_pretrained(model_id).to(device)
  model.generation_config.pad_token_id = model.config.eos_token_id
  print(model.generation_config)
  outputs = model.generate(**inputs, max_length=30, do_sample=False, temperature=None,top_p=None)
  print(outputs.cpu().numpy().tolist())
  # [[128000, 791, 10334, 315, 1375, 44515, 5415, 430, 279, 4732, 315, 3177, 374, 6926, 304, 682, 5905, 14418, 13, 1115, 3445, 430, 422, 499, 527, 7366, 520, 264, 6926, 4732]]
  texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  print(texts[0])
  # The theory of relativity states that the speed of light is constant in all reference frames. This means that if you are moving at a constant speed
  return model, tokenizer, model.config, model.generation_config


if __name__ == "__main__":
  from dataclasses import dataclass
  from typing import Any

  @dataclass
  class Benchmark:
    model: torch.nn.Module
    tokenizer: Any
    config: Any
    generation_config: Any
  # main()
  prompts = ["The theory of relativity states that"]
  # if os.getenv("HF", "0") == "1":
  out = hf_main(prompts)
  bench_hf = Benchmark(*out)
  # else:
  out = my_main(prompts)
  bench_hf = Benchmark(*out)
