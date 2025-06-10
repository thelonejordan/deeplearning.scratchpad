from __future__ import annotations
from typing import Optional

from models.helpers import timeit, Generator
from models.llama2.transformer import Transformer
from models.llama2.generate import generate, Dialog, ChatPrediction
from models.qwen2.config import QwenConfig
from models.qwen2.load import build, ModelOptions, ModelSizes


class Qwen(Generator):
  def __init__(self, model: Transformer, tokenizer, config: QwenConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, model_desc: ModelOptions="qwq", model_size: ModelSizes="32B",
                      preview: bool=True, instruct: bool=False, force_dtype: Optional[str]=None) -> Qwen:
    model, tokenizer, config = build(
      max_seq_len, max_batch_size, model_desc, model_size, preview, instruct, force_dtype=force_dtype
    )
    return Qwen(model, tokenizer, config)

  def chat_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool = False):
    return chat_completion(self, prompts, temperature, top_p, max_gen_len, logprobs)


def chat_completion(generator: Qwen, dialogs: list[Dialog], temperature: float=0.6, top_p: float=0.9,
                    max_gen_len: Optional[int]=None, logprobs: bool=False) -> list[ChatPrediction]:
  if max_gen_len is None:
    max_gen_len = generator.config.max_seq_len - 1
  prompt_tokens = []
  dialogs = generator.tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True)
  prompt_tokens = generator.tokenizer(dialogs)["input_ids"]
  generation_tokens, generation_logprobs = generate(
    generator, prompt_tokens=prompt_tokens, max_gen_len=max_gen_len,
    temperature=temperature, top_p=top_p, logprobs=logprobs,
  )
  generation_texts = generator.tokenizer.batch_decode(generation_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  if logprobs:
    return [
      {
        "generation": {
          "role": "assistant",
          "content": g,
        },
        "tokens": [generator.tokenizer.decode(x) for x in t],
        "logprobs": logprobs_i,
      }
      for t, g, logprobs_i in zip(
        generation_tokens, generation_texts, generation_logprobs
      )
    ]
  return [
    {
      "generation": {
        "role": "assistant",
        "content": g,
      }
    }
    for g in generation_texts
  ]
