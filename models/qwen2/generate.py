from __future__ import annotations
from typing import Optional, Any

from models.helpers import timeit, Generator
from models.llama2.generate import generate, CompletionPrediction, ChatPrediction, Dialog
from models.qwen2.transformer import Transformer
from models.qwen2.config import QwenConfig
from models.qwen2.load import build


class Qwen(Transformer, Generator):
  def __init__(self, *args, **kwargs):
    assert "config" in kwargs and "tokenizer" in kwargs
    self.config: QwenConfig = kwargs.pop("config")
    self.tokenizer = kwargs.pop("tokenizer")
    super().__init__(*args, **kwargs)

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8,
                      repo_id: str="Qwen/Qwen2.5-0.5B", force_dtype: Optional[str]=None) -> Qwen:
    generator, _, __ = build(max_seq_len, max_batch_size, repo_id, force_dtype=force_dtype, model_class=Qwen)
    return generator

  @property
  def G(self):
    # HACK: AttributeError: Qwen2TokenizerFast has no attribute pad_id
    # (File "/workspace/deeplearning.scratchpad/models/llama2/generate.py")
    pad_id = self.tokenizer.encode(self.tokenizer.special_tokens_map['pad_token'])[0]
    eos_id = self.tokenizer.encode(self.tokenizer.special_tokens_map['eos_token'])[0]
    return self, self.tokenizer, self.config.max_seq_len, self.config.max_batch_size, pad_id, eos_id

  def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool = False, echo: bool = False):
    return text_completion(*self.G, prompts, temperature, top_p, max_gen_len, logprobs, echo)

  def chat_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool = False):
    return chat_completion(*self.G, prompts, temperature, top_p, max_gen_len, logprobs)


def text_completion(model: Qwen, tokenizer: Any, max_seq_len: int, max_batch_size: int, pad_id: int, eos_id: int,
                    prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                    max_gen_len: Optional[int]=None, logprobs: bool=False, echo: bool=False) -> list[CompletionPrediction]:
  """
  Perform text completion for a list of prompts using the language generation model.

  Args:
    prompts (list[str]): List of text prompts for completion.
    temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
    top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
    max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
      If not provided, it's set to the model's maximum sequence length minus 1.
    logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
    echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

  Returns:
    list[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

  Note:
    This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
    If logprobs is True, token log probabilities are computed for each generated token.
  """
  if max_gen_len is None:
    max_gen_len = max_seq_len - 1
  prompt_tokens = tokenizer(prompts)["input_ids"]
  generation_tokens, generation_logprobs = generate(
    model, max_seq_len, max_batch_size, pad_id, eos_id,
    prompt_tokens, max_gen_len, temperature, top_p, logprobs, echo,
  )
  generation_texts = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  if logprobs:
    return [
      {
        "generation": g,
        "tokens": [tokenizer.decode([x]) for x in t],
        "logprobs": logprobs_i,
      }
      for g, t, logprobs_i in zip(generation_texts, generation_tokens, generation_logprobs)  # type: ignore
    ]
  return [{"generation": g} for g in generation_texts]


def chat_completion(model: Qwen, tokenizer: Any, max_seq_len: int, max_batch_size: int, pad_id: int, eos_id: int,
                    dialogs: list[Dialog], temperature: float=0.6, top_p: float=0.9,
                    max_gen_len: Optional[int]=None, logprobs: bool=False) -> list[ChatPrediction]:
  if max_gen_len is None:
    max_gen_len = max_seq_len - 1
  prompt_tokens = []
  dialogs = tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True)
  prompt_tokens = tokenizer(dialogs)["input_ids"]
  generation_tokens, generation_logprobs = generate(
    model, max_seq_len, max_batch_size, pad_id, eos_id,
    prompt_tokens, max_gen_len, temperature, top_p, logprobs,
  )
  generation_texts = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  if logprobs:
    return [
      {
        "generation": {
          "role": "assistant",
          "content": g,
        },
        "tokens": [tokenizer.decode(x) for x in t],
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
