from __future__ import annotations
from typing import Optional
from models.llama2.transformer import Transformer
from models.llama2.generate import generate, Dialog, ChatPrediction
from models.qwen.config import QwQConfig
from models.helpers import timeit, Generator
from models.qwen.load import build

class QwQ(Generator):
  def __init__(self, model: Transformer, tokenizer, config: QwQConfig):
    self.model, self.tokenizer, self.config = model, tokenizer, config

  @staticmethod
  @timeit(desc="Load time", ms=False)
  def from_pretrained(max_seq_len: int=512, max_batch_size: int=8, preview: bool=True, force_dtype: Optional[str]=None) -> QwQ:
    model, tokenizer, config = build(max_seq_len, max_batch_size, preview, force_dtype=force_dtype)
    return QwQ(model, tokenizer, config)

  # def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
  #                     max_gen_len: Optional[int]=None, logprobs: bool = False, echo: bool = False):
  #   return text_completion(self, prompts, temperature, top_p, max_gen_len, logprobs, echo)

  def chat_completion(self, prompts: list[str], temperature: float=0.6, top_p: float=0.9,
                      max_gen_len: Optional[int]=None, logprobs: bool = False):
    return chat_completion(self, prompts, temperature, top_p, max_gen_len, logprobs)


def chat_completion(generator: QwQ, dialogs: list[Dialog], temperature: float=0.6, top_p: float=0.9,
                    max_gen_len: Optional[int]=None, logprobs: bool=False) -> list[ChatPrediction]:
  """
  Generate assistant responses for a list of conversational dialogs using the language generation model.

  Args:
    dialogs (list[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
    temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
    top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
    max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
      If not provided, it's set to the model's maximum sequence length minus 1.
    logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

  Returns:
    list[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

  Raises:
    AssertionError: If the last message in a dialog is not from the user.
    AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

  Note:
    This method generates assistant responses for the provided conversational dialogs.
    It employs nucleus sampling to introduce controlled randomness in text generation.
    If logprobs is True, token log probabilities are computed for each generated token.

  """
  if max_gen_len is None:
    max_gen_len = generator.config.max_seq_len - 1
  prompt_tokens = []
  unsafe_requests = []
  dialogs = generator.tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True)
  prompt_tokens = generator.tokenizer(dialogs)
  generation_tokens, generation_logprobs = generate(
    generator, prompt_tokens=prompt_tokens, max_gen_len=max_gen_len,
    temperature=temperature, top_p=top_p, logprobs=logprobs,
  )
  if logprobs:
    return [
      {
        "generation": {
          "role": "assistant",
          "content": generator.tokenizer.decode(t),
        },
        "tokens": [generator.tokenizer.decode(x) for x in t],
        "logprobs": logprobs_i,
      }
      for t, logprobs_i, unsafe in zip(
        generation_tokens, generation_logprobs, unsafe_requests
      )
    ]
  return [
    {
      "generation": {
        "role": "assistant",
        "content": generator.tokenizer.decode(t),
      }
    }
    for t, unsafe in zip(generation_tokens, unsafe_requests)
  ]
