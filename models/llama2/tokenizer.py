from typing import Literal, TypedDict

from models.llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
  role: Role
  content: str


Dialog = list[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


def preprocess_dialog(dialog: Dialog):
  if dialog[0]["role"] == "system":
    dialog = [
      {
        "role": dialog[1]["role"],
        "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
      }
    ] + dialog[2:]
  return dialog


def encode_dialog_prompt(tokenizer: Tokenizer, dialog: Dialog):

  dialog = preprocess_dialog(dialog)
  assert all([msg["role"] == "user" for msg in dialog[::2]]) and all([msg["role"] == "assistant" for msg in dialog[1::2]]), (
    "model only supports 'system', 'user' and 'assistant' roles, "
    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
  )
  dialog_tokens: list[int] = sum(
    [
      tokenizer.encode(
        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ", bos=True, eos=True)
      for prompt, answer in zip(dialog[::2], dialog[1::2])
    ],
    [],
  )
  assert dialog[-1]["role"] == "user", f"Last message must be from user, got {dialog[-1]['role']}"
  dialog_tokens += tokenizer.encode(
    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
    bos=True,
    eos=False,
  )
  return dialog_tokens
