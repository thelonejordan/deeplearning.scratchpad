from pathlib import Path
from sentencepiece import SentencePieceProcessor

class Tokenizer:
  def __init__(self, model_path: str):
    assert Path(model_path).exists(), model_path
    self._model = SentencePieceProcessor(model_file=model_path)
    assert self._model.vocab_size() == self._model.get_piece_size()
    self.n_words = self._model.vocab_size()

  @property
  def bos_id(self) -> int: return self._model.bos_id()
  @property
  def eos_id(self) -> int: return self._model.eos_id()
  @property
  def pad_id(self) -> int: return self._model.pad_id()

  def encode(self, s: str, bos: bool=True, eos: bool=False) -> list[int]:
    assert isinstance(s, str)
    t: list[int] = self._model.encode(s)
    if bos: t = [self.bos_id] + t
    if eos: t = t + [self.eos_id]
    return t

  def decode(self, t: list[int]) -> str:
    return self._model.decode(t)
