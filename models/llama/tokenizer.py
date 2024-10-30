from typing import List
from pathlib import Path
from sentencepiece import SentencePieceProcessor

class Tokenizer:
  def __init__(self, model_path: str):
    assert Path(model_path).exists(), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)
    print(f"Reloaded SentencePiece model from {model_path}")
    self.n_words: int = self.sp_model.vocab_size()
    print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  @property
  def bos_id(self) -> int: return self.sp_model.bos_id()
  @property
  def eos_id(self) -> int: return self.sp_model.eos_id()
  @property
  def pad_id(self) -> int: return self.sp_model.pad_id()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert isinstance(s, str)
    t = self.sp_model.encode(s)
    if bos: t = [self.bos_id] + t
    if eos: t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)
