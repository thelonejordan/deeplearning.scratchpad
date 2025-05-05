from typing import Optional, Callable, cast
import os
from time import perf_counter
from functools import wraps
import torch

def timeit(desc: Optional[str]=None, ms: bool=True):
  desc = "Time elapsed" if desc is None else desc
  def _timeit(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
      start = perf_counter()
      out = func(*args, **kwargs)
      stop = perf_counter()
      diff = stop - start
      text = None
      if ms: text = f"{desc}: {diff*1000:.2f}ms"
      else: text = f"{desc}: {diff:.2f}s"
      print(text)
      return out
    return wrapper
  return _timeit

def set_device(device: Optional[str]=None) -> torch.device:
  if os.getenv("CUDA") == "1": device = 'cuda'
  elif os.getenv("MPS") == "1": device = 'mps'
  elif os.getenv("CPU") == "1": device = 'cpu'
  elif device is None:
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
  print(f'Using device: {device}')
  return torch.device(device)

def set_seed(device: torch.device, seed: Optional[int]=None):
  if seed is None: seed = int(os.getenv("SEED", 420))
  torch.manual_seed(seed)
  if device.type == 'cuda': torch.cuda.manual_seed(seed)
  if device.type == 'mps': torch.mps.manual_seed(seed)


class Generator:
  @property
  def device(self) -> torch.device: return next(cast(torch.nn.Module, self.model).parameters()).device  # type: ignore
  @property
  def dtype(self) -> torch.dtype: return next(cast(torch.nn.Module, self.model).parameters()).dtype  # type: ignore
  def to(self, device: torch.device):
    self.model = cast(torch.nn.Module, self.model).to(device)  # type: ignore
    return self

SAFETENSORS = os.getenv("SAFETENSORS", "1") == "1"
