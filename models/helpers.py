from __future__ import annotations
from typing import Optional, Callable, ClassVar, TypeVar, cast
import os, contextlib, functools
from time import perf_counter
import torch

R = TypeVar('R')

def timeit(desc: Optional[str]=None, ms: bool=True):
  desc = "Time elapsed" if desc is None else desc
  def _decorator(func: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(func)
    def __wrapper(*args, **kwargs):
      start = perf_counter()
      out = func(*args, **kwargs)
      stop = perf_counter()
      diff = stop - start
      text = None
      if ms: text = f"{desc}: {diff*1000:.2f}ms"
      else: text = f"{desc}: {diff:.2f}s"
      print(text)
      return out
    return __wrapper
  return _decorator

def set_device(device: Optional[str]=None, quiet: bool=False) -> torch.device:
  if device is not None: pass
  elif getenv("CUDA") == 1: device = 'cuda'
  elif getenv("MPS") == 1: device = 'mps'
  elif getenv("CPU") == 1: device = 'cpu'
  if device is None:
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
  if not quiet:
    print(f'Using device: {device}')
  os.environ["CUDA"] = str(int(device=="cuda"))
  os.environ["MPS"] = str(int(device=="mps"))
  os.environ["CPU"] = str(int(device=="cpu"))
  return torch.device(device)

def set_seed(device: torch.device, seed: Optional[int]=None):
  if seed is None: seed = getenv("SEED", 420)
  torch.manual_seed(seed)
  if device.type == 'cuda': torch.cuda.manual_seed(seed)
  if device.type == 'mps': torch.mps.manual_seed(seed)


class Generator:
  @property
  def device(self) -> torch.device: return next(cast(torch.nn.Module, self).parameters()).device  # type: ignore
  @property
  def dtype(self) -> torch.dtype: return next(cast(torch.nn.Module, self).parameters()).dtype  # type: ignore


def accept_extra_kwargs(method=True):
  def _decorator(func):
    @functools.wraps(func)
    def __wrapper(*args, **kwargs):
      accept = func.__code__.co_varnames
      if method: accept = accept[1:]  # remove self
      kwargs = {k:v for k,v in kwargs.items() if k in accept}
      return func(*args, **kwargs)
    return __wrapper
  return _decorator


# helpers from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py

@functools.cache
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

class Context(contextlib.ContextDecorator):
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    self.old_context:dict[str, int] = {k:v.value for k,v in ContextVar._cache.items()}
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v
  def __exit__(self, *args):
    for k,v in self.old_context.items(): ContextVar._cache[k].value = v

class ContextVar:
  _cache: ClassVar[dict[str, ContextVar]] = {}
  value: int
  key: str
  def __init__(self, key, default_value):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x


SAFETENSORS = ContextVar("SAFETENSORS", 1)
SDPA = ContextVar("SDPA", 1)
CHAT = ContextVar("CHAT", 0)
TESTING_MINIMAL = ContextVar("TESTING_MINIMAL", 0)
