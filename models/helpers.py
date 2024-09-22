from typing import Optional
from time import perf_counter
from functools import partial
import os, torch

def timer(func, desc, ms):
  desc = "Time elapsed" if desc is None else desc
  def inner(*args, **kwargs):
    start = perf_counter()
    out = func(*args, **kwargs)
    stop = perf_counter()
    diff = stop - start
    text = None
    if ms: text = f"{desc}: {diff*1000:.2f}ms"
    else: text = f"{desc}: {diff:.2f}s"
    print(text)
    return out
  return inner

def timeit(desc=None, ms=True):
  return partial(timer, desc=desc, ms=ms)

def set_device(device: Optional[str]=None) -> torch.device:
  if device is None:
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
    elif torch.backends.mps.is_available():
      device = 'mps'
  print(f'Using device: {device}')
  return torch.device(device)

def set_seed(device: Optional[torch.device]=None, seed: Optional[int]=None):
  if seed is None: seed = int(os.getenv("SEED", 420))
  if device is None: device = torch.device('cpu')
  torch.manual_seed(seed)
  if device.type == 'cuda': torch.cuda.manual_seed(seed)
  if device.type == 'mps': torch.mps.manual_seed(seed)
