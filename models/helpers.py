from time import perf_counter
from functools import partial

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
