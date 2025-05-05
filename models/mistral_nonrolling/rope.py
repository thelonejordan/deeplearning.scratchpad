from typing import Tuple

import torch
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = freqs_cis[:, None, :]
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)
