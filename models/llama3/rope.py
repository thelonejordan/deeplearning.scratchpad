import torch
from torch import Tensor

# https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py


def apply_scaling(freqs: Tensor) -> Tensor:
  # Values obtained from grid search
  scale_factor = 8
  low_freq_factor = 1
  high_freq_factor = 4
  old_context_len = 8192  # original llama3 length

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor

  wavelen = 2 * torch.pi / freqs
  new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
  smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
  return torch.where(
    (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
    (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
    new_freqs,
  )

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, use_scaled: bool=False):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  if use_scaled:
    freqs = apply_scaling(freqs)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)
