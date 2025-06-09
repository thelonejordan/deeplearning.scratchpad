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
