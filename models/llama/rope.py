from typing import Optional, Tuple

import torch
from torch import Tensor, LongTensor


# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  assert dim % 2 == 0, f"dim must be even, {dim=}"
  freqs = torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).neg() / dim) # 1/(theta ^ 2d) for each d < dim/2
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  freqs = torch.outer(t, freqs) # m/(theta ^ 2d) for each m < end, d < dim/2
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, (end, dim/2)
  return freqs_cis

# note: x{q,k} is (bsz, seqlen, n_head, head_dim)
def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(torch.unflatten(xq.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  xk_ = torch.view_as_complex(torch.unflatten(xk.float(), -1, (-1, 2))) # (bsz, seqlen, n_head, head_dim/2)
  assert 1 < xq_.ndim and freqs_cis.size() == (xq_.shape[1], xq_.shape[-1])
  freqs_cis = freqs_cis[:, None, :] # reshape_for_broadcast, (seqlen, 1, head_dim/2)
  xq_out = torch.view_as_real(xq_ * freqs_cis).reshape_as(xq) # (bsz, seqlen, n_head, head_dim)
  xk_out = torch.view_as_real(xk_ * freqs_cis).reshape_as(xk) # (bsz, seqlen, n_head, head_dim)
  return xq_out.type_as(xq), xk_out.type_as(xk)


# TODO: add support for huggingface style RoPE implementation (sliced rotary)

# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py

def _compute_default_rope_parameters(base: float, dim: int, device: Optional[torch.device] = None) -> tuple[Tensor, float]:

  attention_factor = 1.0  # Unused in this type of RoPE

  # Compute the inverse frequencies
  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
  return inv_freq, attention_factor

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

@torch.no_grad()
def rotary_emb_forward(x: Tensor, position_ids: LongTensor, rope_theta: float, head_dim: int):
  inv_freq, attention_scaling = _compute_default_rope_parameters(rope_theta, head_dim, x.device)
  inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
  position_ids_expanded = position_ids[:, None, :].float()

  device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
  with torch.autocast(device_type=device_type, enabled=False):  # Force float32
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

  return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x: Tensor) -> Tensor:
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int=1):
  """Applies Rotary Position Embedding to the query and key tensors.

  Args:
      q (`torch.Tensor`): The query tensor.
      k (`torch.Tensor`): The key tensor.
      cos (`torch.Tensor`): The cosine part of the rotary embedding.
      sin (`torch.Tensor`): The sine part of the rotary embedding.
      unsqueeze_dim (`int`, *optional*, defaults to 1):
          The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
          sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
          that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
          k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
          cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
          the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
  Returns:
      `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed
