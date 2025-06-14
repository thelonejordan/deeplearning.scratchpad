from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor, nn

from models.helpers import set_device


def _compute_default_rope_parameters(dim: int, head_dim: int, rope_theta: float=1000000., partial_rotary_factor: float=1.,
                                     device: Optional[torch.device] = None) -> tuple[Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = rope_theta
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


class Qwen2RotaryEmbedding(nn.Module):
  def __init__(self, dim: int, head_dim: int, max_position_embeddings: int, rope_theta: int, device=None):
    super().__init__()
    self.max_seq_len_cached = max_position_embeddings
    self.original_max_seq_len = max_position_embeddings

    with torch.device(set_device(quiet=True) if device is None else device):
      inv_freq, self.attention_scaling = _compute_default_rope_parameters(dim, head_dim, rope_theta, device=device)
      self.register_buffer("inv_freq", inv_freq, persistent=False)
      self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  def forward(self, x: Tensor, position_ids: Tensor):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: Tensor):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1):
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
