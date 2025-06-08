from typing import Literal
import math
import torch
from torch import Tensor, LongTensor

# TODO: add support for huggingface style RoPE implementation (sliced rotary)

# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py

def _compute_default_rope_parameters(dim: int, base: float, device: torch.device, **_) -> tuple[Tensor, float]:
  """
  Computes the inverse frequencies according to the original RoPE implementation
  Returns:
      Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
      post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
  """
  attention_factor = 1.0  # Unused in this type of RoPE

  # Compute the inverse frequencies
  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
  return inv_freq, attention_factor

def _compute_llama3_parameters(dim: int, base: float, device: torch.device, original_max_position_embeddings: int=8192,
                               factor: float=8.0, low_freq_factor: float=1.0, high_freq_factor:float=4.0) -> tuple[Tensor, float]:
    """
    Computes the inverse frequencies for llama 3.1.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(dim, base, device)

    # factor = factor  # `8` in the original implementation
    # low_freq_factor = low_freq_factor  # `1` in the original implementation
    # high_freq_factor = high_freq_factor  # `4` in the original implementation
    old_context_len = original_max_position_embeddings  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

RopeType = Literal["default", "llama3"]

# LlamaRotaryEmbedding.forward()
@torch.no_grad()
def rotary_emb_forward(x: Tensor, position_ids: LongTensor, dim: int, rope_theta: float,
                       rope_type: RopeType="default", **rope_kwargs) -> tuple[Tensor, Tensor]:
  rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
  inv_freq, attention_scaling = rope_init_fn(rope_theta, dim, x.device, **rope_kwargs)
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


# TODO: add it to the transformer class?
# def rope_usage(self: Transformer, hidden_states: Tensor, position_ids: LongTensor, layer_id: int,
#                dim: int, head_dim: int, rope_theta: float, rope_type: RopeType="default", **rope_kwargs) -> tuple[Tensor, Tensor]:
#   # LlamaModel.forward()
#   # create position embeddings to be shared across the decoder layers
#   position_embeddings = rotary_emb_forward(hidden_states, position_ids, dim, rope_theta, rope_type, **rope_kwargs)

#   # LlamaDecoderLayer.forward()
#   # LlamaAttention.forward()
#   input_shape = hidden_states.shape[:-1]
#   hidden_shape = (*input_shape, -1, head_dim)

#   query_states = self.model.layers[layer_id].self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#   key_states = self.model.layers[layer_id].self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#   cos, sin = position_embeddings
#   query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#   return query_states, key_states
