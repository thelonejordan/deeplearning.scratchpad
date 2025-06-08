import torch
from torch import Tensor

# https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  assert dim % 2 == 0, f"dim must be even, {dim=}"
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))  # 1/(theta ^ 2d) for each d < dim/2
  t = torch.arange(end, device=freqs.device)
  freqs = torch.outer(t, freqs).float()  # m/(theta ^ 2d) for each m < end, d < dim/2
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, (end, dim/2)
  return freqs_cis

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)  # same as freqs_cis[:, None, :]

# note: x{q,k} is (bsz, seqlen, n_head, head_dim)
def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  xq_ = torch.view_as_complex(xq.float().unflatten(dim=-1, sizes=(-1, 2)))  # (bsz, seqlen, n_head, head_dim/2)
  xk_ = torch.view_as_complex(xk.float().unflatten(dim=-1, sizes=(-1, 2)))  # (bsz, seqlen, n_head, head_dim/2)
  assert 1 < xq_.ndim and freqs_cis.size() == (xq_.shape[1], xq_.shape[-1])
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # (seqlen, 1, head_dim/2)
  xq_out = torch.view_as_real(xq_ * freqs_cis).reshape_as(xq)  # (bsz, seqlen, n_head, head_dim)
  xk_out = torch.view_as_real(xk_ * freqs_cis).reshape_as(xk)  # (bsz, seqlen, n_head, head_dim)
  return xq_out.type_as(xq), xk_out.type_as(xk)
