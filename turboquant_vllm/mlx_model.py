"""MLX model-side wrapper for TurboQuant weight compression.

Provides ``TurboQuantMLXLinear`` — a drop-in replacement for
``mlx.nn.Linear`` that stores packed TQ3 weights and dequantizes on
every forward via the primitives in ``mlx_ops``. This is the last
piece needed to serve a TQ3 checkpoint through ``mlx-lm`` on Apple
Silicon without falling back to the CPU path.

v1 scope (this module):
  - Per-Linear wrapper, forward-only
  - Accepts packed weights + shape-gain norms loaded from a TQ3
    checkpoint shard (see ``load_tq3_weights_into_linear``)
  - Shares one ``PolarQuantStateMLX`` instance per (group_size, bits)
    tuple across the whole model

Next (Phase 5):
  - Loader that walks an ``mlx_lm`` model architecture and replaces
    each ``nn.Linear`` with ``TurboQuantMLXLinear`` whose weights come
    from our native TQ3 safetensors shards
  - Integration with ``mlx_lm.server``
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from turboquant_vllm.mlx_ops import (
    PolarQuantStateMLX,
    polar_quant_dequantize_mlx,
    unpack_indices_3bit_mlx,
)
from turboquant_vllm.weight_quant import padded_size


class TurboQuantMLXLinear(nn.Module):
    """MLX Linear that holds packed TQ3 weights and dequantizes on forward.

    The dequant path is:
        packed uint8 -> unpack_indices_3bit -> codebook lookup ->
        inverse WHT -> shape-gain scale -> matmul with the activation.

    Per-forward dequant is the v1 design (same as the PyTorch CPU path);
    a fused Metal dequant-GEMM kernel is a Phase 6 follow-up once quality
    parity is established.

    Args:
        packed_weight: uint8 array of shape ``(out_features, k_packed)``
            where ``k_packed = (padded_in // 8) * 3`` for 3-bit.
        norms: float32 array of shape ``(out_features, n_groups)``
            with shape-gain scales (original_norm / reconstruction_norm).
        state: shared ``PolarQuantStateMLX`` (dim == group_size).
        in_features: original input dim before padding.
        out_features: output dim.
        bias: optional bias array of shape ``(out_features,)``.
    """

    def __init__(
        self,
        packed_weight: mx.array,
        norms: mx.array,
        state: PolarQuantStateMLX,
        in_features: int,
        out_features: int,
        bias: mx.array | None = None,
    ):
        super().__init__()
        self.norms = norms
        self.quant_state = state
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.group_size = state.dim
        self.padded_in, self.n_groups = padded_size(in_features, self.group_size)
        self._pad_needed = self.padded_in > in_features

        # Unpack the packed uint8 weights once at init — indices are immutable
        # and repeated unpacking on the hot path dominated early profiling.
        # Keep packed_weight too so introspection / reload paths can see it.
        self.packed_weight = packed_weight
        self._indices_grouped = unpack_indices_3bit_mlx(
            packed_weight, dim=self.padded_in
        ).reshape(self.out_features * self.n_groups, self.group_size)
        self._norms_flat = norms.reshape(self.out_features * self.n_groups)

    def __call__(self, x: mx.array) -> mx.array:
        w_groups = polar_quant_dequantize_mlx(
            indices=self._indices_grouped,
            norms=self._norms_flat,
            state=self.quant_state,
            output_dtype=x.dtype,
        )
        w = w_groups.reshape(self.out_features, self.padded_in)

        if self._pad_needed:
            x = mx.pad(
                x, [(0, 0)] * (x.ndim - 1) + [(0, self.padded_in - x.shape[-1])]
            )

        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias.astype(out.dtype)
        return out
