"""MLX-native implementations of the TurboQuant dequant pipeline.

Provides the host/GPU-side math needed to unpack + inverse-rotate +
codebook-lookup + shape-gain-scale a 3-bit packed weight into bf16 or
fp16, entirely in MLX. This exists so TQ3 checkpoints can be served on
Apple Silicon through ``mlx-lm`` without falling back to the PyTorch
CPU path (which runs at ~0.008 tok/s for 30B MoE models).

Scope for v1:
  - Pure MLX primitives (``mlx.core``), including the native
    ``mx.hadamard_transform`` op (Metal-optimized on Apple Silicon).
  - Dequant-to-bf16 + ``mx.matmul``. No fused dequant-GEMM yet.
  - Matches ``PolarQuantTorch.dequantize`` numerically within float32
    tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PolarQuantStateMLX:
    """Immutable quantizer state needed by the dequant pipeline.

    Bundled so callers pass a single object instead of six positional
    arguments. Constructed once per (group_size, bits) tuple at model
    load time and reused across every forward.
    """

    signs1: mx.array
    signs2: mx.array
    centroids: mx.array
    dim: int

    @property
    def padded_dim(self) -> int:
        p = 1
        while p < self.dim:
            p <<= 1
        return p


def fast_wht_batch_mlx(x: mx.array) -> mx.array:
    """Batched fast Walsh-Hadamard transform, orthonormally normalized.

    Thin wrapper over MLX's native ``mx.hadamard_transform``, which is
    Metal-optimized on Apple Silicon and handles the ``1/sqrt(n)`` scale
    by default. Kept as its own function so call sites read symmetrically
    with the PyTorch reference path.
    """
    return mx.hadamard_transform(x)


def unpack_indices_3bit_mlx(packed: mx.array, dim: int) -> mx.array:
    """Unpack 3-bit indices from uint8 into int32.

    Layout matches ``turboquant_vllm.weight_quant.unpack_indices``:
    8 values x 3 bits = 24 bits = 3 bytes per group. Values at packed
    positions 2 and 5 cross byte boundaries — that's the only part of
    this math that isn't obvious.
    """
    n_rows, n_packed = packed.shape
    n_groups_of_3 = n_packed // 3

    p = packed.reshape(n_rows, n_groups_of_3, 3).astype(mx.int32)
    b0 = p[:, :, 0]
    b1 = p[:, :, 1]
    b2 = p[:, :, 2]

    v0 = b0 & 0x7
    v1 = (b0 >> 3) & 0x7
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x7  # cross-byte: low 2 bits from b0, high bit from b1
    v3 = (b1 >> 1) & 0x7
    v4 = (b1 >> 4) & 0x7
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x7  # cross-byte: low 1 bit from b1, high 2 bits from b2
    v6 = (b2 >> 2) & 0x7
    v7 = (b2 >> 5) & 0x7

    unpacked = mx.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=-1)
    unpacked = unpacked.reshape(n_rows, n_groups_of_3 * 8)
    return unpacked[:, :dim]


def polar_quant_dequantize_mlx(
    indices: mx.array,
    norms: mx.array,
    state: PolarQuantStateMLX,
    output_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Full dequant pipeline: codebook lookup -> inverse WHT -> shape-gain scale.

    Matches ``PolarQuantTorch.dequantize``. Output is a dense unpacked
    tensor of shape ``(batch, state.dim)`` in ``output_dtype``.
    """
    # Codebook lookup, then pad to power-of-two for the WHT
    y_hat = state.centroids[indices].astype(mx.float32)
    padded_dim = state.padded_dim
    if padded_dim > state.dim:
        y_hat = mx.pad(y_hat, [(0, 0), (0, padded_dim - state.dim)])

    # Inverse rotation: signs2 -> Hadamard -> signs1, slice back to dim
    y_hat = y_hat * state.signs2[None, :]
    y_hat = fast_wht_batch_mlx(y_hat)
    y_hat = y_hat * state.signs1[None, :]
    x_hat_unit = y_hat[:, : state.dim]

    # Shape-gain scaling (Gray 1984): norms stores original / reconstruction,
    # so a single multiply restores the original group magnitude.
    x_hat = x_hat_unit * norms[:, None]

    return x_hat.astype(output_dtype)
