# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant online weight quantization for vLLM.

3-4 bit weight compression via WHT rotation + Lloyd-Max codebook.
Load any BF16 checkpoint, compress weights at startup, serve with
~4x smaller GPU memory. Zero calibration data needed.

Based on TurboQuant (Zandieh et al., ICLR 2026).

Usage:
    vllm serve <model> --quantization turboquant
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Codebook: Lloyd-Max optimal centroids for N(0, 1/d)
# ---------------------------------------------------------------------------


def _gaussian_cond_expect(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] for X ~ N(0, sigma^2)."""
    from scipy import stats

    a_std = a / sigma if math.isfinite(a) else a
    b_std = b / sigma if math.isfinite(b) else b
    if not math.isfinite(a_std):
        prob = stats.norm.cdf(b_std)
    elif not math.isfinite(b_std):
        prob = stats.norm.sf(a_std)
    else:
        prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)
    if prob < 1e-15:
        return (a + b) / 2.0 if math.isfinite(a) and math.isfinite(b) else 0.0
    return sigma * (stats.norm.pdf(a_std) - stats.norm.pdf(b_std)) / prob


def _lloyd_max_centroids(n: int, sigma: float, n_iter: int = 100) -> list[float]:
    """Lloyd's algorithm for optimal scalar quantization of N(0, sigma^2)."""
    from scipy import stats

    boundaries = list(stats.norm.ppf([i / n for i in range(1, n)], scale=sigma))
    centroids = [0.0] * n
    for _ in range(n_iter):
        centroids[0] = _gaussian_cond_expect(sigma, -math.inf, boundaries[0])
        for i in range(1, n - 1):
            centroids[i] = _gaussian_cond_expect(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_cond_expect(sigma, boundaries[-1], math.inf)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n - 1)]
    return sorted(centroids)


def _optimal_centroids(bits: int, dim: int) -> list[float]:
    """Optimal centroids for post-rotation coordinates ~ N(0, 1/d)."""
    n = 1 << bits
    if bits == 1:
        c = math.sqrt(2.0 / (math.pi * dim))
        return [-c, c]
    if bits == 2:
        s = math.sqrt(dim)
        return [-1.51 / s, -0.453 / s, 0.453 / s, 1.51 / s]
    return _lloyd_max_centroids(n, sigma=1.0 / math.sqrt(dim))


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------


def _fast_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched fast WHT. x: (batch, n) where n is power of 2."""
    n = x.shape[1]
    h = 1
    while h < n:
        x_view = x.view(x.shape[0], n // (h * 2), 2, h)
        a = x_view[:, :, 0, :].clone()
        b = x_view[:, :, 1, :].clone()
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        h *= 2
    return x / math.sqrt(n)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# PolarQuant quantizer
# ---------------------------------------------------------------------------

# Cache: (group_size, bits, device_str) → PolarQuant instance
_quantizers: dict[tuple[int, int, str], "_PolarQuant"] = {}


def _get_quantizer(group_size: int, bits: int, device: str) -> "_PolarQuant":
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (group_size, bits, str(dev))
    if key not in _quantizers:
        _quantizers[key] = _PolarQuant(group_size, bits, device=str(dev))
    return _quantizers[key]


class _PolarQuant:
    """WHT rotation + Gaussian Lloyd-Max codebook quantizer."""

    def __init__(self, dim: int, bits: int, seed: int = 42, device: str = "cuda"):
        self.dim = dim
        self.bits = bits
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.padded_dim = _next_pow2(dim)
        self.signs1 = (torch.randint(0, 2, (self.padded_dim,), generator=gen) * 2 - 1).float().to(dev)
        self.signs2 = (torch.randint(0, 2, (self.padded_dim,), generator=gen) * 2 - 1).float().to(dev)

        centroids_list = _optimal_centroids(bits, dim)
        self.centroids = torch.tensor(centroids_list, dtype=torch.float32, device=dev)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=x.device, dtype=x.dtype)
            padded[:, : self.dim] = x
        else:
            padded = x.clone()
        padded *= self.signs1.unsqueeze(0)
        padded = _fast_wht_batch(padded)
        padded *= self.signs2.unsqueeze(0)
        return padded[:, : self.dim]

    def _rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        batch = y.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=y.device, dtype=y.dtype)
            padded[:, : self.dim] = y
        else:
            padded = y.clone()
        padded *= self.signs2.unsqueeze(0)
        padded = _fast_wht_batch(padded)
        padded *= self.signs1.unsqueeze(0)
        return padded[:, : self.dim]

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize grouped vectors. x: (n_groups, group_size). Returns (indices, norms)."""
        x = x.to(device=self.device, dtype=torch.float32)
        norms = torch.linalg.norm(x, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_unit = x / safe_norms.unsqueeze(1)
        y = self._rotate(x_unit)
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        # Norm correction: store original_norm / reconstruction_norm
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        recon_norm = torch.linalg.norm(x_hat_unit, dim=1)
        safe_recon = torch.where(recon_norm > 0, recon_norm, torch.ones_like(recon_norm))
        norms = norms / safe_recon
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize. indices: (n_groups, group_size). Returns (n_groups, group_size)."""
        indices = indices.to(device=self.device)
        norms = norms.to(device=self.device, dtype=torch.float32)
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        return x_hat_unit * norms.unsqueeze(1)


# ---------------------------------------------------------------------------
# Bit packing: pack/unpack quantization indices into uint8
# ---------------------------------------------------------------------------


def _padded_size(dim: int, group_size: int) -> tuple[int, int]:
    """Return (padded_dim, n_groups) for group quantization."""
    padded = ((dim + group_size - 1) // group_size) * group_size
    return padded, padded // group_size


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into uint8."""
    if bits == 4:
        flat = indices.reshape(-1, indices.shape[-1])
        lo = flat[:, 0::2].to(torch.uint8)
        hi = flat[:, 1::2].to(torch.uint8)
        return (lo | (hi << 4)).reshape(indices.shape[0], -1)
    elif bits == 3:
        n_rows, n_cols = indices.shape[0], indices.shape[-1]
        flat = indices.reshape(n_rows, -1).to(torch.uint8)
        pad = (8 - n_cols % 8) % 8
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        n_packed_cols = flat.shape[1] // 8 * 3
        packed = torch.zeros(n_rows, n_packed_cols, dtype=torch.uint8, device=indices.device)
        for i in range(flat.shape[1] // 8):
            v = flat[:, i * 8 : (i + 1) * 8]
            packed[:, i * 3] = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)
            packed[:, i * 3 + 1] = (v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x1) << 7)
            packed[:, i * 3 + 2] = (v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)
        return packed
    elif bits == 2:
        flat = indices.reshape(-1, indices.shape[-1])
        shifts = torch.tensor([0, 2, 4, 6], device=indices.device, dtype=torch.uint8)
        parts = torch.stack([flat[:, i::4].to(torch.uint8) for i in range(4)], dim=-1)
        return (parts << shifts).sum(dim=-1).to(torch.uint8).reshape(indices.shape[0], -1)
    return indices.to(torch.uint8)


def _unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack uint8 packed indices back to int64."""
    if bits == 4:
        flat = packed.reshape(-1, packed.shape[-1])
        lo = (flat & 0x0F).to(torch.int64)
        hi = ((flat >> 4) & 0x0F).to(torch.int64)
        unpacked = torch.zeros(flat.shape[0], flat.shape[1] * 2, dtype=torch.int64, device=packed.device)
        unpacked[:, 0::2] = lo
        unpacked[:, 1::2] = hi
        return unpacked.reshape(packed.shape[0], -1)[:, :dim]
    elif bits == 3:
        flat = packed.reshape(-1, packed.shape[-1])
        n_rows = flat.shape[0]
        n_groups_of_3 = flat.shape[1] // 3
        unpacked = torch.zeros(n_rows, n_groups_of_3 * 8, dtype=torch.int64, device=packed.device)
        for i in range(n_groups_of_3):
            b0 = flat[:, i * 3].to(torch.int64)
            b1 = flat[:, i * 3 + 1].to(torch.int64)
            b2 = flat[:, i * 3 + 2].to(torch.int64)
            unpacked[:, i * 8 + 0] = b0 & 0x7
            unpacked[:, i * 8 + 1] = (b0 >> 3) & 0x7
            unpacked[:, i * 8 + 2] = ((b0 >> 6) | (b1 << 2)) & 0x7
            unpacked[:, i * 8 + 3] = (b1 >> 1) & 0x7
            unpacked[:, i * 8 + 4] = (b1 >> 4) & 0x7
            unpacked[:, i * 8 + 5] = ((b1 >> 7) | (b2 << 1)) & 0x7
            unpacked[:, i * 8 + 6] = (b2 >> 2) & 0x7
            unpacked[:, i * 8 + 7] = (b2 >> 5) & 0x7
        return unpacked[:, :dim]
    elif bits == 2:
        flat = packed.reshape(-1, packed.shape[-1])
        unpacked = torch.zeros(flat.shape[0], flat.shape[1] * 4, dtype=torch.int64, device=packed.device)
        for i in range(4):
            unpacked[:, i::4] = ((flat >> (i * 2)) & 0x03).to(torch.int64)
        return unpacked.reshape(packed.shape[0], -1)[:, :dim]
    return packed.to(torch.int64)


# ---------------------------------------------------------------------------
# Triton kernels (FWHT-on-input GEMM + fused dequant-GEMM)
# ---------------------------------------------------------------------------

# Lazy-loaded to avoid import errors when Triton isn't available
_triton_available: bool | None = None
_tq_fwht_input_fn = None
_tq_fused_gemm_fn = None


def _ensure_triton():
    """Lazy-load Triton kernels. Must be called before forward, not inside."""
    global _triton_available, _tq_fwht_input_fn, _tq_fused_gemm_fn
    if _triton_available is not None:
        return _triton_available
    try:
        # Import from the plugin's Triton ops if available,
        # otherwise fall back to PyTorch dequant path
        from turboquant_vllm.triton_ops import tq_fused_gemm, tq_fwht_input_gemm

        _tq_fused_gemm_fn = tq_fused_gemm
        _tq_fwht_input_fn = tq_fwht_input_gemm
        _triton_available = True
    except (ImportError, Exception):
        _triton_available = False
    return _triton_available


# ---------------------------------------------------------------------------
# TurboQuantOnlineLinearMethod
# ---------------------------------------------------------------------------


class TurboQuantOnlineLinearMethod(QuantizeMethodBase):
    """Online TQ3/TQ4 weight compression for Linear layers.

    Allocates bf16 weight on meta device (zero GPU at init). After
    weight loading materializes the bf16 on GPU, compresses to TQ
    packed format. Forward pass uses Triton dequant-GEMM kernels.
    """

    uses_meta_device: bool = True

    def __init__(self, bits: int = 3, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Guard: called twice (online processing + global sweep)
        if not hasattr(layer, "weight") or layer.weight.numel() == 0:
            return

        weight = layer.weight.data
        bits = self.bits
        group_size = self.group_size

        out_dim, in_dim = weight.shape
        padded_in, n_groups = _padded_size(in_dim, group_size)

        if padded_in > in_dim:
            padded = torch.zeros(out_dim, padded_in, dtype=weight.dtype, device=weight.device)
            padded[:, :in_dim] = weight
        else:
            padded = weight

        grouped = padded.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, str(weight.device))
        indices, norms_raw = quantizer.quantize(grouped)
        packed = _pack_indices(indices, bits)
        norms = norms_raw.reshape(out_dim, n_groups)

        # Keep weight attr for vLLM's MLA post-processing (expects it to exist)
        layer.weight.data = torch.empty(0, device=weight.device, dtype=weight.dtype)
        layer.register_buffer("tq_packed_weight", packed)
        layer.register_buffer("tq_norms", norms)
        layer.register_buffer("tq_signs1", quantizer.signs1)
        layer.register_buffer("tq_signs2", quantizer.signs2)
        layer.register_buffer("tq_centroids", quantizer.centroids)
        layer.tq_in_features = in_dim
        layer.tq_out_features = out_dim
        layer.tq_padded_in = padded_in

        # Cache Triton dispatch (must run before CUDA graph capture)
        _ensure_triton()
        if _triton_available:
            layer._tq_primary_fn = _tq_fwht_input_fn if out_dim >= 4096 else _tq_fused_gemm_fn
            layer._tq_fallback_fn = _tq_fused_gemm_fn if out_dim >= 4096 else _tq_fwht_input_fn
        else:
            layer._tq_primary_fn = None

        del weight, padded, grouped, indices, norms_raw

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer._tq_primary_fn is not None:
            args = (
                x,
                layer.tq_packed_weight,
                layer.tq_norms,
                layer.tq_signs1,
                layer.tq_signs2,
                layer.tq_centroids,
            )
            try:
                return layer._tq_primary_fn(*args, group_size=self.group_size, bits=self.bits, bias=bias)
            except (ValueError, RuntimeError):
                return layer._tq_fallback_fn(*args, group_size=self.group_size, bits=self.bits, bias=bias)

        # PyTorch fallback (no Triton)
        indices = _unpack_indices(layer.tq_packed_weight, self.bits, self.group_size)
        norms_flat = layer.tq_norms.reshape(-1)
        quantizer = _get_quantizer(self.group_size, self.bits, str(x.device))
        w_groups = quantizer.dequantize(indices, norms_flat)
        w_deq = w_groups.reshape(layer.tq_out_features, layer.tq_padded_in)[:, : layer.tq_in_features].to(x.dtype)
        output = torch.matmul(x, w_deq.t())
        if bias is not None:
            output = output + bias
        return output
