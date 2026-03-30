"""Monkey-patch vLLM's flash attention to use TurboQuant+ KV cache.

Intercepts FlashAttentionImpl at two points:
1. do_kv_cache_update: compress K/V with TurboQuant+ before storing
2. forward: decompress from TurboQuant+ before attention computation

The compressed data lives in a sidecar store (Python dict), not in vLLM's
paged cache. vLLM's cache is still allocated and written to (for memory
accounting), but attention reads from our decompressed tensors.

This is a prototype integration. Production would need CUDA kernels:
- CUDA WHT butterfly (128-dim, O(d log d)) for rotation
- CUDA searchsorted on 8-16 centroids for quantization
- Fused dequant inside flash attention tile loop for reads
"""

import torch
import logging
from functools import wraps

from turboquant_vllm.torch_ops import KVCacheCompressorTorch, CompressedKV

logger = logging.getLogger(__name__)

# Per-layer compressor and compressed cache storage
_compressors: dict[int, KVCacheCompressorTorch] = {}
_cache: dict[int, dict[tuple[int, int, int], tuple[CompressedKV, CompressedKV]]] = {}
# _cache[layer_id][(block_idx, offset, head_idx)] = (compressed_k, compressed_v)

_k_bits = 4
_v_bits = 4
_use_cuda = False


def _try_cuda_init() -> bool:
    """Try to load the JIT-compiled CUDA extension. Returns True if available."""
    global _use_cuda
    try:
        from turboquant_vllm.build import build
        build()
        _use_cuda = True
        logger.info("TurboQuant+ using CUDA kernels (175x faster)")
        return True
    except Exception as e:
        logger.info("TurboQuant+ CUDA not available (%s), using PyTorch fallback", e)
        _use_cuda = False
        return False


def _get_compressor(head_dim: int, device: torch.device) -> KVCacheCompressorTorch:
    """One compressor per head_dim (all layers share the same codebook/rotation)."""
    if head_dim not in _compressors:
        _compressors[head_dim] = KVCacheCompressorTorch(
            head_dim, k_bits=_k_bits, v_bits=_v_bits,
            seed=42, device=str(device),
            use_cuda=_use_cuda,
        )
        stats = _compressors[head_dim].memory_stats()
        backend = "CUDA" if _use_cuda else "PyTorch"
        logger.info(
            "TurboQuant+ compressor [%s]: head_dim=%d K=%d-bit(PQ%d+QJL1) V=%d-bit(PQ) → %.1fx compression",
            backend, head_dim, _k_bits, _k_bits - 1, _v_bits, stats["compression_ratio"],
        )
    return _compressors[head_dim]


def _make_patched_cache_update(original_fn):
    """Wrap do_kv_cache_update to compress K/V with TurboQuant+."""

    @wraps(original_fn)
    def patched(self, layer, key, value, kv_cache, slot_mapping):
        # Still call original so vLLM's memory accounting stays correct
        original_fn(self, layer, key, value, kv_cache, slot_mapping)

        # Now compress and store in sidecar
        layer_id = id(layer)
        if layer_id not in _cache:
            _cache[layer_id] = {}

        head_dim = key.shape[-1]
        num_kv_heads = key.shape[1] if key.dim() == 3 else 1
        compressor = _get_compressor(head_dim, key.device)

        num_tokens = slot_mapping.shape[0]
        key_cache, value_cache = kv_cache.unbind(0)
        block_size = key_cache.shape[1]

        for t in range(num_tokens):
            slot = slot_mapping[t].item()
            block_idx = slot // block_size
            offset = slot % block_size

            for h in range(num_kv_heads):
                k_vec = key[t, h].unsqueeze(0)  # (1, head_dim)
                v_vec = value[t, h].unsqueeze(0)

                ck = compressor.compress_k(k_vec)
                cv = compressor.compress_v(v_vec)

                _cache[layer_id][(block_idx, offset, h)] = (ck, cv)

    return patched


def _make_patched_forward(original_fn):
    """Wrap forward to decompress K/V from TurboQuant+ before attention."""

    @wraps(original_fn)
    def patched(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        if kv_cache is None:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

        layer_id = id(layer)
        store = _cache.get(layer_id)
        if not store:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

        head_dim = query.shape[-1]
        compressor = _get_compressor(head_dim, query.device)

        # Decompress all cached K/V back into vLLM's paged cache
        key_cache, value_cache = kv_cache.unbind(0)

        for (block_idx, offset, head_idx), (ck, cv) in store.items():
            k_dec = compressor.decompress_k(ck).squeeze(0)  # (head_dim,)
            v_dec = compressor.decompress_v(cv).squeeze(0)

            key_cache[block_idx, offset, head_idx] = k_dec.to(key_cache.dtype)
            value_cache[block_idx, offset, head_idx] = v_dec.to(value_cache.dtype)

        # Now call original forward with decompressed cache
        return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

    return patched


def patch_vllm_attention(k_bits: int = 4, v_bits: int = 4):
    """Monkey-patch vLLM's FlashAttentionImpl for TurboQuant+ KV cache.

    Call before starting the vLLM engine.

    Algorithm per the turboquant_plus paper:
    - K cache: PolarQuant at (k_bits-1) bits + QJL at 1 bit = k_bits total
      Preserves inner products for attention score computation (Q @ K^T)
    - V cache: PolarQuant MSE-only at v_bits bits
      Preserves MSE for value reconstruction (attn_weights @ V)

    Expected quality impact (from turboquant_plus benchmarks):
    - turbo4 (4-bit): +0.23% PPL
    - turbo3 (3-bit): +1.06% PPL
    - Asymmetric K4/V3: K precision dominates, V can be compressed more
    """
    global _k_bits, _v_bits
    _k_bits = k_bits
    _v_bits = v_bits

    # Try CUDA kernels first, fall back to PyTorch
    _try_cuda_init()

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except ImportError:
        logger.error("Cannot import vLLM FlashAttentionImpl — is vLLM installed?")
        raise

    FlashAttentionImpl.do_kv_cache_update = _make_patched_cache_update(
        FlashAttentionImpl.do_kv_cache_update
    )
    FlashAttentionImpl.forward = _make_patched_forward(
        FlashAttentionImpl.forward
    )

    logger.info(
        "TurboQuant+ patched vLLM: K=%d-bit (PolarQuant %d-bit + QJL 1-bit), V=%d-bit (PolarQuant MSE-only)",
        k_bits, k_bits - 1, v_bits,
    )
