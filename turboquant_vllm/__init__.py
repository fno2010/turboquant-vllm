"""TurboQuant+ KV cache compression for vLLM.

Fused CUDA kernels with PyTorch fallback. 175x faster than pure PyTorch.

Quick start:
    from turboquant_vllm import patch_vllm_attention
    patch_vllm_attention(k_bits=4, v_bits=4)
    # Then start vLLM as usual

For standalone quantization (without vLLM):
    from turboquant_vllm.torch_ops import KVCacheCompressorTorch
    compressor = KVCacheCompressorTorch(head_dim=128, k_bits=4, v_bits=4)
"""

from turboquant_vllm.vllm_patch import patch_vllm_attention
from turboquant_vllm.torch_ops import (
    KVCacheCompressorTorch,
    PolarQuantTorch,
    QJLTorch,
    CompressedKV,
)

__all__ = [
    "patch_vllm_attention",
    "KVCacheCompressorTorch",
    "PolarQuantTorch",
    "QJLTorch",
    "CompressedKV",
]

__version__ = "0.1.0"
