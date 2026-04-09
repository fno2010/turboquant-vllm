# PlanarQuant Rotation for KV Cache Compression

Branch: `feat/planar-rotation-kv`

Adds 2D Givens rotation as a drop-in alternative to Walsh-Hadamard Transform
(WHT) for decorrelating KV cache vectors before quantization. Inspired by
[RotorQuant](https://github.com/scrya-com/rotorquant) (scrya-com/ParaMind2025).

## Why

TurboQuant's WHT applies a d×d structured rotation with O(d log d) butterfly
operations. RotorQuant showed that **simple 2D block-diagonal rotations achieve
comparable or better quality** at O(d) cost — because KV cache vectors live on
low-rank manifolds where local decorrelation suffices.

Their benchmarks on Llama 3.1 8B (RTX 5090, 3-bit symmetric):
- 28% faster decode (119 vs 93 tok/s)
- 5.3x faster prefill (3,822 vs 722 tok/s)
- Slightly better PPL (6.91 vs 7.07)
- 44x fewer rotation parameters (128 vs 16,384)

## What Changed

### `turboquant_vllm/torch_ops.py`

**PolarQuantTorch** now accepts `rotation='planar'` or `rotation='wht'` (default):
- `'wht'`: Original D2 @ H @ D1 structured random rotation. O(d log d).
- `'planar'`: 2D Givens rotation per pair of elements. O(d), 4 FMAs per pair.

New functions `_planar_rotate()` and `_planar_rotate_inverse()` implement the
forward and inverse Givens rotation. Random angles generated at init, stored as
`(cos θ, sin θ)` pairs — 128 parameters for d=128 vs 16,384 for WHT.

**KVCacheCompressorTorch** passes `rotation=` through to PolarQuantTorch.

### `turboquant_vllm/triton_ops.py`

Three new Triton kernels for GPU-accelerated planar quantization:
- `_planar_quantize_kernel`: normalize → rotate → quantize → indices
- `_planar_dequantize_kernel`: indices → inverse rotate → rescale
- Python wrappers: `planar_quantize()`, `planar_dequantize()`

Each kernel processes 2D pairs in parallel with 4 FMAs per pair.

### `turboquant_vllm/vllm_patch.py` + `_vllm_plugin.py`

Rotation mode flows through the full stack:
```
patch_vllm_attention(rotation='planar')
  → env var TQ_KV_ROTATION=planar (survives subprocess spawn)
  → KVCacheCompressorTorch(rotation='planar')
  → PolarQuantTorch(rotation='planar')
```

### `scripts/bench_gemma4_kv.py`

Benchmark script comparing 5 configs on Gemma 4 26B:
BF16 baseline, WHT K3/V3, Planar K3/V3, WHT K4/V3, Planar K4/V3.

## Local Benchmark Results (CPU, d=128)

```
WHT 3-bit MSE:    0.033168
Planar 3-bit MSE: 0.034396  (+3.7%, negligible)

WHT 4-bit MSE:    0.009287
Planar 4-bit MSE: 0.009240  (-0.5%, slightly better)

WHT roundtrip:     121.7 ms  (8192 vectors)
Planar roundtrip:   55.0 ms  (2.2x faster)
```

KV cache compressor integration test (3-bit symmetric):
```
wht     K3 MSE=0.0340  V3 MSE=0.0321
planar  K3 MSE=0.0368  V3 MSE=0.0356
```

## GPU Test Results (2026-04-09)

### Gemma 4 26B on A100 80GB (vLLM 0.19.0)

**BF16 baseline (no KV compression):**
- 15.6 tok/s, 664 tokens in 42.4s
- GPU memory: 75,322 MiB
- Generation quality: correct answers to all 5 prompts

**KV compression configs (WHT and Planar): ALL FAILED**

Root cause: **vLLM V1 engine subprocess issue.** vLLM 0.19.0 spawns
EngineCore as a separate process. Our FlashAttention monkey-patch applies
in the main process but the EngineCore subprocess imports fresh module
instances. The plugin re-registers (env vars propagate, logs confirm
"patched FlashAttention+MLA") but the actual method replacement doesn't
take effect in the subprocess.

This blocks ALL KV cache compression on vLLM 0.19.0, not just planar rotation.
BF16 baseline and weight-only compression (via quant config API) are unaffected.

### Fix Options

1. **`--disable-frontend-multiprocessing`** — use V0 engine (untested on 0.19.0)
2. **Class inheritance** — subclass FlashAttentionImpl instead of monkey-patching
3. **Native `--kv-cache-dtype`** — register via vLLM's KV cache dtype system
   (the approach from the varjoranta/vllm-1 fork)

## How to Use

```python
# KV cache with planar rotation
from turboquant_vllm.vllm_patch import patch_vllm_attention
patch_vllm_attention(k_bits=3, v_bits=3, rotation='planar')

# Or via env vars (for vLLM plugin)
TQ_KV_K_BITS=3 TQ_KV_V_BITS=3 TQ_KV_ROTATION=planar vllm serve ...

# Direct PyTorch usage
from turboquant_vllm.torch_ops import PolarQuantTorch
q = PolarQuantTorch(128, bits=3, rotation='planar', device='cuda')
indices, norms = q.quantize(kv_vectors)
reconstructed = q.dequantize(indices, norms)
```

## The Geometry (Why Block Rotations Work)

WHT is a "sledgehammer" — it globally mixes all d dimensions so every output
depends on every input. This maximally decorrelates but is expensive.

PlanarQuant only mixes pairs of adjacent elements. Much weaker decorrelation
but KV cache vectors don't need full decorrelation — attention heads organize
information in local patterns, so adjacent elements are most correlated.

The result: 2D Givens captures most useful decorrelation at fraction of the
cost. WHT's global mixing over-decorrelates, spreading quantization error
uniformly when localized error would be more benign for attention.

For **weight compression**, WHT may still be better since weight distributions
are more complex than KV cache. Untested.

## Next Steps

1. Fix vLLM V1 subprocess patching (blocks all KV benchmarks)
2. GPU benchmark: WHT vs Planar on Gemma 4 with working KV compression
3. Test planar rotation for weight compression (currently WHT-only)
4. Implement deferred K-cache quantization (RotorQuant's other key innovation)
