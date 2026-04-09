# Native vLLM TQ3 MoE Loading

Branch: `feat/native-vllm-moe-loading`

Adds native vLLM integration for TQ3/TQ4 checkpoints, including FusedMoE
expert layers. Uses vLLM's plugin API — no fork required.

## What changed

### `turboquant_vllm/vllm_quant.py` (280 → 680 lines)

**TurboQuantConfig** — registered via `@register_quantization_config("turboquant")`.
Auto-detected from checkpoint's `config.json` `quantization_config.quant_method`.

**TurboQuantLinearMethod** — handles dense linear layers:
- `create_weights`: creates uint8 `tq_packed` + float32 `tq_norms` parameters
  via `PackedvLLMParameter` with correct TP sharding (packed_factor).
- `process_weights_after_loading`: makes contiguous, provides decompressed
  `weight` attribute for MLA attention fusion (kv_b_proj → W_UV/W_UK_T).
- `apply`: uses decompressed weight if available (fast path for MLA), then
  tries Triton fused kernels, falls back to PyTorch unpack+dequant+matmul.

**TurboQuantMoEMethod** — handles FusedMoE expert layers:
- `create_weights`: creates uint8 packed params (`w13_weight`, `w2_weight`)
  + float32 norms params (`w13_weight_norms`, `w2_weight_norms`). No FP16
  intermediate — packed data loaded directly via FusedMoE.weight_loader.
- `apply(layer, x, topk_weights, topk_ids, shared_experts_input)`: dequants
  all experts to FP16 temporary, calls `fused_experts`, frees temporary.
- Stores `moe_kernel = None` and `moe_quant_config` for v0.19.0 compatibility.

**Weight iterator** (`_patch_weight_name_remapping`):
Three-way routing for checkpoint TQ weight pairs:
1. **LinearBase** (has `.tq_packed` param): reshape packed data, yield directly.
   Stacked params (gate_proj→gate_up_proj) resolved dynamically via model's
   `packed_modules_mapping`.
2. **FusedMoE expert** (has uint8 `w13_weight` param): reshape packed, yield
   as `.weight` (uint8) + `.weight_norms` (float32). Expert params mapping
   substring match handles both: `gate_proj.weight` → `w13_weight` and
   `gate_proj.weight_norms` → `w13_weight_norms`.
3. **Other** (MoE router, etc.): decompress to FP16.

### `turboquant_vllm/weight_quant.py`

Fixed `_get_quantizer` cache key to include device string — prevents CPU
quantizer from being returned for GPU calls (or vice versa).

### `tests/test_vllm_quant.py` (new, 19 tests)

CPU-only tests covering:
- Packed tensor reshape (checkpoint → vLLM layout)
- Bits inference from tensor shapes
- Decompress roundtrip quality
- MoE packed assembly (gate+up stacking, norms, full dequant roundtrip)
- Regex and name matching
- GLM-4.7-Flash weight name routing (10 cases: linear_direct, linear_stacked, moe_expert, decompress)

## Tested on GPU (2026-04-09)

**Model:** GLM-4.7-Flash TQ3 (varjosoft/GLM-4.7-Flash-TQ3)
**GPU:** NVIDIA H100 80GB HBM3 (Verda FIN-02, $2.29/hr)
**vLLM:** 0.19.0, transformers 5.5.1

### What worked

- Plugin auto-registers, quant method auto-detected from config.json
- Model instantiation with TurboQuantLinearMethod + TurboQuantMoEMethod
- Weight loading: packed uint8 data loaded directly into LinearBase and FusedMoE params
- MLA `process_weights_after_loading`: reads decompressed `weight` from kv_b_proj
- KV cache allocation: 996,960 tokens (243x concurrency at 4096 ctx)
- **Generation confirmed**: 5-token and 10-token responses produced (model outputs thinking tokens before answering — GLM-4.7-Flash behavior, not a bug)

### What didn't work / known issues

1. **Performance: ~0.1 tok/s** — Python fallback dequant loop (64 experts × unpack+dequant per forward). Triton kernels fail to import on H100. Need to investigate Triton compatibility.

2. **GPU memory: 76.6 GB** — higher than expected 17 GB. The per-forward dequant of all 64 experts creates large FP16 temporaries that accumulate. Optimization: dequant only top-k experts, or use Triton fused dequant-GEMM.

3. **`--quantization turboquant` CLI flag rejected** — vLLM validates the flag before plugins load. Must rely on auto-detection from config.json. Works, but users can't override.

## vLLM 0.19.0 API notes

- `FusedMoEMethodBase.apply` signature changed: now receives `(layer, x, topk_weights, topk_ids, shared_experts_input)` — routing done by caller.
- Must set `self.moe_kernel = None` and `self.moe_quant_config = None` on init.
- `get_fused_moe_quant_config(layer)` is a new required abstract method.
- `set_weight_attrs` moved to `vllm.model_executor.utils`.
- `FusedMoE.weight_loader` is dtype-agnostic (`copy_()`) — works for uint8→uint8.
- `get_and_maybe_dequant_weights` calls `apply(layer, eye)` as fallback — we provide decompressed `weight` to avoid this expensive path.

## Next steps

1. Fix Triton kernel import on H100 (CUDA 12.8 / sm_90 compatibility)
2. Optimize MoE: dequant only top-k experts per forward
3. Test on GLM-5.1 (309 GB, 8x GPU with TP)
4. Merge to main after performance is acceptable
