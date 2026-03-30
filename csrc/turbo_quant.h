/**
 * TurboQuant+ CUDA kernels for vLLM KV cache compression.
 *
 * Full Algorithm 2 from turboquant_plus:
 *   K cache: PolarQuant(b_k - 1 bits) + QJL(1 bit)
 *   V cache: PolarQuant(b_v bits) MSE-only
 *   Asymmetric K/V: K and V use separate codebooks, rotations, bit widths.
 */

#pragma once
#include <torch/extension.h>

namespace turbo_quant {

/** Initialize codebook and rotation for both K and V (symmetric mode). */
void init(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
);

/** Initialize K cache codebook, rotation, and bit width independently. */
void init_k(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
);

/** Initialize V cache codebook, rotation, and bit width independently. */
void init_v(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
);

/** Upload QJL projection matrix for K cache inner-product preservation. */
void init_qjl(torch::Tensor qjl_matrix);

/** Standalone quantize (PolarQuant, for testing). */
void quantize(torch::Tensor input, torch::Tensor indices, torch::Tensor norms);

/** Standalone dequantize (PolarQuant, for testing). */
void dequantize(torch::Tensor indices, torch::Tensor norms, torch::Tensor output);

/** Fused quantize + pack into vLLM paged cache. Asymmetric K/V supported. */
void reshape_and_cache(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor slot_mapping
);

/** Dequantize from paged cache to contiguous fp16 buffer. */
void dequant_paged_cache(
    torch::Tensor cache, torch::Tensor norms, torch::Tensor output,
    torch::Tensor block_table, int seq_len
);

}  // namespace turbo_quant
