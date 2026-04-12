"""TurboQuant quantization config for native vLLM integration.

Registers as a vLLM quantization method so TQ3/TQ4 checkpoints can be loaded
with tensor parallelism. Uses the same dequant kernels as TurboQuantWrapper.

Usage:
    # Checkpoint must have quantization_config in config.json:
    # {"quantization_config": {"quant_method": "turboquant", "bits": 3, "group_size": 128}}
    #
    # Then just:
    vllm serve ./my-tq3-checkpoint --quantization turboquant
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _lazy_import_vllm():
    """Import vLLM components lazily to avoid import errors when vLLM isn't installed."""
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    )

    return (
        LinearBase,
        LinearMethodBase,
        QuantizationConfig,
        QuantizeMethodBase,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    )


# Deferred class creation — only built when register() is called from the plugin
_registered = False


def register():
    """Register TurboQuant as a vLLM quantization method. Called from the plugin."""
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
    except ImportError:
        logger.debug("vLLM not installed, skipping TurboQuant quant config registration")
        return

    (
        LinearBase,
        LinearMethodBase,
        QuantizationConfig,
        QuantizeMethodBase,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    ) = _lazy_import_vllm()

    from turboquant_vllm.weight_quant import (
        _SKIP_PATTERNS,
        select_bits,
        _get_quantizer,
        unpack_indices,
    )

    @register_quantization_config("turboquant")
    class TurboQuantConfig(QuantizationConfig):
        """Config for TurboQuant weight quantization (TQ3/TQ4)."""

        def __init__(self, bits: int = 3, group_size: int = 128, sensitive_bits: int | None = None):
            super().__init__()
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits

        def __repr__(self) -> str:
            return (
                f"TurboQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                f"sensitive_bits={self.sensitive_bits})"
            )

        def get_name(self) -> str:
            return "turboquant"

        def get_supported_act_dtypes(self) -> list[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70  # Volta and newer

        @staticmethod
        def get_config_filenames() -> list[str]:
            return ["tq_config.json", "quantize_config.json"]

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "TurboQuantConfig":
            bits = cls.get_from_keys_or(config, ["bits"], 3)
            group_size = cls.get_from_keys_or(config, ["group_size"], 128)
            sensitive_bits = cls.get_from_keys_or(config, ["sensitive_bits"], None)
            return cls(bits=bits, group_size=group_size, sensitive_bits=sensitive_bits)

        def get_quant_method(
            self, layer: nn.Module, prefix: str
        ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
            # Native TQ3 checkpoints are decompressed to bf16 during
            # weight loading (see _patch_weight_name_remapping).  All
            # layers receive standard bf16 weights, so we return None
            # to let vLLM use its default unquantized methods.  The
            # runtime plugin (enable_weight_quantization) re-compresses
            # on GPU after loading.
            return None

    class TurboQuantLinearMethod(LinearMethodBase):
        """Linear method that loads TQ3/TQ4 packed weights and dequants on forward."""

        def __init__(self, quant_config: "TurboQuantConfig", bits: int):
            self.quant_config = quant_config
            self.bits = bits
            self.group_size = quant_config.group_size

        def create_weights(
            self,
            layer: nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")

            from fractions import Fraction
            from turboquant_vllm.weight_quant import packed_group_bytes as _pgb

            from turboquant_vllm.weight_quant import padded_size
            padded_in, n_groups = padded_size(input_size_per_partition, self.group_size)
            packed_cols = n_groups * _pgb(self.bits, self.group_size)
            pack_ratio = Fraction(padded_in, packed_cols)

            tq_packed = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    packed_cols,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=pack_ratio,
                weight_loader=weight_loader,
            )

            # Norms: (out_features, n_groups) as float32
            # n_groups = padded_in / group_size, so pack ratio is group_size
            tq_norms = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    n_groups,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.group_size,
                weight_loader=weight_loader,
            )

            layer.register_parameter("tq_packed", tq_packed)
            layer.register_parameter("tq_norms", tq_norms)
            layer._tq_bits = self.bits
            layer._tq_group_size = self.group_size
            layer._tq_in_features = input_size_per_partition
            layer._tq_out_features = output_size_per_partition

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            # Ensure weights are contiguous and on the right device
            layer.tq_packed = nn.Parameter(layer.tq_packed.data.contiguous(), requires_grad=False)
            layer.tq_norms = nn.Parameter(layer.tq_norms.data.contiguous(), requires_grad=False)

            # Eagerly populate the module-level rotation matrix cache so
            # the first forward (potentially the warmup pass before CUDA
            # graph capture) does not hit a cache miss and run a
            # butterfly WHT inside the custom_op body. Mirrors the
            # equivalent eager call in TurboQuantWrapper.__init__; see
            # turboquant_vllm.triton_ops._rotation_matrix_cache for the
            # capture-safety invariant.
            try:
                from turboquant_vllm.triton_ops import _get_cached_rotation_matrix

                quantizer = _get_quantizer(self.group_size, self.bits, str(layer.tq_packed.device))
                _get_cached_rotation_matrix(quantizer.signs1, quantizer.signs2, self.group_size)
            except ImportError:
                pass

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            bits = layer._tq_bits
            group_size = layer._tq_group_size
            in_features = layer._tq_in_features
            out_features = layer._tq_out_features

            # Try Triton fused kernels first (fastest)
            try:
                from turboquant_vllm.triton_ops import tq_fused_gemm, tq_fwht_input_gemm

                quantizer = _get_quantizer(group_size, bits, str(x.device))

                args = (
                    x,
                    layer.tq_packed.data,
                    layer.tq_norms.data,
                    quantizer.signs1,
                    quantizer.signs2,
                    quantizer.centroids,
                )
                kwargs = dict(group_size=group_size, bits=bits, bias=bias)

                primary = tq_fwht_input_gemm if out_features >= 4096 else tq_fused_gemm
                fallback = tq_fused_gemm if out_features >= 4096 else tq_fwht_input_gemm
                try:
                    return primary(*args, **kwargs)
                except (ValueError, RuntimeError):
                    try:
                        return fallback(*args, **kwargs)
                    except (ValueError, RuntimeError):
                        pass
            except ImportError:
                pass

            # Fallback: dequant + matmul
            quantizer = _get_quantizer(group_size, bits, str(x.device))
            indices = unpack_indices(layer.tq_packed.data, bits, group_size)
            norms_flat = layer.tq_norms.data.reshape(-1)
            w_groups = quantizer.dequantize(indices, norms_flat)
            padded_in, _ = padded_size(in_features, group_size)
            w_deq = w_groups.reshape(out_features, padded_in)[:, :in_features]
            w_deq = w_deq.to(x.dtype)

            output = torch.matmul(x, w_deq.t())
            del w_deq
            if bias is not None:
                output = output + bias
            return output

    # ------------------------------------------------------------------
    # FusedMoE: load TQ3-packed expert weights from native checkpoint
    # ------------------------------------------------------------------
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs

    # Imports used by both linear and MoE loaders, resolved once.
    from turboquant_vllm.weight_quant import Compressed3D, packed_group_bytes, padded_size
    from vllm.model_executor.layers.fused_moe import fused_experts

    class TurboQuantFusedMoELoadMethod(FusedMoEMethodBase):
        """Load TQ3-packed MoE expert weights from a native TQ3 checkpoint.

        The checkpoint stores per-expert 2D packed tensors (matching the
        original HF naming, e.g. ``experts.0.gate_proj.weight.tq_packed``).
        vLLM's model-specific ``load_weights`` maps these to our registered
        fused 3D packed parameters via ``expert_params_mapping``, calling
        the weight_loader with ``(param, loaded_weight, name, shard_id,
        expert_id)``.  The custom weight_loader assembles per-expert packed
        data into the correct slot of the fused parameter — no bf16
        intermediate.
        """

        def __init__(self, moe_config, bits: int, group_size: int):
            import threading

            super().__init__(moe_config)
            self.bits = bits
            self.group_size = group_size
            self._scratch_pool = None
            self._w13_comp = None
            self._w2_comp = None
            self._init_lock = threading.Lock()

        def create_weights(
            self,
            layer: nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            bits = self.bits
            gs = self.group_size
            is_act_and_mul = self.moe.is_act_and_mul

            # Fused w13: (num_experts, 2*intermediate, hidden) for gate+up
            # w2:        (num_experts, hidden, intermediate)
            w13_out = (2 if is_act_and_mul else 1) * intermediate_size_per_partition
            w13_in = hidden_size
            w2_out = hidden_size
            w2_in = intermediate_size_per_partition

            def _packed_shape(n_exp, out_dim, in_dim):
                _, n_groups = padded_size(in_dim, gs)
                return (n_exp * out_dim, n_groups * packed_group_bytes(bits, gs))

            def _norms_shape(n_exp, out_dim, in_dim):
                _, n_groups = padded_size(in_dim, gs)
                return (n_exp * out_dim, n_groups)

            orig_weight_loader = extra_weight_attrs.get("weight_loader")

            # --- Per-expert packed weight_loader ---
            # vLLM calls this once per (expert, shard) with the per-expert
            # packed tensor from the checkpoint.  We assemble it into the
            # fused 3D packed array at the right row offset.
            def tq_expert_weight_loader(
                param: nn.Parameter,
                loaded_weight: torch.Tensor,
                weight_name: str,
                shard_id: str = "",
                expert_id: int = 0,
                return_success: bool = False,
            ) -> bool | None:
                # Map global → local expert id
                local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if local_id == -1:
                    return False if return_success else None

                # Determine row range in the fused 2D packed array.
                # The packed array is (num_experts * out_dim, packed_cols),
                # stored in expert-major order.
                if shard_id in ("w1", "w3"):
                    # w13: gate (w1) in first half, up (w3) in second half
                    per_expert_out = w13_out
                    half = per_expert_out // 2 if is_act_and_mul else per_expert_out
                    base = local_id * per_expert_out
                    if shard_id == "w1":
                        row_start = base
                    else:
                        row_start = base + half
                    row_end = row_start + half
                elif shard_id == "w2":
                    row_start = local_id * w2_out
                    row_end = row_start + w2_out
                else:
                    return False if return_success else None

                # TP sharding: the loaded_weight may need narrowing along
                # the output dim for w13, or input dim for w2.
                # For per-expert loading, vLLM handles TP *before* calling
                # our weight_loader (the expert_params_mapping already
                # selects local experts).  The loaded per-expert tensor
                # is the full shard for this TP rank.
                n_rows = row_end - row_start
                if loaded_weight.shape[0] != n_rows:
                    # TP-sharded: take this rank's slice
                    tp_rank = getattr(layer, "tp_rank", 0)
                    start = n_rows * tp_rank
                    loaded_weight = loaded_weight.narrow(0, start, n_rows)

                param.data[row_start:row_end, :loaded_weight.shape[1]] = loaded_weight
                return True if return_success else None

            # Register fused packed parameters
            w13_packed = nn.Parameter(
                torch.zeros(*_packed_shape(num_experts, w13_out, w13_in), dtype=torch.uint8),
                requires_grad=False,
            )
            w2_packed = nn.Parameter(
                torch.zeros(*_packed_shape(num_experts, w2_out, w2_in), dtype=torch.uint8),
                requires_grad=False,
            )
            w13_norms = nn.Parameter(
                torch.zeros(*_norms_shape(num_experts, w13_out, w13_in), dtype=torch.float32),
                requires_grad=False,
            )
            w2_norms = nn.Parameter(
                torch.zeros(*_norms_shape(num_experts, w2_out, w2_in), dtype=torch.float32),
                requires_grad=False,
            )

            layer.register_parameter("w13_tq_packed", w13_packed)
            layer.register_parameter("w13_tq_norms", w13_norms)
            layer.register_parameter("w2_tq_packed", w2_packed)
            layer.register_parameter("w2_tq_norms", w2_norms)
            for p in (w13_packed, w13_norms, w2_packed, w2_norms):
                set_weight_attrs(p, {"weight_loader": tq_expert_weight_loader})

            self._w13_shape = (num_experts, w13_out, w13_in)
            self._w2_shape = (num_experts, w2_out, w2_in)
            self._params_dtype = params_dtype

            # bf16 weight placeholders — zero-size, re-pointed at scratch
            # on first forward.  No weight_loader: these should NOT receive
            # checkpoint data (the packed params handle that).
            w13_weight = nn.Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            w2_weight = nn.Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            layer.register_parameter("w13_weight", w13_weight)
            layer.register_parameter("w2_weight", w2_weight)

        def get_fused_moe_quant_config(self, layer: nn.Module):
            return None

        def _ensure_ready(self, layer: nn.Module):
            """Lazily allocate scratch and build Compressed3D on first forward."""
            if self._scratch_pool is not None:
                return
            with self._init_lock:
                if self._scratch_pool is not None:
                    return
                self._init_scratch(layer)

        def _init_scratch(self, layer: nn.Module):
            device = layer.w13_tq_packed.device
            dtype = self._params_dtype
            w13_s, w2_s = self._w13_shape, self._w2_shape

            self._scratch_pool = {
                "w13": torch.empty(w13_s, dtype=dtype, device=device),
                "w2": torch.empty(w2_s, dtype=dtype, device=device),
                "w13_fp32": torch.empty(w13_s, dtype=torch.float32, device=device),
                "w2_fp32": torch.empty(w2_s, dtype=torch.float32, device=device),
            }
            layer.w13_weight.data = self._scratch_pool["w13"]
            layer.w2_weight.data = self._scratch_pool["w2"]

            # Cache Compressed3D objects — packed data never changes.
            self._w13_comp = Compressed3D.from_packed(
                layer.w13_tq_packed.data, layer.w13_tq_norms.data,
                w13_s, dtype, self.bits, self.group_size,
            )
            self._w2_comp = Compressed3D.from_packed(
                layer.w2_tq_packed.data, layer.w2_tq_norms.data,
                w2_s, dtype, self.bits, self.group_size,
            )

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None = None,
        ):
            self._ensure_ready(layer)
            pool = self._scratch_pool
            self._w13_comp.decompress_into(pool["w13"], fp32_scratch=pool["w13_fp32"])
            self._w2_comp.decompress_into(pool["w2"], fp32_scratch=pool["w2_fp32"])

            return fused_experts(
                x,
                pool["w13"],
                pool["w2"],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=not self.moe.disable_inplace,
            )

    # Patch the weight loader to remap old-style checkpoint names
    # Old: model.layers.0.self_attn.q_proj.weight.tq_packed
    # New: model.layers.0.self_attn.q_proj.tq_packed
    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM")


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to decompress TQ3 weights on load.

    When a native TQ3 checkpoint is loaded, the checkpoint contains
    ``.tq_packed`` / ``.tq_norms`` tensor pairs instead of standard
    ``.weight`` tensors.  This patch collects each pair, decompresses
    to bf16 on CPU, and yields the result with the original weight name.
    vLLM's model-specific weight loaders (stacked qkv, fused gate_up,
    expert assembly) then work unchanged.

    After loading, the runtime plugin re-compresses weights on GPU via
    ``enable_weight_quantization`` — so the bf16 is transient.
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    from turboquant_vllm.weight_quant import (
        unpack_indices,
        _get_quantizer,
        padded_size,
    )

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _decompress_get_all_weights(self, model_config, model):
        # Check if this is a TQ checkpoint
        quant_cfg = getattr(model_config.hf_config, "quantization_config", None)
        if not quant_cfg or quant_cfg.get("quant_method") != "turboquant":
            yield from _original_get_all_weights(self, model_config, model)
            return

        bits = quant_cfg.get("bits", 3)
        group_size = quant_cfg.get("group_size", 128)

        # Collect packed/norms pairs, decompress, yield as bf16.
        # Tensors arrive in checkpoint order — packed and norms for the
        # same weight are adjacent (both in the same shard, consecutive).
        pending_packed = {}  # base_name → packed tensor
        pending_norms = {}   # base_name → norms tensor

        for name, tensor in _original_get_all_weights(self, model_config, model):
            if name.endswith(".weight.tq_packed"):
                base = name[: -len(".tq_packed")]  # e.g. "layers.0.q_proj.weight"
                pending_packed[base] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = name[: -len(".tq_norms")]
                pending_norms[base] = tensor
            else:
                # Regular tensor — yield as-is
                yield name, tensor
                continue

            # Check if we have both packed + norms for this weight
            base_p = name[: -len(".tq_packed")] if name.endswith(".tq_packed") else None
            base_n = name[: -len(".tq_norms")] if name.endswith(".tq_norms") else None
            base = base_p or base_n

            if base in pending_packed and base in pending_norms:
                packed = pending_packed.pop(base)
                norms = pending_norms.pop(base)

                # Decompress on CPU to bf16
                quantizer = _get_quantizer(group_size, bits, "cpu")
                indices = unpack_indices(packed, bits, group_size)
                norms_flat = norms.reshape(-1)
                w_groups = quantizer.dequantize(indices, norms_flat)
                n_rows = norms.shape[0]
                padded_in, _ = padded_size(w_groups.shape[-1], group_size)
                # Determine original in_dim from packed shape
                in_dim = padded_in  # might be padded; trimmed by model loader
                w = w_groups.reshape(n_rows, -1)[:, :in_dim].to(torch.bfloat16)

                # Yield with original weight name
                yield base, w
                del packed, norms, indices, w_groups, w

        # Flush any orphaned packed/norms (shouldn't happen with valid checkpoints)
        for base in pending_packed:
            logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        for base in pending_norms:
            logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
