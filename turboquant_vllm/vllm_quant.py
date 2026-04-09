"""TurboQuant quantization config for native vLLM integration.

Registers as a vLLM quantization method so TQ3/TQ4 checkpoints can be loaded
with tensor parallelism. Uses the same dequant kernels as TurboQuantWrapper.

Supports both LinearBase (dense) and FusedMoE (expert) layers.

Usage:
    # Checkpoint must have tq_config.json or quantization_config in config.json:
    # {"quant_method": "turboquant", "bits": 3, "group_size": 128}
    #
    # Then just:
    vllm serve ./my-tq3-checkpoint --quantization turboquant
"""

from __future__ import annotations

import logging
import re
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
        LinearBase, LinearMethodBase, QuantizationConfig, QuantizeMethodBase,
        GroupQuantScaleParameter, ModelWeightParameter, PackedvLLMParameter,
    )


# Deferred class creation — only built when register() is called from the plugin
_registered = False

def _infer_bits_from_packed(packed: torch.Tensor, norms: torch.Tensor,
                            group_size: int = 128) -> int:
    """Infer quantization bits from packed/norms tensor shapes."""
    n_groups = norms.shape[-1]
    if n_groups == 0:
        return 3
    packed_cols = packed.shape[-1]
    packed_per_group = packed_cols // max(n_groups, 1)
    expected_3bit = group_size * 3 // 8  # 48 for group_size=128
    expected_4bit = group_size // 2       # 64 for group_size=128
    if packed_per_group == expected_3bit:
        return 3
    elif packed_per_group == expected_4bit:
        return 4
    return 3  # default


def _decompress_tq_to_fp16(packed: torch.Tensor, norms: torch.Tensor,
                            group_size: int = 128) -> torch.Tensor:
    """Decompress a TQ packed weight pair to FP16 on CPU.

    packed: [out * n_groups, packed_per_group], norms: [out, n_groups].
    Returns FP16 [out, padded_in].
    """
    from turboquant_vllm.weight_quant import unpack_indices, _get_quantizer

    out_features = norms.shape[0]
    n_groups = norms.shape[1]
    bits = _infer_bits_from_packed(packed, norms, group_size)

    quantizer = _get_quantizer(group_size, bits, "cpu")
    # packed is [out*n_groups, packed_per_group], each row = one group
    indices = unpack_indices(packed, bits, group_size)
    norms_flat = norms.reshape(-1)
    w_groups = quantizer.dequantize(indices, norms_flat)
    padded_in = n_groups * group_size
    w_deq = w_groups.reshape(out_features, padded_in)
    return w_deq.to(torch.float16)


def _reshape_packed_for_vllm(packed: torch.Tensor, norms: torch.Tensor):
    """Reshape checkpoint [out*n_groups, ppg] to vLLM [out, n_groups*ppg]."""
    return packed.reshape(norms.shape[0], -1)


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
        LinearBase, LinearMethodBase, QuantizationConfig, QuantizeMethodBase,
        GroupQuantScaleParameter, ModelWeightParameter, PackedvLLMParameter,
    ) = _lazy_import_vllm()

    from turboquant_vllm.weight_quant import (
        _SKIP_PATTERNS, select_bits, _get_quantizer, unpack_indices,
        pack_indices,
    )

    # Optional FusedMoE support
    try:
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoE, FusedMoEMethodBase,
        )
        _has_fused_moe = True
    except ImportError:
        FusedMoE = None
        FusedMoEMethodBase = None
        _has_fused_moe = False

    @register_quantization_config("turboquant")
    class TurboQuantConfig(QuantizationConfig):
        """Config for TurboQuant weight quantization (TQ3/TQ4)."""

        def __init__(self, bits: int = 3, group_size: int = 128,
                     sensitive_bits: int | None = None):
            super().__init__()
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits

        def __repr__(self) -> str:
            return (f"TurboQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                    f"sensitive_bits={self.sensitive_bits})")

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
            if isinstance(layer, LinearBase):
                # Skip layers that shouldn't be quantized
                if any(p in prefix.lower() for p in _SKIP_PATTERNS):
                    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                    return UnquantizedLinearMethod()
                layer_bits = select_bits(prefix, self.bits, self.sensitive_bits)
                return TurboQuantLinearMethod(self, layer_bits)
            if _has_fused_moe and isinstance(layer, FusedMoE):
                layer_bits = select_bits(prefix, self.bits, self.sensitive_bits)
                return TurboQuantMoEMethod(self, layer_bits)
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

            # Packed indices: (out_features, n_groups * packed_group_bytes) as uint8
            # Loaded directly from TQ checkpoint — no FP16 intermediate.
            padded_in = ((input_size_per_partition + self.group_size - 1)
                         // self.group_size) * self.group_size
            n_groups = padded_in // self.group_size

            if self.bits == 4:
                packed_cols = n_groups * (self.group_size // 2)
            elif self.bits == 3:
                packed_cols = n_groups * (self.group_size * 3 // 8)
            else:
                packed_cols = n_groups * self.group_size

            from fractions import Fraction
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
            layer.tq_packed = nn.Parameter(
                layer.tq_packed.data.contiguous(), requires_grad=False)
            layer.tq_norms = nn.Parameter(
                layer.tq_norms.data.contiguous(), requires_grad=False)

            # Provide decompressed 'weight' for consumers that need it
            # (e.g., MLA attention fuses kv_b_proj.weight into W_UV/W_UK_T).
            # This runs once at load time, not per forward pass.
            bits = layer._tq_bits
            group_size = layer._tq_group_size
            in_features = layer._tq_in_features
            out_features = layer._tq_out_features
            padded_in = ((in_features + group_size - 1) // group_size) * group_size
            n_groups = padded_in // group_size
            packed_per_group = layer.tq_packed.data.shape[1] // n_groups

            packed_grouped = layer.tq_packed.data.reshape(
                out_features * n_groups, packed_per_group)
            quantizer = _get_quantizer(group_size, bits,
                                       str(layer.tq_packed.data.device))
            indices = unpack_indices(packed_grouped, bits, group_size)
            norms_flat = layer.tq_norms.data.reshape(-1)
            w_groups = quantizer.dequantize(indices, norms_flat)
            w_deq = w_groups.reshape(out_features, padded_in)[:, :in_features]
            layer.weight = nn.Parameter(w_deq.to(layer.tq_packed.data.device),
                                        requires_grad=False)

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

            # If decompressed weight exists (set by process_weights_after_loading
            # for MLA consumption), use it directly — avoids expensive dequant.
            w_deq = getattr(layer, 'weight', None)
            if w_deq is not None and isinstance(w_deq, nn.Parameter):
                output = torch.matmul(x, w_deq.data.to(x.dtype).t())
                if bias is not None:
                    output = output + bias
                return output

            # Try Triton fused kernels (fastest for regular forward)
            try:
                from turboquant_vllm.triton_ops import tq_fused_gemm, tq_fwht_input_gemm
                quantizer = _get_quantizer(group_size, bits, str(x.device))

                args = (x, layer.tq_packed.data, layer.tq_norms.data,
                        quantizer.signs1, quantizer.signs2, quantizer.centroids)
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
            padded_in = ((in_features + group_size - 1) // group_size) * group_size
            n_groups = padded_in // group_size
            packed_per_group = layer.tq_packed.data.shape[1] // n_groups

            packed_grouped = layer.tq_packed.data.reshape(
                out_features * n_groups, packed_per_group
            )
            quantizer = _get_quantizer(group_size, bits, str(x.device))
            indices = unpack_indices(packed_grouped, bits, group_size)
            norms_flat = layer.tq_norms.data.reshape(-1)
            w_groups = quantizer.dequantize(indices, norms_flat)
            w_fp = w_groups.reshape(out_features, padded_in)[:, :in_features]
            w_fp = w_fp.to(x.dtype)

            output = torch.matmul(x, w_fp.t())
            del w_fp
            if bias is not None:
                output = output + bias
            return output

    # ── FusedMoE support ──────────────────────────────────────────────

    if _has_fused_moe:
        class TurboQuantMoEMethod(FusedMoEMethodBase):
            """MoE method for TQ3/TQ4 checkpoints.

            Stores expert weights as packed uint8 (no FP16 intermediate).
            FusedMoE.weight_loader copies packed data directly into uint8
            params. Forward pass dequants all experts to FP16, calls
            fused_experts, then frees the temporary.

            Memory: ~12.7 GB packed for GLM-4.7-Flash (vs 53 GB FP16).
            """

            def __init__(self, quant_config: "TurboQuantConfig", bits: int):
                self.quant_config = quant_config
                self.bits = bits
                self.group_size = quant_config.group_size
                self.moe_quant_config = None  # set lazily
                self.moe_kernel = None  # not using modular kernel system

            def get_fused_moe_quant_config(self, layer: nn.Module):
                from vllm.model_executor.layers.fused_moe.config import (
                    FUSED_MOE_UNQUANTIZED_CONFIG,
                )
                self.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
                return FUSED_MOE_UNQUANTIZED_CONFIG

            def create_weights(
                self,
                layer: nn.Module,
                num_experts: int,
                hidden_size: int,
                intermediate_size_per_partition: int,
                params_dtype: torch.dtype,
                **extra_weight_attrs,
            ):
                group_size = self.group_size
                bits = self.bits

                # Compute packed dimensions for the input (hidden) dimension
                padded_hidden = ((hidden_size + group_size - 1)
                                 // group_size) * group_size
                ng_hidden = padded_hidden // group_size
                if bits == 3:
                    ppg_hidden = group_size * 3 // 8
                elif bits == 4:
                    ppg_hidden = group_size // 2
                else:
                    ppg_hidden = group_size

                # Packed dimensions for intermediate dimension (w2 input)
                padded_inter = ((intermediate_size_per_partition + group_size - 1)
                                // group_size) * group_size
                ng_inter = padded_inter // group_size
                if bits == 3:
                    ppg_inter = group_size * 3 // 8
                elif bits == 4:
                    ppg_inter = group_size // 2
                else:
                    ppg_inter = group_size

                from vllm.model_executor.utils import set_weight_attrs

                # w13 (gate_up): per-expert packed [out*ng, ppg]
                # Reshaped to [out, total_packed_cols] in the weight iterator.
                # FusedMoE.weight_loader stacks gate(shard=0) + up(shard=1)
                # along dim=1 (output rows), so w13 = [2*inter, packed_cols].
                w13_weight = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        ng_hidden * ppg_hidden,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                )
                w13_weight_norms = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        ng_hidden,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                # w2 (down): [hidden, packed_cols_inter]
                w2_weight = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        hidden_size,
                        ng_inter * ppg_inter,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                )
                w2_weight_norms = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        hidden_size,
                        ng_inter,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                set_weight_attrs(w13_weight, extra_weight_attrs)
                set_weight_attrs(w13_weight_norms, extra_weight_attrs)
                set_weight_attrs(w2_weight, extra_weight_attrs)
                set_weight_attrs(w2_weight_norms, extra_weight_attrs)

                layer.register_parameter("w13_weight", w13_weight)
                layer.register_parameter("w13_weight_norms", w13_weight_norms)
                layer.register_parameter("w2_weight", w2_weight)
                layer.register_parameter("w2_weight_norms", w2_weight_norms)

                layer._tq_bits = bits
                layer._tq_group_size = group_size
                layer._tq_hidden_size = hidden_size
                layer._tq_inter_size = intermediate_size_per_partition

            def process_weights_after_loading(self, layer: nn.Module) -> None:
                # Data loaded directly as packed uint8 — just ensure contiguous
                for attr in ("w13_weight", "w13_weight_norms",
                             "w2_weight", "w2_weight_norms"):
                    param = getattr(layer, attr)
                    setattr(layer, attr, nn.Parameter(
                        param.data.contiguous(), requires_grad=False))

            def apply(
                self,
                layer: nn.Module,
                x: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                shared_experts_input: torch.Tensor | None = None,
            ) -> torch.Tensor:
                from vllm.model_executor.layers.fused_moe import fused_experts

                # Dequant all experts to FP16 temporary
                w13 = self._dequant_experts(
                    layer.w13_weight, layer.w13_weight_norms,
                    layer._tq_hidden_size,
                    2 * layer._tq_inter_size,
                    layer, x.dtype)
                w2 = self._dequant_experts(
                    layer.w2_weight, layer.w2_weight_norms,
                    layer._tq_inter_size,
                    layer._tq_hidden_size,
                    layer, x.dtype)

                result = fused_experts(
                    hidden_states=x, w1=w13, w2=w2,
                    topk_weights=topk_weights, topk_ids=topk_ids,
                )
                del w13, w2
                return result

            def _dequant_experts(
                self,
                packed_3d: torch.Tensor,
                norms_3d: torch.Tensor,
                in_features: int,
                out_features: int,
                layer: nn.Module,
                dtype: torch.dtype,
            ) -> torch.Tensor:
                """Dequant [E, out, packed_cols] → [E, out, in] FP16."""
                bits = layer._tq_bits
                group_size = layer._tq_group_size
                num_experts = packed_3d.shape[0]
                device = packed_3d.device
                padded_in = ((in_features + group_size - 1)
                             // group_size) * group_size
                n_groups = padded_in // group_size
                packed_per_group = packed_3d.shape[2] // n_groups

                quantizer = _get_quantizer(group_size, bits, str(device))
                result = torch.empty(num_experts, out_features, in_features,
                                     dtype=dtype, device=device)

                for e in range(num_experts):
                    # Reshape [out, total_packed_cols] → [out*ng, ppg]
                    pe = packed_3d[e].reshape(
                        out_features * n_groups, packed_per_group)
                    indices = unpack_indices(pe, bits, group_size)
                    norms_flat = norms_3d[e].reshape(-1)
                    w_groups = quantizer.dequantize(indices, norms_flat)
                    w_deq = w_groups.reshape(out_features, padded_in)
                    result[e] = w_deq[:, :in_features].to(dtype)

                return result
    else:
        # Dummy so references don't break when FusedMoE not available
        TurboQuantMoEMethod = None  # noqa: N806

    # Patch the weight loader for TQ checkpoint compatibility
    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM"
                " (FusedMoE=%s)", "yes" if _has_fused_moe else "no")


# ── Weight iterator remapping ─────────────────────────────────────────

# Matches TQ tensor names: model.layers.0.self_attn.q_proj.weight.tq_packed
_TQ_WEIGHT_RE = re.compile(r'^(.+)\.weight\.(tq_packed|tq_norms)$')


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator for TQ checkpoint compatibility.

    Three cases for TQ weight pairs (tq_packed + tq_norms):
    1. LinearBase layers (has .tq_packed param): yield reshaped packed data
       directly. Model's stacked_params_mapping + PackedvLLMParameter handle
       fusion (gate_proj+up_proj → gate_up_proj) and TP sharding.
    2. FusedMoE expert layers (has w13_weight uint8 param): yield reshaped
       packed as .weight (uint8), norms as .weight_norms (float32).
       FusedMoE.weight_loader copies into per-expert slots via copy_().
    3. Other layers (MoE router, etc.): decompress to FP16.
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _remapped_get_all_weights(self, model_config, model):
        # Build set of param names for TQ-quantized LinearBase layers.
        tq_param_names = {
            name for name, _ in model.named_parameters()
            if name.endswith(".tq_packed")
        }
        # Build set of param names for TQ-quantized MoE layers.
        # These have uint8 w13_weight/w2_weight params.
        moe_param_names = set()
        for name, param in model.named_parameters():
            if (name.endswith(".w13_weight") or name.endswith(".w2_weight")):
                if param.dtype == torch.uint8:
                    moe_param_names.add(name)
        # Also track norms params for MoE
        moe_norms_names = {
            name for name, _ in model.named_parameters()
            if name.endswith(".w13_weight_norms") or name.endswith(".w2_weight_norms")
        }

        # Extract stacked params mapping (ckpt_name → fused_name).
        stacked_mapping: dict[str, str] = {}
        for mod_name, mod in model.named_modules():
            if hasattr(mod, 'packed_modules_mapping'):
                for fused, sources in mod.packed_modules_mapping.items():
                    for src in sources:
                        stacked_mapping[src] = fused

        pending: dict[str, dict[str, torch.Tensor]] = {}

        def _can_load_packed_linear(prefix: str) -> bool:
            """Check if a LinearBase TQ param exists for this weight."""
            if prefix + "tq_packed" in tq_param_names:
                return True
            for ckpt_name, fused_name in stacked_mapping.items():
                search = ckpt_name + "."
                if search in prefix:
                    alt = prefix.replace(search, fused_name + ".") + "tq_packed"
                    if alt in tq_param_names:
                        return True
            return False

        def _is_moe_expert(prefix: str) -> bool:
            """Check if this weight belongs to a FusedMoE expert layer."""
            return bool(re.search(r'\.experts\.\d+\.', prefix))

        def _flush_pair(prefix: str, pair: dict):
            packed = pair["tq_packed"]
            norms = pair["tq_norms"]

            if _can_load_packed_linear(prefix):
                # Case 1: LinearBase — yield packed directly
                reshaped_packed = _reshape_packed_for_vllm(packed, norms)
                yield prefix + "tq_packed", reshaped_packed
                yield prefix + "tq_norms", norms
            elif _is_moe_expert(prefix):
                # Case 2: FusedMoE expert — yield packed as .weight (uint8)
                # and norms as .weight_norms (float32).
                # FusedMoE.weight_loader handles per-expert → fused 3D.
                # expert_params_mapping: "gate_proj.weight" → "w13_weight"
                # Substring match also catches "gate_proj.weight_norms" →
                # "w13_weight_norms".
                reshaped_packed = _reshape_packed_for_vllm(packed, norms)
                yield prefix + "weight", reshaped_packed
                yield prefix + "weight_norms", norms
            else:
                # Case 3: unquantized layer (MoE router, etc.) — FP16
                fp16 = _decompress_tq_to_fp16(packed, norms)
                yield prefix + "weight", fp16

        for name, tensor in _original_get_all_weights(self, model_config, model):
            m = _TQ_WEIGHT_RE.match(name)
            if m:
                prefix = m.group(1) + "."
                suffix = m.group(2)
                pending.setdefault(prefix, {})[suffix] = tensor
                if len(pending[prefix]) == 2:
                    yield from _flush_pair(prefix, pending.pop(prefix))
            else:
                yield name, tensor

        # Flush any remaining buffered pairs
        for base, pair in pending.items():
            if len(pair) == 2:
                yield from _flush_pair(base, pair)
            else:
                logger.warning("Incomplete TQ pair for %s (got %s)",
                               base, list(pair.keys()))
                for suffix, t in pair.items():
                    yield f"{base}{suffix}", t

    DefaultModelLoader.get_all_weights = _remapped_get_all_weights
