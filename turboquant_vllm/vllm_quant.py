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
from typing import TYPE_CHECKING, Any, Iterable, Union

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
    )

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
                # Determine bits for this layer
                layer_bits = select_bits(prefix, self.bits, self.sensitive_bits)
                return TurboQuantLinearMethod(self, layer_bits)
            # Check for FusedMoE (MoE layers like GLM-5.1's 256-expert MoE)
            try:
                from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            except ImportError:
                return None
            if isinstance(layer, FusedMoE):
                # Get the FusedMoEConfig from the layer's quant_method
                # (created during FusedMoE.__init__)
                moe_config = layer.moe_config
                return TurboQuantMoEMethod(self, self.bits, moe_config)
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

            # Packed indices: shape depends on bits and group_size
            # For group_size=128, bits=3: each group of 128 indices packs into 48 bytes (128*3/8)
            # Layout: (out_features, n_groups * packed_group_bytes) as uint8
            padded_in = ((input_size_per_partition + self.group_size - 1)
                         // self.group_size) * self.group_size
            n_groups = padded_in // self.group_size

            if self.bits == 4:
                # 4-bit: 2 indices per byte → packed_cols = group_size // 2
                packed_cols = n_groups * (self.group_size // 2)
            elif self.bits == 3:
                # 3-bit: 8 indices per 3 bytes → packed_cols = group_size * 3 // 8
                packed_cols = n_groups * (self.group_size * 3 // 8)
            else:
                # Fallback: 1 byte per index
                packed_cols = n_groups * self.group_size

            # packed_factor: ratio of uncompressed input dim to packed dim
            # vLLM uses this to correctly shard packed weights in TP
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
            layer.tq_packed = nn.Parameter(
                layer.tq_packed.data.contiguous(), requires_grad=False)
            layer.tq_norms = nn.Parameter(
                layer.tq_norms.data.contiguous(), requires_grad=False)

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
            quantizer = _get_quantizer(group_size, bits, str(x.device))
            indices = unpack_indices(layer.tq_packed.data, bits, group_size)
            norms_flat = layer.tq_norms.data.reshape(-1)
            w_groups = quantizer.dequantize(indices, norms_flat)
            padded_in = ((in_features + group_size - 1) // group_size) * group_size
            w_deq = w_groups.reshape(out_features, padded_in)[:, :in_features]
            w_deq = w_deq.to(x.dtype)

            output = torch.matmul(x, w_deq.t())
            del w_deq
            if bias is not None:
                output = output + bias
            return output

    # ------------------------------------------------------------------
    # TurboQuantMoEMethod: TQ3/TQ4 weight compression for FusedMoE layers.
    # GLM-5.1 MoE uses independent gate/up/down projections (NOT fused w13),
    # and the computation is: h = gate(x) * silu(up(x)) (element-wise), then h @ down(x).
    # This is semantically incompatible with vLLM's fused_experts kernel.
    # We decompress only top-k experts (8 out of 256) to avoid OOM.
    # ------------------------------------------------------------------

    # Import FusedMoE base classes needed for TurboQuantMoEMethod
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    class TurboQuantMoEMethod(FusedMoEMethodBase):
        """MoE quantization method that decompresses only top-k experts.

        GLM-5.1 has 256 routed experts with 8 active per token. Decompressing
        all 256 experts in BF16 would require ~17 GB just for weights.
        Instead, we keep all 256 experts in TQ3/TQ4 packed format (~1.5 GB)
        and decompress only the 8 active experts per forward pass.
        """

        def __init__(self, quant_config: "TurboQuantConfig", bits: int,
                     moe: "FusedMoEConfig"):  # FusedMoEConfig from vLLM
            super().__init__(moe)
            self.quant_config = quant_config
            self.bits = bits
            self.group_size = quant_config.group_size

        def get_fused_moe_quant_config(self, layer) -> "FusedMoEQuantConfig":
            # Return UNQUANTIZED so vLLM's modular kernel system doesn't try
            # to dequantize our already-compressed weights. We handle
            # decompression ourselves in apply().
            return FUSED_MOE_UNQUANTIZED_CONFIG

        def create_weights(self, layer, num_experts: int, hidden_size: int,
                           intermediate_size_per_partition: int,
                           params_dtype: torch.dtype, **extra_weight_attrs):
            """Register gate_packed, up_packed, down_packed as uint8 packed params.

            GLM checkpoint: experts.{E}.gate_proj/up_proj/down_proj.weight.tq_packed
            We store each as (num_experts, dim, hidden_size) packed uint8.
            """
            weight_loader = extra_weight_attrs.get("weight_loader")
            assert weight_loader is not None, "weight_loader required"

            # Compute packed dimensions for each projection
            # Each proj: (intermediate_size, hidden_size) → TQ packed uint8
            for proj_name, dim_size in [
                ("gate_packed", intermediate_size_per_partition),
                ("up_packed", intermediate_size_per_partition),
                ("down_packed", hidden_size),
            ]:
                padded_in = ((dim_size + self.group_size - 1) // self.group_size) * self.group_size
                n_groups = padded_in // self.group_size
                if self.bits == 3:
                    packed_cols = n_groups * (self.group_size * 3 // 8)
                elif self.bits == 4:
                    packed_cols = n_groups * (self.group_size // 2)
                else:
                    packed_cols = n_groups * self.group_size

                from fractions import Fraction
                pack_ratio = Fraction(padded_in, packed_cols)

                packed_param = PackedvLLMParameter(
                    data=torch.empty(
                        num_experts * dim_size,
                        packed_cols,
                        dtype=torch.uint8,
                    ),
                    input_dim=1,
                    output_dim=0,
                    packed_dim=1,
                    packed_factor=pack_ratio,
                    weight_loader=weight_loader,
                )

                # Norms: (num_experts * dim_size, n_groups) float32
                norms_param = PackedvLLMParameter(
                    data=torch.empty(
                        num_experts * dim_size,
                        n_groups,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    packed_dim=1,
                    packed_factor=self.group_size,
                    weight_loader=weight_loader,
                )

                layer.register_parameter(proj_name, packed_param)
                layer.register_parameter(f"{proj_name}_norms", norms_param)

            layer._tq_moe_bits = self.bits
            layer._tq_moe_group_size = self.group_size
            layer._tq_moe_intermediate = intermediate_size_per_partition
            layer._tq_moe_hidden = hidden_size
            layer._tq_moe_num_experts = num_experts

        def _decompress_topk_experts(self, layer, topk_ids: torch.Tensor,
                                      x: torch.Tensor):
            """Decompress only the topk_ids experts from packed format.

            Returns (gate_deq, up_deq, down_deq) each of shape (num_topk, dim, hidden).
            """
            bits = layer._tq_moe_bits
            group_size = layer._tq_moe_group_size
            inter_size = layer._tq_moe_intermediate
            hidden_size = layer._tq_moe_hidden

            unique_ids = topk_ids.unique().tolist()
            num_topk = len(unique_ids)

            device = x.device
            quantizer = _get_quantizer(group_size, bits, str(device))

            # Decompress gate, up, down for the unique experts in topk_ids
            results = {}
            for proj_name, dim_size in [
                ("gate_packed", inter_size),
                ("up_packed", inter_size),
                ("down_packed", hidden_size),
            ]:
                packed = getattr(layer, proj_name).data
                norms = getattr(layer, f"{proj_name}_norms").data

                # Extract rows for the topk experts
                # Each expert occupies dim_size rows
                indices = torch.tensor(
                    [eid * dim_size for eid in unique_ids],
                    dtype=torch.long, device=device
                )
                # Expand to all rows of each expert
                idx_expanded = indices.unsqueeze(1).repeat(1, dim_size).flatten()
                offset = torch.arange(dim_size, device=device).unsqueeze(0) * num_topk
                row_idx = (idx_expanded.unsqueeze(1) + offset).flatten()

                proj_packed = packed[row_idx]  # (num_topk * dim_size, packed_cols)
                proj_norms = norms[row_idx]    # (num_topk * dim_size, n_groups)

                # Decompress
                padded_in = ((dim_size + group_size - 1) // group_size) * group_size
                indices_unpacked = unpack_indices(proj_packed, bits, group_size)
                n_groups = padded_in // group_size
                proj_groups = quantizer.dequantize(
                    indices_unpacked, proj_norms.reshape(-1)
                )  # (num_topk * dim_size * n_groups,)

                proj_deq = proj_groups.reshape(num_topk * dim_size, padded_in)[:, :dim_size]
                results[proj_name] = proj_deq.reshape(num_topk, dim_size, dim_size).to(x.dtype)

            return results["gate_packed"], results["up_packed"], results["down_packed"]

        def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  shared_experts_input: torch.Tensor | None) -> torch.Tensor:
            """GLM-5.1 MoE computation: gate * silu(up) element-wise, then @ down.

            Only decompresses the experts in topk_ids (typically 8 out of 256).
            topk_ids: (total_tokens, num_topk) expert indices
            topk_weights: (total_tokens, num_topk) weights for each expert
            """
            # Decompress only the unique experts present in this batch
            gate_w, up_w, down_w = self._decompress_topk_experts(layer, topk_ids, x)
            num_topk = gate_w.shape[0]
            # unique_ids[k] = expert ID for the k-th decompressed expert (sorted)
            unique_ids = topk_ids.unique().tolist()
            # Build lookup: expert_id → index in decompressed weights
            eid_to_idx = {eid: k for k, eid in enumerate(unique_ids)}

            # x: (seq_len, batch, hidden) or (batch, hidden)
            orig_shape = x.shape
            if x.dim() == 3:
                x = x.transpose(0, 1).reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                x = x.reshape(-1, x.shape[-1])
            # x: (total_tokens, hidden_size)

            total_tokens = x.shape[0]
            output = torch.zeros_like(x)

            # Iterate: for each token i and each topk slot k,
            # look up the expert for that slot and compute its contribution
            for i in range(total_tokens):
                for k in range(num_topk):
                    eid = topk_ids[i, k].item()
                    w = topk_weights[i, k]
                    if w == 0:
                        continue
                    idx = eid_to_idx[eid]
                    gate = gate_w[idx]
                    up = up_w[idx]
                    down = down_w[idx]
                    # GLM: h = gate * silu(up) element-wise, then @ down
                    h_gate = torch.matmul(x[i], gate.t())  # (intermediate,)
                    h_up = torch.matmul(x[i], up.t())       # (intermediate,)
                    h_up_act = h_up * torch.sigmoid(h_up)
                    h = h_gate * h_up_act
                    out = torch.matmul(h, down.t()) * w
                    output[i] += out

            # Restore original shape
            if len(orig_shape) == 3:
                output = output.reshape(orig_shape[1], orig_shape[0], -1).transpose(0, 1)
            elif len(orig_shape) == 2:
                output = output.reshape(orig_shape)

            return output

        def load_weights(self, layer, weights) -> Iterable[str]:
            """Custom weight loading for GLM MoE checkpoints.

            GLM has: experts.{E}.gate_proj.weight.tq_packed (per expert, 2D).
            We map these to our gate_packed/up_packed/down_packed params.
            """
            num_experts = layer._tq_moe_num_experts
            hidden_size = layer._tq_moe_hidden
            inter_size = layer._tq_moe_intermediate

            # Build mapping: checkpoint name → (param_name, expert_id)
            # GLM: experts.{E}.gate_proj → experts.gate_packed[E * inter_size : (E+1) * inter_size]
            #      experts.{E}.up_proj → experts.up_packed[E * inter_size : ...]
            #      experts.{E}.down_proj → experts.down_packed[E * hidden_size : ...]
            expert_params_mapping = []
            for expert_id in range(num_experts):
                expert_prefix = f"experts.{expert_id}."
                for proj_name, dim_size in [
                    ("gate_proj", inter_size),
                    ("up_proj", inter_size),
                    ("down_proj", hidden_size),
                ]:
                    weight_name = f"{expert_prefix}{proj_name}.weight"
                    # Map to our param: gate_packed, up_packed, or down_packed
                    if proj_name == "gate_proj":
                        param_base = "gate_packed"
                    elif proj_name == "up_proj":
                        param_base = "up_packed"
                    else:
                        param_base = "down_packed"
                    expert_params_mapping.append((param_base, weight_name, expert_id))

            for expert_name, loaded_weight in weights:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in expert_name:
                        continue
                    # weight_name found in expert_name — load it
                    param = getattr(layer, param_name)
                    dim_size = inter_size if "gate" in param_name or "up" in param_name else hidden_size

                    # loaded_weight: (dim_size, hidden_size) TQ packed
                    # Store at row offset: expert_id * dim_size
                    start_row = expert_id * dim_size
                    end_row = start_row + dim_size

                    if loaded_weight.dim() == 2:
                        param.data[start_row:end_row] = loaded_weight.contiguous()
                    else:
                        # Should not happen for GLM (experts are 2D)
                        param.data[start_row:end_row] = loaded_weight.squeeze(0).contiguous()

                    yield expert_name

    # ------------------------------------------------------------------
    # Patch FusedMoE.load_weights to delegate to quant_method's loader
    # when it's a TurboQuantMoEMethod.
    # ------------------------------------------------------------------
    _original_fused_moe_load_weights = FusedMoE.load_weights

    def _tq_fused_moe_load_weights(self, weights):
        if isinstance(self.quant_method, TurboQuantMoEMethod):
            # Use our custom loader that understands GLM TQ checkpoint format
            return self.quant_method.load_weights(self, weights)
        return _original_fused_moe_load_weights(self, weights)

    FusedMoE.load_weights = _tq_fused_moe_load_weights

    # Patch the weight loader to remap old-style checkpoint names
    # Old: model.layers.0.self_attn.q_proj.weight.tq_packed
    # New: model.layers.0.self_attn.q_proj.tq_packed
    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM")


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to remap old TQ checkpoint names."""
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _remapped_get_all_weights(self, model_config, model):
        for name, tensor in _original_get_all_weights(self, model_config, model):
            # Remap old-style .weight.tq_packed → .tq_packed
            if ".weight.tq_packed" in name:
                name = name.replace(".weight.tq_packed", ".tq_packed")
            elif ".weight.tq_norms" in name:
                name = name.replace(".weight.tq_norms", ".tq_norms")
            yield name, tensor

    DefaultModelLoader.get_all_weights = _remapped_get_all_weights
