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
            """Register w13 (gate+up fused) and w2 (down) TQ packed params.

            expert_params_mapping (from SharedFusedMoE.make_expert_params_mapping)
            expects params named w13_weight and w2_weight so that the standard
            vLLM weight loading path (deepseek_v2.load_weights →
            expert_params_mapping → param.weight_loader) finds them.

            GLM checkpoint stores separate gate_proj/up_proj/down_proj per expert.
            We concatenate gate+up into w13 and keep down as w2.
            No TP sharding — expert parallelism handles distribution.

            We register as plain torch.nn.Parameter (NOT PackedvLLMParameter)
            to avoid LinearMethod.weight_loader trying to apply TP sharding.
            """
            inter_size = intermediate_size_per_partition

            # Helper to compute packed dimensions
            def _packed_dims(dim_size):
                padded_in = ((dim_size + self.group_size - 1) // self.group_size) * self.group_size
                n_groups = padded_in // self.group_size
                if self.bits == 3:
                    packed_cols = n_groups * (self.group_size * 3 // 8)
                elif self.bits == 4:
                    packed_cols = n_groups * (self.group_size // 2)
                else:
                    packed_cols = n_groups * self.group_size
                return packed_cols, n_groups, padded_in

            # w13 = [gate; up] — stored as (num_experts * 2 * inter, packed_w13)
            packed_w13, n_groups_w13, padded_in_w13 = _packed_dims(inter_size)
            w13_rows = num_experts * 2 * inter_size

            # Register params and attach weight_loader method + layer reference.
            # vLLM's deepseek_v2.load_weights calls param.weight_loader(...) for
            # each (pth_path, param_name, expert_id, shard_id) in
            # expert_params_mapping. We need weight_loader attached to each param.
            w13_weight = torch.nn.Parameter(
                torch.empty(w13_rows, packed_w13, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            w13_weight.weight_loader = self.weight_loader
            w13_weight._tq_layer = layer

            w13_norms = torch.nn.Parameter(
                torch.empty(w13_rows, n_groups_w13, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_norms", w13_norms)
            w13_norms.weight_loader = self.weight_loader
            w13_norms._tq_layer = layer

            # w2 = down — stored as (num_experts * inter, packed_w2)
            packed_w2, n_groups_w2, padded_in_w2 = _packed_dims(hidden_size)
            w2_rows = num_experts * inter_size

            w2_weight = torch.nn.Parameter(
                torch.empty(w2_rows, packed_w2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            w2_weight.weight_loader = self.weight_loader
            w2_weight._tq_layer = layer

            w2_norms = torch.nn.Parameter(
                torch.empty(w2_rows, n_groups_w2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_norms", w2_norms)
            w2_norms.weight_loader = self.weight_loader
            w2_norms._tq_layer = layer

            layer._tq_moe_bits = self.bits
            layer._tq_moe_group_size = self.group_size
            layer._tq_moe_intermediate = inter_size
            layer._tq_moe_hidden = hidden_size
            layer._tq_moe_num_experts = num_experts

        def _decompress_topk_experts(self, layer, topk_ids: torch.Tensor,
                                      x: torch.Tensor):
            """Decompress only the topk_ids experts from packed format.

            w13_weight stores [gate; up] concatenated: each expert has
            2*inter_size rows (gate first, up second).
            w2_weight stores down_proj: each expert has inter_size rows.

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

            padded_in_w13 = ((inter_size + group_size - 1) // group_size) * group_size
            padded_in_w2 = ((hidden_size + group_size - 1) // group_size) * group_size

            # Decompress all gate+up rows from w13 at once, then split
            # w13 layout per expert: [gate; up] = 2*inter_size rows
            # For gate of expert E at position k: rows (2*E + k) * inter_size + j
            # For up of expert E at position k: rows (2*E + 1 + k) * inter_size + j
            w13_packed = layer.w13_weight.data
            w13_norms = layer.w13_norms.data
            w2_packed = layer.w2_weight.data
            w2_norms = layer.w2_norms.data

            k_indices = torch.arange(num_topk, device=device, dtype=torch.long)
            gate_base = torch.tensor([eid * 2 * inter_size for eid in unique_ids],
                                     dtype=torch.long, device=device)
            j_indices = torch.arange(inter_size, device=device, dtype=torch.long)
            gate_row_idx = (gate_base.unsqueeze(1) + j_indices.unsqueeze(0)).flatten()
            up_base = gate_base + inter_size
            up_row_idx = (up_base.unsqueeze(1) + j_indices.unsqueeze(0)).flatten()

            gate_packed = w13_packed[gate_row_idx]
            gate_norms = w13_norms[gate_row_idx]
            up_packed = w13_packed[up_row_idx]
            up_norms = w13_norms[up_row_idx]

            # Decompress gate: shape (num_topk, inter_size, inter_size)
            gate_indices = unpack_indices(gate_packed, bits, group_size)
            gate_groups = quantizer.dequantize(gate_indices, gate_norms.reshape(-1))
            gate_deq = gate_groups.reshape(num_topk * inter_size,
                                          padded_in_w13)[:, :inter_size]
            gate_w = gate_deq.reshape(num_topk, inter_size, inter_size).to(x.dtype)

            # Decompress up: shape (num_topk, inter_size, inter_size)
            up_indices = unpack_indices(up_packed, bits, group_size)
            up_groups = quantizer.dequantize(up_indices, up_norms.reshape(-1))
            up_deq = up_groups.reshape(num_topk * inter_size,
                                       padded_in_w13)[:, :inter_size]
            up_w = up_deq.reshape(num_topk, inter_size, inter_size).to(x.dtype)

            # Decompress down from w2: shape (num_topk, hidden_size, inter_size)
            # w2 layout per expert: inter_size rows
            # For down of expert E at position k: rows (E + k) * inter_size + j
            down_base = torch.tensor([eid * inter_size for eid in unique_ids],
                                      dtype=torch.long, device=device)
            down_row_idx = (down_base.unsqueeze(1) + j_indices.unsqueeze(0)).flatten()
            down_packed = w2_packed[down_row_idx]
            down_norms = w2_norms[down_row_idx]
            down_indices = unpack_indices(down_packed, bits, group_size)
            down_groups = quantizer.dequantize(down_indices, down_norms.reshape(-1))
            down_deq = down_groups.reshape(num_topk * inter_size,
                                          padded_in_w2)[:, :hidden_size]
            down_w = down_deq.reshape(num_topk, hidden_size, inter_size).to(x.dtype)

            return gate_w, up_w, down_w

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
                    gate = gate_w[idx]    # (inter, hidden)
                    up = up_w[idx]        # (inter, hidden)
                    down = down_w[idx]     # (hidden, inter) — GLM layout
                    # GLM: h = gate * silu(up) element-wise, then @ down
                    h_gate = torch.matmul(x[i], gate.t())  # (inter,)
                    h_up = torch.matmul(x[i], up.t())       # (inter,)
                    h_up_act = h_up * torch.sigmoid(h_up)
                    h = h_gate * h_up_act
                    # down.t() = (inter, hidden) so h @ down.t() = (inter,) @ (inter, hidden) = (hidden,)
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

            GLM checkpoint stores separate gate_proj/up_proj/down_proj per expert.
            expert_params_mapping (from SharedFusedMoE.make_expert_params_mapping)
            expects params named w13_weight (gate+up fused) and w2_weight (down).

            Our w13_weight stores [gate; up] concatenated: each expert occupies
            2*intermediate rows. w2_weight stores down: each expert occupies
            intermediate rows.
            """
            hidden_size = layer._tq_moe_hidden
            inter_size = layer._tq_moe_intermediate

            # Mapping from GLM checkpoint name patterns to our storage
            # GLM: experts.{E}.gate_proj.weight.tq_packed → w13_weight
            #       experts.{E}.up_proj.weight.tq_packed   → w13_weight
            #       experts.{E}.down_proj.weight.tq_packed → w2_weight
            for expert_name, loaded_weight in weights:
                # Determine projection type and weight type (packed or norms)
                proj_type = None  # "gate", "up", "down"
                weight_type = None  # "packed" or "norms"

                for pt in ("gate_proj", "up_proj", "down_proj"):
                    if f".{pt}.weight.tq_packed" in expert_name:
                        proj_type = pt.replace("_proj", "")
                        weight_type = "packed"
                        break
                    elif f".{pt}.weight.tq_norms" in expert_name:
                        proj_type = pt.replace("_proj", "")
                        weight_type = "norms"
                        break

                if proj_type is None:
                    continue  # Not a MoE expert weight

                # Extract global expert ID
                experts_pos = expert_name.find("experts.")
                if experts_pos == -1:
                    continue
                start = experts_pos + len("experts.")
                end = expert_name.find(".", start)
                global_expert_id = int(expert_name[start:end])

                # Convert global → local via expert_map
                if layer._expert_map is not None:
                    local_idx = layer._expert_map[global_expert_id].item()
                    if local_idx < 0:
                        continue
                else:
                    local_idx = global_expert_id

                # Copy to the correct location in our w13_weight or w2_weight
                if proj_type in ("gate", "up"):
                    # w13_weight: each expert has 2*inter_size rows
                    # gate → first inter_size rows, up → second inter_size rows
                    base_row = local_idx * 2 * inter_size
                    if proj_type == "gate":
                        dest_row = base_row  # first half
                    else:
                        dest_row = base_row + inter_size  # second half

                    param_name = ("w13_weight" if weight_type == "packed"
                                  else "w13_norms")
                    dim_size = inter_size
                else:  # down
                    param_name = ("w2_weight" if weight_type == "packed"
                                  else "w2_norms")
                    dest_row = local_idx * hidden_size
                    dim_size = hidden_size

                param = getattr(layer, param_name)
                if loaded_weight.dim() == 2:
                    param.data[dest_row:dest_row + dim_size] = loaded_weight.contiguous()
                else:
                    param.data[dest_row:dest_row + dim_size] = loaded_weight.squeeze(0).contiguous()

                yield expert_name

        def weight_loader(self, param, loaded_weight, name_mapped,
                         shard_id=None, expert_id=None, return_success=False):
            """Intercept weight loading from expert_params_mapping loop.

            expert_params_mapping maps checkpoint gate_proj → w13_weight.
            We handle the GLM separate gate/up → w13 concatenation here.

            Called from deepseek_v2.load_weights → expert_params_mapping loop
            via param.weight_loader(param, loaded_weight, name_mapped,
                                   shard_id=shard_id, expert_id=expert_id,
                                   return_success=True).

            The layer is accessed via param._tq_layer (set in create_weights).
            """
            layer = getattr(param, '_tq_layer', None)
            if layer is None:
                return False if return_success else None

            # Determine projection type from shard_id
            # shard_id: "w1"=gate, "w2"=down, "w3"=up
            if shard_id == "w2":
                param_name = "w2_weight"
                dest_row = expert_id * layer._tq_moe_hidden
                dim_size = layer._tq_moe_hidden
            elif shard_id in ("w1", "w3"):
                param_name = "w13_weight"
                base_row = expert_id * 2 * layer._tq_moe_intermediate
                dest_row = base_row if shard_id == "w1" else base_row + layer._tq_moe_intermediate
                dim_size = layer._tq_moe_intermediate
            else:
                return False if return_success else None

            target_param = getattr(layer, param_name, None)
            if target_param is None:
                return False if return_success else None

            if loaded_weight.dim() == 2:
                target_param.data[dest_row:dest_row + dim_size] = loaded_weight.contiguous()
            else:
                target_param.data[dest_row:dest_row + dim_size] = loaded_weight.squeeze(0).contiguous()

            return True if return_success else None

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
