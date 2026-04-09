"""Tests for vllm_quant.py — TQ3/TQ4 native vLLM integration.

CPU-only tests validating:
- Packed tensor reshape (checkpoint → vLLM layout)
- Weight iterator routing (LinearBase vs MoE expert vs router)
- MoE packed data assembly (per-expert → fused 3D)
- Dequant roundtrip through the full pipeline
- Expert params mapping substring trick (weight_norms)

Run: pytest tests/test_vllm_quant.py -v
"""

import re
import pytest
import torch

from turboquant_vllm.vllm_quant import (
    _infer_bits_from_packed,
    _decompress_tq_to_fp16,
    _reshape_packed_for_vllm,
    _TQ_WEIGHT_RE,
)
from turboquant_vllm.weight_quant import (
    _get_quantizer,
    pack_indices,
    unpack_indices,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_tq_pair(out_features, in_features, bits=3, group_size=128):
    """Create a (packed, norms) pair as save_tq3_checkpoint would."""
    q = _get_quantizer(group_size, bits, "cpu")
    w = torch.randn(out_features, in_features)
    padded_in = ((in_features + group_size - 1) // group_size) * group_size
    n_groups = padded_in // group_size

    if padded_in > in_features:
        w_pad = torch.zeros(out_features, padded_in)
        w_pad[:, :in_features] = w
    else:
        w_pad = w

    grouped = w_pad.reshape(-1, group_size)
    indices, norms_raw = q.quantize(grouped, norm_correction=True)
    packed = pack_indices(indices, bits)  # [out*ng, ppg]
    norms = norms_raw.reshape(out_features, n_groups)
    return w, packed, norms


# ── Reshape tests ────────────────────────────────────────────────────

class TestReshapePackedForVllm:

    def test_basic_shape(self):
        packed = torch.randint(0, 255, (32, 48), dtype=torch.uint8)
        norms = torch.randn(16, 2)
        result = _reshape_packed_for_vllm(packed, norms)
        assert result.shape == (16, 96)

    def test_data_preserved(self):
        packed = torch.randint(0, 255, (32, 48), dtype=torch.uint8)
        norms = torch.randn(16, 2)
        result = _reshape_packed_for_vllm(packed, norms)
        # Should be equivalent to reshape(16, -1)
        assert torch.equal(result, packed.reshape(16, -1))

    def test_single_group(self):
        # in_features = 128 = group_size, so n_groups = 1
        packed = torch.randint(0, 255, (8, 48), dtype=torch.uint8)
        norms = torch.randn(8, 1)
        result = _reshape_packed_for_vllm(packed, norms)
        assert result.shape == (8, 48)
        assert torch.equal(result, packed)


# ── Bits inference ───────────────────────────────────────────────────

class TestInferBits:

    def test_3bit(self):
        # 3-bit: ppg = 128*3/8 = 48
        packed = torch.zeros(16, 48, dtype=torch.uint8)
        norms = torch.zeros(16, 1)
        assert _infer_bits_from_packed(packed, norms, 128) == 3

    def test_4bit(self):
        # 4-bit: ppg = 128/2 = 64
        packed = torch.zeros(16, 64, dtype=torch.uint8)
        norms = torch.zeros(16, 1)
        assert _infer_bits_from_packed(packed, norms, 128) == 4

    def test_multigroup_3bit(self):
        # 2 groups: ppg=48, total_packed_cols=96
        packed = torch.zeros(32, 48, dtype=torch.uint8)
        norms = torch.zeros(16, 2)
        assert _infer_bits_from_packed(packed, norms, 128) == 3


# ── Decompress roundtrip ────────────────────────────────────────────

class TestDecompressTqToFp16:

    def test_roundtrip_shape(self):
        w, packed, norms = _make_tq_pair(16, 256)
        decompressed = _decompress_tq_to_fp16(packed, norms)
        assert decompressed.shape == (16, 256)
        assert decompressed.dtype == torch.float16

    def test_roundtrip_quality(self):
        w, packed, norms = _make_tq_pair(32, 512)
        decompressed = _decompress_tq_to_fp16(packed, norms)
        mse = ((w.float() - decompressed.float()) ** 2).mean().item()
        assert mse < 0.1, f"MSE too high: {mse}"

    def test_single_group(self):
        w, packed, norms = _make_tq_pair(8, 128)
        decompressed = _decompress_tq_to_fp16(packed, norms)
        assert decompressed.shape == (8, 128)


# ── MoE packed assembly ─────────────────────────────────────────────

class TestMoEPackedAssembly:
    """Test that per-expert packed data can be assembled into fused 3D params."""

    def test_w13_gate_up_stacking(self):
        """gate_proj + up_proj packed data stacks correctly into w13."""
        out_dim, in_dim = 16, 256
        num_experts = 4

        _, gate_packed, gate_norms = _make_tq_pair(out_dim, in_dim)
        _, up_packed, up_norms = _make_tq_pair(out_dim, in_dim)

        gate_reshaped = _reshape_packed_for_vllm(gate_packed, gate_norms)
        up_reshaped = _reshape_packed_for_vllm(up_packed, up_norms)

        # Simulate w13_weight param: [E, 2*out, packed_cols]
        packed_cols = gate_reshaped.shape[1]
        w13 = torch.zeros(num_experts, 2 * out_dim, packed_cols, dtype=torch.uint8)

        # Simulate weight_loader for expert 0
        expert_id = 0
        shard_size = out_dim
        w13[expert_id, :shard_size, :].copy_(gate_reshaped)  # shard=0
        w13[expert_id, shard_size:, :].copy_(up_reshaped)     # shard=1

        assert (w13[expert_id, :shard_size] != 0).any()
        assert (w13[expert_id, shard_size:] != 0).any()
        assert (w13[1] == 0).all()  # other experts still zero

    def test_norms_stacking(self):
        """Norms stack the same way as packed data."""
        out_dim, in_dim = 16, 256
        _, _, gate_norms = _make_tq_pair(out_dim, in_dim)
        _, _, up_norms = _make_tq_pair(out_dim, in_dim)

        n_groups = gate_norms.shape[1]
        w13_norms = torch.zeros(4, 2 * out_dim, n_groups)

        w13_norms[0, :out_dim, :].copy_(gate_norms)
        w13_norms[0, out_dim:, :].copy_(up_norms)

        assert w13_norms[0].abs().sum() > 0
        assert w13_norms[1].abs().sum() == 0

    def test_dequant_after_assembly(self):
        """Full roundtrip: quantize → pack → reshape → assemble → dequant."""
        out_dim, in_dim, bits, gs = 16, 256, 3, 128
        q = _get_quantizer(gs, bits, "cpu")
        padded_in = ((in_dim + gs - 1) // gs) * gs
        n_groups = padded_in // gs

        # Original weight
        w_orig = torch.randn(out_dim, in_dim)
        w_pad = torch.zeros(out_dim, padded_in)
        w_pad[:, :in_dim] = w_orig

        # Quantize
        grouped = w_pad.reshape(-1, gs)
        indices, norms_raw = q.quantize(grouped, norm_correction=True)
        packed = pack_indices(indices, bits)
        norms = norms_raw.reshape(out_dim, n_groups)

        # Reshape to vLLM layout
        packed_vllm = _reshape_packed_for_vllm(packed, norms)

        # Simulate what _dequant_experts does
        ppg = packed_vllm.shape[1] // n_groups
        pe = packed_vllm.reshape(out_dim * n_groups, ppg)
        indices_out = unpack_indices(pe, bits, gs)
        norms_flat = norms.reshape(-1)
        w_groups = q.dequantize(indices_out, norms_flat)
        w_deq = w_groups.reshape(out_dim, padded_in)[:, :in_dim]

        mse = ((w_orig.float() - w_deq.float()) ** 2).mean().item()
        assert mse < 0.1, f"MSE too high: {mse}"


# ── Regex and name matching ──────────────────────────────────────────

class TestWeightNameMatching:

    def test_tq_weight_re_nonexpert(self):
        name = "model.layers.0.self_attn.q_proj.weight.tq_packed"
        m = _TQ_WEIGHT_RE.match(name)
        assert m is not None
        assert m.group(1) == "model.layers.0.self_attn.q_proj"
        assert m.group(2) == "tq_packed"

    def test_tq_weight_re_expert(self):
        name = "model.layers.10.mlp.experts.0.gate_proj.weight.tq_packed"
        m = _TQ_WEIGHT_RE.match(name)
        assert m is not None
        assert m.group(1) == "model.layers.10.mlp.experts.0.gate_proj"

    def test_tq_weight_re_norms(self):
        name = "model.layers.0.self_attn.kv_b_proj.weight.tq_norms"
        m = _TQ_WEIGHT_RE.match(name)
        assert m is not None
        assert m.group(2) == "tq_norms"

    def test_tq_weight_re_no_match(self):
        assert _TQ_WEIGHT_RE.match("model.layers.0.norm.weight") is None
        assert _TQ_WEIGHT_RE.match("model.embed_tokens.weight") is None

    def test_expert_detection(self):
        """The is_expert check in _flush_pair."""
        expert = "model.layers.10.mlp.experts.0.gate_proj."
        nonexpert = "model.layers.0.self_attn.q_proj."
        router = "model.layers.1.mlp.gate."

        assert bool(re.search(r'\.experts\.\d+\.', expert))
        assert not bool(re.search(r'\.experts\.\d+\.', nonexpert))
        assert not bool(re.search(r'\.experts\.\d+\.', router))

    def test_expert_params_mapping_substring(self):
        """Verify that 'gate_proj.weight' matches inside 'gate_proj.weight_norms'."""
        weight_name = "gate_proj.weight"

        packed_name = "experts.0.gate_proj.weight"
        norms_name = "experts.0.gate_proj.weight_norms"

        # Both should match
        assert weight_name in packed_name
        assert weight_name in norms_name

        # Replacement should produce correct param names
        assert packed_name.replace(weight_name, "w13_weight") == "experts.0.w13_weight"
        assert norms_name.replace(weight_name, "w13_weight") == "experts.0.w13_weight_norms"


# ── Checkpoint name inventory ────────────────────────────────────────

class TestCheckpointNameRouting:
    """Test that checkpoint weight names get routed to the correct case."""

    def _classify(self, prefix: str, tq_param_names: set, stacked_mapping: dict):
        """Simplified version of _flush_pair's routing logic."""
        # Case 1: LinearBase
        tq_name = prefix + "tq_packed"
        if tq_name in tq_param_names:
            return "linear_direct"
        for ckpt_name, fused_name in stacked_mapping.items():
            search = ckpt_name + "."
            if search in prefix:
                alt = prefix.replace(search, fused_name + ".") + "tq_packed"
                if alt in tq_param_names:
                    return "linear_stacked"

        # Case 2: MoE expert
        if bool(re.search(r'\.experts\.\d+\.', prefix)):
            return "moe_expert"

        # Case 3: decompress
        return "decompress"

    def test_glm47_routing(self):
        """Simulate GLM-4.7-Flash weight routing."""
        # Params that exist on the model
        tq_params = {
            # Dense layers (layer 0-2) have fused projections
            "model.layers.0.self_attn.fused_qkv_a_proj.tq_packed",
            "model.layers.0.self_attn.kv_b_proj.tq_packed",
            "model.layers.0.self_attn.q_b_proj.tq_packed",
            "model.layers.0.self_attn.o_proj.tq_packed",
            "model.layers.0.mlp.gate_up_proj.tq_packed",
            "model.layers.0.mlp.down_proj.tq_packed",
            # MoE layers (3+) — same attention but no dense MLP
            "model.layers.3.self_attn.fused_qkv_a_proj.tq_packed",
            "model.layers.3.self_attn.kv_b_proj.tq_packed",
            "model.layers.3.self_attn.q_b_proj.tq_packed",
            "model.layers.3.self_attn.o_proj.tq_packed",
        }

        stacked = {
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "q_a_proj": "fused_qkv_a_proj",
            "kv_a_proj_with_mqa": "fused_qkv_a_proj",
        }

        # Dense layer attention — stacked
        assert self._classify(
            "model.layers.0.self_attn.q_a_proj.", tq_params, stacked
        ) == "linear_stacked"
        assert self._classify(
            "model.layers.0.self_attn.kv_a_proj_with_mqa.", tq_params, stacked
        ) == "linear_stacked"

        # Dense layer attention — direct
        assert self._classify(
            "model.layers.0.self_attn.kv_b_proj.", tq_params, stacked
        ) == "linear_direct"
        assert self._classify(
            "model.layers.0.self_attn.o_proj.", tq_params, stacked
        ) == "linear_direct"

        # Dense MLP — stacked
        assert self._classify(
            "model.layers.0.mlp.gate_proj.", tq_params, stacked
        ) == "linear_stacked"
        assert self._classify(
            "model.layers.0.mlp.up_proj.", tq_params, stacked
        ) == "linear_stacked"

        # Dense MLP — direct
        assert self._classify(
            "model.layers.0.mlp.down_proj.", tq_params, stacked
        ) == "linear_direct"

        # MoE expert — should route to moe_expert
        assert self._classify(
            "model.layers.3.mlp.experts.0.gate_proj.", tq_params, stacked
        ) == "moe_expert"
        assert self._classify(
            "model.layers.3.mlp.experts.5.down_proj.", tq_params, stacked
        ) == "moe_expert"

        # MoE router — should decompress to FP16
        assert self._classify(
            "model.layers.3.mlp.gate.", tq_params, stacked
        ) == "decompress"

        # MoE layer attention — direct (same as dense)
        assert self._classify(
            "model.layers.3.self_attn.kv_b_proj.", tq_params, stacked
        ) == "linear_direct"
