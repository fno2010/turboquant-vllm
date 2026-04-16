"""End-to-end test: load a synthetic TQ3 checkpoint via mlx_loader.

Builds a tiny mlx_lm Qwen2 model, compresses its Linear weights to TQ3
native format, writes a checkpoint to a temp dir, loads it back through
``load_tq3_model``, and verifies the reconstructed forward output
matches a PyTorch reference that dequantizes the packed weights and
runs a plain matmul.

If this test passes, the MLX serving path for dense TQ3 checkpoints
is end-to-end validated. Gemma 4 / MoE loaders are a follow-up.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen2 import Model, ModelArgs

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _tiny_args() -> "ModelArgs":
    return ModelArgs(
        model_type="qwen2",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=True,
    )


def _qwen2_config_dict(args: "ModelArgs") -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "rms_norm_eps": args.rms_norm_eps,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": args.rope_theta,
        "tie_word_embeddings": args.tie_word_embeddings,
    }


def _compress_linear_weight(w_np: np.ndarray, bits: int, group_size: int, seed: int):
    """Compress a (out, in) weight tensor to TQ3 packed + norms."""
    from turboquant_vllm.torch_ops import PolarQuantTorch
    from turboquant_vllm.weight_quant import pack_indices, padded_size

    out_features, in_features = w_np.shape
    padded_in, n_groups = padded_size(in_features, group_size)

    w_pt = torch.from_numpy(w_np).float()
    if padded_in > in_features:
        padded = torch.zeros(out_features, padded_in, dtype=w_pt.dtype)
        padded[:, :in_features] = w_pt
    else:
        padded = w_pt

    pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=seed, device="cpu")
    grouped = padded.reshape(-1, group_size)
    indices, norms = pq.quantize(grouped, norm_correction=True)
    packed = pack_indices(indices, bits)
    norms = norms.reshape(out_features, n_groups)

    return packed, norms, pq


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestMLXLoader(unittest.TestCase):
    """Round-trip: MLX model -> compress -> save -> load -> forward."""

    def _write_tq3_checkpoint(self, tmp_path: Path, bits: int = 3, group_size: int = 128):
        """Save a tiny Qwen2 TQ3 native checkpoint. Return (ref_params, packed_keys)."""
        from turboquant_vllm.weight_quant import unpack_indices as _unpack

        args = _tiny_args()
        mx.random.seed(0)
        model = Model(args)
        mx.eval(model.parameters())

        # Snapshot every parameter (name -> mx.array) so we can either save
        # it uncompressed (non-Linears) or compress it (Linears).
        def flatten(tree, prefix=""):
            out = {}
            if isinstance(tree, dict):
                for k, v in tree.items():
                    out.update(flatten(v, f"{prefix}{k}."))
            elif isinstance(tree, list):
                for i, v in enumerate(tree):
                    out.update(flatten(v, f"{prefix}{i}."))
            elif isinstance(tree, mx.array):
                out[prefix.rstrip(".")] = tree
            return out

        params = flatten(model.parameters())

        linear_paths = {name for name, m in model.named_modules() if isinstance(m, nn.Linear)}

        weights_to_save: dict[str, torch.Tensor] = {}
        # Keep a reconstructed-weight dict for the PyTorch reference so the
        # test compares against the same numerical path the loader will take.
        reconstructed: dict[str, np.ndarray] = {}
        seed = 42

        for name, arr in params.items():
            # path looks like "model.layers.0.self_attn.q_proj.weight"
            if name.endswith(".weight") and name.rsplit(".", 1)[0] in linear_paths:
                layer_path = name.rsplit(".", 1)[0]
                w_np = np.array(arr)
                packed, norms, pq = _compress_linear_weight(w_np, bits, group_size, seed)
                weights_to_save[f"{layer_path}.weight.tq_packed"] = packed
                weights_to_save[f"{layer_path}.weight.tq_norms"] = norms

                # Reconstruct for reference forward
                indices = _unpack(packed, bits, group_size)
                w_groups = pq.dequantize(indices, norms.reshape(-1))
                w_deq = w_groups.reshape(w_np.shape[0], -1)[:, : w_np.shape[1]]
                reconstructed[f"{layer_path}.weight"] = w_deq.numpy().astype(np.float32)
            else:
                # Keep as-is (embed_tokens, norm.weight, etc.)
                # Convert mx.array -> torch.Tensor for safetensors.save_file
                np_arr = np.array(arr)
                weights_to_save[name] = torch.from_numpy(np_arr)

        # Save safetensors
        from safetensors.torch import save_file

        save_file(weights_to_save, str(tmp_path / "model.safetensors"))

        # config.json
        with open(tmp_path / "config.json", "w") as f:
            json.dump(_qwen2_config_dict(args), f)

        # tq_config.json (the marker)
        with open(tmp_path / "tq_config.json", "w") as f:
            json.dump(
                {
                    "format": "tq3_native",
                    "bits": bits,
                    "group_size": group_size,
                    "quantizer_seed": seed,
                    "compressed_layers": len(linear_paths),
                },
                f,
            )

        return args, reconstructed

    def test_load_tq3_model_dense_qwen2(self):
        """Round-trip test: compress + save + load + forward parity."""
        import tempfile

        from turboquant_vllm.mlx_loader import load_tq3_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args, reconstructed = self._write_tq3_checkpoint(tmp_path)

            # Load via our loader
            model_tq, config = load_tq3_model(str(tmp_path))
            self.assertEqual(config["model_type"], "qwen2")

            # Build a reference model: same architecture with the
            # reconstructed FP32 weights. Our loader uses the PolarQuant
            # dequant path, so the "reference" must use the SAME decoded
            # weights (not the originals) — otherwise we'd be testing the
            # accuracy of quantization, not the fidelity of the loader.
            mx.random.seed(0)
            model_ref = Model(args)
            mx.eval(model_ref.parameters())

            # Patch reference model with the reconstructed weights so the
            # two paths differ only in how the weight is stored at rest.
            for name, w_np in reconstructed.items():
                parent = model_ref
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                getattr(parent, parts[-1])  # sanity: attribute exists
                setattr(parent, parts[-1], mx.array(w_np.astype(np.float32)))

            mx.eval(model_ref.parameters())

            # Run forward on both
            tokens = mx.array([[1, 2, 3, 4, 5]])
            out_ref = model_ref(tokens)
            out_tq = model_tq(tokens)
            mx.eval(out_ref, out_tq)

            np.testing.assert_allclose(
                np.array(out_tq).astype(np.float32),
                np.array(out_ref).astype(np.float32),
                rtol=5e-3,
                atol=5e-3,
            )

    def test_rejects_non_tq3_checkpoint(self):
        """Missing tq_config.json must raise."""
        import tempfile

        from turboquant_vllm.mlx_loader import load_tq3_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "tq_config.json"):
                load_tq3_model(str(tmp_path))

    def test_rejects_unknown_format(self):
        """tq_config.format != tq3_native must raise."""
        import tempfile

        from turboquant_vllm.mlx_loader import load_tq3_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "config.json").write_text("{}")
            (tmp_path / "tq_config.json").write_text(json.dumps({"format": "tq4_future", "bits": 4, "group_size": 128}))
            with self.assertRaisesRegex(ValueError, "Unsupported TQ format"):
                load_tq3_model(str(tmp_path))


if __name__ == "__main__":
    unittest.main()
