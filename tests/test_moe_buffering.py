"""Test the custom MoE buffering logic used by TurboQuantOnlineMoEMethod.

The buffering replaces vLLM's initialize_online_processing for FusedMoE
layers because CopyCounter may not track copy_() into meta tensors
correctly. This test verifies the buffering logic independently of vLLM.
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn


class FakeFusedMoELayer(nn.Module):
    """Minimal stand-in for vLLM's FusedMoE with expert assembly."""

    def __init__(self, num_experts: int, out_dim: int, in_dim: int):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(num_experts, 2 * out_dim, in_dim), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.zeros(num_experts, in_dim, out_dim), requires_grad=False)
        # Attach weight_loaders that do expert assembly
        self.w13_weight.weight_loader = self._make_expert_loader("w13_weight")
        self.w2_weight.weight_loader = self._make_expert_loader("w2_weight")

    def _make_expert_loader(self, param_name: str):
        def loader(param, loaded_weight, expert_id=0, shard_id="gate"):
            if param_name == "w13_weight":
                if shard_id == "gate":
                    param.data[expert_id, : loaded_weight.shape[0]] = loaded_weight
                else:  # up
                    offset = param.shape[1] // 2
                    param.data[expert_id, offset : offset + loaded_weight.shape[0]] = loaded_weight
            else:
                param.data[expert_id] = loaded_weight

        return loader


class TestMoEBuffering(unittest.TestCase):
    def test_buffer_tracks_numel_correctly(self):
        """Buffer should track total loaded numel from tensor sizes."""
        layer = FakeFusedMoELayer(num_experts=4, out_dim=8, in_dim=16)
        total_numel = sum(p.numel() for p in layer.parameters())

        # Move to meta (simulating what create_weights does)
        orig_loaders = {}
        param_shapes = {}
        param_dtypes = {}
        for name, param in list(layer.named_parameters()):
            orig_loaders[name] = param.weight_loader
            param_shapes[name] = tuple(param.shape)
            param_dtypes[name] = param.dtype
            meta = nn.Parameter(torch.empty_like(param, device="meta"), requires_grad=False)
            meta.weight_loader = param.weight_loader
            delattr(layer, name)
            layer.register_parameter(name, meta)

        # Set up buffering
        buffer = []
        loaded_numel = [0]
        materialized = [False]
        completion_callback = mock.Mock()

        def make_buffering_loader(pname, orig):
            def loader(*args, **kwargs):
                if materialized[0]:
                    return orig(*args, **kwargs)
                loaded = args[1] if len(args) > 1 else None
                numel = loaded.numel() if isinstance(loaded, torch.Tensor) else 0
                buffer.append((pname, args, kwargs))
                loaded_numel[0] += numel
                if loaded_numel[0] >= total_numel:
                    materialized[0] = True
                    completion_callback()

            return loader

        for pname, param in layer.named_parameters():
            if pname in orig_loaders:
                param.weight_loader = make_buffering_loader(pname, orig_loaders[pname])

        # Simulate loading experts one by one
        for expert_id in range(4):
            gate = torch.randn(8, 16)
            up = torch.randn(8, 16)
            down = torch.randn(16, 8)
            layer.w13_weight.weight_loader(layer.w13_weight, gate, expert_id=expert_id, shard_id="gate")
            layer.w13_weight.weight_loader(layer.w13_weight, up, expert_id=expert_id, shard_id="up")
            layer.w2_weight.weight_loader(layer.w2_weight, down, expert_id=expert_id)

        # Verify
        self.assertEqual(loaded_numel[0], total_numel)
        self.assertTrue(materialized[0])
        completion_callback.assert_called_once()
        self.assertEqual(len(buffer), 12)  # 4 experts × 3 (gate+up+down)

    def test_buffer_does_not_fire_early(self):
        """Buffer should not fire completion before all data arrives."""
        layer = FakeFusedMoELayer(num_experts=4, out_dim=8, in_dim=16)
        total_numel = sum(p.numel() for p in layer.parameters())

        buffer = []
        loaded_numel = [0]
        fired = [False]

        def make_loader(pname):
            def loader(*args, **kwargs):
                loaded = args[1] if len(args) > 1 else None
                numel = loaded.numel() if isinstance(loaded, torch.Tensor) else 0
                buffer.append((pname, args, kwargs))
                loaded_numel[0] += numel
                if loaded_numel[0] >= total_numel:
                    fired[0] = True

            return loader

        for pname, param in layer.named_parameters():
            param.weight_loader = make_loader(pname)

        # Load only 2 of 4 experts
        for expert_id in range(2):
            layer.w13_weight.weight_loader(layer.w13_weight, torch.randn(8, 16), expert_id=expert_id, shard_id="gate")
            layer.w13_weight.weight_loader(layer.w13_weight, torch.randn(8, 16), expert_id=expert_id, shard_id="up")
            layer.w2_weight.weight_loader(layer.w2_weight, torch.randn(16, 8), expert_id=expert_id)

        self.assertFalse(fired[0])
        self.assertLess(loaded_numel[0], total_numel)

    def test_meta_tensor_numel_matches_real(self):
        """Meta tensor numel should equal real tensor numel."""
        shapes = [(64, 128, 256), (64, 256, 128), (64,)]
        for shape in shapes:
            real = torch.empty(shape)
            meta = torch.empty(shape, device="meta")
            self.assertEqual(real.numel(), meta.numel())


if __name__ == "__main__":
    unittest.main(verbosity=2)
