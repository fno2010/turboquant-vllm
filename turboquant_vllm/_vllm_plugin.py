"""vLLM plugin: auto-registers TQ weight compression in all processes.

Activated via environment variables set by enable_weight_quantization().
This plugin is loaded by vLLM in the main process AND in spawned
subprocesses (V1 engine), so the monkey-patch survives multiprocessing spawn.

Environment variables:
    TQ_WEIGHT_BITS: quantization bits (2-8), triggers the hook
    TQ_WEIGHT_GROUP_SIZE: group size (default 128)
"""
import os


def register():
    """Called by vLLM's plugin loader in every process."""
    bits = os.environ.get("TQ_WEIGHT_BITS")
    if bits is None:
        return

    bits = int(bits)
    group_size = int(os.environ.get("TQ_WEIGHT_GROUP_SIZE", "128"))

    try:
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading as _original,
        )
    except ImportError:
        return

    from turboquant_vllm.weight_quant import _replace_linear_layers

    # Only patch once per process (idempotent check)
    if getattr(register, "_patched", False):
        return
    register._patched = True

    def patched_process_weights(model, model_config, target_device):
        _original(model, model_config, target_device)
        _replace_linear_layers(model, bits=bits, group_size=group_size, min_size=128)

    import vllm.model_executor.model_loader.utils as loader_utils
    loader_utils.process_weights_after_loading = patched_process_weights
