"""vLLM plugin: auto-registers TQ weight compression in all processes.

Activated via environment variables set by enable_weight_quantization().
This plugin is loaded by vLLM in the main process AND in spawned
subprocesses (V1 engine), so the monkey-patch survives multiprocessing spawn.

Environment variables:
    TQ_WEIGHT_BITS: quantization bits (2-8), triggers the hook
    TQ_WEIGHT_GROUP_SIZE: group size (default 128)
"""
import logging
import os

logger = logging.getLogger("turboquant_vllm.plugin")

_patched = False


def register():
    """Called by vLLM's plugin loader in every process."""
    global _patched

    bits = os.environ.get("TQ_WEIGHT_BITS")
    if bits is None:
        logger.debug("TQ_WEIGHT_BITS not set, plugin inactive")
        return

    if _patched:
        return
    _patched = True

    bits = int(bits)
    group_size = int(os.environ.get("TQ_WEIGHT_GROUP_SIZE", "128"))

    try:
        from turboquant_vllm.weight_quant import patch_vllm_loader
        patch_vllm_loader(bits=bits, group_size=group_size, min_size=128)
        logger.info("TurboQuant TQ%d-g%d weight compression registered (pid=%d)",
                     bits, group_size, os.getpid())
    except ImportError as e:
        logger.warning("Failed to register TurboQuant plugin: %s", e)
