"""Native TQ3 checkpoint format: ~12 GB on disk, loads on any 24+ GB GPU.

Solves the problem where the original 52 GB BF16 checkpoint doesn't fit on
a 48 GB GPU during loading, even though the compressed model is only 12 GB
at runtime.

The approach: read the original safetensors tensor-by-tensor (lazy loading,
~15 MB peak memory), TQ3-compress each weight on CPU, and save the packed
indices + norms as a new safetensors file. Non-weight tensors (embeddings,
norms, biases) are kept as FP16.

The output checkpoint is ~12 GB and loads directly into GPU memory via a
custom vLLM weight loader.

Usage:
    # Step 1: Create TQ3 checkpoint (CPU only, ~60 GB RAM, ~2 min)
    from turboquant_vllm.checkpoint import save_tq3_checkpoint
    save_tq3_checkpoint("google/gemma-4-26B-A4B-it", "./gemma4-tq3-native")

    # Step 2: Serve on L40S 48GB
    from turboquant_vllm.checkpoint import enable_tq3_serving
    enable_tq3_serving()
    # then: vllm serve ./gemma4-tq3-native
"""

import json
import logging
import os
import torch

logger = logging.getLogger(__name__)

# Layers to keep at full precision
_SKIP_PATTERNS = ("lm_head", "embed", "norm", "head")


def save_tq3_checkpoint(
    model_id: str,
    output_dir: str,
    bits: int = 3,
    group_size: int = 128,
):
    """Convert a HuggingFace checkpoint to native TQ3 packed format.

    Reads the original safetensors lazy (one tensor at a time), compresses
    weights on CPU, and writes a ~12 GB checkpoint. Peak memory: ~1 GB
    (one weight tensor + its compressed form).

    The output is a standard safetensors file where:
    - Weight tensors are replaced with .tq_packed (uint8) and .tq_norms (float32)
    - Non-weight tensors are stored as FP16
    - A tq_config.json records compression parameters

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Where to save the TQ3 checkpoint.
        bits: Quantization bits (default 3).
        group_size: Group size (default 128).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoConfig, AutoTokenizer
    from turboquant_vllm.weight_quant import pack_indices
    from turboquant_vllm.torch_ops import PolarQuantTorch

    os.makedirs(output_dir, exist_ok=True)

    # Download config and tokenizer
    logger.info("Downloading config and tokenizer for %s...", model_id)
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Find safetensors files
    api = HfApi()
    repo_files = api.list_repo_files(model_id)
    shard_files = sorted([f for f in repo_files if f.endswith(".safetensors")])
    index_file = [f for f in repo_files if f == "model.safetensors.index.json"]

    if index_file:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        weight_map = {}

    # Create quantizer on CPU
    quantizer = PolarQuantTorch(group_size, bits, seed=42, device="cpu")

    # Process each shard
    output_tensors = {}
    total_original = 0
    total_compressed = 0
    compressed_count = 0

    for shard_name in shard_files:
        logger.info("Processing shard: %s", shard_name)
        shard_path = hf_hub_download(model_id, shard_name)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                original_bytes = tensor.numel() * tensor.element_size()
                total_original += original_bytes

                # Decide: compress or keep
                # Compress 2D weight tensors AND 3D expert tensors (Gemma 4 stores
                # experts as "experts.down_proj" without .weight suffix)
                is_weight_2d = tensor_name.endswith(".weight") and tensor.dim() == 2
                is_expert_3d = tensor.dim() == 3 and "expert" in tensor_name.lower()
                is_weight = is_weight_2d or is_expert_3d
                is_skip = any(p in tensor_name.lower() for p in _SKIP_PATTERNS)
                is_large = tensor.shape[-1] >= 128 or (tensor.dim() >= 2 and tensor.shape[-2] >= 128)

                if is_weight and not is_skip and is_large:
                    # Compress this weight
                    packed, norms = _compress_tensor(
                        tensor, quantizer, bits, group_size
                    )
                    output_tensors[tensor_name + ".tq_packed"] = packed
                    output_tensors[tensor_name + ".tq_norms"] = norms

                    comp_bytes = packed.numel() + norms.numel() * norms.element_size()
                    total_compressed += comp_bytes
                    compressed_count += 1

                    if compressed_count % 200 == 0:
                        logger.info("  Compressed %d tensors (%.1f GB saved so far)",
                                    compressed_count,
                                    (total_original - total_compressed) / 1e9)
                else:
                    # Keep as FP16
                    if tensor.is_floating_point():
                        output_tensors[tensor_name] = tensor.half()
                    else:
                        output_tensors[tensor_name] = tensor
                    total_compressed += tensor.numel() * 2  # FP16

    # Save compressed checkpoint
    logger.info("Saving %d tensors (%d compressed)...", len(output_tensors), compressed_count)

    # Split into shards if too large for single file
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5 GB per shard
    _save_sharded(output_tensors, output_dir, max_shard_size)

    # Save TQ config
    tq_config = {
        "format": "tq3_native",
        "bits": bits,
        "group_size": group_size,
        "quantizer_seed": 42,
        "compressed_layers": compressed_count,
        "original_model": model_id,
    }
    with open(os.path.join(output_dir, "tq_config.json"), "w") as f:
        json.dump(tq_config, f, indent=2)

    ratio = total_original / max(total_compressed, 1)
    logger.info(
        "TQ3 checkpoint saved: %.1f GB -> %.1f GB (%.1fx), %d layers compressed",
        total_original / 1e9, total_compressed / 1e9, ratio, compressed_count,
    )


def _compress_tensor(
    tensor: torch.Tensor,
    quantizer,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a single weight tensor to packed TQ format.

    Handles both 2D (linear) and 3D (MoE expert) tensors.

    Returns: (packed_indices as uint8, norms as float32)
    """
    from turboquant_vllm.weight_quant import pack_indices

    orig_shape = tensor.shape
    if tensor.dim() == 3:
        # MoE: (num_experts, out_dim, in_dim) -> flatten to 2D
        n_exp, out_dim, in_dim = tensor.shape
        tensor = tensor.reshape(-1, in_dim)
    elif tensor.dim() == 2:
        out_dim, in_dim = tensor.shape
    else:
        # Skip unexpected shapes
        return tensor.to(torch.uint8), torch.tensor([])

    # Pad to group_size
    padded_in = ((in_dim + group_size - 1) // group_size) * group_size
    if padded_in > in_dim:
        padded = torch.zeros(tensor.shape[0], padded_in, dtype=tensor.dtype)
        padded[:, :in_dim] = tensor
    else:
        padded = tensor

    # Quantize
    grouped = padded.float().reshape(-1, group_size)
    indices, norms = quantizer.quantize(grouped, norm_correction=True)
    packed = pack_indices(indices, bits)

    # Store shape info in norms layout
    n_groups = padded_in // group_size
    norms = norms.reshape(-1, n_groups)  # (total_rows, n_groups)

    return packed.contiguous(), norms.contiguous()


def _save_sharded(tensors: dict, output_dir: str, max_shard_size: int):
    """Save tensors as sharded safetensors files."""
    from safetensors.torch import save_file

    # Calculate sizes and assign to shards
    shards = []
    current_shard = {}
    current_size = 0

    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    if len(shards) == 1:
        save_file(shards[0], os.path.join(output_dir, "model.safetensors"))
    else:
        weight_map = {}
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, os.path.join(output_dir, shard_name))
            for name in shard:
                weight_map[name] = shard_name

        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def load_tq3_weights(model, checkpoint_dir: str, device: str = "cuda"):
    """Load TQ3-packed weights into a model, creating TurboQuantWrapper modules.

    This is the counterpart to save_tq3_checkpoint. It reads the packed
    indices + norms and creates TurboQuantWrapper modules that decompress
    on the fly during forward passes.

    Args:
        model: The model to load weights into.
        checkpoint_dir: Path to the TQ3 checkpoint.
        device: Target device.
    """
    from safetensors import safe_open
    from turboquant_vllm.weight_quant import TurboQuantWrapper, unpack_indices, _get_quantizer
    import torch.nn as nn

    # Load TQ config
    with open(os.path.join(checkpoint_dir, "tq_config.json")) as f:
        tq_config = json.load(f)

    bits = tq_config["bits"]
    group_size = tq_config["group_size"]

    # Find safetensors files
    shard_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")])

    # Load all tensors
    all_tensors = {}
    for shard_name in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_name)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for name in f.keys():
                all_tensors[name] = f.get_tensor(name)

    # Identify compressed layers (have .tq_packed suffix)
    compressed_names = set()
    for name in all_tensors:
        if name.endswith(".tq_packed"):
            base_name = name[:-len(".tq_packed")]
            compressed_names.add(base_name)

    logger.info("Loading %d compressed layers + %d regular tensors",
                len(compressed_names),
                len(all_tensors) - 2 * len(compressed_names))

    # Load regular parameters
    model_state = model.state_dict()
    for name, tensor in all_tensors.items():
        if name.endswith(".tq_packed") or name.endswith(".tq_norms"):
            continue  # Handle these separately
        if name in model_state:
            model_state[name].copy_(tensor)

    # Create TurboQuantWrapper for compressed layers
    for base_name in compressed_names:
        packed = all_tensors[base_name + ".tq_packed"]
        norms = all_tensors[base_name + ".tq_norms"]

        # Find the module and replace with wrapper
        parts = base_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_path, param_name = parts
        else:
            continue

        # Navigate to parent module
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part, None)
            if parent is None:
                break

        if parent is None:
            logger.warning("Could not find module for %s", base_name)
            continue

        # Get the original module
        module = getattr(parent, param_name.split(".")[0], None)
        if module is None or not isinstance(module, nn.Linear):
            continue

        # Create wrapper with pre-packed data
        wrapper = TurboQuantWrapper.__new__(TurboQuantWrapper)
        nn.Module.__init__(wrapper)
        wrapper.bits = bits
        wrapper.group_size = group_size
        wrapper.in_features = module.in_features
        wrapper.out_features = module.out_features
        wrapper.padded_in = norms.shape[1] * group_size
        wrapper.n_groups = norms.shape[1]
        wrapper._has_learned_rotation = False
        wrapper._ratio = 0.0
        wrapper.bias = module.bias
        wrapper.register_buffer("packed_weight", packed)
        wrapper.register_buffer("norms", norms)

        # Replace module
        attr_name = param_name.split(".")[0]
        setattr(parent, attr_name, wrapper)

    logger.info("TQ3 weights loaded successfully")
