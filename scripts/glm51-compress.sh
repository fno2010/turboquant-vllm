#!/usr/bin/env bash
# Step 1: Create native TQ3 checkpoint for GLM-5.1 (754B)
#
# Runs on a CPU instance (no GPU needed). Streams shards from HuggingFace,
# compresses one tensor at a time, deletes input shards after processing.
#
# Resources: ~8 GB RAM peak, ~200 GB disk (output + temp)
# Time: ~2-4 hours (download-bound on 282 shards)
#
# Usage: run on a Verda CPU instance after setup.
#   bash /root/turboquant-vllm/scripts/glm51-compress.sh

set -euo pipefail
export PATH="/root/.local/bin:$PATH"

MODEL_ID="${MODEL_ID:-zai-org/GLM-5.1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/glm51-tq3-native}"
BITS="${BITS:-3}"
GROUP_SIZE="${GROUP_SIZE:-128}"
LOG="/tmp/glm51-compress.log"
RESULT="/tmp/glm51-compress-result.txt"

rm -f "$LOG" "$RESULT"
exec > >(tee -a "$LOG") 2>&1

echo "=== GLM-5.1 TQ3 Checkpoint Creation ==="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Bits: $BITS, Group size: $GROUP_SIZE"
echo "Date: $(date -u)"
echo "Disk free: $(df -h /root --output=avail | tail -1)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Pre-flight: verify imports
echo "Phase 1: Import check..."
python3 -c "
from turboquant_vllm.checkpoint import save_tq3_checkpoint
from turboquant_vllm.torch_ops import PolarQuantTorch
print('Imports OK')
" || { echo "FAIL: import check" > "$RESULT"; exit 1; }

# Pre-flight: verify HuggingFace access
echo "Phase 2: HuggingFace access check..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('$MODEL_ID')
safetensors = [f for f in files if f.endswith('.safetensors')]
print(f'Found {len(safetensors)} safetensor shards')
assert len(safetensors) > 200, f'Expected 282 shards, got {len(safetensors)}'
print('HuggingFace access OK')
" || { echo "FAIL: HuggingFace access" > "$RESULT"; exit 1; }

# Compress
echo ""
echo "Phase 3: Compressing (this takes 2-4 hours)..."
echo "Start: $(date -u)"

python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from turboquant_vllm.checkpoint import save_tq3_checkpoint
save_tq3_checkpoint(
    model_id='$MODEL_ID',
    output_dir='$OUTPUT_DIR',
    bits=$BITS,
    group_size=$GROUP_SIZE,
)
print('Compression complete!')
" || { echo "FAIL: compression" > "$RESULT"; exit 1; }

echo ""
echo "End: $(date -u)"
echo "Disk used by output: $(du -sh $OUTPUT_DIR | cut -f1)"
echo ""

# Verify output
echo "Phase 4: Verification..."
python3 -c "
import os, json
output_dir = '$OUTPUT_DIR'

# Check config
config = json.load(open(os.path.join(output_dir, 'config.json')))
print(f'Model type: {config.get(\"model_type\", \"unknown\")}')
print(f'Quant config: {config.get(\"quantization_config\", {})}')

# Check TQ config
tq = json.load(open(os.path.join(output_dir, 'tq_config.json')))
print(f'TQ format: {tq[\"format\"]}')
print(f'Compressed layers: {tq[\"compressed_layers\"]}')
print(f'Bits: {tq[\"bits\"]}, Group size: {tq[\"group_size\"]}')

# Count output files
safetensors = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
print(f'Output shards: {len(safetensors)}')

# Check index
idx_path = os.path.join(output_dir, 'model.safetensors.index.json')
if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
    total_gb = idx['metadata']['total_size'] / 1e9
    print(f'Total checkpoint size: {total_gb:.1f} GB')
    tq_packed = sum(1 for k in idx['weight_map'] if '.tq_packed' in k)
    tq_norms = sum(1 for k in idx['weight_map'] if '.tq_norms' in k)
    print(f'TQ tensors: {tq_packed} packed + {tq_norms} norms')
" || echo "WARNING: verification failed (non-fatal)"

echo ""
echo "PASS: GLM-5.1 TQ3 checkpoint created at $OUTPUT_DIR" > "$RESULT"
cat "$RESULT"
echo ""
echo "Next: upload with publish_model.py upload $OUTPUT_DIR varjosoft/GLM-5.1-TQ3-native"
