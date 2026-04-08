#!/bin/bash
# Self-contained test: create TQ3 checkpoint from GLM-4.7-Flash and verify
# Run with nohup on a GPU server. Logs to /root/glm47_test.log
set -e

LOG=/root/glm47_test.log
exec > >(tee -a "$LOG") 2>&1

echo "=== GLM-4.7-Flash TQ3 Checkpoint Test ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "Disk: $(df -h / | tail -1)"
echo ""

# Setup
source /root/env/bin/activate
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"

# Phase 1: Create TQ3 checkpoint (streaming)
echo "=== Phase 1: Creating TQ3 checkpoint ==="
echo "Peak RAM will be ~1 tensor (~6 MB for this model)"
echo ""

python3 -u -c "
import logging, time, os, psutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

process = psutil.Process()

t0 = time.time()
from turboquant_vllm.checkpoint import save_tq3_checkpoint

ram_before = process.memory_info().rss / 1e9
save_tq3_checkpoint('zai-org/GLM-4.7-Flash', '/root/glm47-tq3', bits=3, group_size=128)
ram_after = process.memory_info().rss / 1e9

elapsed = time.time() - t0
print(f'Checkpoint created in {elapsed:.0f}s')
print(f'RAM: {ram_before:.1f} GB before -> {ram_after:.1f} GB after (peak ~{ram_after:.1f} GB)')

# Check output
total = 0
for f in sorted(os.listdir('/root/glm47-tq3')):
    size = os.path.getsize(os.path.join('/root/glm47-tq3', f))
    total += size
    print(f'  {f}: {size/1e9:.2f} GB')
print(f'Total output: {total/1e9:.1f} GB')
"

echo ""
echo "=== Phase 2: Load and verify quality ==="

python3 -u -c "
import logging, time, torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

t0 = time.time()
from turboquant_vllm.checkpoint import load_tq3_model
model, tokenizer = load_tq3_model('/root/glm47-tq3', device='cuda')
print(f'Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB')

prompts = [
    'What is 2+2? Answer with just the number.',
    'What is the capital of Finland?',
    'Explain gravity in one sentence.',
]
for p in prompts:
    chat = [{'role': 'user', 'content': p}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=50, do_sample=False)
    resp = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'Q: {p}')
    print(f'A: {resp}')
    print()

print(f'Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
print('QUALITY CHECK DONE')
"

echo ""
echo "=== Extrapolation for GLM-5.1 ==="
python3 -c "
import os, json

# Measure actual compression ratio
total_out = sum(os.path.getsize(os.path.join('/root/glm47-tq3', f))
    for f in os.listdir('/root/glm47-tq3') if f.endswith('.safetensors'))
total_in = 31.2e9  # known BF16 size
ratio = total_in / total_out

print(f'GLM-4.7-Flash: {total_in/1e9:.1f} GB -> {total_out/1e9:.1f} GB ({ratio:.1f}x)')
print()

glm51_bf16 = 1508e9
glm51_tq3 = glm51_bf16 / ratio
print(f'GLM-5.1 extrapolation:')
print(f'  BF16: {glm51_bf16/1e9:.0f} GB')
print(f'  TQ3:  {glm51_tq3/1e9:.0f} GB ({ratio:.1f}x)')
print(f'  Disk needed: ~{(glm51_bf16 + glm51_tq3)/1e9:.0f} GB (original + compressed)')
"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"
