#!/bin/bash
# Self-contained test: TQ3 + DFlash on Qwen3-8B
# Run with nohup on a GPU server. Logs to /root/dflash_tq3_test.log
set -e

LOG=/root/dflash_tq3_test.log
> "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo "  DFlash + TQ3 Benchmark: Qwen3-8B"
echo "============================================="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# Setup
apt-get update -qq
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-dev python3.12-venv git > /dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"

python3.12 -m venv /root/env
source /root/env/bin/activate

pip install -q --upgrade pip
pip install -q torch 'transformers>=4.57' scipy datasets
pip install -q 'turboquant-plus-vllm @ git+https://github.com/varjoranta/turboquant-vllm.git'
pip install -q 'dflash @ git+https://github.com/z-lab/dflash.git'
pip uninstall -y turboquant-vllm 2>/dev/null || true

echo ""
echo "Versions:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import dflash; print('DFlash: OK')"
python3 -c "import turboquant_vllm; print('TurboQuant: OK')"

echo ""
echo "=== Running benchmark ==="
python3 -u /root/turboquant-vllm/scripts/benchmark_dflash.py \
    --target Qwen/Qwen3-8B \
    --draft z-lab/Qwen3-8B-DFlash-b16 \
    --dataset simple \
    --max-samples 5 \
    --max-new-tokens 128

echo ""
echo "=== GPU Stats ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "============================================="
echo "  BENCHMARK COMPLETE"
echo "============================================="
echo "Finished: $(date)"
