#!/usr/bin/env bash
# Local launcher: rsync test files to Verda instance, start the GPU test
# in background, return PID + log path immediately.
#
# Principles:
#   - NEVER inline multi-line scripts via ssh heredoc — rsync and execute.
#   - ALWAYS nohup on remote + disown so local ssh can disconnect safely.
#   - ALWAYS return a "how to poll" command so the user can check progress.
#
# Usage:
#   ./tests/gpu/run_on_verda.sh <ip>             # default: Qwen2.5-0.5B + tq3
#   ./tests/gpu/run_on_verda.sh <ip> <model>     # override model
#   MODEL=... KV_DTYPE=tq4 ./tests/gpu/run_on_verda.sh <ip>
#
# After launch, poll with:
#   ssh -i ~/.ssh/id_ed25519_varjosoft_hez root@<ip> 'tail -n30 /tmp/tq-test.log'
#   ssh -i ~/.ssh/id_ed25519_varjosoft_hez root@<ip> 'cat /tmp/tq-test-result.txt'

set -euo pipefail

IP="${1:?Usage: $0 <verda-ip> [model]}"
MODEL_ARG="${2:-${MODEL:-Qwen/Qwen2.5-0.5B}}"
KV_DTYPE="${KV_DTYPE:-tq3}"

SSH_KEY="$HOME/.ssh/id_ed25519_varjosoft_hez"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
RSYNC="rsync -az --exclude .venv --exclude .git --exclude __pycache__ -e 'ssh -i $SSH_KEY -o StrictHostKeyChecking=no'"

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DIR="\$HOME/turboquant-vllm"

echo "=== Remote GPU test launcher ==="
echo "instance: $IP"
echo "model:    $MODEL_ARG"
echo "kv dtype: $KV_DTYPE"
echo

# Check reachability first
if ! $SSH root@"$IP" 'echo ok' >/dev/null 2>&1; then
    echo "ERROR: cannot ssh root@$IP"
    exit 1
fi

echo "[1/3] rsync source tree to remote"
eval "$RSYNC" "$HERE/" "root@$IP:~/turboquant-vllm/"

echo "[2/3] verify test script is syntactically valid on remote"
$SSH root@"$IP" "bash -n ~/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh" \
    || { echo "FAIL: remote script has syntax errors"; exit 1; }

echo "[3/3] launch test in background (nohup + disown)"
# Pass env vars via a wrapper: write them to a file on remote, source it.
# This sidesteps all shell-quoting issues with multi-word values.
$SSH root@"$IP" "cat > /tmp/tq-test.env <<ENVEOF
export MODEL='$MODEL_ARG'
export KV_DTYPE='$KV_DTYPE'
ENVEOF
chmod +x ~/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh
nohup bash -c 'source /tmp/tq-test.env && ~/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh' > /tmp/tq-launcher.log 2>&1 &
disown
echo \$! > /tmp/tq-launcher.pid
sleep 0.5
echo 'launcher PID='\$(cat /tmp/tq-launcher.pid)"

echo
echo "=== launched ==="
echo "Poll log:     ssh -i $SSH_KEY root@$IP 'tail -n40 /tmp/tq-test.log'"
echo "Poll server:  ssh -i $SSH_KEY root@$IP 'tail -n40 /tmp/tq-server.log'"
echo "Final result: ssh -i $SSH_KEY root@$IP 'cat /tmp/tq-test-result.txt'"
echo "Is running:   ssh -i $SSH_KEY root@$IP 'pgrep -af tq-test || echo \"not running\"'"
