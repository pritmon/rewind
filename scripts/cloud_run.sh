#!/usr/bin/env bash
# Run on a fresh cloud GPU box (PyTorch image, CUDA available), from the repo root.
#   bash scripts/cloud_run.sh bench     # 200-step timing run (~5 min): check speed and memory first
#   bash scripts/cloud_run.sh train     # full run on all visible GPUs
#   bash scripts/cloud_run.sh eval      # rediscovery test on the trained model
set -euo pipefail
GPUS=$(nvidia-smi -L | wc -l)
mkdir -p logs
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen

pip install -q -e ".[dev]"
[ -f data/primekg.pt ] || bash scripts/download_primekg.sh

case "${1:-}" in
  bench)
    sed 's/^  steps: .*/  steps: 200/; s/^  log_every: .*/  log_every: 50/; s|checkpoints/gpu.pt|checkpoints/bench.pt|' \
      configs/gpu.yaml > configs/bench.yaml
    torchrun --nproc_per_node="$GPUS" -m rewind.train --config configs/bench.yaml 2>&1 | tee "logs/bench_${GPUS}gpu.log"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
    ;;
  train)
    torchrun --nproc_per_node="$GPUS" -m rewind.train --config configs/gpu.yaml 2>&1 | tee "logs/gpu_train_${GPUS}gpu.log"
    ;;
  eval)
    python -u -m rewind.eval.rediscovery --checkpoint checkpoints/gpu.pt 2>&1 | tee logs/gpu_eval.log
    ;;
  *)
    echo "usage: $0 bench|train|eval" >&2; exit 1
    ;;
esac
