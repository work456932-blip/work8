#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {anet|tacos|mad} [seed ...]" >&2
  exit 2
fi

DATASET="$1"
shift
if [[ $# -eq 0 ]]; then
  SEEDS=(2024 2025 2026)
else
  SEEDS=("$@")
fi

for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash scripts/train.sh \
    "configs/${DATASET}.yaml" seed="$SEED" name="${DATASET}_seed${SEED}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash scripts/eval.sh \
    "configs/${DATASET}.yaml" "results/${DATASET}_seed${SEED}/models/last.pth" \
    seed="$SEED" name="${DATASET}_seed${SEED}"
done

METRICS=()
for SEED in "${SEEDS[@]}"; do
  METRICS+=("results/${DATASET}_seed${SEED}/metrics.json")
done
python tools/aggregate_results.py "${METRICS[@]}" \
  --output "results/${DATASET}_three_seed_summary.json"
