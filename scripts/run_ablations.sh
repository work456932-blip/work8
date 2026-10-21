#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-anet}"
SEED="${2:-2024}"
# Supplementary Table S5: no-FP, fixed diagnostic seed.
for VARIANT in survvtg hem direct_mt hazard_only no_cure no_state no_monotonic no_prefix no_hard_negative; do
  RUN_NAME="${DATASET}_${VARIANT}_seed${SEED}"
  EXTRA=(variant="$VARIANT" model.future_prediction=false train.future_loss_weight=0.0)
  bash scripts/train.sh "configs/${DATASET}.yaml" seed="$SEED" name="$RUN_NAME" "${EXTRA[@]}"
  bash scripts/eval.sh "configs/${DATASET}.yaml" "results/${RUN_NAME}/models/last.pth" \
    seed="$SEED" name="$RUN_NAME" "${EXTRA[@]}"
done
# The no-gate row REUSES the full checkpoint; no second training run.
FULL="${DATASET}_survvtg_seed${SEED}"
bash scripts/eval.sh "configs/${DATASET}.yaml" "results/${FULL}/models/last.pth" \
  seed="$SEED" name="${FULL}_no_gate" model.future_prediction=false \
  train.future_loss_weight=0.0 eval.cure_gate=false eval.explicit_refusal=false
