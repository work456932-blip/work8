#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT [OmegaConf overrides...]" >&2
  exit 2
fi
CONFIG=$1
CKPT=$2

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
evaluate.py \
  config="$CONFIG" \
  eval.ckpt="$CKPT" \
  "${@:3}"
