#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [OmegaConf overrides...]" >&2
  exit 2
fi
CONFIG=$1

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
train.py \
  config="$CONFIG" \
  "${@:2}"
