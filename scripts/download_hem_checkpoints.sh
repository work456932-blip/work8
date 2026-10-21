#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/checkpoints/hem}"
mkdir -p "$OUTPUT_DIR"
gdown --folder \
  "https://drive.google.com/drive/folders/1XZ9VyZRJ0IQ6nWnULaPf7jqcPHtae_kw" \
  -O "$OUTPUT_DIR"
echo "HEM checkpoints downloaded under $OUTPUT_DIR"
