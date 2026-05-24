#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a model directory, e.g. runs/colab_l4/best}"
OUTPUT="${OUTPUT:-benchmark.json}"
python main.py evaluate --checkpoint "$CHECKPOINT" --output "$OUTPUT"
