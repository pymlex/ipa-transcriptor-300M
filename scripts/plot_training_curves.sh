#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/plot_training_curves.py
