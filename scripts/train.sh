#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -a
if [ -f .env ]; then
  source .env
fi
set +a
python -u main.py train --run-name "${RUN_NAME:-}"
