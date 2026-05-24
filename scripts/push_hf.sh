#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to runs/<run>/best}"
REPO_ID="${HF_REPO_ID:-}"
python main.py push --checkpoint "$CHECKPOINT" ${REPO_ID:+--repo-id "$REPO_ID"}
