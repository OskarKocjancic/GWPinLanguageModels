#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
BASE_OUT_DIR="${BASE_OUT_DIR:-$ROOT_DIR/out}"

run_scenario() {
  local name="$1"
  shift

  echo "Running scenario: ${name}"
  "${PYTHON_BIN}" "$ROOT_DIR/src/train.py" \
    --data-dir "$DATA_DIR" \
    --out-dir "$BASE_OUT_DIR/${name}" \
    "$@"
}

run_scenario baseline \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --max-iters 2000

run_scenario smaller_model \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --max-iters 2000

run_scenario shorter_training \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --max-iters 500
