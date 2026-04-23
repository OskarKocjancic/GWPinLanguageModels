#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_OUT_DIR="${BASE_OUT_DIR:-$ROOT_DIR/out}"
PROMPT_OUT_DIR="${PROMPT_OUT_DIR:-$BASE_OUT_DIR}"

run_prompt_scenario() {
  local name="$1"
  local ckpt_path="$2"
  shift 2

  mkdir -p "$PROMPT_OUT_DIR/$name"
  echo "Running prompt scenario: ${name}"
  "${PYTHON_BIN}" "$ROOT_DIR/src/prompt.py" \
    --out-dir "$PROMPT_OUT_DIR/$name" \
    --ckpt-path "$ckpt_path" \
    "$@" | tee "$PROMPT_OUT_DIR/$name/prompt_output.txt"
}

run_prompt_scenario baseline \
  "$BASE_OUT_DIR/baseline/ckpt.pt" \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --top-k 50

run_prompt_scenario smaller_model \
  "$BASE_OUT_DIR/smaller_model/ckpt.pt" \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --top-k 50

run_prompt_scenario shorter_training \
  "$BASE_OUT_DIR/shorter_training/ckpt.pt" \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --top-k 50
