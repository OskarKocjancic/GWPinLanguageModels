#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-$ROOT_DIR/out}"
PROMPT_OUTPUT_FILE="${PROMPT_OUTPUT_FILE:-$TRAIN_OUT_DIR/prompt_output.txt}"
PROMPT_TEXT="${PROMPT_TEXT:-To be, or not to be}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURES=(0.5 1.0)
TOP_KS=(25 50)

run_prompt_scenario() {
  local scenario_name="$1"
  local ckpt_path="$2"
  local temperature="$3"
  local top_k="$4"

  echo "Running prompt scenario: ${scenario_name} | temperature=${temperature} | top_k=${top_k}"
  {
    printf '\n=== checkpoint=%s temperature=%s top_k=%s ===\n' "$scenario_name" "$temperature" "$top_k"
    "${PYTHON_BIN}" "$ROOT_DIR/src/prompt.py" \
      --out-dir "$TRAIN_OUT_DIR" \
      --ckpt-path "$ckpt_path" \
      --prompt "$PROMPT_TEXT" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$temperature" \
      --top-k "$top_k" \
      --name "${scenario_name}_temp${temperature}_topk${top_k}"
  } >> "$PROMPT_OUTPUT_FILE"
}

mapfile -t CHECKPOINTS < <(find "$TRAIN_OUT_DIR" -maxdepth 1 -type f -name '*.pt' | sort)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoint files found under $TRAIN_OUT_DIR"
  exit 1
fi

mkdir -p "$TRAIN_OUT_DIR"
: > "$PROMPT_OUTPUT_FILE"

for ckpt_path in "${CHECKPOINTS[@]}"; do
  scenario_name="$(basename "${ckpt_path%.pt}")"
  for temperature in "${TEMPERATURES[@]}"; do
    for top_k in "${TOP_KS[@]}"; do
      run_prompt_scenario \
        "$scenario_name" \
        "$ckpt_path" \
        "$temperature" \
        "$top_k"
    done
  done
done
