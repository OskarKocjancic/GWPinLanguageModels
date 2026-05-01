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
    "$@"
}



# Breakpoint 1: ~400k params (0.5x baseline)
run_scenario bp01_baseline \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 256 \
  --name bp01_baseline

run_scenario bp01_more_heads \
  --n-layer 2 \
  --n-head 8 \
  --n-embd 128 \
  --block-size 256 \
  --name bp01_more_heads

run_scenario bp01_larger_block_size \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 512 \
  --name bp01_larger_block_size

run_scenario bp01_larger_embeddings \
  --n-layer 1 \
  --n-head 4 \
  --n-embd 176 \
  --block-size 256 \
  --name bp01_larger_embeddings

# Breakpoint 2: ~595k params (0.75x baseline)
run_scenario bp02_baseline \
  --n-layer 3 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 256 \
  --name bp02_baseline

run_scenario bp02_more_heads \
  --n-layer 3 \
  --n-head 8 \
  --n-embd 128 \
  --block-size 256 \
  --name bp02_more_heads

run_scenario bp02_larger_block_size \
  --n-layer 3 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 512 \
  --name bp02_larger_block_size

run_scenario bp02_larger_embeddings \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 176 \
  --block-size 256 \
  --name bp02_larger_embeddings

# Breakpoint 3: ~793k params (baseline)
run_scenario bp03_baseline \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 256 \
  --name bp03_baseline

run_scenario bp03_more_heads \
  --n-layer 4 \
  --n-head 8 \
  --n-embd 128 \
  --block-size 256 \
  --name bp03_more_heads

run_scenario bp03_larger_block_size \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 512 \
  --name bp03_larger_block_size

run_scenario bp03_larger_embeddings \
  --n-layer 1 \
  --n-head 4 \
  --n-embd 256 \
  --block-size 256 \
  --name bp03_larger_embeddings

# Breakpoint 4: ~1.19M params (1.5x baseline)
run_scenario bp04_baseline \
  --n-layer 6 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 256 \
  --name bp04_baseline

run_scenario bp04_more_heads \
  --n-layer 6 \
  --n-head 8 \
  --n-embd 128 \
  --block-size 256 \
  --name bp04_more_heads

run_scenario bp04_larger_block_size \
  --n-layer 6 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 512 \
  --name bp04_larger_block_size

run_scenario bp04_larger_embeddings \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 224 \
  --block-size 256 \
  --name bp04_larger_embeddings

# Breakpoint 5: ~1.59M params (2x baseline)
run_scenario bp05_baseline \
  --n-layer 8 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 256 \
  --name bp05_baseline

run_scenario bp05_more_heads \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 128 \
  --block-size 256 \
  --name bp05_more_heads

run_scenario bp05_larger_block_size \
  --n-layer 8 \
  --n-head 4 \
  --n-embd 128 \
  --block-size 512 \
  --name bp05_larger_block_size

run_scenario bp05_larger_embeddings \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 256 \
  --block-size 256 \
  --name bp05_larger_embeddings

