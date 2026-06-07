#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATA_PATH [--open-world] [extra train.py args...]"
  exit 1
fi

DATA_PATH=$1
shift
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

run_one() {
  local model=$1
  local feature=$2
  local seq_length=$3
  local batch_size=$4
  local lr0=$5
  local decay=$6
  shift 6
  python "${PROJECT_ROOT}/src/train.py" \
    --data-path "${DATA_PATH}" \
    --model "${model}" \
    --feature-type "${feature}" \
    --seq-length "${seq_length}" \
    --batch-size "${batch_size}" \
    --lr0 "${lr0}" \
    --weight-decay "${decay}" \
    --mon-classes "${MON_CLASSES:-100}" \
    --mon-inst "${MON_INST:-100}" \
    --unmon-inst "${UNMON_INST:-10000}" \
    --page-per-class "${PAGE_PER_CLASS:-1}" \
    --epochs "${EPOCHS:-50}" \
    --workers "${WORKERS:-6}" \
    --model-path "${MODEL_PATH:-./checkpoints}" \
    --verbose \
    "$@"
}

run_one wfcat iat 1800 64 0.001 0.0005 "$@"
run_one rf tam 1800 200 0.0005 0.001 "$@"
run_one df direction 10000 64 0.002 0 "$@"
run_one df directional_timing 10000 64 0.002 0 "$@"
run_one varcnn directional_timing 10000 64 0.002 0.0005 "$@"
run_one ares direction 10000 256 0.002 0.001 "$@"
run_one tmwf direction 30720 256 0.002 0.0005 "$@"
