#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATA_PATH [extra train.py args...]"
  exit 1
fi

DATA_PATH=$1
shift
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/src/train.py" \
  --data-path "${DATA_PATH}" \
  --model "${MODEL:-wfcat}" \
  --feature-type "${FEATURE_TYPE:-iat}" \
  --mon-classes "${MON_CLASSES:-4}" \
  --mon-inst "${MON_INST:-100}" \
  --page-per-class "${PAGE_PER_CLASS:-1}" \
  --seq-length "${SEQ_LENGTH:-1800}" \
  --iat-bins "${IAT_BINS:-9}" \
  --time-window "${TIME_WINDOW:-0.044}" \
  --num-kernels "${NUM_KERNELS:-4}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr0 "${LR0:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0005}" \
  --workers "${WORKERS:-6}" \
  --model-path "${MODEL_PATH:-./checkpoints}" \
  --verbose \
  "$@"
