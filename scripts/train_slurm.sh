#!/usr/bin/env bash
#SBATCH --job-name=wfcat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/wfcat_%j.out
#SBATCH --error=logs/wfcat_%j.err

set -euo pipefail

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "Set DATA_PATH before sbatch, e.g.: DATA_PATH=/path/to/data sbatch scripts/train_slurm.sh"
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "${PROJECT_ROOT}"
mkdir -p logs checkpoints
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/src/train.py" \
  --data-path "${DATA_PATH}" \
  --model "${MODEL:-wfcat}" \
  --feature-type "${FEATURE_TYPE:-iat}" \
  --mon-classes "${MON_CLASSES:-100}" \
  --mon-inst "${MON_INST:-100}" \
  --unmon-inst "${UNMON_INST:-10000}" \
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
  ${OPEN_WORLD:+--open-world} \
  ${ONE_FOLD:+--one-fold} \
  ${NOSAVE:+--nosave} \
  --verbose
