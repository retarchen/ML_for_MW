#!/usr/bin/env bash
set -euo pipefail

# Detached/no-stop launcher for scripts/train_hi_cnn.py.
# Edit the variables below, or override them from the command line:
#   RUN_NAME=my_run FCNM_ERROR_FLOOR=0.02 bash scripts/run_train_hi_cnn.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-cnn}"

RUN_NAME="${RUN_NAME:-cnn_run_$(date +%Y%m%d_%H%M%S)}"
SUBSET_SIZE="${SUBSET_SIZE:--1}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
TB_COLUMN="${TB_COLUMN:-3}"
NUM_LAYERS="${NUM_LAYERS:-8}"
RHI_TARGET_TRANSFORM="${RHI_TARGET_TRANSFORM:-log}"
FCNM_ERROR_FLOOR="${FCNM_ERROR_FLOOR:-0}"
FCNM_ZERO_LOSS_WEIGHT="${FCNM_ZERO_LOSS_WEIGHT:-3.0}"

LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

setsid bash -lc "
source '${CONDA_SH}'
conda activate '${ENV_NAME}'
export MPLCONFIGDIR=/tmp
cd '${PROJECT_DIR}'
python -u scripts/train_hi_cnn.py \
  --subset-size '${SUBSET_SIZE}' \
  --epochs '${EPOCHS}' \
  --patience '${PATIENCE}' \
  --batch-size '${BATCH_SIZE}' \
  --device '${DEVICE}' \
  --tb-column '${TB_COLUMN}' \
  --num-layers '${NUM_LAYERS}' \
  --rhi-target-transform '${RHI_TARGET_TRANSFORM}' \
  --fcnm-error-floor '${FCNM_ERROR_FLOOR}' \
  --fcnm-zero-loss-weight '${FCNM_ZERO_LOSS_WEIGHT}' \
  --run-name '${RUN_NAME}'
" > "${LOG_FILE}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "Started train_hi_cnn.py"
echo "  PID: ${pid}"
echo "  run name: ${RUN_NAME}"
echo "  log: ${LOG_FILE}"
echo "  pid file: ${PID_FILE}"
echo "  results: ${PROJECT_DIR}/results/${RUN_NAME}/"
echo "  figs: ${PROJECT_DIR}/figs/${RUN_NAME}/"
echo
echo "Monitor with:"
echo "  tail -f '${LOG_FILE}'"
