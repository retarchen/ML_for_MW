#!/usr/bin/env bash
set -euo pipefail

# Detached/no-stop launcher for scripts/train_HI_cnn_GASKAP_LMC.py.
# This trains on the simulation spectra, then applies the trained model to
# the matched GASKAP LMC observation spectra as an external prediction set.
# Override settings like:
#   RUN_NAME=sim_to_GASKAP_LMC EPOCHS=100 DEVICE=cuda bash scripts/run_train_HI_cnn_GASKAP_LMC.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-cnn}"

RUN_NAME="${RUN_NAME:-sim_to_GASKAP_LMC_$(date +%Y%m%d_%H%M%S)}"
SUBSET_SIZE="${SUBSET_SIZE:--1}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
TB_COLUMN="${TB_COLUMN:-3}"
INPUT_MODE="${INPUT_MODE:-raw}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-9}"
GRID_MODE="${GRID_MODE:-overlap}"
GRID_SIZE="${GRID_SIZE:-0}"
RHI_TARGET_TRANSFORM="${RHI_TARGET_TRANSFORM:-log}"
RHI_TAIL_LOSS_WEIGHT="${RHI_TAIL_LOSS_WEIGHT:-0}"
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
python -u scripts/train_HI_cnn_GASKAP_LMC.py \
  --subset-size '${SUBSET_SIZE}' \
  --epochs '${EPOCHS}' \
  --patience '${PATIENCE}' \
  --batch-size '${BATCH_SIZE}' \
  --device '${DEVICE}' \
  --tb-column '${TB_COLUMN}' \
  --input-mode '${INPUT_MODE}' \
  --smooth-window '${SMOOTH_WINDOW}' \
  --grid-mode '${GRID_MODE}' \
  --grid-size '${GRID_SIZE}' \
  --rhi-target-transform '${RHI_TARGET_TRANSFORM}' \
  --rhi-tail-loss-weight '${RHI_TAIL_LOSS_WEIGHT}' \
  --fcnm-error-floor '${FCNM_ERROR_FLOOR}' \
  --fcnm-zero-loss-weight '${FCNM_ZERO_LOSS_WEIGHT}' \
  --run-name '${RUN_NAME}'
" > "${LOG_FILE}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "Started train_HI_cnn_GASKAP_LMC.py"
echo "  PID: ${pid}"
echo "  run name: ${RUN_NAME}"
echo "  log: ${LOG_FILE}"
echo "  pid file: ${PID_FILE}"
echo "  results: ${PROJECT_DIR}/results/${RUN_NAME}/"
echo "  figs: ${PROJECT_DIR}/figs/${RUN_NAME}/"
echo
echo "Monitor with:"
echo "  tail -f '${LOG_FILE}'"
