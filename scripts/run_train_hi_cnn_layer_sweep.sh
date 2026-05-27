#!/usr/bin/env bash
set -euo pipefail

# Detached/no-stop sequential sweep for scripts/train_hi_cnn.py.
# Runs one GPU training job at a time for the requested CNN depths.
#
# Example:
#   LAYERS="2 3 4 5 6 7 8" SUBSET_SIZE=-1 DEVICE=cuda bash scripts/run_train_hi_cnn_layer_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-cnn}"

LAYERS="${LAYERS:-2 3 4 5 6 7 8}"
RUN_PREFIX="${RUN_PREFIX:-layer_sweep}"
SUBSET_SIZE="${SUBSET_SIZE:--1}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
TB_COLUMN="${TB_COLUMN:-3}"
RHI_TARGET_TRANSFORM="${RHI_TARGET_TRANSFORM:-log}"
FCNM_ERROR_FLOOR="${FCNM_ERROR_FLOOR:-0}"
FCNM_ZERO_LOSS_WEIGHT="${FCNM_ZERO_LOSS_WEIGHT:-3.0}"

LOG_DIR="${PROJECT_DIR}/logs"
LAYERS_TAG="$(echo "${LAYERS}" | tr ' ' '_' | tr -cd '[:alnum:]_')"
LOG_FILE="${LOG_DIR}/${RUN_PREFIX}_layers_${LAYERS_TAG}.log"
PID_FILE="${LOG_DIR}/${RUN_PREFIX}_layers_${LAYERS_TAG}.pid"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

nohup setsid bash -lc "
source '${CONDA_SH}'
conda activate '${ENV_NAME}'
export MPLCONFIGDIR=/tmp
cd '${PROJECT_DIR}'

echo 'Starting sequential CNN layer sweep'
echo 'Layers: ${LAYERS}'
echo 'Subset size: ${SUBSET_SIZE}'
echo 'Device: ${DEVICE}'

for layer in ${LAYERS}; do
  echo
  echo '============================================================'
  echo \"Starting layer count: \${layer}\"
  echo '============================================================'
  python -u scripts/train_hi_cnn.py \
    --subset-size '${SUBSET_SIZE}' \
    --epochs '${EPOCHS}' \
    --patience '${PATIENCE}' \
    --batch-size '${BATCH_SIZE}' \
    --device '${DEVICE}' \
    --tb-column '${TB_COLUMN}' \
    --num-layers \"\${layer}\" \
    --rhi-target-transform '${RHI_TARGET_TRANSFORM}' \
    --fcnm-error-floor '${FCNM_ERROR_FLOOR}' \
    --fcnm-zero-loss-weight '${FCNM_ZERO_LOSS_WEIGHT}' \
    --run-name \"${RUN_PREFIX}_layers\${layer}\"
done

echo
echo 'Layer sweep complete.'
" > "${LOG_FILE}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "Started sequential train_hi_cnn.py layer sweep"
echo "  PID: ${pid}"
echo "  layers: ${LAYERS}"
echo "  log: ${LOG_FILE}"
echo "  pid file: ${PID_FILE}"
echo
echo "Monitor with:"
echo "  tail -f '${LOG_FILE}'"
