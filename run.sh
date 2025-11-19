#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$1"                 # e.g. ml_1.py / ml_2.py / ml_3.py
TAG="${SCRIPT%.py}"         # tag used in filenames: ml_1, ml_2, ml_3

# Optional: silence PATH warnings
export PATH="$HOME/.local/bin:$PATH"

python -V
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir numpy==1.26.4 pandas astropy==6.0.1 scikit-learn matplotlib==3.8.4

mkdir -p input
tar -xzf syn_HI_spec_z.tar.gz -C input      # CSVs
cp fcnm_RHI_z.fits input/                   # FITS


export DATAPATH_BASE="$PWD/input"


echo "[run.sh] running ${SCRIPT} (tag=${TAG})"
python -u "train_script/${SCRIPT}"
