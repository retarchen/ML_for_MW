# HI Spectra CNN Regression

This repository trains and evaluates a PyTorch 1D CNN for predicting two HI
emission-spectrum regression targets:

```text
input : TB emission spectrum, shape (N, L)
output: [fCNM, RHI]
```

The current main training script is:

```text
scripts/train_hi_tpcnet_cnn.py
```

The CNN is a small-scale, practical implementation inspired by the 1D CNN idea
in Appendix A / Figure A1 of the TPCNet paper. It is not intended to reproduce
the full paper training setup exactly.

## Quick Start

Run the default full-data CNN training in detached/no-stop mode:

```bash
cd /path/to/ML_MW
RUN_NAME=my_cnn_run bash scripts/run_train_hi_tpcnet_cnn.sh
```

This keeps running after the terminal closes. Outputs are written to:

```text
logs/my_cnn_run.log
logs/my_cnn_run.pid
results/my_cnn_run/
figs/my_cnn_run/
```

Monitor the run:

```bash
tail -f logs/my_cnn_run.log
```

Check whether it is still running:

```bash
ps -ef | grep train_hi_tpcnet_cnn.py | grep -v grep
```

Check GPU usage:

```bash
nvidia-smi
```

## Data

By default, `train_hi_tpcnet_cnn.py` looks for data relative to the repository
parent directory:

```text
../data/MW/fcnm_RHI_z.fits
../data/MW/syn_HI_spec_z/
```

You can override these paths with command-line flags:

```bash
--fits-path /path/to/fcnm_RHI_z.fits
--csv-dir /path/to/syn_HI_spec_z
```

The FITS file is expected to contain:

```text
HDU 1: fCNM map
HDU 2: RHI map
```

Spectra files are expected to be named by pixel coordinate, for example:

```text
139_48.csv.gz
```

The script maps each filename to the flattened FITS target index using:

```text
target_index = row * n_cols + col
```

This avoids the common bug where simple string sorting misaligns spectra and
targets.

The monolithic script currently reads the no-noise TB column:

```python
df.iloc[:, 3]
```

For noisy TB or configurable TB columns, use the separated pipeline:

```text
scripts/seperate_scripts/run_pipeline.py
```

with `--tb-column 1` for noisy TB or `--tb-column 3` for no-noise TB.

## Model

The CNN uses:

- 8 1D convolution layers
- BatchNorm after each convolution
- ReLU activations
- no pooling layers
- one shared CNN backbone
- one final linear layer outputting `[fCNM, RHI]`

Kernel sizes alternate:

```text
7, 33, 7, 33, 7, 33, 7, 33
```

Channel counts decrease by 8 each layer:

```text
64, 56, 48, 40, 32, 24, 16, 8
```

## Recommended Run Commands

Full no-noise run with fCNM error-floor handling:

```bash
cd /path/to/ML_MW
RUN_NAME=no_noise_full_fcnm_floor \
FCNM_ERROR_FLOOR=0.02 \
FCNM_ZERO_LOSS_WEIGHT=2.0 \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

Small debug run:

```bash
cd /path/to/ML_MW
RUN_NAME=debug_small \
SUBSET_SIZE=1000 \
EPOCHS=2 \
PATIENCE=2 \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

CPU debug run:

```bash
cd /path/to/ML_MW
RUN_NAME=cpu_debug \
SUBSET_SIZE=500 \
EPOCHS=1 \
DEVICE=cpu \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

## Direct Python Run

If you do not use the launcher script, activate the `cnn` conda environment
first:

```bash
cd /path/to/ML_MW
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate cnn
export MPLCONFIGDIR=/tmp

python -u scripts/train_hi_tpcnet_cnn.py \
  --subset-size -1 \
  --epochs 100 \
  --patience 15 \
  --batch-size 256 \
  --device cuda \
  --rhi-target-transform log \
  --fcnm-error-floor 0.02 \
  --fcnm-zero-loss-weight 2.0 \
  --run-name my_cnn_run
```

Detached/no-stop version:

```bash
setsid bash -lc 'source "${HOME}/miniconda3/etc/profile.d/conda.sh" && conda activate cnn && export MPLCONFIGDIR=/tmp && python -u scripts/train_hi_tpcnet_cnn.py --subset-size -1 --epochs 100 --patience 15 --batch-size 256 --device cuda --rhi-target-transform log --fcnm-error-floor 0.02 --fcnm-zero-loss-weight 2.0 --run-name my_cnn_run' > logs/my_cnn_run.log 2>&1 < /dev/null & echo $!
```

## Important Parameters

### Run And Data

`--run-name`

Name of the output subdirectory under `results/` and `figs/`.

`--fits-path`

Path to the FITS target file.

`--csv-dir`

Directory containing spectra CSV/CSV.GZ files.

`--subset-size`

Number of spectra to use. Use all spectra with:

```bash
--subset-size -1
```

`--seed`

Random seed for reproducible subset and split.

### Split

Defaults:

```bash
--train-frac 0.70
--val-frac 0.15
--test-frac 0.15
```

These must sum to `1.0`.

### Training

Common training controls:

```bash
--epochs 100
--patience 15
--batch-size 256
--lr 1e-3
--weight-decay 1e-4
--device cuda
```

Use `--device cuda` to force GPU. Use `--device cpu` for CPU tests.

### Normalization

Input and target normalization are on by default:

```bash
--normalize-inputs
--normalize-targets
```

Disable them with:

```bash
--no-normalize-inputs
--no-normalize-targets
```

Usually keep both enabled.

### RHI

RHI is concentrated near 1 with a high tail, so the default is to train on
`log(RHI)`:

```bash
--rhi-target-transform log
```

Use raw RHI only for comparison:

```bash
--rhi-target-transform raw
```

Optional high-RHI weighting:

```bash
--rhi-tail-loss-weight 0
```

The default `0` disables extra tail weighting.

### fCNM Error Floor

Many samples have true fCNM equal to zero or below the physical/error floor.
These points can appear as a vertical band near true fCNM = 0 in true-vs-pred
plots.

Treat sub-floor fCNM as zero:

```bash
--fcnm-error-floor 0.02
```

Disable the floor:

```bash
--fcnm-error-floor 0
```

Increase the fCNM loss weight for true-zero or below-floor samples:

```bash
--fcnm-zero-loss-weight 2.0
```

Disable extra weighting:

```bash
--fcnm-zero-loss-weight 1.0
```

Physical prediction constraints are on by default:

```text
fCNM clipped to [0, 1]
RHI clipped to >= 0
```

Disable them with:

```bash
--no-physical-constraints
```

Predicted fCNM below the floor is snapped to zero by default. Disable with:

```bash
--no-snap-fcnm-below-floor
```

### Imbalance Sampler

The imbalance sampler is on by default:

```bash
--use-imbalance-sampler
```

Disable it:

```bash
--no-imbalance-sampler
```

Control binning:

```bash
--imbalance-bins 10
```

## Outputs

For `--run-name my_cnn_run`, results are saved under:

```text
results/my_cnn_run/
figs/my_cnn_run/
```

Main CSV/model outputs:

```text
results/my_cnn_run/sampled_indices.csv
results/my_cnn_run/split_indices.csv
results/my_cnn_run/training_metrics.csv
results/my_cnn_run/test_predictions.csv
results/my_cnn_run/best_model.pt
```

Figures:

```text
figs/my_cnn_run/true_vs_pred_fcnm.png
figs/my_cnn_run/true_vs_pred_rhi.png
figs/my_cnn_run/true_vs_pred_combined.png
figs/my_cnn_run/true_vs_pred_rhi_log.png
```

`test_predictions.csv` contains:

```text
sample_index
true_fcnm
pred_fcnm
true_rhi
pred_rhi
```

`training_metrics.csv` stores epoch-by-epoch loss, validation RMSE, and learning
rate for plotting learning curves.

## Repository Layout

```text
scripts/train_hi_tpcnet_cnn.py          Main monolithic CNN training script
scripts/run_train_hi_tpcnet_cnn.sh      Detached/no-stop launcher
scripts/sample.py                       Earlier experimental script
scripts/seperate_scripts/               Modular version of the CNN pipeline
results/                                CSV outputs and model checkpoints
figs/                                   Saved figures
logs/                                   Training logs
```

## Troubleshooting

If GPU memory stays at zero, the script may still be reading CSV files. Loading
many compressed spectra from mounted drives or network filesystems can take a
long time.

If a run stops when the terminal closes, use:

```bash
bash scripts/run_train_hi_tpcnet_cnn.sh
```

or the detached `setsid` command shown above.

If CUDA is not used, check:

```bash
nvidia-smi
```

and verify that PyTorch in the `cnn` environment sees CUDA:

```bash
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate cnn
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you want to compare noisy and no-noise TB columns without editing code, use
the separated runner:

```bash
python -u scripts/seperate_scripts/run_pipeline.py --tb-column 1 --run-name noisy_tb
python -u scripts/seperate_scripts/run_pipeline.py --tb-column 3 --run-name no_noise_tb
```
