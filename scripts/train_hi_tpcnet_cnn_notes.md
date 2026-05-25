# Notes for `train_hi_tpcnet_cnn.py`

This note explains the CNN training script:

```text
scripts/train_hi_tpcnet_cnn.py
```

This is the main 1D CNN script for predicting two regression targets from HI emission spectra:

```text
input:  TB spectrum, shape (N, L)
output: [fCNM, RHI]
```

The current default data paths are:

```text
FITS targets: /mnt/c/Users/retar/Desktop/research/ML/data/MW/fcnm_RHI_z.fits
Spectra CSVs: /mnt/c/Users/retar/Desktop/research/ML/data/MW/syn_HI_spec_z
```

The script currently reads the no-noise TB column:

```python
df.iloc[:, 3]
```

If you want noisy TB instead, use the separated pipeline with `--tb-column 1`, or edit the TB column in `train_hi_tpcnet_cnn.py`.

## What the Script Does

1. Loads fCNM and RHI from the FITS file.
2. Lists spectra CSV/CSV.GZ files in numeric `row_col` order.
3. Maps filenames like `139_48.csv.gz` to the correct flattened FITS target index.
4. Selects a random subset, or all spectra with `--subset-size -1`.
5. Splits data into train/validation/test.
6. Loads selected spectra.
7. Optionally normalizes TB inputs and targets.
8. Trains an 8-layer 1D CNN.
9. Evaluates on the test set.
10. Saves predictions, metrics, model checkpoint, and plots.

## CNN Architecture

The model is inspired by the TPCNet Appendix/Figure A1 idea:

- 8 convolutional layers
- 1D convolutions
- ReLU activations
- BatchNorm after each convolution and before ReLU
- alternating kernel sizes:

```text
7, 33, 7, 33, 7, 33, 7, 33
```

- decreasing channel pattern:

```text
64, 56, 48, 40, 32, 24, 16, 8
```

- no pooling layers
- one shared CNN backbone
- one final linear output layer for:

```text
[fCNM, RHI]
```

## Easiest Way to Run

Use the launcher script:

```bash
cd /mnt/c/Users/retar/Desktop/research/ML/ML_MW
bash scripts/run_train_hi_tpcnet_cnn.sh
```

This runs in detached/no-stop mode, so it should keep running after you close the terminal.

To choose your own run name:

```bash
cd /mnt/c/Users/retar/Desktop/research/ML/ML_MW
RUN_NAME=my_cnn_run bash scripts/run_train_hi_tpcnet_cnn.sh
```

To change common parameters:

```bash
RUN_NAME=my_cnn_run \
EPOCHS=100 \
PATIENCE=15 \
BATCH_SIZE=256 \
DEVICE=cuda \
FCNM_ERROR_FLOOR=0.02 \
FCNM_ZERO_LOSS_WEIGHT=2.0 \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

Outputs will be saved to:

```text
results/my_cnn_run/
figs/my_cnn_run/
logs/my_cnn_run.log
logs/my_cnn_run.pid
```

Monitor the run with:

```bash
tail -f logs/my_cnn_run.log
```

Check if it is running:

```bash
ps -ef | grep train_hi_tpcnet_cnn.py | grep -v grep
```

Check GPU usage:

```bash
nvidia-smi
```

## Direct Python Command

If you do not use the `.sh` launcher, run inside the `cnn` conda environment:

```bash
cd /mnt/c/Users/retar/Desktop/research/ML/ML_MW
source /home/retar/miniconda3/etc/profile.d/conda.sh
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

For no-stop mode directly:

```bash
setsid bash -lc 'source /home/retar/miniconda3/etc/profile.d/conda.sh && conda activate cnn && export MPLCONFIGDIR=/tmp && python -u scripts/train_hi_tpcnet_cnn.py --subset-size -1 --epochs 100 --patience 15 --batch-size 256 --device cuda --rhi-target-transform log --fcnm-error-floor 0.02 --fcnm-zero-loss-weight 2.0 --run-name my_cnn_run' > logs/my_cnn_run.log 2>&1 < /dev/null & echo $!
```

## Parameters You Will Most Often Change

### Run/output parameters

`--run-name`

Name for output subdirectories under `results/` and `figs/`.

Example:

```bash
--run-name no_noise_test1
```

`--output-root`

Base directory where `results/` and `figs/` are created. Usually leave this unchanged.

### Data parameters

`--fits-path`

Path to the FITS file containing fCNM and RHI.

`--csv-dir`

Directory containing spectra CSV files.

`--subset-size`

Number of spectra to randomly use.

Examples:

```bash
--subset-size 20000
--subset-size -1
```

Use `-1` for all spectra.

`--seed`

Random seed for reproducible subset/split.

Default:

```bash
--seed 42
```

### Split parameters

Defaults are:

```bash
--train-frac 0.70
--val-frac 0.15
--test-frac 0.15
```

These should sum to `1.0`.

### Training parameters

`--epochs`

Maximum number of training epochs.

`--patience`

Early stopping patience. Training stops if validation loss does not improve for this many epochs.

`--batch-size`

Batch size. Increase if GPU memory allows; decrease if you get CUDA memory errors.

`--lr`

Learning rate. Default:

```bash
--lr 1e-3
```

`--weight-decay`

AdamW weight decay. Default:

```bash
--weight-decay 1e-4
```

`--device`

Choose:

```bash
--device cuda
--device cpu
--device auto
```

Use `cuda` when you want to force GPU.

### Normalization parameters

Input normalization is ON by default:

```bash
--normalize-inputs
```

Disable it with:

```bash
--no-normalize-inputs
```

Target normalization is ON by default:

```bash
--normalize-targets
```

Disable it with:

```bash
--no-normalize-targets
```

Usually keep both ON.

### RHI parameters

`--rhi-target-transform`

Choices:

```bash
--rhi-target-transform log
--rhi-target-transform raw
```

Use `log` for the current problem because RHI is concentrated near 1 with a high tail.

`--rhi-tail-loss-weight`

Extra weighting for high-RHI samples. Default is off:

```bash
--rhi-tail-loss-weight 0
```

Only increase this if you specifically want to emphasize rare high-RHI values.

### fCNM floor parameters

These were added because many true fCNM values are exactly zero or below the physical/error floor, producing a vertical band in true-vs-predicted plots.

`--fcnm-error-floor`

Treat true fCNM below this value as zero during training/evaluation.

Example:

```bash
--fcnm-error-floor 0.02
```

Set to `0` to disable:

```bash
--fcnm-error-floor 0
```

`--fcnm-zero-loss-weight`

Extra fCNM loss weight for true-zero or below-floor fCNM samples.

Example:

```bash
--fcnm-zero-loss-weight 2.0
```

Use `1.0` to disable extra weighting:

```bash
--fcnm-zero-loss-weight 1.0
```

`--apply-physical-constraints`

On by default. During metrics/plots, it constrains predictions to physical ranges:

```text
fCNM in [0, 1]
RHI >= 0
```

Disable with:

```bash
--no-physical-constraints
```

`--snap-fcnm-below-floor`

On by default. If predicted fCNM is below `--fcnm-error-floor`, it is set to zero before metrics/plots.

Disable with:

```bash
--no-snap-fcnm-below-floor
```

### Imbalance parameters

The imbalance sampler is ON by default:

```bash
--use-imbalance-sampler
```

Disable it with:

```bash
--no-imbalance-sampler
```

`--imbalance-bins`

Number of bins per target for the sampler. Default:

```bash
--imbalance-bins 10
```

## Output Files

For a run named `my_cnn_run`, the script writes:

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

The prediction CSV contains:

```text
sample_index
true_fcnm
pred_fcnm
true_rhi
pred_rhi
```

## Suggested Starting Commands

Full no-noise run:

```bash
RUN_NAME=no_noise_full_fcnm_floor \
FCNM_ERROR_FLOOR=0.02 \
FCNM_ZERO_LOSS_WEIGHT=2.0 \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

Small quick test:

```bash
RUN_NAME=debug_small \
SUBSET_SIZE=1000 \
EPOCHS=2 \
PATIENCE=2 \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

CPU test:

```bash
RUN_NAME=cpu_debug \
SUBSET_SIZE=500 \
EPOCHS=1 \
DEVICE=cpu \
bash scripts/run_train_hi_tpcnet_cnn.sh
```

## Practical Notes

- The script may look slow at first because reading many CSV/CSV.GZ files from `/mnt/c` is slow.
- GPU memory will usually stay near zero until data loading finishes and training epochs begin.
- Use a unique `--run-name` for each experiment so old figures and CSVs are not overwritten.
- If the terminal closes, use the `.sh` launcher or the `setsid` command so training keeps running.
