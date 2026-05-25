#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run noised TB, change weight loss
import os, glob, math, json, re
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from astropy.io import fits
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
#DATAPATH_BASE = os.environ.get("DATAPATH_BASE", "input")
DATAPATH_BASE = os.environ.get("DATAPATH_BASE", "/mnt/c/Users/retar/Desktop/research/ML/data/MW")
FITS_PATH     = os.path.join(DATAPATH_BASE, 'fcnm_RHI_z.fits')
CSV_DIR       = os.path.join(DATAPATH_BASE, 'syn_HI_spec_z')

USE_SHAPE_CHANNEL = False   # TB-only: add a second channel = z-scored shape per spectrum
TEST_SIZE = 0.15
VAL_SIZE  = 0.15
BATCH_SIZE = 512
MAX_EPOCHS = 300
PATIENCE   = 30 # early stopping patience

# CPU threading
torch.backends.cudnn.benchmark = True
# PyTorch 2.x (optional but helpful)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# CPU threading: be polite on shared nodes
torch.set_num_threads(min(8, os.cpu_count() or 8))
torch.set_num_interop_threads(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------- IO helpers ----------------
def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path, label):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{label} not found: {path}")


def parse_spectrum_coord(path):
    """Parse row/column coordinates from names like 139_48.csv.gz."""
    match = re.match(r"^(\d+)_(\d+)\.csv(?:\.gz)?$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_spectra_files(data_path):
    """Return usable spectra files in numeric row/column order.

    Plain sorted(glob(...)) can mis-order row_col filenames, which breaks the
    one-to-one alignment between TB spectra and flattened FITS targets.
    """
    require_dir(data_path, "CSV directory")
    files = []
    skipped_tiny = 0
    for entry in os.scandir(data_path):
        if not entry.is_file() or not (entry.name.endswith(".csv") or entry.name.endswith(".csv.gz")):
            continue
        if entry.stat().st_size < 100:
            skipped_tiny += 1
            continue
        files.append(entry.path)

    if not files:
        raise FileNotFoundError(f"No usable .csv or .csv.gz spectra files found in: {data_path}")
    if skipped_tiny:
        print(f"Skipped {skipped_tiny} tiny/empty spectra files.")

    coords = [parse_spectrum_coord(fp) for fp in files]
    if not all(coord is not None for coord in coords):
        bad = [os.path.basename(fp) for fp, coord in zip(files, coords) if coord is None]
        raise ValueError(f"Cannot align spectra to FITS targets; bad filename examples: {bad[:5]}")

    return [fp for _, fp in sorted(zip(coords, files), key=lambda item: item[0])]


def load_aligned_data(fits_path, data_path, tb_column=1):
    """Load TB, fCNM, and RHI with correct row_col filename alignment.

    This keeps the rest of sample.py unchanged: TB still uses column 1
    for the noisy spectra, and targets below are still stacked as [RHI, fcnm].
    """
    require_file(fits_path, "FITS file")
    with fits.open(fits_path) as hdul:
        if len(hdul) <= 2:
            raise ValueError(f"Expected fCNM in HDU 1 and RHI in HDU 2, got {len(hdul)} HDUs.")
        if hdul[1].data.shape != hdul[2].data.shape:
            raise ValueError(f"Target shape mismatch: {hdul[1].data.shape} vs {hdul[2].data.shape}")
        target_shape = hdul[1].data.shape
        fcnm_map = hdul[1].data.flatten().astype(np.float32)
        rhi_map = hdul[2].data.flatten().astype(np.float32)

    files = list_spectra_files(data_path)
    n_rows, n_cols = target_shape
    tb_spectra = []
    fcnm_values = []
    rhi_values = []
    target_indices = []
    expected_length = None

    for i, fp in enumerate(files, start=1):
        row, col = parse_spectrum_coord(fp)
        if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
            raise ValueError(f"Spectrum coordinate {(row, col)} outside FITS target shape {target_shape}: {fp}")
        target_index = row * n_cols + col

        df = pd.read_csv(fp)
        if df.shape[1] <= tb_column:
            raise ValueError(f"CSV has {df.shape[1]} columns, cannot read column {tb_column}: {fp}")
        spectrum = df.iloc[:, tb_column].to_numpy(dtype=np.float32)

        if expected_length is None:
            expected_length = spectrum.shape[0]
            print(f"Inferred spectrum length: {expected_length} channels")
        if spectrum.shape[0] != expected_length:
            raise ValueError(f"Expected spectrum length {expected_length}, got {spectrum.shape[0]} in {fp}")

        tb_spectra.append(spectrum)
        fcnm_values.append(fcnm_map[target_index])
        rhi_values.append(rhi_map[target_index])
        target_indices.append(target_index)

        if i == 1 or i % 1000 == 0 or i == len(files):
            print(f"Loaded {i}/{len(files)} spectra", flush=True)

    return (
        np.asarray(tb_spectra, dtype=np.float32),
        np.asarray(fcnm_values, dtype=np.float32),
        np.asarray(rhi_values, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int64),
    )

# ---------------- Read data ----------------
TB, fcnm, RHI, sample_indices = load_aligned_data(FITS_PATH, CSV_DIR, tb_column=1)   # noisy TB column

# Basic filtering
mask = (RHI < 2) & (fcnm > 0)
TB, fcnm, RHI = TB[mask], fcnm[mask], RHI[mask]
sample_indices = sample_indices[mask]
print('finish loading data')

# ---------------- Inputs/targets ----------------
# TB-only; optionally add "shape" channel derived from TB
tb = TB.astype(np.float32)
if USE_SHAPE_CHANNEL:
    tb_mean = tb.mean(axis=1, keepdims=True)
    tb_std  = tb.std(axis=1, keepdims=True) + 1e-6
    tb_shape = (tb - tb_mean) / tb_std
    X = np.stack([tb, tb_shape], axis=1)  # (N, 2, 201)
    CIN = 2
else:
    X = tb                                  # (N, 201)
    CIN = 1

y = np.stack([RHI, fcnm], axis=1).astype(np.float32)  # (N, 2)

# ---------------- Stratified split on (RHI,fcnm) ----------------
def stratify_labels(r, f, bins=12, min_count=2, verbose=True):
    for nb in range(bins, 1, -1):
        qr = np.unique(np.quantile(r, np.linspace(0, 1, nb + 1)))
        qf = np.unique(np.quantile(f, np.linspace(0, 1, nb + 1)))
        br = np.clip(np.digitize(r, qr[1:-1], right=True), 0, len(qr)-2)
        bf = np.clip(np.digitize(f, qf[1:-1], right=True), 0, len(qf)-2)
        nb_r, nb_f = len(qr)-1, len(qf)-1
        labels = (br * nb_f + bf).astype(np.int64)
        counts = np.bincount(labels, minlength=nb_r*nb_f)
        if counts.min() >= min_count:
            if verbose:
                print(f"[Stratify] {nb_r}x{nb_f} bins OK (min={counts.min()}, max={counts.max()})")
            return labels
    # fallback 1D
    n = r.shape[0]
    r1 = np.argsort(np.argsort(r)) / max(n-1, 1)
    r2 = np.argsort(np.argsort(f)) / max(n-1, 1)
    t = 0.5 * (r1 + r2)
    q = np.unique(np.quantile(t, np.linspace(0,1,8)))
    labs = np.clip(np.digitize(t, q[1:-1], right=True), 0, len(q)-2).astype(np.int64)
    print("[Stratify:Fallback] 1D bins =", len(q)-1)
    return labs

labels = stratify_labels(RHI, fcnm, bins=15, min_count=2, verbose=True)

# Train/Val/Test split (stratified on joint labels)
sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
trainval_idx, test_idx = next(sss_outer.split(np.zeros(len(labels)), labels))
sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=43)
train_idx, val_idx = next(sss_inner.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx   = trainval_idx[val_idx]

print(f"Split sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# ---------------- Target normalization (train-only) ----------------
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y[train_idx])
y_val   = y_scaler.transform(y[val_idx])
y_test  = y_scaler.transform(y[test_idx])

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

# ---------------- Dataset ----------------
class EmissionDataset(Dataset):
    def __init__(self, X, y):
        X = torch.from_numpy(X).float()
        if X.ndim == 2:  # (N, L) -> (N,1,L)
            X = X.unsqueeze(1)
        self.X = X
        self.y = torch.from_numpy(y).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = EmissionDataset(X_train, y_train)
val_ds   = EmissionDataset(X_val,   y_val)
test_ds  = EmissionDataset(X_test,  y_test)

# ---------------- Sampler: balance **fcnm** bins in TRAIN ----------------
# This focuses sampling on the hard target without touching val/test.
q = np.unique(np.quantile(fcnm, np.linspace(0,1,21)))              # 20 bins
f_bins_all = np.clip(np.digitize(fcnm, q[1:-1], right=True), 0, len(q)-2).astype(np.int64)
train_bins = f_bins_all[train_idx]
u, c = np.unique(train_bins, return_counts=True)
wmap = {b: n for b, n in zip(u, c)}
weights = np.array([1.0 / wmap[b] for b in train_bins], dtype=np.float64)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# (Optional) quick audit:
# print("Sampler (fcnm) counts sample:", Counter(np.random.choice(train_bins, size=2000, p=weights/weights.sum())))

num_workers = min(16, os.cpu_count())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=num_workers, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

# ---------------- Model (BatchNorm keeps intensity info) ----------------
class CNN1DRegressor(nn.Module):
    def __init__(self, cin, out_dim=2,
                 chs=(180, 160, 128, 96, 64, 32),
                 ks=(33, 25, 33, 7, 20, 3),
                 p_drop=0.1):
        super().__init__()

        assert len(chs) == len(ks), "chs and ks must have same length"
        layers = []
        c_in = cin

        for i, (c_out, k) in enumerate(zip(chs, ks)):
            pool = (i < 3)   # apply pooling only in first few blocks
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(c_out), 
                nn.ReLU(inplace=True),
            ]
            if i in [0, 2, 4]:
                layers.append(nn.MaxPool1d(2))
            
            # Dropout only in later layers
            if i >= 2 and p_drop > 0:
                layers.append(nn.Dropout(p_drop))
                
            # if pool:
            #     layers.append(nn.MaxPool1d(2))
            # if p_drop > 0:
            #     layers.append(nn.Dropout(p_drop))
            c_in = c_out

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        z = self.features(x)
        return self.head(z)

model = CNN1DRegressor(cin=CIN).to(device)

# ---------------- Training setup ----------------
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=[2.0, 1.0]):  # [RHI_weight, fcnm_weight]
        super().__init__()
        self.weights = torch.tensor(weights)
    
    def forward(self, pred, target):
        self.weights = self.weights.to(pred.device)
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights
        return weighted_mse.mean()

criterion = WeightedMSELoss(weights=[2.0, 1.0])  # emphasize RHI
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6, verbose=True)

# ---------------- (Optional) tiny overfit sanity check ----------------
TINY_OVERFIT = False
if TINY_OVERFIT:
    tiny = np.random.choice(len(train_ds), size=512, replace=False)
    train_ds = torch.utils.data.Subset(train_ds, tiny)
    # simple loader, no sampler
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)

# ---------------- Train / validate with early stopping ----------------
def run_epoch(dl, train=False):
    model.train(mode=train)
    total, n = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)

@torch.no_grad()
def epoch_rmse(dl):
    model.eval()
    P, T = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        P.append(model(xb).cpu().numpy())
        T.append(yb.cpu().numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    # back to original units
    P = y_scaler.inverse_transform(P)
    T = y_scaler.inverse_transform(T)
    rmse = np.sqrt(np.mean((P - T)**2, axis=0))
    return rmse  # [RHI, fcnm]

best_val = float("inf")
best_state = None
patience_ctr = 0

# Baseline RMSE (predict train mean)
y_mean_train = y[train_idx].mean(axis=0)
baseline_rmse = np.sqrt(np.mean((y[val_idx] - y_mean_train)**2, axis=0))
print("Baseline RMSE (val) — RHI {:.4f}, fcnm {:.4f}".format(baseline_rmse[0], baseline_rmse[1]))

history = []  # will store dicts per epoch

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)

    tr_rmse = epoch_rmse(train_loader)
    va_rmse = epoch_rmse(val_loader)

    curr_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:03d} | train {tr_loss:.4f} "
        f"(RMSE RHI={tr_rmse[0]:.4f}, fcnm={tr_rmse[1]:.4f}) | "
        f"val {val_loss:.4f} "
        f"(RMSE RHI={va_rmse[0]:.4f}, fcnm={va_rmse[1]:.4f}) | "
        f"lr={curr_lr:g}")

    # --- log one row ---
    history.append({
        "epoch": epoch,
        "train_loss": float(tr_loss),
        "val_loss": float(val_loss),
        "train_rmse_RHI": float(tr_rmse[0]),
        "train_rmse_fcnm": float(tr_rmse[1]),
        "val_rmse_RHI": float(va_rmse[0]),
        "val_rmse_fcnm": float(va_rmse[1]),
        "lr": float(curr_lr),
    })

    if val_loss + 1e-6 < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

# ---------------- Evaluate, save CSV, and parity plot ----------------
@torch.no_grad()
def collect_preds(dl, model, y_scaler=None):
    model.eval()
    preds, trues = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        trues = y_scaler.inverse_transform(trues)
    return preds, trues

pred_test, true_test = collect_preds(test_loader, model, y_scaler=y_scaler)

os.makedirs("results", exist_ok=True)
log_path = os.path.join("results", f"2_training_log{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
pd.DataFrame(history).to_csv(log_path, index=False)
print(f"Saved training log to: {log_path}")

pd.DataFrame({
    "RHI_true":  true_test[:, 0],
    "RHI_pred":  pred_test[:, 0],
    "fcnm_true": true_test[:, 1],
    "fcnm_pred": pred_test[:, 1],
}).to_csv(f"results/2_test_predictions{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
print("Saved predictions to results/2_test_predictions.csv")

# parity plots
fig, axes = plt.subplots(1, 2, figsize=(11,5))
names = ["RHI", "fcnm"]
for i, ax in enumerate(axes):
    x = true_test[:, i]; yhat = pred_test[:, i]
    lo, hi = float(min(x.min(), yhat.min())), float(max(x.max(), yhat.max()))
    ax.scatter(x, yhat, s=10, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], ls="--", c="black")
    ax.set_xlabel(f"True {names[i]}"); ax.set_ylabel(f"Pred {names[i]}")
    ax.set_title(f"Pred vs True: {names[i]}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

os.makedirs("figs", exist_ok=True)
png_path = os.path.join("figs", f"2_pred_vs_true_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to: {png_path}")
