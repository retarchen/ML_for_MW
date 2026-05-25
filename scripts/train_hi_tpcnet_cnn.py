#!/usr/bin/env python3
"""
Train and evaluate a small paper-inspired 1D CNN on HI emission spectra.

Scientific target:
    input  TB spectra: (N, L), where L is the number of spectral channels
    output regression targets: [fCNM, RHI]

This is inspired by the 1D CNN idea in Fig. A1 / Appendix A of
"TPCNet: Representation learning for HI mapping", but it is intentionally a
small, practical baseline for local spectra rather than a full reproduction of
the paper training setup.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.io import fits
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATAPATH_BASE = os.environ.get("DATAPATH_BASE", str(PROJECT_ROOT.parent / "data" / "MW"))
# Note: the prompt showed os.path.join(DATAPATH_BASE, "/fcnm_RHI_z.fits").
# A leading slash would ignore DATAPATH_BASE, so the default below uses the
# intended path under DATAPATH_BASE.
FITS_PATH = os.path.join(DATAPATH_BASE, "fcnm_RHI_z.fits")
CSV_DIR = os.path.join(DATAPATH_BASE, "syn_HI_spec_z")


@dataclass
class TrainConfig:
    fits_path: str
    csv_dir: str
    output_root: Path
    run_name: str
    subset_size: int
    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    num_workers: int
    use_imbalance_sampler: bool
    imbalance_bins: int
    normalize_inputs: bool
    normalize_targets: bool
    rhi_target_transform: str
    rhi_tail_loss_weight: float
    fcnm_error_floor: float
    fcnm_zero_loss_weight: float
    apply_physical_constraints: bool
    snap_fcnm_below_floor: bool
    input_mode: str
    smooth_window: int
    device: str


# -----------------------------------------------------------------------------
# Reproducibility and validation
# -----------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{label} not found: {path}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def parse_spectrum_coord(path: str) -> Optional[Tuple[int, int]]:
    """Parse row/column coordinates from names like 139_48.csv.gz."""
    match = re.match(r"^(\d+)_(\d+)\.csv(?:\.gz)?$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_spectra_files(data_path: str) -> List[str]:
    """Return spectra files in numeric row/column order.

    Plain lexicographic sorting is wrong for these filenames:
    0_100.csv.gz would come before 0_11.csv.gz. That scrambles the
    one-to-one alignment with flattened FITS targets and can make the model
    collapse toward mean predictions.
    """
    require_dir(data_path, "CSV directory")

    # os.scandir avoids two large glob passes over this Windows-mounted
    # directory, which can be noticeably slow for >100k compressed spectra.
    skipped_tiny = 0
    files = []
    for entry in os.scandir(data_path):
        if not entry.is_file() or not (entry.name.endswith(".csv") or entry.name.endswith(".csv.gz")):
            continue

        # During spectra generation, placeholder or interrupted .csv.gz files
        # can exist with zero bytes. Pandas raises EmptyDataError on these.
        # Skip tiny files so an "all spectra" run means all complete/usable
        # spectra currently present.
        if entry.stat().st_size < 100:
            skipped_tiny += 1
            continue
        files.append(entry.path)

    if not files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found in: {data_path}")
    if skipped_tiny:
        print(f"Skipped {skipped_tiny} tiny/empty spectra files that are likely incomplete.")

    coords = [parse_spectrum_coord(path) for path in files]
    if all(coord is not None for coord in coords):
        files = [path for _, path in sorted(zip(coords, files), key=lambda item: item[0])]
    else:
        files.sort()
    return files


def target_indices_from_spectra_files(files: List[str], target_shape: Tuple[int, int]) -> np.ndarray:
    """Map row_col spectra filenames onto flattened FITS target indices."""
    n_rows, n_cols = target_shape
    indices: List[int] = []
    bad_names: List[str] = []

    for path in files:
        coord = parse_spectrum_coord(path)
        if coord is None:
            bad_names.append(os.path.basename(path))
            continue
        row, col = coord
        if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
            raise ValueError(
                f"Spectrum filename coordinate out of target bounds {target_shape}: {path}"
            )
        indices.append(row * n_cols + col)

    if bad_names:
        preview = ", ".join(bad_names[:5])
        raise ValueError(
            "Cannot map spectra to FITS targets because some filenames are not row_col CSVs: "
            f"{preview}"
        )

    return np.asarray(indices, dtype=np.int64)


def load_spectra_files(files: List[str], expected_length: Optional[int] = None) -> np.ndarray:
    """Load TB spectra from an already-selected list of CSV files.

    The TB column follows the user's local code: df.iloc[:, 3].
    """

    tb_spectra: List[np.ndarray] = []
    for i, fp in enumerate(files, start=1):
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV file {fp}: {exc}") from exc

        if df.shape[1] <= 3:
            raise ValueError(
                f"CSV file has {df.shape[1]} columns, but TB expects column index 3: {fp}"
            )

        spectrum = df.iloc[:, 3].to_numpy(dtype=np.float32)
        if expected_length is None:
            expected_length = int(spectrum.shape[0])
            print(f"Inferred spectrum length: {expected_length} channels")
        if spectrum.shape[0] != expected_length:
            raise ValueError(
                f"Expected spectrum length {expected_length}, got {spectrum.shape[0]} in {fp}"
            )
        tb_spectra.append(spectrum)

        if i == 1 or i % 1000 == 0 or i == len(files):
            print(f"Loaded {i}/{len(files)} spectra", flush=True)

    tb = np.asarray(tb_spectra, dtype=np.float32)
    return tb


def load_spectra(
    data_path: str, expected_length: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """Load every available spectrum.

    This helper is kept for completeness, but the main training path below
    samples indices first and then reads only selected files. That is much
    faster for small subset experiments with many compressed CSV spectra.
    """
    files = list_spectra_files(data_path)
    return load_spectra_files(files, expected_length), files


def load_targets(fits_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    require_file(fits_path, "FITS file")

    try:
        with fits.open(fits_path) as hdul:
            if len(hdul) <= 2:
                raise ValueError(
                    f"Expected fCNM in HDU 1 and RHI in HDU 2, but FITS has {len(hdul)} HDUs."
                )
            if hdul[1].data.shape != hdul[2].data.shape:
                raise ValueError(
                    f"Target HDU shape mismatch: fCNM={hdul[1].data.shape}, RHI={hdul[2].data.shape}"
                )
            target_shape = tuple(int(v) for v in hdul[1].data.shape)
            if len(target_shape) != 2:
                raise ValueError(f"Expected 2D target maps, got shape {target_shape}")
            fcnm = hdul[1].data.flatten().astype(np.float32)
            rhi = hdul[2].data.flatten().astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to read FITS targets from {fits_path}: {exc}") from exc

    return fcnm, rhi, target_shape


def load_aligned_data(config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Reading targets from: {config.fits_path}")
    fcnm, rhi, _ = load_targets(config.fits_path)

    print(f"Reading spectra from: {config.csv_dir}")
    tb, _ = load_spectra(config.csv_dir)

    if tb.ndim != 2:
        raise ValueError(f"Expected TB shape (N, L), got {tb.shape}")

    if len(fcnm) != len(rhi):
        raise ValueError(f"Target length mismatch: len(fCNM)={len(fcnm)}, len(RHI)={len(rhi)}")

    # Your spectra and target pixels are aligned one-to-one in the same order.
    # While the CSV generation is still running, use as many spectra as exist.
    if len(tb) < len(fcnm):
        print(
            f"TB has {len(tb)} spectra but FITS has {len(fcnm)} target pixels; "
            f"using the first {len(tb)} targets to match current spectra."
        )
        fcnm = fcnm[: len(tb)]
        rhi = rhi[: len(tb)]
    elif len(tb) > len(fcnm):
        raise ValueError(
            f"TB has more spectra than targets: len(TB)={len(tb)}, len(fCNM)={len(fcnm)}"
        )

    if not (len(tb) == len(fcnm) == len(rhi)):
        raise ValueError(
            f"Aligned length check failed: len(TB)={len(tb)}, "
            f"len(fCNM)={len(fcnm)}, len(RHI)={len(rhi)}"
        )

    print(f"Data loaded: TB={tb.shape}, fCNM={fcnm.shape}, RHI={rhi.shape}")
    return tb, fcnm, rhi


# -----------------------------------------------------------------------------
# Subset, split, and summaries
# -----------------------------------------------------------------------------


def select_random_subset(n_samples: int, subset_size: int, seed: int) -> np.ndarray:
    """Return sorted original indices for a reproducible random subset.

    Use --subset-size -1 to use all available samples.
    """
    if subset_size == -1 or subset_size >= n_samples:
        subset = np.arange(n_samples, dtype=np.int64)
    elif subset_size <= 0:
        raise ValueError("--subset-size must be positive, or -1 for all samples.")
    else:
        rng = np.random.default_rng(seed)
        subset = rng.choice(n_samples, size=subset_size, replace=False)
        subset = np.sort(subset.astype(np.int64))
    return subset


def create_splits(
    subset_indices: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Split fractions must sum to 1.0, got train+val+test={total:.6f}"
        )

    rng = np.random.default_rng(seed)
    shuffled = subset_indices.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Split too small: train={n_train}, val={n_val}, test={n_test}. "
            "Increase --subset-size."
        )

    train_idx = np.sort(shuffled[:n_train])
    val_idx = np.sort(shuffled[n_train : n_train + n_val])
    test_idx = np.sort(shuffled[n_train + n_val :])
    return train_idx, val_idx, test_idx


def target_summary(name: str, fcnm: np.ndarray, rhi: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    f = fcnm[indices]
    r = rhi[indices]
    stats: Dict[str, float] = {"split_size": float(len(indices))}
    for label, arr in (("fCNM", f), ("RHI", r)):
        stats[f"{label}_min"] = float(np.min(arr))
        stats[f"{label}_p05"] = float(np.percentile(arr, 5))
        stats[f"{label}_median"] = float(np.median(arr))
        stats[f"{label}_mean"] = float(np.mean(arr))
        stats[f"{label}_std"] = float(np.std(arr))
        stats[f"{label}_p95"] = float(np.percentile(arr, 95))
        stats[f"{label}_max"] = float(np.max(arr))

    print(f"\nTarget summary: {name} (n={len(indices)})")
    print(
        "  fCNM: "
        f"min={stats['fCNM_min']:.5g}, p05={stats['fCNM_p05']:.5g}, "
        f"median={stats['fCNM_median']:.5g}, mean={stats['fCNM_mean']:.5g}, "
        f"std={stats['fCNM_std']:.5g}, p95={stats['fCNM_p95']:.5g}, "
        f"max={stats['fCNM_max']:.5g}"
    )
    print(
        "  RHI : "
        f"min={stats['RHI_min']:.5g}, p05={stats['RHI_p05']:.5g}, "
        f"median={stats['RHI_median']:.5g}, mean={stats['RHI_mean']:.5g}, "
        f"std={stats['RHI_std']:.5g}, p95={stats['RHI_p95']:.5g}, "
        f"max={stats['RHI_max']:.5g}"
    )
    return stats


def save_indices(
    results_dir: Path,
    subset_indices: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Path, Path]:
    subset_path = results_dir / "sampled_indices.csv"
    split_path = results_dir / "split_indices.csv"

    pd.DataFrame({"sample_index": subset_indices}).to_csv(subset_path, index=False)

    split_df = pd.concat(
        [
            pd.DataFrame({"sample_index": train_idx, "split": "train"}),
            pd.DataFrame({"sample_index": val_idx, "split": "val"}),
            pd.DataFrame({"sample_index": test_idx, "split": "test"}),
        ],
        ignore_index=True,
    )
    split_df.to_csv(split_path, index=False)
    return subset_path, split_path


# -----------------------------------------------------------------------------
# Dataset and DataLoader
# -----------------------------------------------------------------------------


class HISpectraDataset(Dataset):
    """PyTorch Dataset for TB spectra and [fCNM, RHI] targets."""

    def __init__(
        self,
        tb_subset: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        row_indices: np.ndarray,
        sample_indices: np.ndarray,
    ):
        # Inputs can be either (N, L) for one channel or (N, C, L) for
        # raw+smoothed spectra. Normalization, if enabled, is fitted upstream
        # on the train split only.
        x = torch.from_numpy(tb_subset[row_indices].astype(np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected input shape (N, L) or (N, C, L), got {tuple(x.shape)}")
        self.x = x

        self.y = torch.from_numpy(targets[row_indices].astype(np.float32))
        self.weights = torch.from_numpy(weights[row_indices].astype(np.float32))
        self.indices = torch.from_numpy(sample_indices.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.indices[idx], self.weights[idx]


def make_imbalance_sampler(
    raw_targets_subset: np.ndarray,
    train_rows: np.ndarray,
    n_bins: int,
) -> WeightedRandomSampler:
    """Create inverse-frequency sample weights from joint target bins.

    This is optional and is meant to counter target imbalance when predictions
    collapse toward common values such as fCNM ~ 0 and RHI ~ 1. It changes only
    the train sampling distribution; validation and test remain untouched.
    """
    if n_bins < 2:
        raise ValueError("--imbalance-bins must be >= 2")

    f = raw_targets_subset[train_rows, 0]
    r = raw_targets_subset[train_rows, 1]

    f_edges = np.linspace(float(np.min(f)), float(np.max(f)), n_bins + 1)
    r_edges = np.linspace(float(np.min(r)), float(np.max(r)), n_bins + 1)
    if np.allclose(f_edges[0], f_edges[-1]) or np.allclose(r_edges[0], r_edges[-1]):
        raise ValueError("Cannot create imbalance bins because a target is constant in train split.")

    f_bin = np.clip(np.digitize(f, f_edges[1:-1], right=True), 0, n_bins - 1)
    r_bin = np.clip(np.digitize(r, r_edges[1:-1], right=True), 0, n_bins - 1)
    joint_bin = f_bin * n_bins + r_bin

    _, inverse, counts = np.unique(joint_bin, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights = weights / np.mean(weights)

    print(
        f"Using imbalance sampler with {len(counts)} occupied joint bins "
        f"out of {n_bins * n_bins}; min/max bin counts={counts.min()}/{counts.max()}"
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def create_dataloaders(
    tb_subset: np.ndarray,
    targets_subset_for_training: np.ndarray,
    raw_targets_subset: np.ndarray,
    target_indices_subset: np.ndarray,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # tb_subset is stored in the same order as subset_idx. Convert original
    # sample indices back to rows in that compact array for each split.
    train_rows = np.searchsorted(subset_idx, train_idx)
    val_rows = np.searchsorted(subset_idx, val_idx)
    test_rows = np.searchsorted(subset_idx, test_idx)

    target_weights = np.ones_like(targets_subset_for_training, dtype=np.float32)
    if config.rhi_tail_loss_weight > 0:
        train_rhi = raw_targets_subset[train_rows, 1]
        rhi_p99 = float(np.percentile(train_rhi, 99))
        scale = max(rhi_p99 - 1.0, 1e-6)
        tail_strength = np.clip((raw_targets_subset[:, 1] - 1.0) / scale, 0.0, 1.0)
        target_weights[:, 1] = 1.0 + config.rhi_tail_loss_weight * tail_strength.astype(np.float32)
        print(
            "RHI tail loss weighting: ON "
            f"(p99={rhi_p99:.4g}, max RHI weight={target_weights[:, 1].max():.3g})"
        )
    else:
        print("RHI tail loss weighting: OFF")

    if config.fcnm_zero_loss_weight <= 0:
        raise ValueError("--fcnm-zero-loss-weight must be positive.")
    if not np.isclose(config.fcnm_zero_loss_weight, 1.0):
        threshold = max(float(config.fcnm_error_floor), 0.0)
        zero_like = raw_targets_subset[:, 0] <= threshold
        target_weights[zero_like, 0] *= float(config.fcnm_zero_loss_weight)
        print(
            "fCNM zero/floor loss weighting: ON "
            f"(threshold <= {threshold:.6g}, weight={config.fcnm_zero_loss_weight:.4g}, "
            f"n={int(zero_like.sum())})"
        )
    else:
        print("fCNM zero/floor loss weighting: OFF")

    train_ds = HISpectraDataset(
        tb_subset,
        targets_subset_for_training,
        target_weights,
        train_rows,
        target_indices_subset[train_rows],
    )
    unit_weights = np.ones_like(targets_subset_for_training, dtype=np.float32)
    val_ds = HISpectraDataset(
        tb_subset,
        targets_subset_for_training,
        unit_weights,
        val_rows,
        target_indices_subset[val_rows],
    )
    test_ds = HISpectraDataset(
        tb_subset,
        targets_subset_for_training,
        unit_weights,
        test_rows,
        target_indices_subset[test_rows],
    )

    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if config.use_imbalance_sampler:
        sampler = make_imbalance_sampler(raw_targets_subset, train_rows, config.imbalance_bins)
        shuffle = False

    pin_memory = resolve_device(config.device).type == "cuda" if config.device != "auto" else torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


@dataclass
class Standardization:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


def fit_standardization(values: np.ndarray, axis: int = 0) -> Standardization:
    mean = values.mean(axis=axis, keepdims=True).astype(np.float32)
    std = values.std(axis=axis, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return Standardization(mean=mean, std=std)


def maybe_normalize_inputs(
    tb_subset: np.ndarray,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, Optional[Standardization]]:
    if not enabled:
        print("Input normalization: OFF")
        return tb_subset.astype(np.float32), None

    train_rows = np.searchsorted(subset_idx, train_idx)
    scaler = fit_standardization(tb_subset[train_rows], axis=0)
    normalized = scaler.transform(tb_subset)
    print("Input normalization: ON, fitted per velocity channel on train split")
    return normalized, scaler


def moving_average_spectra(tb: np.ndarray, window: int) -> np.ndarray:
    """Smooth spectra along velocity with an edge-padded moving average."""
    if window <= 1:
        return tb.astype(np.float32)
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd so smoothing is centered.")

    pad = window // 2
    padded = np.pad(tb.astype(np.float32), ((0, 0), (pad, pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    smoothed = (cumsum[:, window:] - cumsum[:, :-window]) / float(window)
    return smoothed.astype(np.float32)


def prepare_input_channels(tb: np.ndarray, mode: str, smooth_window: int) -> np.ndarray:
    """Create model inputs from noisy spectra.

    raw_plus_smooth is intended for noisy TB: the raw channel preserves sharp
    spectral information, while the smoothed channel gives the CNN a denoised
    view that often helps intensity-sensitive targets such as RHI.
    """
    if mode == "raw":
        print("Input mode: raw spectrum only")
        return tb.astype(np.float32)

    smoothed = moving_average_spectra(tb, smooth_window)
    if mode == "smooth":
        print(f"Input mode: smoothed spectrum only (window={smooth_window})")
        return smoothed
    if mode == "raw_plus_smooth":
        print(f"Input mode: raw noisy spectrum + smoothed spectrum (window={smooth_window})")
        return np.stack([tb.astype(np.float32), smoothed], axis=1)
    raise ValueError(f"Unknown input mode: {mode}")


def maybe_normalize_targets(
    raw_targets_subset: np.ndarray,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, Optional[Standardization]]:
    if not enabled:
        print("Target normalization: OFF")
        return raw_targets_subset.astype(np.float32), None

    train_rows = np.searchsorted(subset_idx, train_idx)
    scaler = fit_standardization(raw_targets_subset[train_rows], axis=0)
    normalized = scaler.transform(raw_targets_subset)
    print(
        "Target normalization: ON, fitted on train split "
        f"(mean fCNM/RHI={scaler.mean.ravel()}, std={scaler.std.ravel()})"
    )
    return normalized, scaler


def transform_rhi_target(raw_targets_subset: np.ndarray, mode: str) -> np.ndarray:
    """Return targets used for optimization.

    RHI is positive and strongly concentrated near 1 with a long high tail.
    Optimizing log(RHI) reduces domination by rare extreme values and usually
    gives the network finer resolution for the scientifically common/moderate
    range, especially RHI < 5.
    """
    targets = raw_targets_subset.astype(np.float32).copy()
    if mode == "raw":
        print("RHI target transform: raw RHI")
        return targets
    if mode == "log":
        if np.any(targets[:, 1] <= 0):
            raise ValueError("Cannot use --rhi-target-transform log because some RHI values <= 0.")
        targets[:, 1] = np.log(targets[:, 1]).astype(np.float32)
        print("RHI target transform: log(RHI) for training, exp() for evaluation")
        return targets
    raise ValueError(f"Unknown RHI target transform: {mode}")


def apply_fcnm_error_floor(raw_targets_subset: np.ndarray, floor: float) -> np.ndarray:
    """Treat fCNM values below the chosen error/detection floor as zero.

    Below this floor, fCNM is interpreted as consistent with zero rather than
    as a precise positive measurement. Use --fcnm-error-floor 0 to disable.
    """
    targets = raw_targets_subset.astype(np.float32).copy()
    if floor <= 0:
        print("fCNM error floor: OFF")
        return targets
    mask = targets[:, 0] < floor
    targets[mask, 0] = 0.0
    print(
        f"fCNM error floor: values < {floor:.6g} set to 0 "
        f"({int(mask.sum())}/{len(mask)} samples)"
    )
    return targets


def inverse_transform_rhi_target(values: np.ndarray, mode: str) -> np.ndarray:
    restored = values.astype(np.float32).copy()
    if mode == "log":
        restored[:, 1] = np.exp(restored[:, 1]).astype(np.float32)
    elif mode != "raw":
        raise ValueError(f"Unknown RHI target transform: {mode}")
    return restored


def apply_physical_prediction_constraints(
    values: np.ndarray,
    fcnm_error_floor: float,
    snap_fcnm_below_floor: bool,
) -> np.ndarray:
    """Keep inverse-transformed predictions in the physical target domain."""
    constrained = values.astype(np.float32).copy()
    constrained[:, 0] = np.clip(constrained[:, 0], 0.0, 1.0)
    constrained[:, 1] = np.maximum(constrained[:, 1], 0.0)
    if snap_fcnm_below_floor and fcnm_error_floor > 0:
        constrained[constrained[:, 0] < fcnm_error_floor, 0] = 0.0
    return constrained


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class TPCNetInspiredCNN(nn.Module):
    """Eight-layer no-pooling 1D CNN for spectra regression.

    Paper-inspired choices:
      - 8 convolutional layers.
      - ReLU activations.
      - BatchNorm after each convolution and before activation.
      - Alternating kernel sizes 7 and 33.
      - No pooling layers.
      - One shared CNN backbone and one final linear layer producing
        [fCNM, RHI].

    Approximation:
      - The paper states filters are reduced by 8 after each convolution.
        Here we use a compact decreasing pattern
        64, 56, 48, 40, 32, 24, 16, 8 for a small local test.
      - The final head is a simple flatten + linear regression layer.
    """

    def __init__(self, input_length: int, in_channels: int = 1, out_dim: int = 2):
        super().__init__()
        channels = [64, 56, 48, 40, 32, 24, 16, 8]
        kernels = [7, 33, 7, 33, 7, 33, 7, 33]

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out, kernel_size in zip(channels, kernels):
            layers.extend(
                [
                    nn.Conv1d(
                        c_in,
                        c_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] * input_length, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        flat = features.flatten(start_dim=1)
        return self.head(flat)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def batch_to_device(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, indices, weights = batch
    return x.to(device), y.to(device), indices.to(device), weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    target_scaler: Optional[Standardization] = None,
    rhi_target_transform: str = "raw",
    apply_physical_constraints: bool = True,
    fcnm_error_floor: float = 0.0,
    snap_fcnm_below_floor: bool = True,
) -> Tuple[float, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_seen = 0
    squared_error_sum = np.zeros(2, dtype=np.float64)

    for batch in loader:
        x, y, _, weights = batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss_by_target = criterion(pred, y)
        loss = (loss_by_target * weights).mean()

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        n_seen += batch_size

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        if target_scaler is not None:
            pred_np = target_scaler.inverse_transform(pred_np)
            y_np = target_scaler.inverse_transform(y_np)
        pred_np = inverse_transform_rhi_target(pred_np, rhi_target_transform)
        y_np = inverse_transform_rhi_target(y_np, rhi_target_transform)
        if apply_physical_constraints:
            pred_np = apply_physical_prediction_constraints(
                pred_np, fcnm_error_floor, snap_fcnm_below_floor
            )
        err = pred_np - y_np
        squared_error_sum += np.sum(err * err, axis=0)

    mean_loss = total_loss / max(n_seen, 1)
    rmse = np.sqrt(squared_error_sum / max(n_seen, 1))
    return mean_loss, rmse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    target_scaler: Optional[Standardization],
    results_dir: Path,
) -> Tuple[nn.Module, List[Dict[str, float]], float]:
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, config.patience // 3))

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    metrics_path = results_dir / "training_metrics.csv"
    checkpoint_path = results_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            target_scaler=target_scaler,
            rhi_target_transform=config.rhi_target_transform,
            apply_physical_constraints=config.apply_physical_constraints,
            fcnm_error_floor=config.fcnm_error_floor,
            snap_fcnm_below_floor=config.snap_fcnm_below_floor,
        )

        with torch.no_grad():
            val_loss, val_rmse = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                target_scaler=target_scaler,
                rhi_target_transform=config.rhi_target_transform,
                apply_physical_constraints=config.apply_physical_constraints,
                fcnm_error_floor=config.fcnm_error_floor,
                snap_fcnm_below_floor=config.snap_fcnm_below_floor,
            )

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse_fcnm": float(val_rmse[0]),
            "val_rmse_rhi": float(val_rmse[1]),
            "learning_rate": current_lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6g} | "
            f"val_loss={val_loss:.6g} | "
            f"val_RMSE_fCNM={val_rmse[0]:.6g} | "
            f"val_RMSE_RHI={val_rmse[1]:.6g} | "
            f"lr={current_lr:.3g}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_val_loss": float(best_val_loss),
                    "epoch": int(epoch),
                    "config": vars(config),
                    "target_scaler_mean": None
                    if target_scaler is None
                    else target_scaler.mean.astype(np.float32),
                    "target_scaler_std": None
                    if target_scaler is None
                    else target_scaler.std.astype(np.float32),
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history, best_val_loss


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Optional[Standardization] = None,
    rhi_target_transform: str = "raw",
    apply_physical_constraints: bool = True,
    fcnm_error_floor: float = 0.0,
    snap_fcnm_below_floor: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    indices: List[np.ndarray] = []

    for batch in loader:
        x, y, batch_indices, _ = batch_to_device(batch, device)
        pred = model(x)
        pred_np = pred.cpu().numpy()
        true_np = y.cpu().numpy()
        if target_scaler is not None:
            pred_np = target_scaler.inverse_transform(pred_np)
            true_np = target_scaler.inverse_transform(true_np)
        pred_np = inverse_transform_rhi_target(pred_np, rhi_target_transform)
        true_np = inverse_transform_rhi_target(true_np, rhi_target_transform)
        if apply_physical_constraints:
            pred_np = apply_physical_prediction_constraints(
                pred_np, fcnm_error_floor, snap_fcnm_below_floor
            )
        preds.append(pred_np)
        trues.append(true_np)
        indices.append(batch_indices.cpu().numpy())

    return np.concatenate(indices), np.concatenate(trues), np.concatenate(preds)


def regression_metrics(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - truth
    rmse = np.sqrt(np.mean(err * err, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return {
        "rmse_fcnm": float(rmse[0]),
        "rmse_rhi": float(rmse[1]),
        "mae_fcnm": float(mae[0]),
        "mae_rhi": float(mae[1]),
    }


def print_prediction_diagnostics(
    truth: np.ndarray,
    pred: np.ndarray,
    fcnm_error_floor: float = 0.0,
) -> None:
    print("\nPrediction diagnostics")
    for i, label in enumerate(("fCNM", "RHI")):
        t = truth[:, i]
        p = pred[:, i]
        print(
            f"  {label} truth: min={t.min():.6g}, max={t.max():.6g}, mean={t.mean():.6g}"
        )
        print(
            f"  {label} pred : min={p.min():.6g}, max={p.max():.6g}, mean={p.mean():.6g}"
        )
    threshold = max(float(fcnm_error_floor), 0.0)
    zero_like = truth[:, 0] <= threshold
    if np.any(zero_like):
        p0 = pred[zero_like, 0]
        print(
            f"  fCNM true<=floor ({threshold:.6g}): n={int(zero_like.sum())}, "
            f"pred median={np.median(p0):.6g}, p95={np.percentile(p0, 95):.6g}, max={p0.max():.6g}"
        )


# -----------------------------------------------------------------------------
# Outputs and plots
# -----------------------------------------------------------------------------


def save_training_history(results_dir: Path, history: List[Dict[str, float]]) -> Path:
    path = results_dir / "training_metrics.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def save_predictions(
    results_dir: Path,
    sample_indices: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> Path:
    path = results_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "sample_index": sample_indices.astype(np.int64),
            "true_fcnm": truth[:, 0],
            "pred_fcnm": pred[:, 0],
            "true_rhi": truth[:, 1],
            "pred_rhi": pred[:, 1],
        }
    ).to_csv(path, index=False)
    return path


def plot_true_vs_pred(
    truth: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, float],
    figs_dir: Path,
    fcnm_error_floor: float = 0.0,
) -> List[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: List[Path] = []

    plot_specs = [
        ("fCNM", 0, metrics["rmse_fcnm"], figs_dir / "true_vs_pred_fcnm.png"),
        ("RHI", 1, metrics["rmse_rhi"], figs_dir / "true_vs_pred_rhi.png"),
    ]

    for label, col, rmse, path in plot_specs:
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        add_scatter_panel(ax, truth[:, col], pred[:, col], label, rmse, fcnm_error_floor)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    combined_path = figs_dir / "true_vs_pred_combined.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    add_scatter_panel(axes[0], truth[:, 0], pred[:, 0], "fCNM", metrics["rmse_fcnm"], fcnm_error_floor)
    add_scatter_panel(axes[1], truth[:, 1], pred[:, 1], "RHI", metrics["rmse_rhi"], fcnm_error_floor)
    fig.tight_layout()
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined_path)

    rhi_log_path = figs_dir / "true_vs_pred_rhi_log.png"
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    add_scatter_panel(ax, truth[:, 1], pred[:, 1], "RHI", metrics["rmse_rhi"], fcnm_error_floor)
    positive = (truth[:, 1] > 0) & (pred[:, 1] > 0)
    if np.any(positive):
        lo = float(min(np.min(truth[positive, 1]), np.min(pred[positive, 1])))
        hi = float(max(np.max(truth[positive, 1]), np.max(pred[positive, 1])))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(max(lo * 0.9, 1e-3), hi * 1.1)
        ax.set_ylim(max(lo * 0.9, 1e-3), hi * 1.1)
        ax.set_title(f"RHI: true vs predicted, log scale\nRMSE = {metrics['rmse_rhi']:.4g}")
    fig.tight_layout()
    fig.savefig(rhi_log_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(rhi_log_path)

    return paths


def add_scatter_panel(
    ax: plt.Axes,
    truth: np.ndarray,
    pred: np.ndarray,
    label: str,
    rmse: float,
    fcnm_error_floor: float = 0.0,
) -> None:
    ax.scatter(truth, pred, s=12, alpha=0.45, linewidths=0)

    lo = float(min(np.min(truth), np.min(pred)))
    hi = float(max(np.max(truth), np.max(pred)))
    if math.isclose(lo, hi):
        pad = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.05
    else:
        pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, label="1:1")
    if label == "fCNM" and fcnm_error_floor > 0:
        ax.axvline(fcnm_error_floor, color="tab:red", linestyle=":", linewidth=1.1, label="fCNM floor")
        ax.axhline(fcnm_error_floor, color="tab:red", linestyle=":", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label}: true vs predicted\nRMSE = {rmse:.4g}")
    ax.legend(frameon=False, loc="best")
    ax.tick_params(axis="both", labelsize=10)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a paper-inspired 8-layer 1D CNN on HI emission spectra."
    )
    parser.add_argument("--fits-path", default=FITS_PATH, help="Path to fcnm_RHI_z.fits")
    parser.add_argument("--csv-dir", default=CSV_DIR, help="Directory containing spectra CSV files")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT),
        help="Directory where results/ and figs/ will be created",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "Optional subdirectory name under results/ and figs/. "
            "Use this to keep runs separate, e.g. --run-name noisy_tb."
        ),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=20000,
        help="Random subset size. Use -1 to use all loaded spectra.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-imbalance-sampler",
        action="store_true",
        help="Enable inverse-frequency weighted sampler from binned targets.",
    )
    parser.add_argument(
        "--no-imbalance-sampler",
        dest="use_imbalance_sampler",
        action="store_false",
        help="Disable the default imbalance sampler and use plain shuffled batches.",
    )
    parser.set_defaults(use_imbalance_sampler=True)
    parser.add_argument(
        "--imbalance-bins",
        type=int,
        default=10,
        help="Number of linear bins per target for the optional imbalance sampler.",
    )
    parser.add_argument(
        "--rhi-tail-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra RHI MSE weight for high-RHI samples. "
            "Default 0 disables this; use only if you explicitly want to emphasize the high-RHI tail."
        ),
    )
    parser.add_argument(
        "--fcnm-error-floor",
        type=float,
        default=0.0,
        help=(
            "Treat true fCNM values below this error/detection floor as zero. "
            "Use 0 to disable. Example: --fcnm-error-floor 0.02"
        ),
    )
    parser.add_argument(
        "--fcnm-zero-loss-weight",
        type=float,
        default=2.0,
        help=(
            "Multiplier for the fCNM MSE term when true fCNM is zero or below "
            "--fcnm-error-floor. Use 1 to disable."
        ),
    )
    parser.add_argument(
        "--apply-physical-constraints",
        dest="apply_physical_constraints",
        action="store_true",
        help="Clamp evaluated predictions to physical ranges: fCNM in [0, 1], RHI >= 0.",
    )
    parser.add_argument(
        "--no-physical-constraints",
        dest="apply_physical_constraints",
        action="store_false",
        help="Disable physical postprocessing of predictions before metrics/plots.",
    )
    parser.set_defaults(apply_physical_constraints=True)
    parser.add_argument(
        "--snap-fcnm-below-floor",
        dest="snap_fcnm_below_floor",
        action="store_true",
        help="If --fcnm-error-floor > 0, set predicted fCNM below the floor to 0 before metrics/plots.",
    )
    parser.add_argument(
        "--no-snap-fcnm-below-floor",
        dest="snap_fcnm_below_floor",
        action="store_false",
        help="Keep positive predicted fCNM values below the floor instead of snapping them to zero.",
    )
    parser.set_defaults(snap_fcnm_below_floor=True)
    parser.add_argument(
        "--normalize-inputs",
        dest="normalize_inputs",
        action="store_true",
        help="Enable train-fitted per-channel TB standardization.",
    )
    parser.add_argument(
        "--no-normalize-inputs",
        dest="normalize_inputs",
        action="store_false",
        help="Disable input standardization and use raw TB.",
    )
    parser.set_defaults(normalize_inputs=True)
    parser.add_argument(
        "--normalize-targets",
        dest="normalize_targets",
        action="store_true",
        help="Enable train-fitted fCNM/RHI standardization for optimization.",
    )
    parser.add_argument(
        "--no-normalize-targets",
        dest="normalize_targets",
        action="store_false",
        help="Disable target standardization and optimize raw targets.",
    )
    parser.set_defaults(normalize_targets=True)
    parser.add_argument(
        "--rhi-target-transform",
        choices=("log", "raw"),
        default="log",
        help="Train RHI as log(RHI) by default to improve the common/moderate RHI range.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("raw", "smooth", "raw_plus_smooth"),
        default="raw",
        help="Input channels. raw_plus_smooth is useful when training on noisy TB.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd moving-average window for --input-mode smooth/raw_plus_smooth.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses GPU if available, otherwise CPU.",
    )

    args = parser.parse_args()
    return TrainConfig(
        fits_path=args.fits_path,
        csv_dir=args.csv_dir,
        output_root=Path(args.output_root).resolve(),
        run_name=args.run_name.strip(),
        subset_size=args.subset_size,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_imbalance_sampler=args.use_imbalance_sampler,
        imbalance_bins=args.imbalance_bins,
        normalize_inputs=args.normalize_inputs,
        normalize_targets=args.normalize_targets,
        rhi_target_transform=args.rhi_target_transform,
        rhi_tail_loss_weight=args.rhi_tail_loss_weight,
        fcnm_error_floor=args.fcnm_error_floor,
        fcnm_zero_loss_weight=args.fcnm_zero_loss_weight,
        apply_physical_constraints=args.apply_physical_constraints,
        snap_fcnm_below_floor=args.snap_fcnm_below_floor,
        input_mode=args.input_mode,
        smooth_window=args.smooth_window,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    set_random_seed(config.seed)

    device = resolve_device(config.device)
    print(f"Using device: {device}")

    if config.run_name:
        results_dir = config.output_root / "results" / config.run_name
        figs_dir = config.output_root / "figs" / config.run_name
    else:
        results_dir = config.output_root / "results"
        figs_dir = config.output_root / "figs"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading targets from: {config.fits_path}")
    fcnm, rhi, target_shape = load_targets(config.fits_path)
    if len(fcnm) != len(rhi):
        raise ValueError(f"Target length mismatch: len(fCNM)={len(fcnm)}, len(RHI)={len(rhi)}")

    print(f"Listing spectra in: {config.csv_dir}")
    spectra_files = list_spectra_files(config.csv_dir)
    print(f"Found {len(spectra_files)} spectra files")
    spectra_target_indices = target_indices_from_spectra_files(spectra_files, target_shape)
    print(
        "Mapped spectra filenames to flattened FITS target indices "
        f"using target shape {target_shape}"
    )
    if len(np.unique(spectra_target_indices)) != len(spectra_target_indices):
        raise ValueError("Duplicate row_col filenames map to duplicate FITS target indices.")

    subset_idx = select_random_subset(len(spectra_files), config.subset_size, config.seed)
    train_idx, val_idx, test_idx = create_splits(
        subset_idx, config.train_frac, config.val_frac, config.test_frac, config.seed
    )
    subset_target_idx = spectra_target_indices[subset_idx]
    train_target_idx = spectra_target_indices[train_idx]
    val_target_idx = spectra_target_indices[val_idx]
    test_target_idx = spectra_target_indices[test_idx]

    print(
        f"\nSplit sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    target_summary("selected subset", fcnm, rhi, subset_target_idx)
    target_summary("train", fcnm, rhi, train_target_idx)
    target_summary("validation", fcnm, rhi, val_target_idx)
    target_summary("test", fcnm, rhi, test_target_idx)

    subset_path, split_path = save_indices(
        results_dir, subset_target_idx, train_target_idx, val_target_idx, test_target_idx
    )

    selected_files = [spectra_files[i] for i in subset_idx]
    print(f"\nReading only selected subset spectra: {len(selected_files)} files")
    tb_subset = load_spectra_files(selected_files)
    if tb_subset.ndim != 2:
        raise ValueError(f"Expected selected TB shape (N, L), got {tb_subset.shape}")
    if len(tb_subset) != len(subset_idx):
        raise ValueError(
            f"Selected TB length mismatch: len(TB)={len(tb_subset)}, len(subset)={len(subset_idx)}"
        )
    print(f"Selected TB loaded: {tb_subset.shape}")
    tb_subset = prepare_input_channels(
        tb_subset, config.input_mode, config.smooth_window
    )
    print(f"Model input array prepared: {tb_subset.shape}")

    raw_targets_subset = np.stack(
        [fcnm[subset_target_idx], rhi[subset_target_idx]], axis=1
    ).astype(np.float32)
    raw_targets_subset = apply_fcnm_error_floor(
        raw_targets_subset, config.fcnm_error_floor
    )
    transformed_targets_subset = transform_rhi_target(
        raw_targets_subset, config.rhi_target_transform
    )
    tb_subset, _ = maybe_normalize_inputs(
        tb_subset, subset_idx, train_idx, config.normalize_inputs
    )
    targets_subset_for_training, target_scaler = maybe_normalize_targets(
        transformed_targets_subset, subset_idx, train_idx, config.normalize_targets
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        tb_subset,
        targets_subset_for_training,
        raw_targets_subset,
        subset_target_idx,
        subset_idx,
        train_idx,
        val_idx,
        test_idx,
        config,
    )

    input_length = tb_subset.shape[-1]
    in_channels = tb_subset.shape[1] if tb_subset.ndim == 3 else 1
    model = TPCNetInspiredCNN(input_length=input_length, in_channels=in_channels).to(device)
    print(model)

    model, history, best_val_loss = train_model(
        model, train_loader, val_loader, config, device, target_scaler, results_dir
    )
    metrics_path = save_training_history(results_dir, history)

    test_indices, test_truth, test_pred = collect_predictions(
        model,
        test_loader,
        device,
        target_scaler,
        config.rhi_target_transform,
        config.apply_physical_constraints,
        config.fcnm_error_floor,
        config.snap_fcnm_below_floor,
    )
    metrics = regression_metrics(test_truth, test_pred)
    predictions_path = save_predictions(results_dir, test_indices, test_truth, test_pred)
    figure_paths = plot_true_vs_pred(test_truth, test_pred, metrics, figs_dir, config.fcnm_error_floor)

    print(
        f"\nTest RMSE fCNM: {metrics['rmse_fcnm']:.6g}\n"
        f"Test RMSE RHI : {metrics['rmse_rhi']:.6g}\n"
        f"Test MAE  fCNM: {metrics['mae_fcnm']:.6g}\n"
        f"Test MAE  RHI : {metrics['mae_rhi']:.6g}"
    )
    print_prediction_diagnostics(test_truth, test_pred, config.fcnm_error_floor)

    print("\nSummary")
    print(f"  device: {device}")
    print(f"  subset size: {len(subset_idx)}")
    print(f"  train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"  best validation loss: {best_val_loss:.6g}")
    print(f"  final test RMSE fCNM: {metrics['rmse_fcnm']:.6g}")
    print(f"  final test RMSE RHI : {metrics['rmse_rhi']:.6g}")
    print(f"  sampled indices CSV: {subset_path}")
    print(f"  split indices CSV: {split_path}")
    print(f"  training metrics CSV: {metrics_path}")
    print(f"  prediction CSV: {predictions_path}")
    print("  figures:")
    for path in figure_paths:
        print(f"    {path}")


if __name__ == "__main__":
    main()



# Example detached run:
# RUN_NAME=no_noise_tb_fcnm_floor_002 bash scripts/run_train_hi_tpcnet_cnn.sh
