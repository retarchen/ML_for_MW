#!/usr/bin/env python3
"""Data loading, preprocessing, splitting, and DataLoader construction."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class Standardization:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{label} not found: {path}")


def parse_spectrum_coord(path: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d+)_(\d+)\.csv(?:\.gz)?$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_spectra_files(data_path: str) -> List[str]:
    """Return complete spectra files in numeric row/column order."""
    require_dir(data_path, "CSV directory")
    skipped_tiny = 0
    files: List[str] = []
    for entry in os.scandir(data_path):
        if not entry.is_file() or not (entry.name.endswith(".csv") or entry.name.endswith(".csv.gz")):
            continue
        if entry.stat().st_size < 100:
            skipped_tiny += 1
            continue
        files.append(entry.path)

    if not files:
        raise FileNotFoundError(f"No usable .csv or .csv.gz files found in: {data_path}")
    if skipped_tiny:
        print(f"Skipped {skipped_tiny} tiny/empty spectra files that are likely incomplete.")

    coords = [parse_spectrum_coord(path) for path in files]
    if all(coord is not None for coord in coords):
        files = [path for _, path in sorted(zip(coords, files), key=lambda item: item[0])]
    else:
        files.sort()
    return files


def target_indices_from_spectra_files(files: List[str], target_shape: Tuple[int, int]) -> np.ndarray:
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
            raise ValueError(f"Spectrum coordinate out of bounds {target_shape}: {path}")
        indices.append(row * n_cols + col)

    if bad_names:
        raise ValueError(
            "Cannot map spectra to FITS targets because some filenames are not row_col CSVs: "
            + ", ".join(bad_names[:5])
        )
    return np.asarray(indices, dtype=np.int64)


def load_targets(fits_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    require_file(fits_path, "FITS file")
    with fits.open(fits_path) as hdul:
        if len(hdul) <= 2:
            raise ValueError(f"Expected fCNM in HDU 1 and RHI in HDU 2, got {len(hdul)} HDUs.")
        if hdul[1].data.shape != hdul[2].data.shape:
            raise ValueError(f"Target shape mismatch: {hdul[1].data.shape} vs {hdul[2].data.shape}")
        target_shape = tuple(int(v) for v in hdul[1].data.shape)
        if len(target_shape) != 2:
            raise ValueError(f"Expected 2D target maps, got {target_shape}")
        fcnm = hdul[1].data.flatten().astype(np.float32)
        rhi = hdul[2].data.flatten().astype(np.float32)
    return fcnm, rhi, target_shape


def load_spectra_files(
    files: List[str],
    tb_column: int,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """Load selected spectra using a configurable TB column.

    Use tb_column=1 for noisy TB and tb_column=3 for the no-noise TB column.
    The spectral channel count L is inferred from the first selected file.
    """
    tb_spectra: List[np.ndarray] = []
    for i, fp in enumerate(files, start=1):
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV file {fp}: {exc}") from exc

        if df.shape[1] <= tb_column:
            raise ValueError(f"CSV has {df.shape[1]} columns, cannot read column {tb_column}: {fp}")

        spectrum = df.iloc[:, tb_column].to_numpy(dtype=np.float32)
        if expected_length is None:
            expected_length = int(spectrum.shape[0])
            print(f"Inferred spectrum length: {expected_length} channels")
        if spectrum.shape[0] != expected_length:
            raise ValueError(f"Expected spectrum length {expected_length}, got {spectrum.shape[0]} in {fp}")
        tb_spectra.append(spectrum)

        if i == 1 or i % 1000 == 0 or i == len(files):
            print(f"Loaded {i}/{len(files)} spectra", flush=True)
    return np.asarray(tb_spectra, dtype=np.float32)


def select_random_subset(n_samples: int, subset_size: int, seed: int) -> np.ndarray:
    if subset_size == -1 or subset_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if subset_size <= 0:
        raise ValueError("--subset-size must be positive, or -1 for all samples.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=subset_size, replace=False).astype(np.int64))


def create_splits(
    subset_indices: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    shuffled = subset_indices.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    if min(n_train, n_val, n - n_train - n_val) <= 0:
        raise ValueError("Split too small. Increase --subset-size.")
    return (
        np.sort(shuffled[:n_train]),
        np.sort(shuffled[n_train : n_train + n_val]),
        np.sort(shuffled[n_train + n_val :]),
    )


def target_summary(name: str, fcnm: np.ndarray, rhi: np.ndarray, indices: np.ndarray) -> None:
    print(f"\nTarget summary: {name} (n={len(indices)})")
    for label, arr in (("fCNM", fcnm[indices]), ("RHI", rhi[indices])):
        print(
            f"  {label}: min={np.min(arr):.5g}, p05={np.percentile(arr, 5):.5g}, "
            f"median={np.median(arr):.5g}, mean={np.mean(arr):.5g}, "
            f"std={np.std(arr):.5g}, p95={np.percentile(arr, 95):.5g}, max={np.max(arr):.5g}"
        )


def save_indices(
    results_dir: Path,
    subset_indices: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    subset_path = results_dir / "sampled_indices.csv"
    split_path = results_dir / "split_indices.csv"
    pd.DataFrame({"sample_index": subset_indices}).to_csv(subset_path, index=False)
    pd.concat(
        [
            pd.DataFrame({"sample_index": train_idx, "split": "train"}),
            pd.DataFrame({"sample_index": val_idx, "split": "val"}),
            pd.DataFrame({"sample_index": test_idx, "split": "test"}),
        ],
        ignore_index=True,
    ).to_csv(split_path, index=False)
    return subset_path, split_path


def moving_average_spectra(tb: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return tb.astype(np.float32)
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd.")
    pad = window // 2
    padded = np.pad(tb.astype(np.float32), ((0, 0), (pad, pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    return ((cumsum[:, window:] - cumsum[:, :-window]) / float(window)).astype(np.float32)


def prepare_input_channels(tb: np.ndarray, mode: str, smooth_window: int) -> np.ndarray:
    if mode == "raw":
        print("Input mode: raw spectrum only")
        return tb.astype(np.float32)
    smoothed = moving_average_spectra(tb, smooth_window)
    if mode == "smooth":
        print(f"Input mode: smoothed spectrum only (window={smooth_window})")
        return smoothed
    if mode == "raw_plus_smooth":
        print(f"Input mode: raw spectrum + smoothed spectrum (window={smooth_window})")
        return np.stack([tb.astype(np.float32), smoothed], axis=1)
    raise ValueError(f"Unknown input mode: {mode}")


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
    print("Input normalization: ON, fitted on train split")
    return scaler.transform(tb_subset), scaler


def transform_rhi_target(raw_targets: np.ndarray, mode: str) -> np.ndarray:
    targets = raw_targets.astype(np.float32).copy()
    if mode == "raw":
        print("RHI target transform: raw RHI")
        return targets
    if mode == "log":
        if np.any(targets[:, 1] <= 0):
            raise ValueError("Cannot train log(RHI) because at least one RHI <= 0.")
        targets[:, 1] = np.log(targets[:, 1]).astype(np.float32)
        print("RHI target transform: log(RHI) for training, exp() for evaluation")
        return targets
    raise ValueError(f"Unknown RHI target transform: {mode}")


def apply_fcnm_error_floor(raw_targets: np.ndarray, floor: float) -> np.ndarray:
    """Treat fCNM values below the chosen error/detection floor as zero.

    This is a scientific approximation for the vertical true-fCNM=0 band:
    below the floor, fCNM is not interpreted as a precise positive fraction.
    Use --fcnm-error-floor 0 to disable it.
    """
    targets = raw_targets.astype(np.float32).copy()
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
    """Apply simple physical constraints after inverse target transforms.

    The model head is intentionally linear for stable regression, so this
    postprocessing keeps evaluation/plots in the physical domain.
    """
    constrained = values.astype(np.float32).copy()
    constrained[:, 0] = np.clip(constrained[:, 0], 0.0, 1.0)
    constrained[:, 1] = np.maximum(constrained[:, 1], 0.0)
    if snap_fcnm_below_floor and fcnm_error_floor > 0:
        constrained[constrained[:, 0] < fcnm_error_floor, 0] = 0.0
    return constrained


def maybe_normalize_targets(
    targets: np.ndarray,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, Optional[Standardization]]:
    if not enabled:
        print("Target normalization: OFF")
        return targets.astype(np.float32), None
    train_rows = np.searchsorted(subset_idx, train_idx)
    scaler = fit_standardization(targets[train_rows], axis=0)
    print(f"Target normalization: ON (mean={scaler.mean.ravel()}, std={scaler.std.ravel()})")
    return scaler.transform(targets), scaler


class HISpectraDataset(Dataset):
    def __init__(
        self,
        tb_subset: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        row_indices: np.ndarray,
        sample_indices: np.ndarray,
    ):
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

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.indices[idx], self.weights[idx]


def make_imbalance_sampler(raw_targets_subset: np.ndarray, train_rows: np.ndarray, n_bins: int) -> WeightedRandomSampler:
    f = raw_targets_subset[train_rows, 0]
    r = raw_targets_subset[train_rows, 1]
    f_edges = np.linspace(float(np.min(f)), float(np.max(f)), n_bins + 1)
    r_edges = np.linspace(float(np.min(r)), float(np.max(r)), n_bins + 1)
    if np.allclose(f_edges[0], f_edges[-1]) or np.allclose(r_edges[0], r_edges[-1]):
        raise ValueError("Cannot create imbalance bins because a target is constant.")
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
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def create_dataloaders(
    tb_subset: np.ndarray,
    targets_for_training: np.ndarray,
    raw_targets_subset: np.ndarray,
    target_indices_subset: np.ndarray,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_imbalance_sampler: bool,
    imbalance_bins: int,
    rhi_tail_loss_weight: float,
    fcnm_error_floor: float,
    fcnm_zero_loss_weight: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_rows = np.searchsorted(subset_idx, train_idx)
    val_rows = np.searchsorted(subset_idx, val_idx)
    test_rows = np.searchsorted(subset_idx, test_idx)

    target_weights = np.ones_like(targets_for_training, dtype=np.float32)
    if rhi_tail_loss_weight > 0:
        train_rhi = raw_targets_subset[train_rows, 1]
        rhi_p99 = float(np.percentile(train_rhi, 99))
        scale = max(rhi_p99 - 1.0, 1e-6)
        tail_strength = np.clip((raw_targets_subset[:, 1] - 1.0) / scale, 0.0, 1.0)
        target_weights[:, 1] = 1.0 + rhi_tail_loss_weight * tail_strength.astype(np.float32)
        print(f"RHI tail loss weighting: ON (p99={rhi_p99:.4g}, max weight={target_weights[:, 1].max():.3g})")
    else:
        print("RHI tail loss weighting: OFF")

    if fcnm_zero_loss_weight <= 0:
        raise ValueError("--fcnm-zero-loss-weight must be positive.")
    if not np.isclose(fcnm_zero_loss_weight, 1.0):
        threshold = max(float(fcnm_error_floor), 0.0)
        zero_like = raw_targets_subset[:, 0] <= threshold
        target_weights[zero_like, 0] *= float(fcnm_zero_loss_weight)
        print(
            f"fCNM zero/floor loss weighting: ON "
            f"(threshold <= {threshold:.6g}, weight={fcnm_zero_loss_weight:.4g}, "
            f"n={int(zero_like.sum())})"
        )
    else:
        print("fCNM zero/floor loss weighting: OFF")

    unit_weights = np.ones_like(targets_for_training, dtype=np.float32)
    train_ds = HISpectraDataset(tb_subset, targets_for_training, target_weights, train_rows, target_indices_subset[train_rows])
    val_ds = HISpectraDataset(tb_subset, targets_for_training, unit_weights, val_rows, target_indices_subset[val_rows])
    test_ds = HISpectraDataset(tb_subset, targets_for_training, unit_weights, test_rows, target_indices_subset[test_rows])

    sampler = None
    shuffle = True
    if use_imbalance_sampler:
        sampler = make_imbalance_sampler(raw_targets_subset, train_rows, imbalance_bins)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
