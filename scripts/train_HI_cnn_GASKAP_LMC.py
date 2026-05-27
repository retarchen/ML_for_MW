#!/usr/bin/env python3
"""Train on simulated HI spectra, then apply the CNN to GASKAP LMC spectra.

This script intentionally keeps the same simulation training path as
scripts/train_hi_cnn.py. The real GASKAP/LMC observations are treated as an
external application set: they are not used for fitting, validation, early
stopping, or model selection.

Workflow:
    1. Load simulation targets from the FITS file and simulation spectra from
       the synthetic spectra directory.
    2. Randomly split the simulation spectra into train/validation/test.
    3. Train the TPCNet-inspired 1D CNN on the simulation train split.
    4. Evaluate on the held-out simulation test split.
    5. Interpolate all matched GASKAP/LMC spectra onto the same number of
       channels as the simulation input and run external predictions.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_hi_cnn as simcnn  # noqa: E402


DEFAULT_SIM_DATA_ROOT = PROJECT_ROOT.parent / "data" / "MW"
DEFAULT_SIM_FITS = DEFAULT_SIM_DATA_ROOT / "fcnm_RHI_z.fits"
DEFAULT_SIM_CSV_DIR = DEFAULT_SIM_DATA_ROOT / "syn_HI_spec_z"
DEFAULT_GASKAP_ROOT = PROJECT_ROOT.parent / "data" / "GASKAP_LMC"
DEFAULT_GASKAP_TARGET_CSV = DEFAULT_GASKAP_ROOT / "Fulldata_for_ml.csv"
DEFAULT_GASKAP_SPECTRA_DIR = DEFAULT_GASKAP_ROOT / "LMC_spectra"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} not found: {path}")


def load_gaskap_target_table(target_csv: Path) -> pd.DataFrame:
    require_file(target_csv, "GASKAP target CSV")
    table = pd.read_csv(target_csv)
    required = {"Name", "f_c", "R_HI"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"GASKAP target CSV missing required columns: {sorted(missing)}")

    table = table[["Name", "f_c", "R_HI"]].copy()
    table["Name"] = table["Name"].astype(str)
    table["f_c"] = pd.to_numeric(table["f_c"], errors="coerce")
    table["R_HI"] = pd.to_numeric(table["R_HI"], errors="coerce")
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=["Name", "f_c", "R_HI"])

    if table["Name"].duplicated().any():
        examples = table.loc[table["Name"].duplicated(), "Name"].head().tolist()
        raise ValueError(f"Duplicate Name values in GASKAP target CSV, examples: {examples}")
    if table.empty:
        raise ValueError("No usable GASKAP target rows after filtering finite f_c and R_HI.")
    return table


def list_gaskap_spectra(spectra_dir: Path) -> Dict[str, Path]:
    require_dir(spectra_dir, "GASKAP spectra directory")
    files = sorted(spectra_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No GASKAP spectra CSV files found in: {spectra_dir}")
    spectra_by_name = {path.stem: path for path in files}
    if len(spectra_by_name) != len(files):
        raise ValueError("Duplicate GASKAP spectra file stems found.")
    return spectra_by_name


def read_gaskap_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    required = {"velocity_TB", "TB"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Spectrum {path.name} missing required columns: {sorted(missing)}")

    velocity = pd.to_numeric(table["velocity_TB"], errors="coerce").to_numpy(dtype=np.float64)
    tb = pd.to_numeric(table["TB"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(velocity) & np.isfinite(tb)
    velocity = velocity[valid]
    tb = tb[valid]
    if len(velocity) < 2:
        raise ValueError(f"Spectrum {path.name} has fewer than 2 valid channels.")

    order = np.argsort(velocity)
    return velocity[order], tb[order]


def make_velocity_grid(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mode: str,
    grid_size: int,
    velocity_min: Optional[float],
    velocity_max: Optional[float],
) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")

    mins = np.asarray([values[0][0] for values in spectra.values()], dtype=np.float64)
    maxs = np.asarray([values[0][-1] for values in spectra.values()], dtype=np.float64)

    if velocity_min is not None or velocity_max is not None:
        vmin = float(np.min(mins) if velocity_min is None else velocity_min)
        vmax = float(np.max(maxs) if velocity_max is None else velocity_max)
    elif mode == "overlap":
        vmin = float(np.max(mins))
        vmax = float(np.min(maxs))
    elif mode == "union":
        vmin = float(np.min(mins))
        vmax = float(np.max(maxs))
    else:
        raise ValueError(f"Unknown velocity grid mode: {mode}")

    if not vmin < vmax:
        raise ValueError(f"Invalid GASKAP velocity grid range: {vmin} to {vmax}")
    return np.linspace(vmin, vmax, grid_size, dtype=np.float32)


def load_gaskap_external_set(
    target_csv: Path,
    spectra_dir: Path,
    grid_size: int,
    grid_mode: str,
    velocity_min: Optional[float],
    velocity_max: Optional[float],
    fill_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    target_table = load_gaskap_target_table(target_csv)
    spectra_by_name = list_gaskap_spectra(spectra_dir)

    target_names = set(target_table["Name"].astype(str))
    spectra_names = set(spectra_by_name)
    missing_spectra = sorted(target_names - spectra_names)
    extra_spectra = sorted(spectra_names - target_names)
    if missing_spectra:
        raise ValueError(f"GASKAP target rows without spectra files, examples: {missing_spectra[:5]}")
    if extra_spectra:
        print(f"Ignoring {len(extra_spectra)} GASKAP spectra without target rows.")

    selected = target_table[target_table["Name"].isin(spectra_names)].copy()
    selected = selected.sort_values("Name").reset_index(drop=True)
    names = selected["Name"].astype(str).tolist()

    raw_spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    lengths: List[int] = []
    for name in names:
        velocity, tb = read_gaskap_spectrum(spectra_by_name[name])
        raw_spectra[name] = (velocity, tb)
        lengths.append(len(tb))

    unique_lengths = sorted(set(lengths))
    preview = ", ".join(str(v) for v in unique_lengths[:12])
    more = "" if len(unique_lengths) <= 12 else ", ..."
    print(f"GASKAP native channel lengths: {preview}{more}")

    velocity_grid = make_velocity_grid(raw_spectra, grid_mode, grid_size, velocity_min, velocity_max)
    tb_rows: List[np.ndarray] = []
    for name in names:
        velocity, tb = raw_spectra[name]
        interp = np.interp(velocity_grid, velocity, tb, left=fill_value, right=fill_value)
        tb_rows.append(interp.astype(np.float32))

    x = np.asarray(tb_rows, dtype=np.float32)
    y = selected[["f_c", "R_HI"]].to_numpy(dtype=np.float32)
    print(f"Loaded GASKAP external set: X={x.shape}, targets={y.shape}")
    print(
        f"GASKAP interpolation grid: n={len(velocity_grid)}, "
        f"min={velocity_grid.min():.6g}, max={velocity_grid.max():.6g}, mode={grid_mode}"
    )
    return x, y, velocity_grid, names


def target_summary_from_array(name: str, targets: np.ndarray) -> None:
    print(f"\nTarget summary: {name} (n={len(targets)})")
    for col, label in enumerate(("fCNM/f_c", "RHI/R_HI")):
        values = targets[:, col]
        print(
            f"  {label}: min={values.min():.5g}, p05={np.percentile(values, 5):.5g}, "
            f"median={np.median(values):.5g}, mean={values.mean():.5g}, "
            f"std={values.std():.5g}, p95={np.percentile(values, 95):.5g}, max={values.max():.5g}"
        )


def save_velocity_grid(results_dir: Path, velocity_grid: np.ndarray) -> Path:
    path = results_dir / "gaskap_velocity_grid.csv"
    pd.DataFrame({"velocity_TB": velocity_grid}).to_csv(path, index=False)
    return path


def save_prediction_csv(
    path: Path,
    indices: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    names: Optional[List[str]] = None,
    original_truth: Optional[np.ndarray] = None,
) -> Path:
    data = {
        "sample_index": indices.astype(np.int64),
        "true_fcnm": truth[:, 0],
        "pred_fcnm": pred[:, 0],
        "true_rhi": truth[:, 1],
        "pred_rhi": pred[:, 1],
    }
    if names is not None:
        data = {"Name": names, **data}
    if original_truth is not None:
        data["original_true_fcnm"] = original_truth[:, 0]
        data["original_true_rhi"] = original_truth[:, 1]
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def plot_external_true_vs_pred(
    truth: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, float],
    figs_dir: Path,
    floor: float,
    prefix: str,
) -> List[Path]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: List[Path] = []
    specs = [
        ("fCNM/f_c", 0, metrics["rmse_fcnm"], figs_dir / f"{prefix}_true_vs_pred_fcnm.png"),
        ("RHI/R_HI", 1, metrics["rmse_rhi"], figs_dir / f"{prefix}_true_vs_pred_rhi.png"),
    ]
    for label, col, rmse, path in specs:
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        simcnn.add_scatter_panel(ax, truth[:, col], pred[:, col], label, rmse, floor)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    combined_path = figs_dir / f"{prefix}_true_vs_pred_combined.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    simcnn.add_scatter_panel(axes[0], truth[:, 0], pred[:, 0], "fCNM/f_c", metrics["rmse_fcnm"], floor)
    simcnn.add_scatter_panel(axes[1], truth[:, 1], pred[:, 1], "RHI/R_HI", metrics["rmse_rhi"], floor)
    fig.tight_layout()
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined_path)
    return paths


def infer_with_targets(
    model: torch.nn.Module,
    x: np.ndarray,
    raw_targets: np.ndarray,
    device: torch.device,
    config: simcnn.TrainConfig,
    target_scaler: Optional[simcnn.Standardization],
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_for_training = simcnn.transform_rhi_target(raw_targets, config.rhi_target_transform)
    if target_scaler is not None:
        y_for_training = target_scaler.transform(y_for_training)

    rows = np.arange(len(x), dtype=np.int64)
    weights = np.ones_like(y_for_training, dtype=np.float32)
    dataset = simcnn.HISpectraDataset(x, y_for_training, weights, rows, rows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return simcnn.collect_predictions(
        model,
        loader,
        device,
        target_scaler=target_scaler,
        rhi_target_transform=config.rhi_target_transform,
        apply_physical_constraints=config.apply_physical_constraints,
        fcnm_error_floor=config.fcnm_error_floor,
        snap_fcnm_below_floor=config.snap_fcnm_below_floor,
    )


def build_sim_config(args: argparse.Namespace) -> simcnn.TrainConfig:
    return simcnn.TrainConfig(
        fits_path=str(Path(args.fits_path).resolve()),
        csv_dir=str(Path(args.csv_dir).resolve()),
        output_root=Path(args.output_root).resolve(),
        run_name=args.run_name.strip(),
        tb_column=args.tb_column,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HI CNN on simulation data and apply it to GASKAP LMC observations."
    )
    parser.add_argument("--fits-path", default=str(DEFAULT_SIM_FITS), help="Simulation fcnm_RHI_z.fits")
    parser.add_argument("--csv-dir", default=str(DEFAULT_SIM_CSV_DIR), help="Simulation spectra CSV directory")
    parser.add_argument("--gaskap-target-csv", default=str(DEFAULT_GASKAP_TARGET_CSV))
    parser.add_argument("--gaskap-spectra-dir", default=str(DEFAULT_GASKAP_SPECTRA_DIR))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT))
    parser.add_argument("--run-name", default="sim_to_GASKAP_LMC")
    parser.add_argument("--tb-column", type=int, default=3, help="Simulation TB column: 1 noisy, 3 no-noise.")
    parser.add_argument("--subset-size", type=int, default=-1, help="Simulation subset size; -1 uses all usable spectra.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use-imbalance-sampler", action="store_true")
    parser.add_argument("--no-imbalance-sampler", dest="use_imbalance_sampler", action="store_false")
    parser.set_defaults(use_imbalance_sampler=True)
    parser.add_argument("--imbalance-bins", type=int, default=10)
    parser.add_argument("--normalize-inputs", dest="normalize_inputs", action="store_true")
    parser.add_argument("--no-normalize-inputs", dest="normalize_inputs", action="store_false")
    parser.set_defaults(normalize_inputs=True)
    parser.add_argument("--normalize-targets", dest="normalize_targets", action="store_true")
    parser.add_argument("--no-normalize-targets", dest="normalize_targets", action="store_false")
    parser.set_defaults(normalize_targets=True)
    parser.add_argument("--rhi-target-transform", choices=("log", "raw"), default="log")
    parser.add_argument("--rhi-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--fcnm-error-floor", type=float, default=0.0)
    parser.add_argument("--fcnm-zero-loss-weight", type=float, default=3.0)
    parser.add_argument("--apply-physical-constraints", dest="apply_physical_constraints", action="store_true")
    parser.add_argument("--no-physical-constraints", dest="apply_physical_constraints", action="store_false")
    parser.set_defaults(apply_physical_constraints=True)
    parser.add_argument("--snap-fcnm-below-floor", dest="snap_fcnm_below_floor", action="store_true")
    parser.add_argument("--no-snap-fcnm-below-floor", dest="snap_fcnm_below_floor", action="store_false")
    parser.set_defaults(snap_fcnm_below_floor=True)
    parser.add_argument("--input-mode", choices=("raw", "smooth", "raw_plus_smooth"), default="raw")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--grid-mode", choices=("overlap", "union"), default="overlap")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=0,
        help="GASKAP interpolation channel count. Default 0 matches the simulation input length.",
    )
    parser.add_argument("--velocity-min", type=float, default=None)
    parser.add_argument("--velocity-max", type=float, default=None)
    parser.add_argument("--fill-value", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_sim_config(args)
    simcnn.set_random_seed(config.seed)
    device = simcnn.resolve_device(config.device)
    print(f"Using device: {device}")
    print("Training source: simulation spectra")
    print("External application set: GASKAP LMC observations")

    results_dir = config.output_root / "results" / config.run_name
    figs_dir = config.output_root / "figs" / config.run_name
    sim_figs_dir = figs_dir / "simulation_test"
    gaskap_figs_dir = figs_dir / "gaskap_lmc_external"
    results_dir.mkdir(parents=True, exist_ok=True)
    sim_figs_dir.mkdir(parents=True, exist_ok=True)
    gaskap_figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading simulation targets from: {config.fits_path}")
    fcnm, rhi, target_shape = simcnn.load_targets(config.fits_path)
    if len(fcnm) != len(rhi):
        raise ValueError(f"Target length mismatch: len(fCNM)={len(fcnm)}, len(RHI)={len(rhi)}")

    print(f"Listing simulation spectra in: {config.csv_dir}")
    spectra_files = simcnn.list_spectra_files(config.csv_dir)
    print(f"Found {len(spectra_files)} simulation spectra files")
    spectra_target_indices = simcnn.target_indices_from_spectra_files(spectra_files, target_shape)
    if len(np.unique(spectra_target_indices)) != len(spectra_target_indices):
        raise ValueError("Duplicate simulation spectra filenames map to duplicate FITS target indices.")

    subset_idx = simcnn.select_random_subset(len(spectra_files), config.subset_size, config.seed)
    train_idx, val_idx, test_idx = simcnn.create_splits(
        subset_idx, config.train_frac, config.val_frac, config.test_frac, config.seed
    )
    subset_target_idx = spectra_target_indices[subset_idx]
    train_target_idx = spectra_target_indices[train_idx]
    val_target_idx = spectra_target_indices[val_idx]
    test_target_idx = spectra_target_indices[test_idx]

    print(f"\nSimulation split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    simcnn.target_summary("simulation selected subset", fcnm, rhi, subset_target_idx)
    simcnn.target_summary("simulation train", fcnm, rhi, train_target_idx)
    simcnn.target_summary("simulation validation", fcnm, rhi, val_target_idx)
    simcnn.target_summary("simulation test", fcnm, rhi, test_target_idx)

    subset_path, split_path = simcnn.save_indices(
        results_dir, subset_target_idx, train_target_idx, val_target_idx, test_target_idx
    )

    selected_files = [spectra_files[i] for i in subset_idx]
    print(f"\nReading simulation subset spectra: {len(selected_files)} files, tb_column={config.tb_column}")
    tb_subset = simcnn.load_spectra_files(selected_files, tb_column=config.tb_column)
    if tb_subset.ndim != 2:
        raise ValueError(f"Expected simulation TB shape (N, L), got {tb_subset.shape}")
    tb_subset = simcnn.prepare_input_channels(tb_subset, config.input_mode, config.smooth_window)
    print(f"Simulation model input array: {tb_subset.shape}")

    raw_targets_subset = np.stack([fcnm[subset_target_idx], rhi[subset_target_idx]], axis=1).astype(np.float32)
    raw_targets_subset = simcnn.apply_fcnm_error_floor(raw_targets_subset, config.fcnm_error_floor)
    transformed_targets_subset = simcnn.transform_rhi_target(raw_targets_subset, config.rhi_target_transform)
    tb_subset, input_scaler = simcnn.maybe_normalize_inputs(
        tb_subset, subset_idx, train_idx, config.normalize_inputs
    )
    targets_subset_for_training, target_scaler = simcnn.maybe_normalize_targets(
        transformed_targets_subset, subset_idx, train_idx, config.normalize_targets
    )

    train_loader, val_loader, test_loader = simcnn.create_dataloaders(
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
    model = simcnn.TPCNetInspiredCNN(input_length=input_length, in_channels=in_channels).to(device)
    print(model)

    model, history, best_val_loss = simcnn.train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
        target_scaler,
        results_dir,
    )
    metrics_path = simcnn.save_training_history(results_dir, history)

    sim_test_indices, sim_test_truth, sim_test_pred = simcnn.collect_predictions(
        model,
        test_loader,
        device,
        target_scaler=target_scaler,
        rhi_target_transform=config.rhi_target_transform,
        apply_physical_constraints=config.apply_physical_constraints,
        fcnm_error_floor=config.fcnm_error_floor,
        snap_fcnm_below_floor=config.snap_fcnm_below_floor,
    )
    sim_metrics = simcnn.regression_metrics(sim_test_truth, sim_test_pred)
    sim_predictions_path = save_prediction_csv(
        results_dir / "simulation_test_predictions.csv",
        sim_test_indices,
        sim_test_truth,
        sim_test_pred,
    )
    sim_figure_paths = simcnn.plot_true_vs_pred(
        sim_test_truth, sim_test_pred, sim_metrics, sim_figs_dir, config.fcnm_error_floor
    )

    print(
        f"\nSimulation test RMSE fCNM: {sim_metrics['rmse_fcnm']:.6g}\n"
        f"Simulation test RMSE RHI : {sim_metrics['rmse_rhi']:.6g}\n"
        f"Simulation test MAE  fCNM: {sim_metrics['mae_fcnm']:.6g}\n"
        f"Simulation test MAE  RHI : {sim_metrics['mae_rhi']:.6g}"
    )
    simcnn.print_prediction_diagnostics(sim_test_truth, sim_test_pred, config.fcnm_error_floor)

    gaskap_grid_size = input_length if args.grid_size == 0 else args.grid_size
    if gaskap_grid_size != input_length:
        raise ValueError(
            f"GASKAP grid size ({gaskap_grid_size}) must equal simulation input length ({input_length}) "
            "for this CNN head. Use --grid-size 0 or the same value as the simulation channels."
        )

    print("\nLoading GASKAP LMC observations for external prediction")
    gaskap_x_raw, gaskap_targets_original, velocity_grid, gaskap_names = load_gaskap_external_set(
        target_csv=Path(args.gaskap_target_csv).resolve(),
        spectra_dir=Path(args.gaskap_spectra_dir).resolve(),
        grid_size=gaskap_grid_size,
        grid_mode=args.grid_mode,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        fill_value=args.fill_value,
    )
    target_summary_from_array("GASKAP external truth", gaskap_targets_original)
    velocity_grid_path = save_velocity_grid(results_dir, velocity_grid)

    gaskap_x = simcnn.prepare_input_channels(gaskap_x_raw, config.input_mode, config.smooth_window)
    if input_scaler is not None:
        print("Applying simulation train-split input scaler to GASKAP spectra.")
        gaskap_x = input_scaler.transform(gaskap_x)
    else:
        print("Input normalization is OFF; using interpolated GASKAP TB directly.")
    if gaskap_x.shape[-1] != input_length:
        raise ValueError(f"GASKAP input length {gaskap_x.shape[-1]} does not match model input length {input_length}.")
    if (gaskap_x.ndim == 3 and gaskap_x.shape[1] != in_channels) or (gaskap_x.ndim == 2 and in_channels != 1):
        raise ValueError("GASKAP input channel count does not match the simulation-trained model.")

    gaskap_targets_used = simcnn.apply_fcnm_error_floor(
        gaskap_targets_original, config.fcnm_error_floor
    )
    gaskap_indices, gaskap_truth, gaskap_pred = infer_with_targets(
        model,
        gaskap_x,
        gaskap_targets_used,
        device,
        config,
        target_scaler,
        config.batch_size,
        config.num_workers,
    )
    gaskap_metrics = simcnn.regression_metrics(gaskap_truth, gaskap_pred)
    gaskap_predictions_path = save_prediction_csv(
        results_dir / "gaskap_lmc_predictions.csv",
        gaskap_indices,
        gaskap_truth,
        gaskap_pred,
        names=gaskap_names,
        original_truth=gaskap_targets_original,
    )
    gaskap_figure_paths = plot_external_true_vs_pred(
        gaskap_truth,
        gaskap_pred,
        gaskap_metrics,
        gaskap_figs_dir,
        config.fcnm_error_floor,
        prefix="gaskap_lmc",
    )

    print(
        f"\nGASKAP external RMSE f_c : {gaskap_metrics['rmse_fcnm']:.6g}\n"
        f"GASKAP external RMSE R_HI: {gaskap_metrics['rmse_rhi']:.6g}\n"
        f"GASKAP external MAE  f_c : {gaskap_metrics['mae_fcnm']:.6g}\n"
        f"GASKAP external MAE  R_HI: {gaskap_metrics['mae_rhi']:.6g}"
    )
    simcnn.print_prediction_diagnostics(gaskap_truth, gaskap_pred, config.fcnm_error_floor)

    print("\nSummary")
    print(f"  device: {device}")
    print(f"  training data: simulation")
    print(f"  external application data: GASKAP LMC")
    print(f"  simulation subset size: {len(subset_idx)}")
    print(f"  simulation train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"  GASKAP external samples: {len(gaskap_names)}")
    print(f"  best simulation validation loss: {best_val_loss:.6g}")
    print(f"  simulation test RMSE fCNM: {sim_metrics['rmse_fcnm']:.6g}")
    print(f"  simulation test RMSE RHI : {sim_metrics['rmse_rhi']:.6g}")
    print(f"  GASKAP external RMSE f_c : {gaskap_metrics['rmse_fcnm']:.6g}")
    print(f"  GASKAP external RMSE R_HI: {gaskap_metrics['rmse_rhi']:.6g}")
    print(f"  sampled/split simulation CSVs: {subset_path}, {split_path}")
    print(f"  training metrics CSV: {metrics_path}")
    print(f"  simulation prediction CSV: {sim_predictions_path}")
    print(f"  GASKAP prediction CSV: {gaskap_predictions_path}")
    print(f"  GASKAP velocity grid CSV: {velocity_grid_path}")
    print("  simulation figures:")
    for path in sim_figure_paths:
        print(f"    {path}")
    print("  GASKAP figures:")
    for path in gaskap_figure_paths:
        print(f"    {path}")


if __name__ == "__main__":
    main()
