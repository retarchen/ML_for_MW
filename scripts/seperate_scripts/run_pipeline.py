#!/usr/bin/env python3
"""End-to-end runner combining data processing, model training, and plotting."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from .data_processing import (
        create_dataloaders,
        create_splits,
        apply_fcnm_error_floor,
        list_spectra_files,
        load_spectra_files,
        load_targets,
        maybe_normalize_inputs,
        maybe_normalize_targets,
        prepare_input_channels,
        save_indices,
        select_random_subset,
        set_random_seed,
        target_indices_from_spectra_files,
        target_summary,
        transform_rhi_target,
    )
    from .model_training import TPCNetInspiredCNN, collect_predictions, resolve_device, train_model
    from .plotting import (
        plot_true_vs_pred,
        print_prediction_diagnostics,
        regression_metrics,
        save_predictions,
        save_training_history,
    )
except ImportError:
    from data_processing import (
        create_dataloaders,
        create_splits,
        apply_fcnm_error_floor,
        list_spectra_files,
        load_spectra_files,
        load_targets,
        maybe_normalize_inputs,
        maybe_normalize_targets,
        prepare_input_channels,
        save_indices,
        select_random_subset,
        set_random_seed,
        target_indices_from_spectra_files,
        target_summary,
        transform_rhi_target,
    )
    from model_training import TPCNetInspiredCNN, collect_predictions, resolve_device, train_model
    from plotting import (
        plot_true_vs_pred,
        print_prediction_diagnostics,
        regression_metrics,
        save_predictions,
        save_training_history,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATAPATH_BASE = "/mnt/c/Users/retar/Desktop/research/ML/data/MW"
FITS_PATH = os.path.join(DATAPATH_BASE, "fcnm_RHI_z.fits")
CSV_DIR = os.path.join(DATAPATH_BASE, "syn_HI_spec_z")


@dataclass
class PipelineConfig:
    fits_path: str
    csv_dir: str
    output_root: Path
    run_name: str
    tb_column: int
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


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Train/evaluate modular HI spectra CNN.")
    parser.add_argument("--fits-path", default=FITS_PATH)
    parser.add_argument("--csv-dir", default=CSV_DIR)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--tb-column", type=int, default=3, help="1=noisy TB, 3=no-noise TB in current CSVs.")
    parser.add_argument("--subset-size", type=int, default=20000, help="Use -1 for all usable spectra.")
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
    parser.add_argument("--apply-physical-constraints", dest="apply_physical_constraints", action="store_true")
    parser.add_argument("--no-physical-constraints", dest="apply_physical_constraints", action="store_false")
    parser.set_defaults(apply_physical_constraints=True)
    parser.add_argument("--snap-fcnm-below-floor", dest="snap_fcnm_below_floor", action="store_true")
    parser.add_argument("--no-snap-fcnm-below-floor", dest="snap_fcnm_below_floor", action="store_false")
    parser.set_defaults(snap_fcnm_below_floor=True)
    parser.add_argument("--input-mode", choices=("raw", "smooth", "raw_plus_smooth"), default="raw")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    return PipelineConfig(
        fits_path=args.fits_path,
        csv_dir=args.csv_dir,
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
    print(f"Listing spectra in: {config.csv_dir}")
    spectra_files = list_spectra_files(config.csv_dir)
    print(f"Found {len(spectra_files)} spectra files")
    spectra_target_indices = target_indices_from_spectra_files(spectra_files, target_shape)
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

    print(f"\nSplit sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    target_summary("selected subset", fcnm, rhi, subset_target_idx)
    target_summary("train", fcnm, rhi, train_target_idx)
    target_summary("validation", fcnm, rhi, val_target_idx)
    target_summary("test", fcnm, rhi, test_target_idx)
    subset_path, split_path = save_indices(results_dir, subset_target_idx, train_target_idx, val_target_idx, test_target_idx)

    selected_files = [spectra_files[i] for i in subset_idx]
    print(f"\nReading selected spectra: {len(selected_files)} files, tb_column={config.tb_column}")
    tb_subset = load_spectra_files(selected_files, tb_column=config.tb_column)
    if tb_subset.ndim != 2:
        raise ValueError(f"Expected selected TB shape (N, L), got {tb_subset.shape}")
    print(f"Selected TB loaded: {tb_subset.shape}")
    tb_subset = prepare_input_channels(tb_subset, config.input_mode, config.smooth_window)
    print(f"Model input array prepared: {tb_subset.shape}")

    raw_targets_subset = np.stack([fcnm[subset_target_idx], rhi[subset_target_idx]], axis=1).astype(np.float32)
    raw_targets_subset = apply_fcnm_error_floor(raw_targets_subset, config.fcnm_error_floor)
    transformed_targets_subset = transform_rhi_target(raw_targets_subset, config.rhi_target_transform)
    tb_subset, _ = maybe_normalize_inputs(tb_subset, subset_idx, train_idx, config.normalize_inputs)
    targets_for_training, target_scaler = maybe_normalize_targets(
        transformed_targets_subset, subset_idx, train_idx, config.normalize_targets
    )

    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader = create_dataloaders(
        tb_subset,
        targets_for_training,
        raw_targets_subset,
        subset_target_idx,
        subset_idx,
        train_idx,
        val_idx,
        test_idx,
        config.batch_size,
        config.num_workers,
        pin_memory,
        config.use_imbalance_sampler,
        config.imbalance_bins,
        config.rhi_tail_loss_weight,
        config.fcnm_error_floor,
        config.fcnm_zero_loss_weight,
    )

    input_length = tb_subset.shape[-1]
    in_channels = tb_subset.shape[1] if tb_subset.ndim == 3 else 1
    model = TPCNetInspiredCNN(input_length=input_length, in_channels=in_channels).to(device)
    print(model)

    model, history, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.epochs,
        patience=config.patience,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
        target_scaler=target_scaler,
        rhi_target_transform=config.rhi_target_transform,
        apply_physical_constraints=config.apply_physical_constraints,
        fcnm_error_floor=config.fcnm_error_floor,
        snap_fcnm_below_floor=config.snap_fcnm_below_floor,
        results_dir=results_dir,
        config_dict=asdict(config),
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
    print(f"  sampled indices CSV: {subset_path}")
    print(f"  split indices CSV: {split_path}")
    print(f"  training metrics CSV: {metrics_path}")
    print(f"  prediction CSV: {predictions_path}")
    print("  figures:")
    for path in figure_paths:
        print(f"    {path}")


if __name__ == "__main__":
    main()
