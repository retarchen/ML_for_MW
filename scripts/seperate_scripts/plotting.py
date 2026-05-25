#!/usr/bin/env python3
"""Metrics, CSV saving, prediction diagnostics, and parity plotting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def print_prediction_diagnostics(truth: np.ndarray, pred: np.ndarray, fcnm_error_floor: float = 0.0) -> None:
    print("\nPrediction diagnostics")
    for i, label in enumerate(("fCNM", "RHI")):
        t = truth[:, i]
        p = pred[:, i]
        print(f"  {label} truth: min={t.min():.6g}, max={t.max():.6g}, mean={t.mean():.6g}")
        print(f"  {label} pred : min={p.min():.6g}, max={p.max():.6g}, mean={p.mean():.6g}")
    threshold = max(float(fcnm_error_floor), 0.0)
    zero_like = truth[:, 0] <= threshold
    if np.any(zero_like):
        p0 = pred[zero_like, 0]
        print(
            f"  fCNM true<=floor ({threshold:.6g}): n={int(zero_like.sum())}, "
            f"pred median={np.median(p0):.6g}, p95={np.percentile(p0, 95):.6g}, max={p0.max():.6g}"
        )


def save_training_history(results_dir: Path, history: List[Dict[str, float]]) -> Path:
    path = results_dir / "training_metrics.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def save_predictions(results_dir: Path, sample_indices: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> Path:
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
    pad = 1.0 if math.isclose(lo, hi) and math.isclose(lo, 0.0) else 0.04 * max(hi - lo, abs(lo), 1e-6)
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


def plot_true_vs_pred(
    truth: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, float],
    figs_dir: Path,
    fcnm_error_floor: float = 0.0,
) -> List[Path]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: List[Path] = []

    specs = [
        ("fCNM", 0, metrics["rmse_fcnm"], figs_dir / "true_vs_pred_fcnm.png"),
        ("RHI", 1, metrics["rmse_rhi"], figs_dir / "true_vs_pred_rhi.png"),
    ]
    for label, col, rmse, path in specs:
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
    positive = (truth[:, 1] > 0) & (pred[:, 1] > 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    add_scatter_panel(ax, truth[:, 1], pred[:, 1], "RHI", metrics["rmse_rhi"], fcnm_error_floor)
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
