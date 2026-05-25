#!/usr/bin/env python3
"""Model definition, training loop, checkpointing, and prediction collection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from .data_processing import (
        Standardization,
        apply_physical_prediction_constraints,
        inverse_transform_rhi_target,
    )
except ImportError:
    from data_processing import Standardization, apply_physical_prediction_constraints, inverse_transform_rhi_target


class TPCNetInspiredCNN(nn.Module):
    """Eight-layer no-pooling 1D CNN for spectra regression."""

    def __init__(self, input_length: int, in_channels: int = 1, out_dim: int = 2):
        super().__init__()
        channels = [64, 56, 48, 40, 32, 24, 16, 8]
        kernels = [7, 33, 7, 33, 7, 33, 7, 33]

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out, kernel_size in zip(channels, kernels):
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] * input_length, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).flatten(start_dim=1))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def batch_to_device(batch, device: torch.device):
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

    return total_loss / max(n_seen, 1), np.sqrt(squared_error_sum / max(n_seen, 1))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    target_scaler: Optional[Standardization],
    rhi_target_transform: str,
    apply_physical_constraints: bool,
    fcnm_error_floor: float,
    snap_fcnm_below_floor: bool,
    results_dir: Path,
    config_dict: Dict,
) -> Tuple[nn.Module, List[Dict[str, float]], float]:
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, patience // 3))

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    metrics_path = results_dir / "training_metrics.csv"
    checkpoint_path = results_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            target_scaler,
            rhi_target_transform,
            apply_physical_constraints,
            fcnm_error_floor,
            snap_fcnm_below_floor,
        )
        with torch.no_grad():
            val_loss, val_rmse = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                None,
                target_scaler,
                rhi_target_transform,
                apply_physical_constraints,
                fcnm_error_floor,
                snap_fcnm_below_floor,
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
            f"Epoch {epoch:03d} | train_loss={train_loss:.6g} | val_loss={val_loss:.6g} | "
            f"val_RMSE_fCNM={val_rmse[0]:.6g} | val_RMSE_RHI={val_rmse[1]:.6g} | lr={current_lr:.3g}",
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
                    "config": config_dict,
                    "target_scaler_mean": None if target_scaler is None else target_scaler.mean.astype(np.float32),
                    "target_scaler_std": None if target_scaler is None else target_scaler.std.astype(np.float32),
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
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
