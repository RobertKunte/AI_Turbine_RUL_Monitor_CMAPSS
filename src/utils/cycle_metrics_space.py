"""
Cycle Metrics Space Utilities.

Provides space-aware normalization/denormalization for cycle branch metrics.
This ensures consistent comparison in either scaled or raw space.

Space Contract:
- "scaled": Predictions and targets are in StandardScaler space (mean≈0, std≈1)
- "raw": Predictions and targets are in physical units (°R, PSIA)

Usage:
    from src.utils.cycle_metrics_space import normalize_cycle_pred, compute_cycle_metrics

    # Normalize raw predictions to scaled space
    pred_scaled = normalize_cycle_pred(pred_raw, cond_ids, scaler_stats)

    # Compute metrics in specified space
    metrics = compute_cycle_metrics(pred, target, sensor_names, space="scaled")
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def normalize_cycle_pred(
    pred_raw: np.ndarray,
    cond_ids: np.ndarray,
    scaler_stats: Tuple[np.ndarray, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize raw cycle predictions to scaled space using condition-wise stats.
    
    Args:
        pred_raw: Raw predictions (N, T, n_targets) or (N, n_targets)
        cond_ids: Condition IDs (N,) for each sample
        scaler_stats: Tuple of (mean, std) arrays, each (num_conditions, n_targets)
        eps: Small value for numerical stability
        
    Returns:
        pred_scaled: Normalized predictions in same shape as input
    """
    mean, std = scaler_stats
    
    # Get per-sample mean/std based on condition
    sample_mean = mean[cond_ids]  # (N, n_targets)
    sample_std = std[cond_ids]    # (N, n_targets)
    
    # Handle sequence dimension
    if pred_raw.ndim == 3:
        # (N, T, n_targets) - broadcast over T
        sample_mean = sample_mean[:, np.newaxis, :]  # (N, 1, n_targets)
        sample_std = sample_std[:, np.newaxis, :]
    
    return (pred_raw - sample_mean) / (sample_std + eps)


def denormalize_cycle_target(
    target_scaled: np.ndarray,
    cond_ids: np.ndarray,
    scaler_stats: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Denormalize scaled cycle targets back to raw space.
    
    Args:
        target_scaled: Scaled targets (N, T, n_targets) or (N, n_targets)
        cond_ids: Condition IDs (N,) for each sample
        scaler_stats: Tuple of (mean, std) arrays, each (num_conditions, n_targets)
        
    Returns:
        target_raw: Denormalized targets in same shape as input
    """
    mean, std = scaler_stats
    
    sample_mean = mean[cond_ids]
    sample_std = std[cond_ids]
    
    if target_scaled.ndim == 3:
        sample_mean = sample_mean[:, np.newaxis, :]
        sample_std = sample_std[:, np.newaxis, :]
    
    return target_scaled * sample_std + sample_mean


def compute_cycle_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    sensor_names: List[str],
    space: Literal["scaled", "raw"] = "scaled",
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-sensor cycle metrics with explicit space annotation.
    
    SPACE CONTRACT: pred and target MUST be in the same space.
    If space="scaled", both should be normalized. If space="raw", both should be raw.
    
    Args:
        pred: Predictions (N, T, n_targets) or (N*T, n_targets)
        target: Targets, same shape as pred
        sensor_names: List of sensor names matching n_targets dimension
        space: "scaled" or "raw" - annotates the reported metrics
        mask: Optional valid mask (N, T) to exclude padded timesteps
        
    Returns:
        Dict with per-sensor metrics, each containing:
        - MSE, MAE, RMSE, std, mean_residual
        - space: annotation of which space these metrics are in
    """
    # Flatten to (N*T, n_targets) for computation
    if pred.ndim == 3:
        N, T, D = pred.shape
        pred_flat = pred.reshape(-1, D)
        target_flat = target.reshape(-1, D)
        mask_flat = mask.reshape(-1) if mask is not None else None
    else:
        pred_flat = pred
        target_flat = target
        mask_flat = mask.reshape(-1) if mask is not None and mask.ndim > 1 else mask
    
    result = {}
    
    for i, name in enumerate(sensor_names):
        if i >= pred_flat.shape[1]:
            break
            
        p = pred_flat[:, i]
        t = target_flat[:, i]
        
        # Apply mask if provided
        if mask_flat is not None:
            valid_idx = mask_flat > 0.5
            p = p[valid_idx]
            t = t[valid_idx]
        
        if len(p) == 0:
            result[name] = {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0, "std": 0.0, "space": space}
            continue
        
        residuals = p - t
        
        result[name] = {
            "MSE": float(np.mean(residuals ** 2)),
            "MAE": float(np.mean(np.abs(residuals))),
            "RMSE": float(np.sqrt(np.mean(residuals ** 2))),
            "std": float(np.std(residuals)),
            "mean_residual": float(np.mean(residuals)),
            "space": space,
        }
    
    # Overall metrics
    if pred_flat.shape[1] > 0:
        all_residuals = pred_flat - target_flat
        if mask_flat is not None:
            valid_idx = mask_flat > 0.5
            all_residuals = all_residuals[valid_idx]
        
        result["_overall"] = {
            "MSE": float(np.mean(all_residuals ** 2)),
            "MAE": float(np.mean(np.abs(all_residuals))),
            "RMSE": float(np.sqrt(np.mean(all_residuals ** 2))),
            "space": space,
        }
    
    return result


def format_cycle_metrics_for_print(
    metrics: Dict[str, Dict[str, Any]],
    sensor_names: Optional[List[str]] = None,
) -> str:
    """Format cycle metrics dict for console printing.
    
    Returns string like: "RMSE_scaled: T24: 0.0812, T30: 0.0945, P30: 0.0723, T50: 0.1234"
    """
    if not metrics:
        return "Cycle metrics: N/A"
    
    space = metrics.get("_overall", {}).get("space", "unknown")
    
    if sensor_names is None:
        sensor_names = [k for k in metrics.keys() if not k.startswith("_")]
    
    parts = []
    for name in sensor_names:
        if name in metrics:
            rmse = metrics[name].get("RMSE", 0.0)
            parts.append(f"{name}: {rmse:.4f}")
    
    return f"RMSE_{space}: " + ", ".join(parts)


def validate_cycle_space_config(
    space: str,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]],
    cond_ids: Optional[np.ndarray],
    context: str = "cycle_metrics",
) -> None:
    """Fail-fast validation for space configuration.
    
    Raises:
        ValueError: If space="scaled" but missing scaler_stats or cond_ids
    """
    if space == "scaled":
        missing = []
        if scaler_stats is None:
            missing.append("scaler_stats")
        if cond_ids is None:
            missing.append("cond_ids")
        
        if missing:
            raise ValueError(
                f"[{context}] space='scaled' but missing: {missing}. "
                f"Either provide scaler_stats and cond_ids for normalization, "
                f"or set space='raw' if targets are in raw physical units."
            )
