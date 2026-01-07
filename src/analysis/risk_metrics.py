"""
Overestimation Risk Metrics for RUL Prediction.

This module provides metrics specifically designed to quantify overestimation risk,
which is critical for FD004 safety analysis. Overestimating RUL can lead to
delayed maintenance and potential failures.

Metrics computed:
- over_rate_10/20: Fraction of predictions overestimating by >10/20 cycles
- under_rate_10: Fraction of predictions underestimating by >10 cycles
- p95_over: 95th percentile of overestimation (pred - true)
- max_over: Maximum overestimation
- mean_over_pos: Mean overestimation for positive errors only
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, list, "torch.Tensor"]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert input to numpy array, handling torch tensors."""
    if arr is None:
        return np.array([])
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr).flatten()


def compute_risk_metrics_last(
    y_true_last: ArrayLike,
    y_pred_last: ArrayLike,
) -> Dict[str, float]:
    """Compute overestimation risk metrics for LAST predictions (one per engine).
    
    Args:
        y_true_last: True RUL at last timestep per engine (N,)
        y_pred_last: Predicted RUL at last timestep per engine (N,)
        
    Returns:
        Dictionary with risk metrics (all plain Python floats)
    """
    y_true = _to_numpy(y_true_last)
    y_pred = _to_numpy(y_pred_last)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        warnings.warn("Empty arrays passed to compute_risk_metrics_last")
        return _empty_risk_metrics()
    
    if len(y_true) != len(y_pred):
        warnings.warn(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")
        return _empty_risk_metrics()
    
    # Error = pred - true (positive = overestimation)
    error = y_pred - y_true
    n = len(error)
    
    # Basic stats
    mean_err = float(np.mean(error))
    std_err = float(np.std(error)) if n > 1 else 0.0
    
    # Overestimation rates
    over_rate_10 = float(np.mean(error > 10))
    over_rate_20 = float(np.mean(error > 20))
    under_rate_10 = float(np.mean(error < -10))
    
    # Percentiles
    p95_over = float(np.percentile(error, 95))
    p99_over = float(np.percentile(error, 99)) if n >= 10 else float("nan")
    max_over = float(np.max(error))
    
    # Mean positive error (overestimation only)
    pos_mask = error > 0
    if pos_mask.any():
        mean_over_pos = float(np.mean(error[pos_mask]))
    else:
        mean_over_pos = 0.0
    
    # Absolute error stats
    abs_err = np.abs(error)
    p95_abs_err = float(np.percentile(abs_err, 95)) if n >= 5 else float("nan")
    
    return {
        "mean_err": mean_err,
        "std_err": std_err,
        "over_rate_10": over_rate_10,
        "over_rate_20": over_rate_20,
        "under_rate_10": under_rate_10,
        "p95_over": p95_over,
        "p99_over": p99_over,
        "max_over": max_over,
        "mean_over_pos": mean_over_pos,
        "p95_abs_err": p95_abs_err,
    }


def compute_risk_metrics_all(
    y_true_all: ArrayLike,
    y_pred_all: ArrayLike,
) -> Dict[str, float]:
    """Compute overestimation risk metrics for ALL predictions (all timesteps).
    
    This provides a broader view of model behavior across all prediction points,
    not just the final (most critical) timestep.
    
    Args:
        y_true_all: True RUL values (flattened across timesteps)
        y_pred_all: Predicted RUL values (same shape)
        
    Returns:
        Dictionary with risk metrics (all plain Python floats)
    """
    y_true = _to_numpy(y_true_all)
    y_pred = _to_numpy(y_pred_all)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        warnings.warn("Empty arrays passed to compute_risk_metrics_all")
        return _empty_risk_metrics()
    
    if len(y_true) != len(y_pred):
        warnings.warn(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")
        return _empty_risk_metrics()
    
    # Reuse same logic as _last
    error = y_pred - y_true
    n = len(error)
    
    mean_err = float(np.mean(error))
    std_err = float(np.std(error)) if n > 1 else 0.0
    
    over_rate_10 = float(np.mean(error > 10))
    over_rate_20 = float(np.mean(error > 20))
    under_rate_10 = float(np.mean(error < -10))
    
    p95_over = float(np.percentile(error, 95))
    p99_over = float(np.percentile(error, 99)) if n >= 100 else float("nan")
    max_over = float(np.max(error))
    
    pos_mask = error > 0
    mean_over_pos = float(np.mean(error[pos_mask])) if pos_mask.any() else 0.0
    
    abs_err = np.abs(error)
    p95_abs_err = float(np.percentile(abs_err, 95)) if n >= 5 else float("nan")
    
    return {
        "mean_err": mean_err,
        "std_err": std_err,
        "over_rate_10": over_rate_10,
        "over_rate_20": over_rate_20,
        "under_rate_10": under_rate_10,
        "p95_over": p95_over,
        "p99_over": p99_over,
        "max_over": max_over,
        "mean_over_pos": mean_over_pos,
        "p95_abs_err": p95_abs_err,
    }


def _empty_risk_metrics() -> Dict[str, float]:
    """Return empty/NaN risk metrics dict for error cases."""
    return {
        "mean_err": float("nan"),
        "std_err": float("nan"),
        "over_rate_10": float("nan"),
        "over_rate_20": float("nan"),
        "under_rate_10": float("nan"),
        "p95_over": float("nan"),
        "p99_over": float("nan"),
        "max_over": float("nan"),
        "mean_over_pos": float("nan"),
        "p95_abs_err": float("nan"),
    }


def format_risk_metrics_for_logging(metrics: Dict[str, float]) -> str:
    """Format risk metrics for console logging."""
    lines = [
        f"  over_rate_10: {metrics.get('over_rate_10', float('nan')):.2%}",
        f"  over_rate_20: {metrics.get('over_rate_20', float('nan')):.2%}",
        f"  under_rate_10: {metrics.get('under_rate_10', float('nan')):.2%}",
        f"  p95_over: {metrics.get('p95_over', float('nan')):.2f}",
        f"  max_over: {metrics.get('max_over', float('nan')):.2f}",
        f"  mean_over_pos: {metrics.get('mean_over_pos', float('nan')):.2f}",
    ]
    return "\n".join(lines)
