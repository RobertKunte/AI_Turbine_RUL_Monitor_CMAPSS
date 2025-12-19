"""
Centralized metrics functions for RUL prediction evaluation.

This module provides consistent implementations of evaluation metrics,
especially the NASA PHM08 score, used across training, evaluation, and diagnostics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, Any


def nasa_phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 Score (centralized implementation).
    
    Formula:
        e = y_pred - y_true
        score_i = exp(-e/13) - 1   if e < 0  (pessimistic, underestimation)
                 = exp( e/10) - 1   if e >= 0 (optimistic, overestimation)
        Total score = Sum(score_i)
    
    Args:
        y_true: True RUL values (array)
        y_pred: Predicted RUL values (array)
    
    Returns:
        NASA score (sum over all samples)
    """
    e = y_pred - y_true
    score = np.where(
        e < 0.0,
        np.exp(-e / 13.0) - 1.0,
        np.exp(e / 10.0) - 1.0,
    )
    return float(score.sum())


def nasa_phm_score_single(true_rul: float, pred_rul: float) -> float:
    """
    NASA PHM08 Score for a single engine (EOL prediction).
    
    Convenience wrapper for nasa_phm_score with single values.
    Use this for per-engine EOL metrics to ensure consistency.
    
    Args:
        true_rul: True RUL at EOL (single value)
        pred_rul: Predicted RUL at EOL (single value)
    
    Returns:
        NASA score contribution for this engine
    """
    return nasa_phm_score(np.array([true_rul]), np.array([pred_rul]))


def compute_eol_errors_and_nasa(
    y_true_eol: np.ndarray,
    y_pred_eol: np.ndarray,
    max_rul: Optional[float] = 125.0,
) -> Dict:
    """
    Compute EOL-based error metrics and NASA scores.
    
    This function computes metrics from EOL predictions only (one value per engine),
    ensuring consistency with evaluate_on_test_data. Applies the same RUL capping
    as evaluate_on_test_data.
    
    Args:
        y_true_eol: True RUL at EOL, shape (num_engines,)
        y_pred_eol: Predicted RUL at EOL, shape (num_engines,)
        max_rul: Maximum RUL for capping (default 125). If None, no capping applied.
    
    Returns:
        Dictionary with:
            - errors: array of errors (pred - true) per engine
            - mean_error: mean error (bias)
            - std_error: standard deviation of errors
            - mean_abs_error: mean absolute error
            - median_error: median error
            - mse: mean squared error
            - rmse: root mean squared error
            - mae: mean absolute error
            - r2: R-squared coefficient
            - nasa_scores: array of NASA scores per engine
            - nasa_mean: mean NASA score
            - nasa_sum: sum of NASA scores
            - nasa_median: median NASA score
            - num_engines: number of engines
    """
    # Ensure arrays
    y_true_eol = np.asarray(y_true_eol).flatten()
    y_pred_eol = np.asarray(y_pred_eol).flatten()
    
    if len(y_true_eol) != len(y_pred_eol):
        raise ValueError(
            f"y_true_eol and y_pred_eol must have same length. "
            f"Got {len(y_true_eol)} and {len(y_pred_eol)}"
        )
    
    # Apply RUL capping (same as evaluate_on_test_data)
    if max_rul is not None:
        y_pred_eol = np.minimum(y_pred_eol, max_rul)
        y_pred_eol = np.maximum(y_pred_eol, 0.0)  # Ensure non-negative
        y_true_eol = np.minimum(y_true_eol, max_rul)
    
    # Compute errors (pred - true) - same convention as evaluate_on_test_data
    errors = y_pred_eol - y_true_eol
    
    # Compute standard metrics (same as evaluate_on_test_data)
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    
    # R² calculation (same as evaluate_on_test_data)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true_eol - y_true_eol.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = float(r2)
    
    # Compute NASA scores per engine using capped values
    nasa_scores = np.array([
        nasa_phm_score_single(true_rul, pred_rul)
        for true_rul, pred_rul in zip(y_true_eol, y_pred_eol)
    ])
    
    return {
        "errors": errors,
        "mean_error": bias,  # bias = mean_error
        "std_error": float(np.std(errors)),
        "mean_abs_error": mae,
        "median_error": float(np.median(errors)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "nasa_scores": nasa_scores,
        "nasa_mean": float(np.mean(nasa_scores)),
        "nasa_sum": float(np.sum(nasa_scores)),
        "nasa_median": float(np.median(nasa_scores)),
        "num_engines": len(y_true_eol),
    }


def _clip_y(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    clip: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """
    Clip y_true/y_pred if requested.

    Args:
        y_true: array-like
        y_pred: array-like
        clip: None (no clipping) or (min_val, max_val)

    Returns:
        (y_true_clipped, y_pred_clipped, meta_clip)
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length. Got {yt.shape[0]} and {yp.shape[0]}.")

    if clip is None:
        return yt, yp, None

    lo, hi = float(clip[0]), float(clip[1])
    yt_c = np.clip(yt, lo, hi)
    yp_c = np.clip(yp, lo, hi)
    meta = {"clip_min": lo, "clip_max": hi}
    return yt_c, yp_c, meta


def compute_last_per_unit_metrics(
    unit_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    clip: Optional[Tuple[float, float]] = None,
    include_r2: bool = True,
    include_nasa: bool = True,
) -> Dict[str, Any]:
    """
    Compute LAST-AVAILABLE-per-unit metrics (truncated-aware, literature-style).

    Definition:
      For each unit_id, take the *last occurrence* in the provided arrays
      (stable last index by scan order), then compute metrics over these per-unit
      last samples.

    Args:
        unit_ids: array of unit ids aligned with y_true/y_pred (shape (N,))
        y_true: true targets (shape (N,))
        y_pred: predictions (shape (N,))
        clip: optional (min,max) applied to y_true and y_pred before metrics
        include_r2: include r2_last
        include_nasa: include nasa_last_sum/nasa_last_mean

    Returns:
        Dict with keys:
          rmse_last, mae_last, bias_last, r2_last (optional),
          nasa_last_sum, nasa_last_mean (optional),
          n_units, max_rul_used (if clip provided), note_last_definition
    """
    u = np.asarray(unit_ids).reshape(-1)
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if u.shape[0] != yt.shape[0] or yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"unit_ids, y_true, y_pred must have same length. "
            f"Got {u.shape[0]}, {yt.shape[0]}, {yp.shape[0]}."
        )

    # stable last index per unit: last occurrence wins
    last_idx: Dict[int, int] = {}
    for i, uid in enumerate(u.tolist()):
        last_idx[int(uid)] = int(i)
    idxs = np.array(sorted(last_idx.values()), dtype=np.int64)

    yt_last = yt[idxs]
    yp_last = yp[idxs]
    yt_last, yp_last, clip_meta = _clip_y(yt_last, yp_last, clip=clip)

    # Base metrics
    errors = yp_last - yt_last
    mse = float(np.mean(errors ** 2)) if errors.size else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
    bias = float(np.mean(errors)) if errors.size else 0.0

    out: Dict[str, Any] = {
        "rmse_last": rmse,
        "mae_last": mae,
        "bias_last": bias,
        "n_units": int(len(last_idx)),
        "note_last_definition": "LAST_AVAILABLE_PER_UNIT (truncated-aware)",
    }
    if clip_meta is not None:
        out["max_rul_used"] = float(clip_meta["clip_max"])

    if include_r2:
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((yt_last - float(np.mean(yt_last))) ** 2)) if yt_last.size else 0.0
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        out["r2_last"] = float(r2)

    if include_nasa:
        # Reuse the centralized NASA implementation via compute_eol_errors_and_nasa
        # If clip is provided, we map it to "max_rul" behavior (lower bound = 0.0).
        max_rul = None
        if clip is not None:
            # compute_eol_errors_and_nasa clamps y_pred to [0,max_rul] and y_true to [0,max_rul]
            max_rul = float(clip[1])
        nasa_stats = compute_eol_errors_and_nasa(yt_last, yp_last, max_rul=max_rul)
        out["nasa_last_sum"] = float(nasa_stats["nasa_sum"])
        out["nasa_last_mean"] = float(nasa_stats["nasa_mean"])

    return out


def compute_all_samples_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    unit_ids: Optional[np.ndarray] = None,
    clip: Optional[Tuple[float, float]] = None,
    include_r2: bool = True,
    include_nasa: bool = True,
) -> Dict[str, Any]:
    """
    Compute ALL-SAMPLES metrics (all windows / all timepoints).

    Args:
        y_true: shape (N,)
        y_pred: shape (N,)
        unit_ids: optional, for debug only (not required for the metrics)
        clip: optional (min,max) applied to y_true and y_pred before metrics
        include_r2: include r2_all
        include_nasa: include nasa_all_sum/nasa_all_mean (interpret as asymmetric cost over all samples)

    Returns:
        Dict with keys:
          rmse_all, mae_all, bias_all, r2_all (optional),
          nasa_all_sum, nasa_all_mean (optional),
          n_samples_all, n_units (if unit_ids provided), max_rul_used (if clip provided)
    """
    yt, yp, clip_meta = _clip_y(y_true, y_pred, clip=clip)
    errors = yp - yt
    mse = float(np.mean(errors ** 2)) if errors.size else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
    bias = float(np.mean(errors)) if errors.size else 0.0

    out: Dict[str, Any] = {
        "rmse_all": rmse,
        "mae_all": mae,
        "bias_all": bias,
        "n_samples_all": int(yt.size),
    }
    if unit_ids is not None:
        try:
            out["n_units"] = int(len(np.unique(np.asarray(unit_ids).reshape(-1))))
        except Exception:
            pass
    if clip_meta is not None:
        out["max_rul_used"] = float(clip_meta["clip_max"])

    if include_r2:
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2)) if yt.size else 0.0
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        out["r2_all"] = float(r2)

    if include_nasa:
        max_rul = None
        if clip is not None:
            max_rul = float(clip[1])
        nasa_stats = compute_eol_errors_and_nasa(yt, yp, max_rul=max_rul)
        out["nasa_all_sum"] = float(nasa_stats["nasa_sum"])
        out["nasa_all_mean"] = float(nasa_stats["nasa_mean"])

    return out

