"""
Centralized metrics functions for RUL prediction evaluation.

This module provides consistent implementations of evaluation metrics,
especially the NASA PHM08 score, used across training, evaluation, and diagnostics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


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

