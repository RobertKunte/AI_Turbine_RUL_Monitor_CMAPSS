# src/eval_utils.py
from __future__ import annotations
import numpy as np

def compute_nasa_score_from_errors(errors: np.ndarray) -> float:
    """
    Compute the NASA PHM08 score from an array of errors e = y_pred - y_true.

    Negative error (early prediction) is penalized mildly,
    positive error (late prediction) is penalized strongly (safety!).
    """
    errors = np.asarray(errors).reshape(-1)
    score = 0.0
    for e in errors:
        if e < 0:
            score += np.exp(-e / 13.0) - 1.0
        else:
            score += np.exp(e / 10.0) - 1.0
    return score


def compute_nasa_score_pairwise(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Convenience wrapper: compute errors and NASA score + some stats.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    errors = y_pred - y_true

    nasa_sum = compute_nasa_score_from_errors(errors)
    return {
        "num_samples": len(errors),
        "mean_error": float(errors.mean()),
        "mean_abs_error": float(np.abs(errors).mean()),
        "nasa_score_sum": float(nasa_sum),
        "nasa_score_mean": float(nasa_sum / len(errors)),
    }
