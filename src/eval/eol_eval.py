from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.metrics import compute_eol_errors_and_nasa


def evaluate_eol_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_rul: float = 125.0,
    clip_y_true: bool = False,
    clip_y_pred: bool = True,
    log_prefix: str = "[eval]",
) -> Dict[str, Any]:
    """
    Single source of truth for scalar EOL/RUL metrics (RMSE/MAE/Bias/R2 + NASA).

    IMPORTANT:
    - This expects both y_true and y_pred to represent the *same scalar target* in cycles.
    - Clipping is explicit via clip_y_true / clip_y_pred.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)

    if clip_y_true:
        yt = np.clip(yt, 0.0, float(max_rul))
    if clip_y_pred:
        yp = np.clip(yp, 0.0, float(max_rul))

    # Range log (helps catch unit mismatches early)
    try:
        print(
            f"{log_prefix} y_true: min={float(np.min(yt)):.2f} max={float(np.max(yt)):.2f} "
            f"| y_pred: min={float(np.min(yp)):.2f} max={float(np.max(yp)):.2f} "
            f"(max_rul={float(max_rul):.2f}, clip_true={clip_y_true}, clip_pred={clip_y_pred})"
        )
    except Exception:
        pass

    errors = yp - yt
    mse = float(np.mean(errors**2)) if errors.size else 0.0
    rmse = float(np.sqrt(mse)) if errors.size else 0.0
    mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
    bias = float(np.mean(errors)) if errors.size else 0.0

    # R²
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    nasa_stats = compute_eol_errors_and_nasa(yt, yp, max_rul=float(max_rul))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R2": r2,
        "nasa_score_sum": float(nasa_stats["nasa_sum"]),
        "nasa_score_mean": float(nasa_stats["nasa_mean"]),
        "nasa_scores": np.asarray(nasa_stats.get("nasa_scores", []), dtype=float),
        "y_true": yt,
        "y_pred": yp,
    }

