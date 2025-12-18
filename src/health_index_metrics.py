"""
Health Index metrics and analysis functions.
Includes detection horizon analysis for HI threshold crossings.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import pandas as pd


def compute_detection_horizon_per_engine(
    hi: np.ndarray,
    rul: np.ndarray,
    unit_ids: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[float, List[float]]:
    """
    Compute detection horizon (RUL at first HI crossing) for each engine.
    
    The detection horizon is the RUL remaining when the Health Index first
    crosses below a given threshold. This is computed using a running-minimum
    of HI to ensure monotonicity.
    
    Args:
        hi: Array of predicted health index values (aligned with rul and unit_ids)
        rul: Array of true RUL values (same shape as hi)
        unit_ids: Array of engine/unit ids aligned with hi/rul
        thresholds: List of HI thresholds, e.g. [0.8, 0.5, 0.2, 0.1]
    
    Returns:
        Dictionary mapping threshold -> list of detection RULs (one per engine
        where crossing occurs). RUL is taken at the FIRST time the *running-min*
        HI falls below the threshold.
    """
    hi = np.asarray(hi, dtype=float)
    rul = np.asarray(rul, dtype=float)
    unit_ids = np.asarray(unit_ids)
    thresholds = list(thresholds)
    
    # Group by engine
    unique_units = np.unique(unit_ids)
    results: Dict[float, List[float]] = {th: [] for th in thresholds}
    
    for uid in unique_units:
        mask = unit_ids == uid
        hi_seq = hi[mask]
        rul_seq = rul[mask]
        
        # Sort by cycle / RUL if not already sorted (ascending RUL)
        # Assume RUL decreases with time; sort by RUL descending -> time ascending
        # (earlier cycles have higher RUL)
        order = np.argsort(-rul_seq)  # descending RUL order
        hi_seq = hi_seq[order]
        rul_seq = rul_seq[order]
        
        # Running-min smoothing (monotone non-increasing HI)
        # Apply running minimum from end to beginning (towards EOL)
        run_min = hi_seq.copy()
        for t in range(len(run_min) - 2, -1, -1):
            run_min[t] = min(run_min[t], run_min[t + 1])
        
        # For each threshold, find first crossing (from healthy to degraded)
        # We want to find when HI crosses FROM ABOVE TO BELOW the threshold
        for th in thresholds:
            # Find where run_min crosses below threshold
            # Start from the beginning (early cycles, high RUL) and find first crossing
            crossing_idx = None
            
            # Check if HI ever goes above this threshold in the sequence
            hi_ever_above = hi_seq.max() > th
            
            # Find the first time point where run_min is below threshold
            below_indices = np.where(run_min <= th)[0]
            
            if len(below_indices) == 0:
                # HI never goes below threshold in this sequence - skip
                continue
            
            first_below_idx = below_indices[0]
            
            if hi_ever_above:
                # HI goes above threshold at some point - find the crossing
                # Check if there's a clear crossing (was above, now below)
                if first_below_idx > 0 and run_min[first_below_idx - 1] > th:
                    # Clear crossing: was above, now below
                    crossing_idx = first_below_idx
                elif first_below_idx == 0:
                    # HI starts at or below threshold, but we know it goes above later
                    # This means the crossing happened before this sequence started.
                    # Use the first point as a conservative estimate.
                    crossing_idx = 0
                else:
                    # No clear crossing found, but HI is below threshold
                    # Use the first point where it's below as a conservative estimate
                    crossing_idx = first_below_idx
            else:
                # HI never goes above threshold in this sequence
                # This means either:
                # 1. HI was always below (crossing happened before sequence)
                # 2. Or this is a partial sequence and we don't have the full trajectory
                # Use the first point where HI is below as a conservative estimate
                crossing_idx = first_below_idx
            
            if crossing_idx is not None:
                # Detection horizon: how many cycles are left at that time?
                det_rul = float(rul_seq[crossing_idx])
                results[th].append(det_rul)
    
    return results


def summarize_detection_horizon(
    horizons: Dict[float, List[float]],
) -> pd.DataFrame:
    """
    Convert detection horizon dict into a summary DataFrame with statistics.
    
    Args:
        horizons: Dictionary mapping threshold -> list of detection RULs
    
    Returns:
        DataFrame with columns: threshold, num_engines, mean_rul, std_rul, min_rul, max_rul
    """
    records = []
    
    for th, values in horizons.items():
        if len(values) == 0:
            records.append(
                {
                    "threshold": th,
                    "num_engines": 0,
                    "mean_rul": np.nan,
                    "std_rul": np.nan,
                    "min_rul": np.nan,
                    "max_rul": np.nan,
                }
            )
            continue
        
        arr = np.asarray(values, dtype=float)
        records.append(
            {
                "threshold": th,
                "num_engines": len(arr),
                "mean_rul": float(np.mean(arr)),
                "std_rul": float(np.std(arr)),
                "min_rul": float(np.min(arr)),
                "max_rul": float(np.max(arr)),
            }
        )
    
    df = pd.DataFrame.from_records(records).sort_values("threshold", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Additional HI/RUL dynamics KPIs (Stage-0 diagnostics)
# ---------------------------------------------------------------------------

def hi_plateau_ratio(hi: np.ndarray, threshold: float = 0.98) -> float:
    """
    Fraction of timesteps where HI is in the (near-)healthy plateau.

    Args:
        hi: HI trajectory (shape (T,))
        threshold: plateau threshold (default 0.98)
    """
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if hi.size == 0:
        return float("nan")
    return float(np.mean(hi > float(threshold)))


def hi_onset_cycle(
    cycles: np.ndarray,
    hi: np.ndarray,
    threshold: float = 0.95,
) -> float:
    """
    First cycle index where HI falls below a threshold (onset of degradation).

    Returns NaN if HI never drops below threshold.
    """
    cycles = np.asarray(cycles, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if cycles.size == 0 or hi.size == 0 or cycles.size != hi.size:
        return float("nan")
    idx = np.where(hi < float(threshold))[0]
    if idx.size == 0:
        return float("nan")
    return float(cycles[int(idx[0])])


def hi_curvature(hi: np.ndarray, abs_mode: bool = True) -> float:
    """
    Mean curvature magnitude of an HI trajectory via second-order differences.

    curvature[t] = HI[t+1] - 2*HI[t] + HI[t-1]
    Returns mean(|curvature|) by default, or mean(curvature^2) if abs_mode=False.
    """
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if hi.size < 3:
        return float("nan")
    curv = hi[2:] - 2.0 * hi[1:-1] + hi[:-2]
    if abs_mode:
        return float(np.mean(np.abs(curv)))
    return float(np.mean(curv ** 2))


def rul_saturation_rate(
    rul_pred: np.ndarray,
    cap: float,
    delta: float = 2.0,
) -> float:
    """
    Fraction of predictions in [cap-delta, cap].

    This is intended to quantify "RUL saturates at cap" behavior.
    """
    rul_pred = np.asarray(rul_pred, dtype=float).reshape(-1)
    if rul_pred.size == 0:
        return float("nan")
    cap = float(cap)
    delta = float(delta)
    lo = cap - delta
    return float(np.mean((rul_pred >= lo) & (rul_pred <= cap)))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit y = a*x + b (least squares) and return slope a.
    Returns NaN if insufficient data or ill-conditioned.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    # Center to improve numerical stability
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 ** 2))
    if denom <= 1e-12:
        return float("nan")
    slope = float(np.sum(x0 * (y - float(np.mean(y)))) / denom)
    return slope


def rul_slope_error(
    cycles: np.ndarray,
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    start_frac: float,
    end_frac: float,
) -> Dict[str, float]:
    """
    Compare fitted linear slopes of predicted vs. true RUL in a lifecycle window.

    Window is defined over cycle fraction in [0,1] relative to last observed cycle.
    Returns dict with true_slope, pred_slope, abs_error.
    """
    cycles = np.asarray(cycles, dtype=float).reshape(-1)
    rul_true = np.asarray(rul_true, dtype=float).reshape(-1)
    rul_pred = np.asarray(rul_pred, dtype=float).reshape(-1)
    if cycles.size < 2 or cycles.size != rul_true.size or cycles.size != rul_pred.size:
        return {"true_slope": float("nan"), "pred_slope": float("nan"), "abs_error": float("nan")}

    last = float(np.max(cycles))
    if last <= 0:
        return {"true_slope": float("nan"), "pred_slope": float("nan"), "abs_error": float("nan")}

    frac = cycles / last
    m = (frac >= float(start_frac)) & (frac <= float(end_frac))
    if np.sum(m) < 2:
        return {"true_slope": float("nan"), "pred_slope": float("nan"), "abs_error": float("nan")}

    x = cycles[m]
    slope_true = linear_slope(x, rul_true[m])
    slope_pred = linear_slope(x, rul_pred[m])
    if np.isnan(slope_true) or np.isnan(slope_pred):
        return {"true_slope": slope_true, "pred_slope": slope_pred, "abs_error": float("nan")}
    return {"true_slope": slope_true, "pred_slope": slope_pred, "abs_error": float(abs(slope_pred - slope_true))}


def reconstruct_rul_trajectory_from_last(
    rul_last: float,
    cycle_last: float,
    cycles: np.ndarray,
    cap: float | None = None,
) -> np.ndarray:
    """
    Reconstruct a right-censored RUL trajectory when only the last-observed-cycle RUL is known.

    NASA CMAPSS test labels provide RUL at the last observed cycle (right-censored).
    For earlier cycles t, the remaining RUL increases by the number of cycles
    between t and the last observed cycle:

        RUL(t) = RUL_last + (cycle_last - cycle_t)

    Args:
        rul_last: True remaining RUL at the last observed cycle (EOL_observed), in cycles.
        cycle_last: The last observed cycle index for the engine.
        cycles: Vector of cycle indices to reconstruct RUL for (same units as cycle_last).
        cap: Optional cap (e.g. 125). If provided, output is clipped to <= cap.

    Returns:
        Reconstructed RUL trajectory aligned with `cycles` (float array).
    """
    cycles_arr = np.asarray(cycles, dtype=float).reshape(-1)
    rul_last_f = float(rul_last)
    cycle_last_f = float(cycle_last)

    # Right-censored reconstruction
    rul = rul_last_f + (cycle_last_f - cycles_arr)

    # Safety: RUL must be non-negative
    rul = np.maximum(rul, 0.0)

    # Optional cap (paper-style capped RUL)
    if cap is not None:
        rul = np.minimum(rul, float(cap))

    return rul

