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

