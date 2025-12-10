from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration dataclass (optional convenience wrapper)
# ---------------------------------------------------------------------------


@dataclass
class HIPhysV3Config:
    """
    Hyperparameters for HI_phys_v3.1 based on residual damage.

    The defaults follow the spec in the project documentation:

      - 13 residual sensors (S2, S3, S4, S7, S11, S12, S13, S14,
        S15, S16, S17, S20, S21)
      - causal median filter with window=5
      - global robust scaling per sensor (median + MAD)
      - per-step damage increment:
            delta_D = alpha_base + alpha_res * mean_s( u_hat_s ** p )
        where u_hat_s are clipped, normalised residual magnitudes
      - global EOL reference via high quantile (q_eol_ref)
    """

    window: int = 5
    alpha_base: float = 0.001
    alpha_res: float = 0.003
    u_max: float = 4.0
    p: float = 1.6
    q_eol_ref: float = 0.98
    eps: float = 1e-8


# Default damage sensor IDs (NASA C-MAPSS indices)
DAMAGE_SENSOR_IDS: Tuple[int, ...] = (
    2,
    3,
    4,
    7,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    20,
    21,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def causal_median_filter_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply a simple causal median filter to a 1D array.

    For each index t, the output uses only values from [t - k, ..., t],
    where k = (window - 1) // 2. At the sequence start, the window is
    truncated as needed.

    NaNs are ignored via np.nanmedian; if all values in the window are NaN,
    the output is NaN at that position.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x.copy()

    if window <= 1:
        return x.copy()

    k = (window - 1) // 2
    y = np.empty_like(x, dtype=float)

    for t in range(n):
        start = max(0, t - k)
        window_vals = x[start : t + 1]
        if window_vals.size == 0 or np.all(np.isnan(window_vals)):
            y[t] = np.nan
        else:
            y[t] = np.nanmedian(window_vals)

    return y


def _find_damage_residual_columns(
    df: pd.DataFrame,
    *,
    prefix: str = "Resid_",
    sensor_ids: Sequence[int] = DAMAGE_SENSOR_IDS,
) -> List[str]:
    """
    Resolve residual column names for the damage sensor set.

    The HealthyTwinRegressor uses original sensor columns as names, e.g.:
        - Sensor2, Sensor3, ...
    and creates residuals as:
        - Resid_Sensor2, Resid_Sensor3, ...

    This helper tries a small set of naming conventions per sensor id
    and keeps only those columns that actually exist in df.
    """
    residual_cols: List[str] = []

    for sid in sensor_ids:
        candidates = [
            f"{prefix}Sensor{sid}",  # Resid_Sensor2, ...
            f"{prefix}S{sid}",  # Resid_S2, ... (fallback if used)
        ]
        chosen: Optional[str] = None
        for name in candidates:
            if name in df.columns:
                chosen = name
                break
        if chosen is not None:
            residual_cols.append(chosen)

    if not residual_cols:
        raise ValueError(
            "[HI_phys_v3] No residual columns for damage set found. "
            f"Expected something like 'Resid_Sensor2', got only: "
            f"{[c for c in df.columns if c.startswith(prefix)][:10]}..."
        )

    return residual_cols


def _compute_robust_stats_per_sensor(
    df: pd.DataFrame,
    residual_cols: Sequence[str],
    eps: float = 1e-8,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute robust (median, MAD-based scale) statistics per residual sensor.

    Returns:
        medians: dict[col] -> median
        scales:  dict[col] -> robust scale (> 0)
    """
    medians: Dict[str, float] = {}
    scales: Dict[str, float] = {}

    for col in residual_cols:
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]  # drop NaNs / infs
        if vals.size == 0:
            medians[col] = 0.0
            scales[col] = 1.0
            continue

        m = float(np.median(vals))
        mad = float(np.median(np.abs(vals - m)))
        if mad < eps:
            # Fall back to standard deviation; if still tiny, use 1.0
            std = float(np.std(vals))
            scale = std if std > eps else 1.0
        else:
            # MAD -> approximate std assuming Gaussian (1.4826 factor)
            scale = 1.4826 * mad

        medians[col] = m
        scales[col] = scale

    return medians, scales


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_hi_phys_v3_from_residuals(
    df_residuals: pd.DataFrame,
    *,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    window: int = 5,
    alpha_base: float = 0.001,
    alpha_res: float = 0.003,
    u_max: float = 4.0,
    p: float = 1.6,
    q_eol_ref: float = 0.98,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Compute HI_phys_v3.1 for all rows in df_residuals.

    Args:
        df_residuals:
            DataFrame containing residuals from the HealthyTwin:
              - unit_col (e.g. "UnitNumber")
              - cycle_col (e.g. "TimeInCycles")
              - residual columns for the 13 damage sensors, e.g. Resid_Sensor2, ...
        unit_col:
            Name of the engine/unit id column.
        cycle_col:
            Name of the time/cycle column.
        window:
            Window size for the causal median filter (must be odd).
        alpha_base:
            Base damage per time step (cycles), independent of residuals.
        alpha_res:
            Scaling factor for residual-based damage increments.
        u_max:
            Clipping threshold for normalised residual magnitudes |z|.
        p:
            Exponent for non-linear amplification of large residuals (e.g. 1.5).
        q_eol_ref:
            Quantile (e.g. 0.95) used to define the EOL reference damage D_eol_ref.
        eps:
            Numerical epsilon for divisions.

    Returns:
        pandas.Series aligned with df_residuals.index containing HI_phys_v3 in [0, 1].
        The series is named "HI_phys_v3".
    """
    if unit_col not in df_residuals.columns or cycle_col not in df_residuals.columns:
        raise ValueError(
            f"[HI_phys_v3] DataFrame must contain '{unit_col}' and "
            f"'{cycle_col}' columns. Available: {list(df_residuals.columns)[:10]}..."
        )

    if df_residuals.empty:
        return pd.Series(
            np.array([], dtype=float), index=df_residuals.index, name="HI_phys_v3"
        )

    # ------------------------------------------------------------------
    # 1) Resolve residual columns for the damage sensor set
    # ------------------------------------------------------------------
    resid_cols = _find_damage_residual_columns(df_residuals)

    # Work on a copy to avoid mutating the input frame
    df_filt = df_residuals[[unit_col, cycle_col] + list(resid_cols)].copy()

    # Ensure deterministic ordering within each engine
    df_filt = df_filt.sort_values([unit_col, cycle_col])

    # ------------------------------------------------------------------
    # 2) Causal median filter per engine and per residual sensor
    # ------------------------------------------------------------------
    for col in resid_cols:
        filtered_values: List[np.ndarray] = []
        index_slices: List[pd.Index] = []

        for uid, g in df_filt.groupby(unit_col, sort=False):
            vals = g[col].to_numpy(dtype=float)
            filt = causal_median_filter_1d(vals, window=window)
            filtered_values.append(filt)
            index_slices.append(g.index)

        # Concatenate in the same grouped order and assign back
        all_filtered = np.concatenate(filtered_values, axis=0)
        all_index = np.concatenate([idx.to_numpy() for idx in index_slices], axis=0)
        # Map back into df_filt via positional alignment
        df_filt.loc[all_index, col] = all_filtered

    # ------------------------------------------------------------------
    # 3) Robust sensor-wise stats (median + MAD-based scale)
    # ------------------------------------------------------------------
    medians, scales = _compute_robust_stats_per_sensor(df_filt, resid_cols, eps=eps)

    # ------------------------------------------------------------------
    # 4) Per-engine cumulative damage
    # ------------------------------------------------------------------
    damage_col = "Damage_phys_v3"
    df_filt[damage_col] = 0.0

    n_sensors = float(len(resid_cols))
    if n_sensors <= 0:
        raise ValueError("[HI_phys_v3] No residual sensors selected for damage set.")

    for uid, g in df_filt.groupby(unit_col, sort=False):
        g_sorted = g.sort_values(cycle_col)
        T = len(g_sorted)
        if T == 0:
            continue

        # Build matrix of normalised, clipped residual magnitudes [T, S]
        u_stack: List[np.ndarray] = []
        for col in resid_cols:
            vals = g_sorted[col].to_numpy(dtype=float)
            m = medians[col]
            s = scales[col]
            z = (vals - m) / (s + eps)
            u = np.abs(z)
            u_clipped = np.minimum(u, u_max)
            # Replace NaNs with zeros (no damage contribution)
            u_clipped = np.nan_to_num(u_clipped, nan=0.0, posinf=u_max, neginf=u_max)
            u_stack.append(u_clipped ** p)

        u_mat = np.stack(u_stack, axis=1)  # [T, S]
        # Mean over sensors
        mean_u_p = np.mean(u_mat, axis=1)  # [T]

        damage_residual_increment = alpha_res * mean_u_p  # [T]
        damage_base_increment = np.full(T, alpha_base, dtype=float)
        delta_D = damage_base_increment + damage_residual_increment  # [T]

        # Cumulative damage per engine (monotonically increasing)
        D = np.cumsum(delta_D)

        df_filt.loc[g_sorted.index, damage_col] = D

    # ------------------------------------------------------------------
    # 5) EOL reference damage via high quantile
    # ------------------------------------------------------------------
    # Final damage per engine = last value in each engine's trajectory
    final_damage = (
        df_filt.sort_values(cycle_col)
        .groupby(unit_col)[damage_col]
        .last()
        .to_numpy(dtype=float)
    )
    if final_damage.size == 0:
        # Degenerate case – no engines
        return pd.Series(
            np.array([], dtype=float), index=df_residuals.index, name="HI_phys_v3"
        )

    D_eol_ref = float(np.quantile(final_damage, q=q_eol_ref))
    if not np.isfinite(D_eol_ref) or D_eol_ref <= 0.0:
        # Fallback: use max damage if quantile is degenerate
        D_eol_ref = float(np.max(final_damage)) if final_damage.size > 0 else 1.0
        if D_eol_ref <= 0.0:
            D_eol_ref = 1.0

    # ------------------------------------------------------------------
    # 6) Health Index in [0, 1], monotone decreasing with damage
    # ------------------------------------------------------------------
    D_all = df_filt[damage_col].to_numpy(dtype=float)
    hi = 1.0 - D_all / (D_eol_ref + eps)
    hi = np.clip(hi, 0.0, 1.0)

    hi_series = pd.Series(hi, index=df_filt.index, name="HI_phys_v3")

    # Reindex to original frame index (df_residuals may not be sorted)
    hi_series = hi_series.reindex(df_residuals.index)

    return hi_series


__all__ = [
    "HIPhysV3Config",
    "compute_hi_phys_v3_from_residuals",
    "causal_median_filter_1d",
]


