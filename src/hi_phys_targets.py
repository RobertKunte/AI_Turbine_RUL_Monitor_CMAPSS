from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_hi_phys_v2(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    time_col: str = "TimeInCycles",
    hpc_eff_col: str = "Effizienz_HPC_Proxy",   # or "HPC_Eff_Proxy" depending on naming
    egt_drift_col: str = "EGT_Drift",
    residual_prefix: str = "Resid_",
    baseline_len: int = 30,
    w_eff: float = 0.4,
    w_egt: float = 0.3,
    w_resid: float = 0.3,
) -> pd.DataFrame:
    """
    Add a physics-based health index HI_phys_v2 to the dataframe.

    The construction follows:
        - d_eff(t)   = max(0, eff_baseline - eff(t))
        - d_egt(t)   = max(0, egt(t) - egt_baseline)
        - d_resid(t) = mean_i |Resid_i(t)|

        Damage_inst(t) = w_eff * d_eff(t) + w_egt * d_egt(t) + w_resid * d_resid(t)
        Damage_smooth  = rolling_mean(Damage_inst, window=30)
        Damage_mono    = cummax(Damage_smooth)
        Damage_norm    = Damage_mono / (max(Damage_mono) + eps)
        HI_phys_v2(t)  = 1 - clip(Damage_norm, 0, 1)

    This yields a strictly decreasing health index in [0,1] per engine.
    """
    df = df.copy()
    if df.empty:
        return df

    # Try to be robust against slight column naming differences
    eff_col = hpc_eff_col
    if eff_col not in df.columns:
        alt = "HPC_Eff_Proxy"
        if alt in df.columns:
            eff_col = alt
        else:
            raise KeyError(f"Neither '{hpc_eff_col}' nor '{alt}' found in dataframe columns.")

    if egt_drift_col not in df.columns:
        raise KeyError(f"Column '{egt_drift_col}' not found in dataframe.")

    resid_cols: list[str] = [c for c in df.columns if c.startswith(residual_prefix)]
    if not resid_cols:
        # No residuals -> fall back to 0 damage component
        resid_cols = []

    hi_phys_values = []

    # Process per unit
    for unit_id, g in df.groupby(unit_col, sort=False):
        g = g.sort_values(time_col).copy()

        # Baselines from early cycles (first baseline_len)
        head = g.head(baseline_len)
        eff_baseline = float(head[eff_col].mean())
        egt_baseline = float(head[egt_drift_col].mean())

        d_eff = np.maximum(0.0, eff_baseline - g[eff_col].to_numpy())
        d_egt = np.maximum(0.0, g[egt_drift_col].to_numpy() - egt_baseline)

        if resid_cols:
            resid_abs = np.abs(g[resid_cols].to_numpy())
            d_resid = resid_abs.mean(axis=1)
        else:
            d_resid = np.zeros(len(g), dtype=float)

        damage_inst = w_eff * d_eff + w_egt * d_egt + w_resid * d_resid

        # Rolling smoothing
        window = 30
        damage_smooth = pd.Series(damage_inst).rolling(window=window, min_periods=1).mean().to_numpy()

        # Monotone cumulative damage (not per-step diff)
        damage_mono = np.maximum.accumulate(damage_smooth)

        # Normalize per engine to [0,1]
        eps = 1e-6
        max_damage = float(damage_mono.max()) + eps
        damage_norm = np.clip(damage_mono / max_damage, 0.0, 1.0)

        hi_phys = 1.0 - damage_norm
        hi_phys_values.append(pd.Series(hi_phys, index=g.index))

    hi_phys_all = pd.concat(hi_phys_values).sort_index()
    df["HI_phys_v2"] = hi_phys_all

    return df

