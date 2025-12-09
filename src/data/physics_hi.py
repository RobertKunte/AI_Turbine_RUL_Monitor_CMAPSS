from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Default channel weights for the combined damage index.
# These are kept as module-level constants so they can be tuned centrally
# or overridden via the damage_feature_weights argument.
DEFAULT_DAMAGE_WEIGHTS: Dict[str, float] = {
    "d_eff": 0.4,
    "d_egt": 0.4,
    "d_resid": 0.2,
}

# Alternative weights for the HI_phys_v2 "instant" damage mixture. These are
# kept separate from DEFAULT_DAMAGE_WEIGHTS so that we can tune them
# independently without affecting the legacy HI implementation.
HI_V2_DAMAGE_WEIGHTS: Dict[str, float] = {
    "d_eff": 0.4,
    "d_egt": 0.3,
    "d_resid": 0.3,
}


def _smooth_series(s: pd.Series, window: int = 15) -> pd.Series:
    """Simple rolling-mean smoother."""
    return s.rolling(window, min_periods=1).mean()


def _enforce_monotone_decreasing(s: pd.Series) -> pd.Series:
    """
    Enforce a monotonically non-increasing profile (no healing).

    Walk forward in time and keep a running minimum.
    """
    out = []
    current = 1.0
    for v in s:
        current = min(current, float(v))
        out.append(current)
    return pd.Series(out, index=s.index)


def _rescale_unit_01(s: pd.Series) -> pd.Series:
    """
    Rescale a per-unit HI trajectory such that:
      - first value is near 1.0
      - last value is near 0.0

    If there is essentially no degradation (flat sequence), return all ones.
    """
    if len(s) == 0:
        return s

    hi_start = float(s.iloc[0])
    hi_end = float(s.iloc[-1])
    if abs(hi_start - hi_end) < 1e-6:
        # No visible degradation -> treat as "always healthy"
        return pd.Series(np.ones_like(s.values, dtype=float), index=s.index)

    hi = (s - hi_end) / (hi_start - hi_end)
    return hi.clip(0.0, 1.0)


def add_physics_hi(
    df: pd.DataFrame,
    unit_col: str = "Unit",
    cycle_col: str = "Cycle",
    cond_col: str = "ConditionID",
    is_training: bool = True,
    damage_feature_weights: Optional[Dict[str, float]] = None,
    global_scalers: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]] | pd.DataFrame:
    """
    Compute a physics-based Health Index HI_phys_final and attach it to df.

    Pipeline:
      1. Per-unit (/per-condition) baselines for key physics proxies.
      2. Non-negative "damage" channels d_eff, d_egt, d_resid.
      3. Robust normalisation of each damage channel using 99th percentile
         scales derived from TRAINING DATA.
      4. Combined damage index with configurable channel weights.
      5. Global damage scaling (again via 99th percentile).
      6. Raw HI = 1 - Damage_norm.
      7. Per-unit smoothing (rolling mean) and monotonic enforcement.
      8. TRAINING ONLY: per-unit rescaling to [1 -> 0] over the full trajectory.
         TEST/INFERENCE: keep the monotone HI directly without any per-unit
         rescaling, to avoid forcing HI to 0 on truncated sequences.

    Args:
        df: Input DataFrame. Must already contain physics features such as
            Effizienz_HPC_Proxy, EGT or EGT_Drift, and residual / twin-residual
            features. The function does NOT create those features.
        unit_col: Name of the engine/unit id column.
        cycle_col: Name of the cycle/time column.
        cond_col: Optional operating-condition id column. If present, it is
            used together with unit_col when computing baselines.
        is_training: If True, fit robust scalers on df and return them.
                     If False, expect global_scalers from a previous training
                     call and apply them without refitting.
        damage_feature_weights: Optional dict overriding DEFAULT_DAMAGE_WEIGHTS.
            Keys: "d_eff", "d_egt", "d_resid".
        global_scalers: Dict of scaling factors learned during training with
            keys: "d_eff", "d_egt", "d_resid", "damage_scale".

    Returns:
        If is_training:
            (df_out, scalers) where:
                - df_out is a copy of df with HI_phys_final (and helper columns).
                - scalers is a dict suitable for reuse at inference time.
        If not is_training:
            df_out only.
    """
    if damage_feature_weights is None:
        damage_feature_weights = dict(DEFAULT_DAMAGE_WEIGHTS)

    required_keys = {"d_eff", "d_egt", "d_resid"}
    if set(damage_feature_weights.keys()) != required_keys:
        # Be strict so that weights are always well-defined.
        raise ValueError(
            f"damage_feature_weights must have keys {required_keys}, "
            f"got {set(damage_feature_weights.keys())}"
        )

    if unit_col not in df.columns or cycle_col not in df.columns:
        raise ValueError(
            f"[add_physics_hi] DataFrame must contain '{unit_col}' and "
            f"'{cycle_col}' columns. Available: {list(df.columns)[:10]}..."
        )

    df_out = df.copy()

    # ------------------------------------------------------------------
    # 1) Per-unit(/per-condition) baselines
    # ------------------------------------------------------------------
    group_cols = [unit_col]
    if cond_col in df_out.columns:
        group_cols.append(cond_col)

    # Efficiency proxy baseline
    hpc_col = "Effizienz_HPC_Proxy"
    if hpc_col in df_out.columns:
        df_out["HPC_eff_baseline"] = (
            df_out.sort_values(cycle_col)
            .groupby(group_cols)[hpc_col]
            .transform(lambda s: s.iloc[:20].mean())
        )
    else:
        # Graceful fallback: no efficiency info -> zero damage channel.
        df_out["HPC_eff_baseline"] = 0.0

    # EGT baseline from raw EGT if available
    if "EGT" in df_out.columns:
        df_out["EGT_baseline"] = (
            df_out.sort_values(cycle_col)
            .groupby(group_cols)["EGT"]
            .transform(lambda s: s.iloc[:20].mean())
        )

    # ------------------------------------------------------------------
    # 2) Damage channels (non-negative)
    # ------------------------------------------------------------------
    # Efficiency damage: loss relative to baseline (higher loss = worse).
    if hpc_col in df_out.columns:
        df_out["d_eff"] = (df_out["HPC_eff_baseline"] - df_out[hpc_col]).clip(lower=0.0)
    else:
        df_out["d_eff"] = 0.0

    # EGT damage: overheating relative to baseline or drift proxy.
    if "EGT_baseline" in df_out.columns and "EGT" in df_out.columns:
        df_out["d_egt"] = (df_out["EGT"] - df_out["EGT_baseline"]).clip(lower=0.0)
    elif "EGT_Drift" in df_out.columns:
        df_out["d_egt"] = df_out["EGT_Drift"].clip(lower=0.0)
    else:
        df_out["d_egt"] = 0.0

    # Residual damage: average absolute residual / twin-residual magnitude.
    residual_cols = [
        c
        for c in df_out.columns
        if ("resid" in c.lower()) or ("residual" in c.lower())
    ]
    if residual_cols:
        df_out["d_resid"] = df_out[residual_cols].abs().mean(axis=1)
    else:
        df_out["d_resid"] = 0.0

    # ------------------------------------------------------------------
    # 3) Robust normalisation using training scalers
    # ------------------------------------------------------------------
    damage_cols = ["d_eff", "d_egt", "d_resid"]
    scalers: Dict[str, float] = {}

    if is_training:
        for col in damage_cols:
            p99 = df_out[col].quantile(0.99)
            scale = float(p99) if p99 > 0 else 1.0
            scalers[col] = scale
            df_out[col + "_n"] = (df_out[col] / scale).clip(0.0, 1.0)
    else:
        if global_scalers is None:
            raise ValueError(
                "[add_physics_hi] Inference mode requires global_scalers from training."
            )
        for col in damage_cols:
            if col not in global_scalers:
                raise KeyError(
                    f"[add_physics_hi] global_scalers missing key '{col}'. "
                    f"Available keys: {list(global_scalers.keys())}"
                )
            scale = float(global_scalers[col]) or 1.0
            df_out[col + "_n"] = (df_out[col] / scale).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 4) Combined damage index + global scaling
    # ------------------------------------------------------------------
    df_out["Damage_raw"] = (
        damage_feature_weights["d_eff"] * df_out["d_eff_n"]
        + damage_feature_weights["d_egt"] * df_out["d_egt_n"]
        + damage_feature_weights["d_resid"] * df_out["d_resid_n"]
    )

    if is_training:
        p99_damage = df_out["Damage_raw"].quantile(0.99)
        scale_damage = float(p99_damage) if p99_damage > 0 else 1.0
        scalers["damage_scale"] = scale_damage
    else:
        if global_scalers is None or "damage_scale" not in global_scalers:
            raise KeyError(
                "[add_physics_hi] Inference mode requires 'damage_scale' in global_scalers."
            )
        scale_damage = float(global_scalers["damage_scale"]) or 1.0

    df_out["Damage_norm"] = (df_out["Damage_raw"] / scale_damage).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 5) Raw HI, smoothing, monotonic enforcement
    # ------------------------------------------------------------------
    df_out["HI_phys_raw"] = 1.0 - df_out["Damage_norm"]

    # Smooth per engine trajectory
    df_out["HI_phys_smooth"] = (
        df_out.sort_values(cycle_col)
        .groupby(unit_col)["HI_phys_raw"]
        .transform(lambda s: _smooth_series(s, window=15))
    )

    # Enforce strictly monotone non-increasing behaviour per engine
    df_out["HI_phys_mono"] = (
        df_out.sort_values(cycle_col)
        .groupby(unit_col)["HI_phys_smooth"]
        .transform(_enforce_monotone_decreasing)
    )

    # ------------------------------------------------------------------
    # 6) Final scaling: TRAIN vs TEST/INFERENCE
    # ------------------------------------------------------------------
    if is_training:
        # TRAINING: rescale each full trajectory to [1 -> 0] so that the learning
        # target is well-conditioned and comparable across engines.
        df_out["HI_phys_final"] = (
            df_out.sort_values(cycle_col)
            .groupby(unit_col)["HI_phys_mono"]
            .transform(_rescale_unit_01)
        )
        return df_out, scalers

    # TEST / INFERENCE: do NOT rescale per engine. This avoids the common PHM
    # pitfall of implicitly forcing HI to 0 on truncated sequences that stop
    # before actual failure.
    df_out["HI_phys_final"] = df_out["HI_phys_mono"]
    return df_out


def add_physics_hi_v2(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    cond_col: str = "ConditionID",
    is_training: bool = True,
    max_rul: float = 125.0,
    rul_col: str = "RUL",
    alpha_hybrid: float = 0.7,
    global_scalers: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]] | pd.DataFrame:
    """
    Compute a robust physics-based Health Index (HI_phys_v2) and an optional
    Hybrid-HI target that blends physics HI with a RUL-based HI proxy.

    Design goals:
      - HI_phys_v2 is derived from degradation proxies + residuals:
          * Effizienz_HPC_Proxy
          * EGT / EGT_Drift
          * Residuals (Twin_*/Resid_*)
      - Damage is smoothed and made monotonically non-decreasing per engine.
      - TRAINING ONLY: trajectories are rescaled per engine to span [1 -> 0]
        (starting near 1, ending near 0).
      - TEST / INFERENCE: no forced 0 at last cycle; we keep the cumulative
        damage profile as-is after normalisation and monotonicity.

    Additionally, for training we construct a Hybrid-HI target:

        HI_target = alpha * HI_phys_v2 + (1 - alpha) * (1 - RUL_norm)

    where RUL_norm = RUL / max_rul. This keeps the target physically grounded
    while making it easier to align with EOL/RUL behaviour.

    Args:
        df: Input DataFrame with physics features and residuals already added.
        unit_col: Engine/unit id column.
        cycle_col: Cycle/time column.
        cond_col: Optional operating-condition column.
        is_training: If True, fit per-channel 99th-percentile scalers and
            perform per-unit 0->1 rescaling. If False, reuse global_scalers and
            skip the per-unit 0->1 rescaling.
        max_rul: Maximal sichtbare RUL (für RUL-Normalisierung im Hybrid-HI).
        rul_col: Name der RUL-Spalte (geclippte RUL).
        alpha_hybrid: Gewichtung physikalisches HI vs. RUL-HI im Hybrid-Target.
        global_scalers: Dict mit Skalen aus dem Training (d_eff/egt/resid).

    Returns:
        TRAINING:
            df_out, scalers dict mit 99%-Skalen je Channel.
        TEST/INFERENCE:
            df_out (nur).
    """
    if unit_col not in df.columns or cycle_col not in df.columns:
        raise ValueError(
            f"[add_physics_hi_v2] DataFrame must contain '{unit_col}' and "
            f"'{cycle_col}' columns. Available: {list(df.columns)[:10]}..."
        )

    df_out = df.copy()

    # ------------------------------------------------------------------
    # 1) Baselines und Damage-Kanäle – wiederverwende Logik aus add_physics_hi
    # ------------------------------------------------------------------
    group_cols = [unit_col]
    if cond_col in df_out.columns:
        group_cols.append(cond_col)

    # HPC-Baseline
    hpc_col = "Effizienz_HPC_Proxy"
    if hpc_col in df_out.columns:
        df_out["HPC_eff_baseline"] = (
            df_out.sort_values(cycle_col)
            .groupby(group_cols)[hpc_col]
            .transform(lambda s: s.iloc[:20].mean())
        )
    else:
        df_out["HPC_eff_baseline"] = 0.0

    # EGT-Baseline
    if "EGT" in df_out.columns:
        df_out["EGT_baseline"] = (
            df_out.sort_values(cycle_col)
            .groupby(group_cols)["EGT"]
            .transform(lambda s: s.iloc[:20].mean())
        )

    # Effizienz-Damage
    if hpc_col in df_out.columns:
        df_out["d_eff"] = (df_out["HPC_eff_baseline"] - df_out[hpc_col]).clip(lower=0.0)
    else:
        df_out["d_eff"] = 0.0

    # EGT-Damage
    if "EGT_baseline" in df_out.columns and "EGT" in df_out.columns:
        df_out["d_egt"] = (df_out["EGT"] - df_out["EGT_baseline"]).clip(lower=0.0)
    elif "EGT_Drift" in df_out.columns:
        df_out["d_egt"] = df_out["EGT_Drift"].clip(lower=0.0)
    else:
        df_out["d_egt"] = 0.0

    # Residual-Damage
    residual_cols = [
        c for c in df_out.columns if ("resid" in c.lower()) or ("residual" in c.lower())
    ]
    if residual_cols:
        df_out["d_resid"] = df_out[residual_cols].abs().mean(axis=1)
    else:
        df_out["d_resid"] = 0.0

    # ------------------------------------------------------------------
    # 2) Robust-Skalierung je Channel (99%-Quantil)
    # ------------------------------------------------------------------
    damage_cols = ["d_eff", "d_egt", "d_resid"]
    scalers: Dict[str, float] = {}

    if is_training:
        for col in damage_cols:
            p99 = df_out[col].quantile(0.99)
            scale = float(p99) if p99 > 0 else 1.0
            scalers[col] = scale
            df_out[col + "_n"] = (df_out[col] / scale).clip(0.0, 1.0)
    else:
        if global_scalers is None:
            raise ValueError(
                "[add_physics_hi_v2] Inference mode requires global_scalers from training."
            )
        for col in damage_cols:
            if col not in global_scalers:
                raise KeyError(
                    f"[add_physics_hi_v2] global_scalers missing key '{col}'. "
                    f"Available keys: {list(global_scalers.keys())}"
                )
            scale = float(global_scalers[col]) or 1.0
            df_out[col + "_n"] = (df_out[col] / scale).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 3) Instantanes Damage-Signal (D_inst)
    # ------------------------------------------------------------------
    w = HI_V2_DAMAGE_WEIGHTS
    df_out["Damage_inst"] = (
        w["d_eff"] * df_out["d_eff_n"]
        + w["d_egt"] * df_out["d_egt_n"]
        + w["d_resid"] * df_out["d_resid_n"]
    )

    # ------------------------------------------------------------------
    # 4) Glätten + monotone kumulative Damage (D_smooth, D_mono)
    # ------------------------------------------------------------------
    # Rolling-Mean über 20 Zyklen pro Unit
    df_out["Damage_smooth"] = (
        df_out.sort_values(cycle_col)
        .groupby(unit_col)["Damage_inst"]
        .transform(lambda s: s.rolling(window=20, min_periods=1).mean())
    )

    # Monotonisierung via kumulativer Max
    df_out["Damage_mono"] = (
        df_out.sort_values(cycle_col)
        .groupby(unit_col)["Damage_smooth"]
        .transform(lambda s: s.cummax())
    )

    # ------------------------------------------------------------------
    # 5) Per-Unit-Rescaling 0->1 (nur Training)
    # ------------------------------------------------------------------
    def _per_unit_rescale_damage(s: pd.Series, start_window: int = 20, end_window: int = 20):
        if len(s) == 0:
            return s
        # Startmittel über erste start_window Punkte
        start_len = min(len(s), start_window)
        D_start = float(s.iloc[:start_len].mean())
        # Endmittel über letzte end_window Punkte
        end_len = min(len(s), end_window)
        D_end = float(s.iloc[-end_len:].mean())

        if D_end > D_start + 1e-6:
            D_unit = (s - D_start) / max(D_end - D_start, 1e-6)
            return D_unit.clip(0.0, 1.0)
        # Kein sichtbarer Schaden -> komplett gesund
        return pd.Series(np.zeros_like(s.values, dtype=float), index=s.index)

    if is_training:
        df_out["Damage_unit"] = (
            df_out.sort_values(cycle_col)
            .groupby(unit_col)["Damage_mono"]
            .transform(_per_unit_rescale_damage)
        )
    else:
        # TEST/Inference: keine erzwungene 0 am Ende; wir verwenden die
        # normalisierte, monotone Damage_mono direkt.
        df_out["Damage_unit"] = df_out["Damage_mono"].clip(0.0, 1.0)

    # Physikalische HI v2
    df_out["HI_phys_v2"] = 1.0 - df_out["Damage_unit"]

    # ------------------------------------------------------------------
    # 6) Hybrid-HI (immer berechnen, sofern RUL verfügbar)
    # ------------------------------------------------------------------
    if rul_col in df_out.columns and max_rul > 0:
        rul_norm = (df_out[rul_col].astype(float) / float(max_rul)).clip(0.0, 1.0)
        df_out["HI_rul_proxy"] = 1.0 - rul_norm
        df_out["HI_target_hybrid"] = (
            float(alpha_hybrid) * df_out["HI_phys_v2"]
            + (1.0 - float(alpha_hybrid)) * df_out["HI_rul_proxy"]
        )

    if is_training:
        return df_out, scalers

    return df_out



