from __future__ import annotations

"""
Global monotone calibration for physics-based health index HI_phys_v3.

This module provides:
  - fit_hi_calibrator:   fit a global monotone mapping f: HI_phys -> HI_cal
  - apply_hi_calibrator: apply the mapping to 1D HI arrays
  - load_hi_calibrator:  load a saved calibrator (joblib)
  - calibrate_hi_array:  convenience wrapper for 1D/2D arrays

It also exposes a small CLI that:
  - rebuilds the FD004 ms_dt_v2 + residual + twin TRAIN pipeline
    used by a given encoder run,
  - fits an IsotonicRegression calibrator on aggregated TRAIN samples
    using targets derived from RUL,
  - saves the calibrator object under the encoder run directory.

Example:

    python -m src.analysis.hi_calibration \\
        --dataset FD004 \\
        --encoder_run fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_hi_calibrator(
    hi_phys: np.ndarray,
    rul: np.ndarray,
    max_rul: float,
) -> IsotonicRegression:
    """
    Fit a global monotone mapping f: HI_phys -> HI_cal on TRAIN data.

    The target health is defined as:
        h_target = clip(1 - rul / max_rul, 0, 1)

    so that:
        - h_target ≈ 1 at the beginning of life (large RUL),
        - h_target ≈ 0 near end-of-life (RUL ≈ 0).

    Args:
        hi_phys: 1D array-like of HI_phys_v3 values (flattened over TRAIN samples).
        rul:     1D array-like of corresponding RUL values.
        max_rul: Maximum RUL used for scaling into [0, 1].

    Returns:
        Fitted IsotonicRegression calibrator mapping HI_phys -> HI_cal in [0, 1].
    """
    hi_phys = np.asarray(hi_phys, dtype=float).reshape(-1)
    rul = np.asarray(rul, dtype=float).reshape(-1)

    if hi_phys.shape[0] != rul.shape[0]:
        raise ValueError(
            f"fit_hi_calibrator: hi_phys and rul must have same length, "
            f"got {hi_phys.shape[0]} vs {rul.shape[0]}"
        )

    if hi_phys.size == 0:
        raise ValueError("fit_hi_calibrator: received empty hi_phys array.")

    # Drop NaNs / infs from both arrays in a leakage-free, global fashion
    mask = np.isfinite(hi_phys) & np.isfinite(rul)
    if not np.any(mask):
        raise ValueError("fit_hi_calibrator: no finite samples after filtering NaNs/Infs.")

    hi_phys_clean = hi_phys[mask]
    rul_clean = rul[mask]

    # Target health in [0, 1], monotone decreasing with degradation
    h_target = 1.0 - rul_clean / float(max_rul)
    h_target = np.clip(h_target, 0.0, 1.0)

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=False,  # health decreases with degradation
        out_of_bounds="clip",
    )
    calibrator.fit(hi_phys_clean, h_target)
    return calibrator


def apply_hi_calibrator(
    hi_phys: np.ndarray,
    calibrator: Any,
) -> np.ndarray:
    """
    Apply a fitted calibrator to HI_phys values to obtain HI_cal in [0, 1].

    Args:
        hi_phys: 1D array-like of HI_phys values.
        calibrator: Fitted IsotonicRegression (or compatible) object.

    Returns:
        NumPy array of calibrated HI values, clipped to [0, 1].
    """
    hi_phys_arr = np.asarray(hi_phys, dtype=float).reshape(-1)

    if hi_phys_arr.size == 0:
        return np.asarray(hi_phys_arr, dtype=float)

    # Replace NaNs with a benign value (e.g., median of finite samples) before calibration
    finite_mask = np.isfinite(hi_phys_arr)
    if not np.any(finite_mask):
        # Degenerate case: all NaNs -> return zeros
        return np.zeros_like(hi_phys_arr, dtype=float)

    median_val = float(np.median(hi_phys_arr[finite_mask]))
    hi_phys_safe = hi_phys_arr.copy()
    hi_phys_safe[~finite_mask] = median_val

    hi_cal = calibrator.predict(hi_phys_safe)
    hi_cal = np.asarray(hi_cal, dtype=float)
    hi_cal = np.clip(hi_cal, 0.0, 1.0)
    return hi_cal


def load_hi_calibrator(calibrator_path: str | Path) -> Any:
    """
    Load a previously saved HI calibrator object via joblib.
    """
    import joblib

    calibrator_path = Path(calibrator_path)
    if not calibrator_path.exists():
        raise FileNotFoundError(f"HI calibrator not found at {calibrator_path}")

    # Fail fast on corrupted / partially-written files (common on interrupted runs)
    try:
        # Also guard against empty files (joblib may raise EOFError)
        if calibrator_path.stat().st_size == 0:
            raise EOFError("calibrator file is empty (0 bytes)")
        return joblib.load(calibrator_path)
    except EOFError as e:
        raise RuntimeError(
            "HI calibrator file exists but is corrupted/empty (EOFError). "
            f"Path: {calibrator_path}\n"
            "Fix: delete the file and refit it via:\n"
            "  python -m src.analysis.hi_calibration --dataset FD004 --encoder_run <BASE_ENCODER_RUN>\n"
            "Then re-run your experiment.\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to load HI calibrator via joblib. "
            f"Path: {calibrator_path}\n"
            "Fix: delete the file and refit it via:\n"
            "  python -m src.analysis.hi_calibration --dataset FD004 --encoder_run <BASE_ENCODER_RUN>\n"
            "Then re-run your experiment.\n"
            f"Original error type: {type(e).__name__}, error: {e}"
        ) from e


def calibrate_hi_array(
    hi_phys_array: np.ndarray,
    calibrator: Any,
) -> np.ndarray:
    """
    Apply a calibrator to a 1D or 2D HI_phys array.

    Supports:
        - 1D: [N]
        - 2D: [N, T]

    For 2D input, the array is flattened to [N*T], calibrated, then reshaped back.
    Output is float32/float64 with values clipped to [0, 1].
    """
    hi_phys_arr = np.asarray(hi_phys_array)

    if hi_phys_arr.ndim == 1:
        hi_cal = apply_hi_calibrator(hi_phys_arr, calibrator)
        return hi_cal.astype(hi_phys_arr.dtype, copy=False)

    if hi_phys_arr.ndim == 2:
        n, t = hi_phys_arr.shape
        hi_flat = hi_phys_arr.reshape(-1)
        hi_cal_flat = apply_hi_calibrator(hi_flat, calibrator)
        hi_cal = hi_cal_flat.reshape(n, t)
        return hi_cal.astype(hi_phys_arr.dtype, copy=False)

    # For higher dimensions, fall back to flattening everything except the last axis
    orig_shape = hi_phys_arr.shape
    last_dim = orig_shape[-1]
    hi_reshaped = hi_phys_arr.reshape(-1, last_dim)
    hi_cal_reshaped = np.empty_like(hi_reshaped, dtype=hi_reshaped.dtype)
    for i in range(hi_reshaped.shape[0]):
        hi_cal_reshaped[i] = apply_hi_calibrator(hi_reshaped[i], calibrator)
    hi_cal = hi_cal_reshaped.reshape(orig_shape)
    return hi_cal.astype(hi_phys_arr.dtype, copy=False)


def hi_cal_v2_from_v1(hi_cal_v1: np.ndarray) -> np.ndarray:
    """
    Convert HI_cal_v1 (0=worst, 1=best) to HI_cal_v2 (1=healthy, 0=EOL).

    Supports both 1D [N] and 2D [N, T] arrays and preserves the input dtype.
    Ensures the result is clipped to [0, 1].
    """
    hi = np.asarray(hi_cal_v1, dtype=float)
    if hi.ndim == 0:
        hi = hi[None]
    # Invert: 1 - h, then clip to [0, 1]
    inv = 1.0 - hi
    inv = np.clip(inv, 0.0, 1.0)
    # Cast back to original dtype if it was float32/float64
    if np.issubdtype(hi_cal_v1.dtype, np.floating):
        inv = inv.astype(hi_cal_v1.dtype, copy=False)
    return inv


def _build_fd004_train_hi_arrays(
    dataset: str,
    encoder_run: str,
    max_rul: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebuild the FD004 ms_dt_v2 + residual + twin TRAIN pipeline for a given
    encoder experiment and extract flattened (HI_phys_v3, RUL) arrays.

    This mirrors the logic used in `src.rul_decoder_training_v1.prepare_fd004_ms_dt_encoder_data`
    but keeps only TRAIN rows and does not build sliding windows.
    """
    import json

    import pandas as pd

    from src.data_loading import load_cmapps_subset
    from src.additional_features import (
        create_physical_features,
        create_all_features,
        FeatureConfig,
        TemporalFeatureConfig,
        PhysicsFeatureConfig,
        build_condition_features,
        create_twin_features,
    )
    from src.config import ResidualFeatureConfig
    from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

    dataset = dataset.upper()
    if dataset != "FD004":
        raise ValueError(
            f"_build_fd004_train_hi_arrays currently supports only FD004, got {dataset}"
        )

    summary_path = (
        Path("results") / dataset.lower() / encoder_run / "summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")

    with open(summary_path, "r") as f:
        summary_cfg = json.load(f)

    # Load raw CMAPSS data (TRAIN only needed here)
    df_train, _, _ = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Physics & feature configs (mirrors rul_decoder_training_v1 and diagnostics)
    name_lower = encoder_run.lower()
    is_phase4_residual = (
        (("phase4" in name_lower) or ("phase5" in name_lower)) and "residual" in name_lower
    ) or ("residual" in name_lower) or ("resid" in name_lower)

    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=is_phase4_residual,
        use_temporal_on_physics=False,
        residual=ResidualFeatureConfig(
            enabled=is_phase4_residual,
            mode="per_engine",
            baseline_len=30,
            include_original=True,
        )
        if is_phase4_residual
        else ResidualFeatureConfig(enabled=False),
    )

    phys_opts = summary_cfg.get("phys_features", {})
    use_phys_condition_vec = phys_opts.get("use_condition_vector", False)
    use_twin_features = phys_opts.get(
        "use_twin_features",
        phys_opts.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_opts.get("twin_baseline_len", 30)
    condition_vector_version = phys_opts.get("condition_vector_version", 2)

    features_cfg = summary_cfg.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    use_temporal_features = features_cfg.get("use_multiscale_features", True)

    windows_short = tuple(ms_cfg.get("windows_short", (5, 10)))
    windows_medium = tuple(ms_cfg.get("windows_medium", ()))
    windows_long = tuple(ms_cfg.get("windows_long", (30,)))
    combined_long = windows_medium + windows_long

    temporal_cfg = TemporalFeatureConfig(
        base_cols=None,
        short_windows=windows_short,
        long_windows=combined_long if combined_long else (30,),
        add_rolling_mean=True,
        add_rolling_std=False,
        add_trend=True,
        add_delta=True,
        delta_lags=(5, 10),
    )
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=use_temporal_features,
        temporal=temporal_cfg,
    )

    # 1) Physics features
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        df_train = build_condition_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin + residuals
    if use_twin_features:
        df_train, _ = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )

    # 4) Temporal / multi-scale features (kept for consistency; not strictly needed for HI)
    df_train = create_all_features(
        df_train,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    # 5) HI_phys_v3: compute if missing
    if "HI_phys_v3" not in df_train.columns:
        hi_v3_series = compute_hi_phys_v3_from_residuals(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_train["HI_phys_v3"] = hi_v3_series

    # Extract arrays (global TRAIN-only; no per-engine normalization here)
    hi_phys = df_train["HI_phys_v3"].to_numpy(dtype=float)

    # Prefer explicit 'RUL' column if present; fall back to 'RUL_raw'
    rul_col = "RUL" if "RUL" in df_train.columns else "RUL_raw"
    rul = df_train[rul_col].to_numpy(dtype=float)

    # Ensure no negative RUL values (just in case)
    rul = np.maximum(rul, 0.0)

    # Cap RUL at max_rul for target construction
    rul = np.minimum(rul, float(max_rul))

    return hi_phys, rul


def main() -> None:
    import argparse
    import joblib

    parser = argparse.ArgumentParser(description="Fit global HI_phys_v3 -> HI_cal_v1 calibrator.")
    parser.add_argument("--dataset", type=str, default="FD004")
    parser.add_argument(
        "--encoder_run",
        type=str,
        default="fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase",
    )
    parser.add_argument("--max_rul", type=float, default=125.0)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset.upper()
    encoder_run = args.encoder_run
    max_rul = float(args.max_rul)

    print("============================================================")
    print(f"[hi_calibration] Fitting calibrator for dataset={dataset}, encoder_run={encoder_run}")
    print("============================================================")

    # 1) Build TRAIN arrays (HI_phys_v3, RUL) using existing pipeline
    hi_phys_train, rul_train = _build_fd004_train_hi_arrays(
        dataset=dataset,
        encoder_run=encoder_run,
        max_rul=max_rul,
    )

    print(
        f"[hi_calibration] TRAIN samples: {hi_phys_train.shape[0]}, "
        f"HI_phys_v3 range: [{np.nanmin(hi_phys_train):.4f}, {np.nanmax(hi_phys_train):.4f}], "
        f"RUL range (capped): [{np.nanmin(rul_train):.1f}, {np.nanmax(rul_train):.1f}]"
    )

    # 2) Fit global monotone calibrator
    calibrator = fit_hi_calibrator(hi_phys_train, rul_train, max_rul=max_rul)

    # 3) Save calibrator object
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = (
            Path("results")
            / dataset.lower()
            / encoder_run
            / f"hi_calibrator_{dataset}.pkl"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_path)

    print(f"[hi_calibration] Saved HI calibrator to: {output_path}")


if __name__ == "__main__":
    main()


