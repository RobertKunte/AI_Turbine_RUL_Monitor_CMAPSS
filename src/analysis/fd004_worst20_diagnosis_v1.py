from __future__ import annotations

"""
FD004 Worst-20 Engine Diagnosis (Encoder v5 / Transformer-Encoder)

This script analyzes the 20 worst FD004 engines (by EOL error) for a given
Transformer-Encoder run (default: fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm).

It:
  - Loads per-engine EOL metrics via the standard inference pipeline
  - Selects the 20 worst late-detection engines and the 20 best engines
  - Rebuilds full FD004 feature pipelines (ms_dt_v2 + residual + Cond_*, Twin_*)
  - Extracts RUL, HI_phys_v3, HI_damage, HI_cal_v2 trajectories per engine
  - Compares ConditionID distribution and basic stats (worst vs rest)
  - Analyzes residual sensors (Resid_Sensor*) for worst vs rest
  - Saves multiple diagnostics plots into the encoder run's results directory
"""

from pathlib import Path
from typing import Dict, List, Tuple

import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.analysis.inference import (
    load_model_from_experiment,
    run_inference_for_experiment,
    EngineEOLMetrics,
    EngineTrajectory,
)
from src.analysis.diagnostics import build_eval_data
from src.additional_features import FeatureConfig, TemporalFeatureConfig
from src.config import PhysicsFeatureConfig


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

RUN_NAME = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm"
DATASET = "FD004"
# k-sweep for uncertainty bands and conservative estimate
K_LIST: List[float] = [1.0, 2.0, 3.0]

def _resolve_run_paths(*, dataset: str, run_name: str) -> Tuple[Path, Path, Path, Path]:
    """
    Resolve standard result paths for a given encoder run.
    Returns:
        results_dir, summary_path, checkpoint_path, scaler_path
    """
    results_dir = Path("results") / dataset.lower() / run_name
    summary_path = results_dir / "summary.json"
    checkpoint_path = results_dir / f"eol_full_lstm_best_{run_name}.pt"
    scaler_path = results_dir / "scaler.pkl"
    return results_dir, summary_path, checkpoint_path, scaler_path


# ---------------------------------------------------------------------------
# 2. Helpers: load summary and per-engine EOL metrics
# ---------------------------------------------------------------------------

def load_encoder_summary(summary_path: Path) -> dict:
    """Load encoder experiment summary.json."""
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    with open(summary_path, "r") as f:
        return json.load(f)


def build_eol_metrics_df(
    experiment_dir: Path,
    dataset_name: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[int, EngineTrajectory], pd.DataFrame]:
    """
    Run standard inference on the TEST set and build a per-engine EOL metrics
    DataFrame, plus trajectories and raw test dataframe.

    Returns:
        eol_df: DataFrame with columns
            - unit_id
            - cond_id
            - num_cycles
            - eol_rul_true
            - eol_rul_pred
            - eol_error
        trajectories: mapping unit_id -> EngineTrajectory
        df_test: raw FD004 test dataframe (from load_cmapps_subset)
    """
    from src.data_loading import load_cmapps_subset

    # 1) Run standard inference to get per-engine EOL metrics + trajectories
    print("[fd004_worst20] Running inference to obtain per-engine EOL metrics...")
    eol_metrics, trajectories = run_inference_for_experiment(
        experiment_dir=experiment_dir,
        split="test",
        device=device,
        return_hi_trajectories=True,
    )

    # 2) Load raw FD004 data to get ConditionID and num_cycles
    print("[fd004_worst20] Loading raw FD004 data to derive ConditionID and num_cycles...")
    _, df_test, _ = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Group by engine for cond_id and trajectory length
    grp = df_test.groupby("UnitNumber")
    cond_id_map = grp["ConditionID"].first().astype(int) if "ConditionID" in df_test.columns else None
    num_cycles_map = grp["TimeInCycles"].count().astype(int)

    def _get_sigma(m: EngineEOLMetrics) -> float:
        """
        Robustly extract predicted sigma for RUL at last observed cycle from metric object.
        Supports multiple historical attribute names.
        """
        for key in ["sigma_rul", "pred_sigma", "rul_sigma", "sigma", "pred_rul_sigma"]:
            if hasattr(m, key):
                val = getattr(m, key)
                if val is None:
                    continue
                try:
                    return float(val)
                except Exception:
                    continue
        return float("nan")

    rows = []
    for m in eol_metrics:
        uid = int(m.unit_id)
        cond_id = int(cond_id_map.loc[uid]) if cond_id_map is not None and uid in cond_id_map.index else -1
        num_cycles = int(num_cycles_map.loc[uid]) if uid in num_cycles_map.index else 0
        mu_last = float(m.pred_rul)
        true_last = float(m.true_rul)
        sigma_last = _get_sigma(m)
        rows.append(
            {
                "unit_id": uid,
                "cond_id": cond_id,
                "num_cycles": num_cycles,
                "eol_rul_true": float(m.true_rul),
                "eol_rul_pred": float(m.pred_rul),
                "eol_error": float(m.error),
                # Aliases for uncertainty-band plots (FD004 test is right-censored)
                "true_rul_last": true_last,
                "mu_last": mu_last,
                "sigma_last": sigma_last,
                "abs_error": float(abs(mu_last - true_last)),
                # Backwards-compatible column name (older versions used eol_sigma)
                "eol_sigma": sigma_last,
            }
        )

    eol_df = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    return eol_df, trajectories, df_test


# ---------------------------------------------------------------------------
# 3. Load full trajectories & features for selected engines
# ---------------------------------------------------------------------------

def build_feature_pipeline_for_fd004(
    summary: dict,
    dataset_name: str,
    max_rul: int = 125,
    past_len: int = 30,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rebuild the FD004 feature pipeline (ms_dt_v2 + residual + Cond_* + Twin_*)
    using the same helper as the main diagnostics (build_eval_data).

    Returns:
        df_test_fe: feature-engineered test dataframe
        feature_cols: list of feature columns used by the encoder
    """
    print("[fd004_worst20] Rebuilding FD004 feature pipeline via build_eval_data...")

    # Reconstruct FeatureConfig from summary["features"] if available
    features_cfg = summary.get("features", {})
    temporal_dict = features_cfg.get("temporal", {})
    temporal_cfg = TemporalFeatureConfig(**temporal_dict) if isinstance(temporal_dict, dict) else TemporalFeatureConfig()
    feature_config = FeatureConfig(
        add_physical_core=features_cfg.get("add_physical_core", True),
        add_temporal_features=features_cfg.get("use_multiscale_features", True),
        temporal=temporal_cfg,
    )

    # PhysicsFeatureConfig: for residual analysis we can use default core physics config.
    physics_config = PhysicsFeatureConfig()

    phys_features = summary.get("phys_features", {})

    # build_eval_data returns many things; we only need feature_cols and df_test_fe
    (
        _X_test_scaled,
        _y_true_eol,
        _y_test_true,
        _unit_ids_test,
        _cond_ids_test,
        _scaler_dict,
        feature_cols,
        df_test_fe,
    ) = build_eval_data(
        dataset_name=dataset_name,
        max_rul=max_rul,
        past_len=past_len,
        feature_config=feature_config,
        physics_config=physics_config,
        phys_features=phys_features,
    )

    # For damage_v3/v4/v5-style experiments, compute HI_phys_v3 on TEST
    # features for diagnostics (true physics-based HI trajectory).
    run_name_lower = summary.get("experiment_name", "").lower()
    if any(tag in run_name_lower for tag in ["damage_v3", "damage_v4", "damage_v5"]):
        try:
            from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

            print("[fd004_worst20] Computing HI_phys_v3 on test data for diagnostics...")
            hi_v3_test = compute_hi_phys_v3_from_residuals(
                df_test_fe,
                unit_col="UnitNumber",
                cycle_col="TimeInCycles",
            )
            df_test_fe["HI_phys_v3"] = hi_v3_test
            print(
                f"[fd004_worst20] HI_phys_v3 (test) stats: "
                f"min={float(np.nanmin(hi_v3_test)):.4f}, "
                f"max={float(np.nanmax(hi_v3_test)):.4f}, "
                f"mean={float(np.nanmean(hi_v3_test)):.4f}"
            )
        except Exception as e:  # pragma: no cover - diagnostics only
            print(f"[fd004_worst20] WARNING: Could not compute HI_phys_v3 for diagnostics: {e}")

    return df_test_fe, feature_cols


def load_fd004_sequences_for_units(
    unit_ids: List[int],
    trajectories: Dict[int, EngineTrajectory],
    df_test_fe: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Build a per-unit dictionary with:
      - 'time':      array of cycle indices
      - 'rul_true_seq': true RUL trajectory (as built by inference helper)
      - 'rul_pred_seq': predicted RUL trajectory (EOL prediction, repeated)
      - 'hi_phys_seq': HI_phys_v3 over time (if available)
      - 'hi_damage_seq': HI_damage over time (if available from trajectories)
      - 'hi_cal_seq': HI_cal_v2 over time (if available from trajectories)
      - 'cond_id': ConditionID (scalar)
      - 'features': full feature matrix [T, F] from df_test_fe
    """
    per_unit: Dict[int, Dict[str, np.ndarray]] = {}

    for uid in unit_ids:
        if uid not in trajectories:
            continue

        traj = trajectories[uid]
        cycles = np.asarray(traj.cycles, dtype=float)

        # RUL trajectories as provided by inference (true_rul and constant pred_rul)
        rul_true_seq = np.asarray(traj.true_rul, dtype=float)
        rul_pred_seq = np.asarray(traj.pred_rul, dtype=float)

        # HI damage / HI_cal from trajectories (may be None)
        hi_damage_seq = (
            np.asarray(traj.hi_damage, dtype=float) if getattr(traj, "hi_damage", None) is not None else None
        )
        hi_cal_seq = (
            np.asarray(traj.hi_cal, dtype=float) if getattr(traj, "hi_cal", None) is not None else None
        )

        # Slice feature-engineered DF for this unit
        g = (
            df_test_fe[df_test_fe["UnitNumber"] == uid]
            .sort_values("TimeInCycles")
            .copy()
        )
        if g.empty:
            continue

        cond_id = int(g["ConditionID"].iloc[0]) if "ConditionID" in g.columns else -1
        features = g[feature_cols].to_numpy(dtype=np.float32)

        # HI_phys_v3 sequence (if available)
        if "HI_phys_v3" in g.columns:
            hi_phys_seq = g["HI_phys_v3"].to_numpy(dtype=float)
        elif "HI_phys_v2" in g.columns:
            hi_phys_seq = g["HI_phys_v2"].to_numpy(dtype=float)
        else:
            hi_phys_seq = np.full(len(g), np.nan, dtype=float)

        per_unit[uid] = {
            "time": cycles,
            "rul_true_seq": rul_true_seq,
            "rul_pred_seq": rul_pred_seq,
            "hi_phys_seq": hi_phys_seq,
            "hi_damage_seq": hi_damage_seq,
            "hi_cal_seq": hi_cal_seq,
            "cond_id": cond_id,
            "features": features,
        }

    return per_unit


# ---------------------------------------------------------------------------
# 5. Trajectory plots
# ---------------------------------------------------------------------------

def plot_rul_hi_trajectories(
    per_unit: Dict[int, Dict[str, np.ndarray]],
    title: str,
    save_path: Path,
    max_engines: int = 10,
    sigma_map: Dict[int, float] | None = None,
    k: float = 1.0,
) -> None:
    """Plot RUL + HI trajectories for up to max_engines units."""
    unit_ids = list(per_unit.keys())[:max_engines]
    if not unit_ids:
        print(f"[fd004_worst20] No units to plot for {save_path.name}")
        return

    n = len(unit_ids)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for idx, uid in enumerate(unit_ids):
        ax = axes[idx]
        data = per_unit[uid]
        t = data["time"]

        # RUL
        ax2 = ax.twinx()
        ax2.plot(t, data["rul_true_seq"], "b-", label="RUL True", linewidth=2, alpha=0.8)
        ax2.plot(t, data["rul_pred_seq"], "r--", label="RUL Pred (μ)", linewidth=2, alpha=0.8)
        # Optional: show μ ± kσ as additional horizontal dashed lines (when sigma exists)
        if sigma_map is not None and uid in sigma_map and np.isfinite(float(sigma_map[uid])):
            sigma = float(sigma_map[uid])
            mu = float(np.asarray(data["rul_pred_seq"], dtype=float)[-1])
            ax2.plot(
                t,
                np.full_like(t, mu - k * sigma, dtype=float),
                "r:",
                linewidth=1.5,
                alpha=0.8,
                label=f"μ-{k}σ",
            )
            ax2.plot(
                t,
                np.full_like(t, mu + k * sigma, dtype=float),
                "r:",
                linewidth=1.5,
                alpha=0.8,
                label=f"μ+{k}σ",
            )
        ax2.set_ylabel("RUL [cycles]", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        # HI_phys / HI_damage / HI_cal on primary axis
        ax.plot(t, data["hi_phys_seq"], "g-", label="HI_phys_v3", linewidth=1.8, alpha=0.8)

        if data.get("hi_damage_seq") is not None:
            ax.plot(t, data["hi_damage_seq"], "m-", label="HI_damage", linewidth=1.5, alpha=0.7)

        if data.get("hi_cal_seq") is not None:
            ax.plot(t, data["hi_cal_seq"], "c--", label="HI_cal_v2", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Health / HI", color="g")
        ax.tick_params(axis="y", labelcolor="g")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Unit {uid}")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Hide unused subplots
    for j in range(len(unit_ids), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved trajectory plot: {save_path}")


# ---------------------------------------------------------------------------
# Truncation / censoring diagnostics (FD004 test is right-censored)
# ---------------------------------------------------------------------------

def plot_truncation_diagnostics(
    df: pd.DataFrame,
    worst_unit_ids: List[int],
    save_path: Path,
) -> None:
    """
    Truncation / censoring diagnostics for FD004 test:
      - hist(true_rul_last) worst20 vs rest
      - hist(num_cycles) worst20 vs rest
      - scatter(true_rul_last vs error_last), highlight worst20
      - scatter(num_cycles vs error_last), highlight worst20
      - boxplot(true_rul_last) worst20 vs rest
      - scatter(true_rul_last vs abs_error_last) + simple linear trendline
    """
    df = df.copy()
    df["group"] = np.where(df["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    df["abs_error"] = df["eol_error"].abs()

    worst = df[df["group"] == "worst20"]
    rest = df[df["group"] == "rest"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.reshape(-1)

    # 1) Histogram of true RUL at last observed cycle
    ax = axes[0]
    ax.hist(rest["eol_rul_true"].dropna().values, bins=20, alpha=0.6, label="rest")
    ax.hist(worst["eol_rul_true"].dropna().values, bins=20, alpha=0.8, label="worst20")
    ax.set_title("True RUL at last observed cycle (test / right-censored)")
    ax.set_xlabel("true_rul_last")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Histogram of num_cycles
    ax = axes[1]
    ax.hist(rest["num_cycles"].dropna().values, bins=20, alpha=0.6, label="rest")
    ax.hist(worst["num_cycles"].dropna().values, bins=20, alpha=0.8, label="worst20")
    ax.set_title("Num cycles in test trajectory (censoring length)")
    ax.set_xlabel("num_cycles")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Scatter: true_rul_last vs error_last
    ax = axes[2]
    ax.scatter(rest["eol_rul_true"], rest["eol_error"], s=18, alpha=0.4, label="rest")
    ax.scatter(worst["eol_rul_true"], worst["eol_error"], s=30, alpha=0.9, label="worst20")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Error vs true_rul_last (pred - true)")
    ax.set_xlabel("true_rul_last")
    ax.set_ylabel("error_last")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4) Scatter: num_cycles vs error_last
    ax = axes[3]
    ax.scatter(rest["num_cycles"], rest["eol_error"], s=18, alpha=0.4, label="rest")
    ax.scatter(worst["num_cycles"], worst["eol_error"], s=30, alpha=0.9, label="worst20")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Error vs num_cycles")
    ax.set_xlabel("num_cycles")
    ax.set_ylabel("error_last")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5) Boxplot: true_rul_last worst vs rest
    ax = axes[4]
    ax.boxplot(
        [
            worst["eol_rul_true"].dropna().values,
            rest["eol_rul_true"].dropna().values,
        ],
        labels=["worst20", "rest"],
        showfliers=True,
    )
    ax.set_title("true_rul_last distribution (worst20 vs rest)")
    ax.grid(True, alpha=0.3)

    # 6) Scatter: true_rul_last vs abs_error_last (+ trendline)
    ax = axes[5]
    ax.scatter(rest["eol_rul_true"], rest["abs_error"], s=18, alpha=0.35, label="rest")
    ax.scatter(worst["eol_rul_true"], worst["abs_error"], s=30, alpha=0.9, label="worst20")
    ax.set_title("|error| vs true_rul_last")
    ax.set_xlabel("true_rul_last")
    ax.set_ylabel("abs_error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Simple linear fit for all points (optional, robust)
    try:
        x = df["eol_rul_true"].to_numpy(dtype=float)
        y = df["abs_error"].to_numpy(dtype=float)
        msk = np.isfinite(x) & np.isfinite(y)
        if int(msk.sum()) > 5:
            a, b = np.polyfit(x[msk], y[msk], 1)
            xs = np.linspace(float(np.nanmin(x[msk])), float(np.nanmax(x[msk])), 100)
            ys = a * xs + b
            ax.plot(xs, ys, linewidth=2, alpha=0.8, color="k")
    except Exception:
        pass

    plt.suptitle("Truncation / Censoring Diagnostics (FD004 test)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved truncation diagnostics plot: {save_path}")


def plot_uncertainty_diagnostics(
    df: pd.DataFrame,
    worst_unit_ids: List[int],
    save_path: Path,
) -> None:
    """
    Uncertainty diagnostics (if encoder predicts sigma at last observed cycle):
      - scatter(sigma_last vs abs_error_last), highlight worst20
      - scatter(sigma_last vs error_last), highlight worst20
      - prints coverage estimates at 1σ and 2σ
    """
    # Prefer sigma_last, fall back to legacy eol_sigma
    sigma_col = "sigma_last" if "sigma_last" in df.columns else ("eol_sigma" if "eol_sigma" in df.columns else None)
    if sigma_col is None:
        print("[fd004_worst20] No sigma_last/eol_sigma column found; skipping uncertainty diagnostics.")
        return

    df = df.copy()
    df["group"] = np.where(df["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    df["abs_error"] = df["eol_error"].abs()
    df["sigma_last"] = pd.to_numeric(df[sigma_col], errors="coerce")

    m = np.isfinite(df["sigma_last"].to_numpy(dtype=float))
    if int(m.sum()) < 5:
        print("[fd004_worst20] Not enough finite sigma values; skipping uncertainty diagnostics.")
        return

    worst = df[df["group"] == "worst20"]
    rest = df[df["group"] == "rest"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(rest["sigma_last"], rest["abs_error"], s=18, alpha=0.4, label="rest")
    ax.scatter(worst["sigma_last"], worst["abs_error"], s=30, alpha=0.9, label="worst20")
    ax.set_title("|error_last| vs sigma_last (encoder)")
    ax.set_xlabel("sigma_last (cycles)")
    ax.set_ylabel("abs_error_last (cycles)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(rest["sigma_last"], rest["eol_error"], s=18, alpha=0.4, label="rest")
    ax.scatter(worst["sigma_last"], worst["eol_error"], s=30, alpha=0.9, label="worst20")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="k")
    ax.set_title("error_last (pred-true) vs sigma_last (encoder)")
    ax.set_xlabel("sigma_last (cycles)")
    ax.set_ylabel("error_last (cycles)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("Uncertainty Diagnostics (FD004 test / right-censored)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved uncertainty diagnostics plot: {save_path}")

    sigma = df["sigma_last"].to_numpy(dtype=float)
    err = df["eol_error"].to_numpy(dtype=float)
    m2 = np.isfinite(sigma) & np.isfinite(err) & (sigma > 0)
    if int(m2.sum()) > 5:
        cov1 = float(np.mean(np.abs(err[m2]) <= 1.0 * sigma[m2]))
        cov2 = float(np.mean(np.abs(err[m2]) <= 2.0 * sigma[m2]))
        print("\n=== Encoder sigma coverage @ last observed cycle (test / right-censored) ===")
        print(f"coverage_1sigma = {cov1:.3f}")
        print(f"coverage_2sigma = {cov2:.3f}")


def plot_rul_band_scatter(
    df: pd.DataFrame,
    worst_unit_ids: List[int],
    save_path: Path,
    k: float = 1.0,
) -> None:
    """
    Scatter: true_rul_last (x) vs mu_last (y) with vertical error bars mu ± k*sigma.
    Highlights worst20 vs rest and overlays y=x.
    """
    d = df.copy()
    d["group"] = np.where(d["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    d["mu_last"] = pd.to_numeric(d.get("mu_last", d.get("eol_rul_pred", np.nan)), errors="coerce")
    d["true_last"] = pd.to_numeric(d.get("true_rul_last", d.get("eol_rul_true", np.nan)), errors="coerce")
    d["sigma_last"] = pd.to_numeric(d.get("sigma_last", np.nan), errors="coerce")

    d_ok = d[np.isfinite(d["sigma_last"].to_numpy(dtype=float))].copy()
    if d_ok.empty:
        print("[fd004_worst20] No sigma_last available, skipping band scatter.")
        return

    rest = d_ok[d_ok["group"] == "rest"]
    worst = d_ok[d_ok["group"] == "worst20"]

    plt.figure(figsize=(9, 8))
    plt.errorbar(
        rest["true_last"],
        rest["mu_last"],
        yerr=k * rest["sigma_last"],
        fmt="o",
        markersize=4,
        alpha=0.35,
        capsize=2,
        label="rest (μ ± kσ)",
    )
    plt.errorbar(
        worst["true_last"],
        worst["mu_last"],
        yerr=k * worst["sigma_last"],
        fmt="o",
        markersize=6,
        alpha=0.9,
        capsize=2,
        label="worst20 (μ ± kσ)",
    )

    x = d_ok["true_last"].to_numpy(dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="perfect (y=x)")

    # Optional conservative points: μ - kσ
    plt.scatter(
        rest["true_last"],
        (rest["mu_last"] - k * rest["sigma_last"]),
        s=10,
        alpha=0.25,
        label="rest (μ-kσ)",
    )
    plt.scatter(
        worst["true_last"],
        (worst["mu_last"] - k * worst["sigma_last"]),
        s=18,
        alpha=0.6,
        label="worst20 (μ-kσ)",
    )

    plt.xlabel("True RUL at last observed cycle")
    plt.ylabel("Predicted RUL (μ) with uncertainty band")
    plt.title(f"FD004 Test (right-censored): True vs Predicted RUL with ±{k}σ band")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved RUL band scatter: {save_path}")


def plot_rul_band_coverage(
    df: pd.DataFrame,
    save_path: Path,
    k_list: List[float] = [1.0, 2.0],
) -> None:
    """
    Coverage vs true_rul_last buckets for bands [μ-kσ, μ+kσ].
    """
    d = df.copy()
    d["mu_last"] = pd.to_numeric(d.get("mu_last", d.get("eol_rul_pred", np.nan)), errors="coerce")
    d["true_last"] = pd.to_numeric(d.get("true_rul_last", d.get("eol_rul_true", np.nan)), errors="coerce")
    d["sigma_last"] = pd.to_numeric(d.get("sigma_last", np.nan), errors="coerce")

    d = d[np.isfinite(d["sigma_last"].to_numpy(dtype=float))].copy()
    if d.empty:
        print("[fd004_worst20] No sigma_last available, skipping coverage plot.")
        return

    bins = [-1, 25, 50, 75, 100, 10_000]
    labels = ["0-25", "25-50", "50-75", "75-100", "100+"]
    d["bin"] = pd.cut(d["true_last"], bins=bins, labels=labels)

    cov: Dict[float, pd.Series] = {}
    for kk in k_list:
        lower = d["mu_last"] - kk * d["sigma_last"]
        upper = d["mu_last"] + kk * d["sigma_last"]
        inside = (d["true_last"] >= lower) & (d["true_last"] <= upper)
        cov[kk] = d.assign(inside=inside).groupby("bin")["inside"].mean()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35 if len(k_list) > 1 else 0.6
    for i, kk in enumerate(k_list):
        vals = [float(cov[kk].get(lbl, np.nan)) for lbl in labels]
        offset = (i - (len(k_list) - 1) / 2) * width
        plt.bar(x + offset, vals, width=width, label=f"coverage (±{kk}σ)")

    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction where true RUL is inside band")
    plt.xlabel("True RUL at last observed cycle bucket")
    plt.title("Uncertainty Coverage vs Horizon (FD004 test / right-censored)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved RUL band coverage: {save_path}")


def compute_k_sweep_metrics(
    df: pd.DataFrame,
    worst_unit_ids: List[int],
    k_list: List[float],
    max_rul: float | None = None,
) -> pd.DataFrame:
    """
    Compute k-sweep metrics for:
      - μ_last (point estimate)
      - RUL_safe(k) = μ_last - k * σ_last

    Metrics are computed for groups: worst20, rest, all.
    FD004 test is right-censored: "last observed cycle" metrics only.
    """
    d = df.copy()
    d["group"] = np.where(d["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    d["mu_last"] = pd.to_numeric(d.get("mu_last", d.get("eol_rul_pred", np.nan)), errors="coerce")
    d["true_last"] = pd.to_numeric(d.get("true_rul_last", d.get("eol_rul_true", np.nan)), errors="coerce")
    d["sigma_last"] = pd.to_numeric(d.get("sigma_last", np.nan), errors="coerce")

    d = d[np.isfinite(d["sigma_last"].to_numpy(dtype=float))].copy()
    if d.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for grp in ["worst20", "rest", "all"]:
        dg = d if grp == "all" else d[d["group"] == grp]
        if dg.empty:
            continue

        mu = dg["mu_last"].to_numpy(dtype=float)
        tru = dg["true_last"].to_numpy(dtype=float)
        sig = dg["sigma_last"].to_numpy(dtype=float)

        err_mu = mu - tru
        mae_mu = float(np.mean(np.abs(err_mu)))
        bias_mu = float(np.mean(err_mu))

        for k in k_list:
            safe = mu - float(k) * sig
            # Keep safe in a plausible RUL range for evaluation (avoids negative RUL artifacts)
            safe = np.maximum(safe, 0.0)
            if max_rul is not None and np.isfinite(max_rul):
                safe = np.minimum(safe, float(max_rul))

            err_safe = safe - tru
            mae_safe = float(np.mean(np.abs(err_safe)))
            bias_safe = float(np.mean(err_safe))

            conservative_rate = float(np.mean(safe <= tru))

            lower = mu - float(k) * sig
            upper = mu + float(k) * sig
            coverage = float(np.mean((tru >= lower) & (tru <= upper)))

            rows.append(
                {
                    "group": grp,
                    "k": float(k),
                    "coverage": coverage,
                    "conservative_rate": conservative_rate,
                    "mae_mu": mae_mu,
                    "bias_mu": bias_mu,
                    "mae_safe": mae_safe,
                    "bias_safe": bias_safe,
                    "n": int(len(dg)),
                }
            )

    return pd.DataFrame(rows)


def plot_k_sweep_summary(df_k: pd.DataFrame, save_path: Path) -> None:
    """Plot coverage and conservative_rate vs k for groups all/rest/worst20."""
    if df_k.empty:
        print("[fd004_worst20] No k-sweep metrics available.")
        return

    plt.figure(figsize=(11, 5))

    # left: coverage
    plt.subplot(1, 2, 1)
    for grp in ["all", "rest", "worst20"]:
        g = df_k[df_k["group"] == grp].sort_values("k")
        if g.empty:
            continue
        plt.plot(g["k"], g["coverage"], marker="o", label=grp)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k")
    plt.ylabel("Coverage: true ∈ [μ−kσ, μ+kσ]")
    plt.title("Coverage vs k (FD004 test / right-censored)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # right: conservative rate
    plt.subplot(1, 2, 2)
    for grp in ["all", "rest", "worst20"]:
        g = df_k[df_k["group"] == grp].sort_values("k")
        if g.empty:
            continue
        plt.plot(g["k"], g["conservative_rate"], marker="o", label=grp)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k")
    plt.ylabel("Conservative rate: (μ−kσ) ≤ true")
    plt.title("Conservatism vs k (FD004 test / right-censored)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved k-sweep summary: {save_path}")


def plot_safe_vs_mu_scatter_k123(
    df: pd.DataFrame,
    worst_unit_ids: List[int],
    save_path: Path,
    k_list: List[float],
    max_rul: float | None = None,
) -> None:
    """
    3-panel scatter (k=1/2/3): x=true_last, y=μ and y=μ−kσ (safe), worst20 highlighted.
    """
    d = df.copy()
    d["group"] = np.where(d["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    d["mu_last"] = pd.to_numeric(d.get("mu_last", d.get("eol_rul_pred", np.nan)), errors="coerce")
    d["true_last"] = pd.to_numeric(d.get("true_rul_last", d.get("eol_rul_true", np.nan)), errors="coerce")
    d["sigma_last"] = pd.to_numeric(d.get("sigma_last", np.nan), errors="coerce")
    d = d[np.isfinite(d["sigma_last"].to_numpy(dtype=float))].copy()
    if d.empty:
        print("[fd004_worst20] No sigma_last available, skipping safe scatter.")
        return

    fig, axes = plt.subplots(1, len(k_list), figsize=(6 * len(k_list), 5))
    if len(k_list) == 1:
        axes = [axes]

    lo = float(np.nanmin(d["true_last"].to_numpy(dtype=float)))
    hi = float(np.nanmax(d["true_last"].to_numpy(dtype=float)))

    for ax, k in zip(axes, k_list):
        safe = d["mu_last"] - float(k) * d["sigma_last"]
        safe = np.maximum(safe.to_numpy(dtype=float), 0.0)
        if max_rul is not None and np.isfinite(max_rul):
            safe = np.minimum(safe, float(max_rul))

        rest = d[d["group"] == "rest"]
        worst = d[d["group"] == "worst20"]

        ax.scatter(rest["true_last"], rest["mu_last"], s=14, alpha=0.25, label="rest μ")
        ax.scatter(rest["true_last"], safe[rest.index], s=14, alpha=0.25, label=f"rest μ-{k}σ")

        ax.scatter(worst["true_last"], worst["mu_last"], s=35, alpha=0.9, label="worst20 μ")
        ax.scatter(worst["true_last"], safe[worst.index], s=35, alpha=0.9, label=f"worst20 μ-{k}σ")

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y=x")
        ax.set_xlabel("True RUL at last observed cycle")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(f"μ vs μ-{k}σ (safe)\n(FD004 test / right-censored)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved safe-vs-mu scatter: {save_path}")


def print_truncation_bucket_stats(eol_df: pd.DataFrame, worst_unit_ids: List[int]) -> None:
    """Bucket analysis by true_rul_last (eol_rul_true) for FD004 test."""
    df = eol_df.copy()
    df["group"] = np.where(df["unit_id"].isin(worst_unit_ids), "worst20", "rest")
    df["abs_error"] = df["eol_error"].abs()

    bins = [-1, 25, 50, 75, 100, 10_000]
    labels = ["0-25", "25-50", "50-75", "75-100", "100+"]
    df["rul_last_bin"] = pd.cut(df["eol_rul_true"], bins=bins, labels=labels)

    agg = df.groupby("rul_last_bin").agg(
        n=("unit_id", "count"),
        mean_error=("eol_error", "mean"),
        median_error=("eol_error", "median"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        worst_frac=("group", lambda s: float((s == "worst20").mean())),
    )
    print("\n=== Truncation bucket stats by true_rul_last (test / right-censored) ===")
    print(agg.to_string())


# ---------------------------------------------------------------------------
# 6. Per-engine summary and stats (worst vs rest)
# ---------------------------------------------------------------------------

def build_engine_level_summary(
    per_unit_all: Dict[int, Dict[str, np.ndarray]],
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge trajectory-based stats into base_df:
      - hi_phys_eol, hi_damage_eol, hi_cal_eol
      - mean/std of HI_phys
    """
    rows = []
    for uid, data in per_unit_all.items():
        hi_phys_seq = np.asarray(data["hi_phys_seq"], dtype=float)
        hi_damage_seq = (
            np.asarray(data["hi_damage_seq"], dtype=float)
            if data.get("hi_damage_seq") is not None
            else None
        )
        hi_cal_seq = (
            np.asarray(data["hi_cal_seq"], dtype=float)
            if data.get("hi_cal_seq") is not None
            else None
        )

        hi_phys_eol = float(hi_phys_seq[-1]) if len(hi_phys_seq) > 0 else np.nan
        hi_phys_mean = float(np.nanmean(hi_phys_seq)) if len(hi_phys_seq) > 0 else np.nan
        hi_phys_std = float(np.nanstd(hi_phys_seq)) if len(hi_phys_seq) > 0 else np.nan

        hi_damage_eol = float(hi_damage_seq[-1]) if hi_damage_seq is not None and len(hi_damage_seq) > 0 else np.nan
        hi_cal_eol = float(hi_cal_seq[-1]) if hi_cal_seq is not None and len(hi_cal_seq) > 0 else np.nan

        rows.append(
            {
                "unit_id": int(uid),
                "hi_phys_eol": hi_phys_eol,
                "hi_phys_mean": hi_phys_mean,
                "hi_phys_std": hi_phys_std,
                "hi_damage_eol": hi_damage_eol,
                "hi_cal_eol": hi_cal_eol,
            }
        )

    extra_df = pd.DataFrame(rows)
    merged = base_df.merge(extra_df, on="unit_id", how="left")
    return merged


def plot_worst_vs_rest_stats(
    df_summary: pd.DataFrame,
    worst_unit_ids: List[int],
    save_path: Path,
) -> None:
    """Compare ConditionID and basic stats for worst-20 vs rest."""
    df = df_summary.copy()
    df["group"] = np.where(df["unit_id"].isin(worst_unit_ids), "worst20", "rest")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(-1)

    # 1) ConditionID distribution
    ax = axes[0]
    for grp, df_g in df.groupby("group"):
        counts = df_g["cond_id"].value_counts().sort_index()
        ax.bar(counts.index + (0.2 if grp == "worst20" else -0.2),
               counts.values,
               width=0.4,
               label=grp)
    ax.set_xlabel("ConditionID")
    ax.set_ylabel("Count")
    ax.set_title("ConditionID distribution (worst20 vs rest)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Helper for boxplot
    def _box(ax, col: str, title: str) -> None:
        data_worst = df.loc[df["group"] == "worst20", col].dropna()
        data_rest = df.loc[df["group"] == "rest", col].dropna()
        ax.boxplot(
            [data_worst.values, data_rest.values],
            labels=["worst20", "rest"],
            showfliers=True,
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # 2) EOL error
    _box(axes[1], "eol_error", "Last-cycle error (pred - true) [test / right-censored]")

    # 3) num_cycles
    _box(axes[2], "num_cycles", "Num cycles per engine")

    # 4) HI_phys_eol
    if "hi_phys_eol" in df.columns:
        _box(axes[3], "hi_phys_eol", "HI_phys_v3 at last observed cycle")

    # 5) HI_damage_eol
    if "hi_damage_eol" in df.columns:
        _box(axes[4], "hi_damage_eol", "HI_damage at last observed cycle")

    # 6) HI_cal_eol
    if "hi_cal_eol" in df.columns:
        _box(axes[5], "hi_cal_eol", "HI_cal_v2 at last observed cycle")

    plt.suptitle("Worst-20 vs Rest – Basic Stats", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved worst-vs-rest stats plot: {save_path}")


# ---------------------------------------------------------------------------
# 7. Residual sensor analysis
# ---------------------------------------------------------------------------

def compute_residual_stats_for_group(
    df_full: pd.DataFrame,
    unit_ids: List[int],
    residual_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-residual column stats for a group of engines:
      - mean_abs_resid
      - std_resid
      - p90_abs_resid
    aggregated over all rows belonging to unit_ids.
    """
    df_group = df_full[df_full["UnitNumber"].isin(unit_ids)].copy()
    stats_rows = []

    for col in residual_cols:
        vals = df_group[col].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        mean_abs = float(np.nanmean(np.abs(vals)))
        std = float(np.nanstd(vals))
        p90 = float(np.nanpercentile(np.abs(vals), 90.0))
        stats_rows.append(
            {
                "residual_col": col,
                "mean_abs_resid": mean_abs,
                "std_resid": std,
                "p90_abs_resid": p90,
            }
        )

    return pd.DataFrame(stats_rows)


def plot_residual_deltas(
    resid_stats_worst: pd.DataFrame,
    resid_stats_rest: pd.DataFrame,
    save_path: Path,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute and plot delta in mean |resid| between worst20 and rest.

    Returns the full delta DataFrame for further textual diagnostics.
    """
    merged = resid_stats_worst.merge(
        resid_stats_rest,
        on="residual_col",
        suffixes=("_worst", "_rest"),
    )
    merged["delta_mean_abs_resid"] = (
        merged["mean_abs_resid_worst"] - merged["mean_abs_resid_rest"]
    )
    merged_sorted = merged.sort_values("delta_mean_abs_resid", ascending=False)

    top = merged_sorted.head(top_k)

    plt.figure(figsize=(10, 5))
    plt.bar(
        range(len(top)),
        top["delta_mean_abs_resid"].values,
        tick_label=[c.replace("Resid_", "") for c in top["residual_col"].values],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ mean |residual| (worst20 - rest)")
    plt.title("Top residual sensors (worst 20 vs rest)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fd004_worst20] Saved residual delta plot: {save_path}")

    return merged_sorted


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main(run_name: str = RUN_NAME, dataset: str = DATASET) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fd004_worst20] Using device: {device}")
    results_dir, summary_path, _, _ = _resolve_run_paths(dataset=dataset, run_name=run_name)
    print(f"[fd004_worst20] Run: {run_name}")
    print(f"[fd004_worst20] Dataset: {dataset}")
    print(f"[fd004_worst20] Results dir: {results_dir}")

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # 1) Load summary
    summary = load_encoder_summary(summary_path)

    # 2) Build per-engine EOL metrics + trajectories
    eol_df, trajectories, df_test_raw = build_eol_metrics_df(
        experiment_dir=results_dir,
        dataset_name=dataset,
        device=device,
    )

    # 3) Select worst 20 and best 20
    worst_late_df = (
        eol_df[eol_df["eol_error"] > 0.0]
        .sort_values("eol_error", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    best_df = (
        eol_df.assign(abs_error=eol_df["eol_error"].abs())
        .sort_values("abs_error", ascending=True)
        .head(20)
        .reset_index(drop=True)
    )

    print("\n=== Worst 20 engines by late EOL error ===")
    print(
        worst_late_df[
            ["unit_id", "cond_id", "num_cycles", "eol_rul_true", "eol_rul_pred", "eol_error"]
        ]
    )

    print("\n=== Best 20 engines by |EOL error| ===")
    print(
        best_df[
            ["unit_id", "cond_id", "num_cycles", "eol_rul_true", "eol_rul_pred", "eol_error"]
        ]
    )

    worst_units = worst_late_df["unit_id"].tolist()
    best_units = best_df["unit_id"].tolist()
    all_units = eol_df["unit_id"].tolist()

    # sigma map for per-engine band overlays
    sigma_map: Dict[int, float] = dict(
        zip(
            eol_df["unit_id"].astype(int).tolist(),
            pd.to_numeric(eol_df.get("sigma_last", np.nan), errors="coerce").astype(float).tolist(),
        )
    )

    max_rul = float(summary.get("max_rul", 125.0) or 125.0)
    sigma_any = bool(np.isfinite(pd.to_numeric(eol_df.get("sigma_last", np.nan), errors="coerce")).any())

    # ------------------------------------------------------------------
    # Truncation / censoring diagnostics (FD004 test is right-censored)
    # ------------------------------------------------------------------
    plot_truncation_diagnostics(
        df=eol_df,
        worst_unit_ids=worst_units,
        save_path=results_dir / "diagnostics_truncation.png",
    )
    print_truncation_bucket_stats(eol_df=eol_df, worst_unit_ids=worst_units)
    plot_uncertainty_diagnostics(
        df=eol_df,
        worst_unit_ids=worst_units,
        save_path=results_dir / "diagnostics_uncertainty.png",
    )

    # Uncertainty band visualizations (μ ± σ) for uncertainty runs
    if sigma_any:
        plot_rul_band_scatter(
            df=eol_df,
            worst_unit_ids=worst_units,
            save_path=results_dir / "diagnostics_rul_band_scatter.png",
            k=1.0,
        )
        plot_rul_band_coverage(
            df=eol_df,
            save_path=results_dir / "diagnostics_rul_band_coverage.png",
            k_list=[1.0, 2.0],
        )

        # k-sweep: conservative RUL_safe(k) = μ - kσ for k in {1,2,3}
        df_k = compute_k_sweep_metrics(
            df=eol_df,
            worst_unit_ids=worst_units,
            k_list=K_LIST,
            max_rul=max_rul,
        )
        if not df_k.empty:
            print("\n=== k-sweep metrics (overall/rest/worst20) ===")
            print(df_k.sort_values(["group", "k"]).to_string(index=False))
            plot_k_sweep_summary(
                df_k=df_k,
                save_path=results_dir / "diagnostics_k_sweep_summary.png",
            )
            plot_safe_vs_mu_scatter_k123(
                df=eol_df,
                worst_unit_ids=worst_units,
                save_path=results_dir / "diagnostics_safe_vs_mu_scatter_k123.png",
                k_list=K_LIST,
                max_rul=max_rul,
            )
        else:
            print("[fd004_worst20] sigma_last present but no valid rows for k-sweep metrics.")
    else:
        print("[fd004_worst20] sigma_last not available -> skipping uncertainty band + k-sweep diagnostics.")

    df_corr = eol_df.copy()
    df_corr["abs_error"] = df_corr["eol_error"].abs()

    def _corr(a: str, b: str) -> float:
        aa = df_corr[a].to_numpy(dtype=float)
        bb = df_corr[b].to_numpy(dtype=float)
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < 3:
            return float("nan")
        return float(np.corrcoef(aa[m], bb[m])[0, 1])

    print("\n=== Correlations (test / right-censored) ===")
    print(f"corr(true_rul_last, error_last)     = {_corr('eol_rul_true','eol_error'):.4f}")
    print(f"corr(true_rul_last, abs_error_last) = {_corr('eol_rul_true','abs_error'):.4f}")
    print(f"corr(num_cycles, error_last)        = {_corr('num_cycles','eol_error'):.4f}")
    print(f"corr(num_cycles, abs_error_last)    = {_corr('num_cycles','abs_error'):.4f}")

    # 4) Feature pipeline for residuals + HI_phys
    df_test_fe, feature_cols = build_feature_pipeline_for_fd004(
        summary=summary,
        dataset_name=dataset,
        max_rul=int(summary.get("max_rul", 125) or 125),
        past_len=30,
    )

    # 5) Per-unit trajectories (worst, best, all)
    per_unit_worst = load_fd004_sequences_for_units(
        unit_ids=worst_units,
        trajectories=trajectories,
        df_test_fe=df_test_fe,
        feature_cols=feature_cols,
    )
    per_unit_best = load_fd004_sequences_for_units(
        unit_ids=best_units,
        trajectories=trajectories,
        df_test_fe=df_test_fe,
        feature_cols=feature_cols,
    )
    per_unit_all = load_fd004_sequences_for_units(
        unit_ids=all_units,
        trajectories=trajectories,
        df_test_fe=df_test_fe,
        feature_cols=feature_cols,
    )

    # 6) Trajectory sanity-check plots
    plot_rul_hi_trajectories(
        per_unit=per_unit_worst,
        title=f"{run_name} – Worst 20 engines (RUL + HI ± σ) [test / right-censored]",
        save_path=results_dir / "diagnostics_worst20_rul_hi.png",
        sigma_map=sigma_map,
        k=1.0,
    )
    plot_rul_hi_trajectories(
        per_unit=per_unit_best,
        title=f"{run_name} – Best 20 engines (RUL + HI ± σ) [test / right-censored]",
        save_path=results_dir / "diagnostics_best20_rul_hi.png",
        sigma_map=sigma_map,
        k=1.0,
    )

    # 7) Per-engine summary + basic stats (worst vs rest)
    df_summary = build_engine_level_summary(per_unit_all, eol_df)
    plot_worst_vs_rest_stats(
        df_summary=df_summary,
        worst_unit_ids=worst_units,
        save_path=results_dir / "diagnostics_worst_vs_rest_stats.png",
    )

    # 8) Residual analysis (worst vs rest)
    residual_cols = [c for c in df_test_fe.columns if c.startswith("Resid_")]
    if residual_cols:
        resid_stats_worst = compute_residual_stats_for_group(
            df_full=df_test_fe,
            unit_ids=worst_units,
            residual_cols=residual_cols,
        )
        resid_stats_rest = compute_residual_stats_for_group(
            df_full=df_test_fe,
            unit_ids=[u for u in all_units if u not in worst_units],
            residual_cols=residual_cols,
        )
        resid_deltas = plot_residual_deltas(
            resid_stats_worst=resid_stats_worst,
            resid_stats_rest=resid_stats_rest,
            save_path=results_dir / "diagnostics_worst20_residual_deltas.png",
            top_k=10,
        )
    else:
        resid_deltas = pd.DataFrame()
        print("[fd004_worst20] No Resid_* columns found in df_test_fe – skipping residual delta plot.")

    # 9) Short textual diagnosis
    print("\n=== ConditionID distribution (worst 20 vs rest) ===")
    print(
        df_summary.assign(group=np.where(df_summary["unit_id"].isin(worst_units), "worst20", "rest"))
        .groupby(["group", "cond_id"])["unit_id"]
        .count()
        .unstack(fill_value=0)
    )

    print("\n=== HI_phys_v3 / HI_damage / HI_cal_v2 at EOL – worst20 vs rest (mean ± std) ===")
    for col in ["hi_phys_eol", "hi_damage_eol", "hi_cal_eol"]:
        if col not in df_summary.columns:
            continue
        stats = (
            df_summary.assign(group=np.where(df_summary["unit_id"].isin(worst_units), "worst20", "rest"))
            .groupby("group")[col]
            .agg(["mean", "std", "median"])
        )
        print(f"\n{col}:")
        print(stats)

    if not resid_deltas.empty:
        print(
            "\n=== Top 5 residual sensors with largest mean |resid| increase (worst 20 vs rest) ==="
        )
        print(
            resid_deltas[
                ["residual_col", "mean_abs_resid_worst", "mean_abs_resid_rest", "delta_mean_abs_resid"]
            ]
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FD004 worst-20 engine diagnosis for a trained encoder run."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=RUN_NAME,
        help="Encoder experiment name (results/<dataset>/<run_name>/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Dataset name (default: FD004).",
    )
    args = parser.parse_args()
    main(run_name=args.run_name, dataset=args.dataset)


