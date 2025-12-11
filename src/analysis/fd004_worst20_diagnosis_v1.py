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

RESULTS_DIR = Path("results") / DATASET.lower() / RUN_NAME
SUMMARY_PATH = RESULTS_DIR / "summary.json"
CHECKPOINT_PATH = RESULTS_DIR / f"eol_full_lstm_best_{RUN_NAME}.pt"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"


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

    rows = []
    for m in eol_metrics:
        uid = int(m.unit_id)
        cond_id = int(cond_id_map.loc[uid]) if cond_id_map is not None and uid in cond_id_map.index else -1
        num_cycles = int(num_cycles_map.loc[uid]) if uid in num_cycles_map.index else 0
        rows.append(
            {
                "unit_id": uid,
                "cond_id": cond_id,
                "num_cycles": num_cycles,
                "eol_rul_true": float(m.true_rul),
                "eol_rul_pred": float(m.pred_rul),
                "eol_error": float(m.error),
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
        ax2.plot(t, data["rul_pred_seq"], "r--", label="RUL Pred", linewidth=2, alpha=0.8)
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
    _box(axes[1], "eol_error", "EOL error (pred - true)")

    # 3) num_cycles
    _box(axes[2], "num_cycles", "Num cycles per engine")

    # 4) HI_phys_eol
    if "hi_phys_eol" in df.columns:
        _box(axes[3], "hi_phys_eol", "HI_phys_v3 at EOL")

    # 5) HI_damage_eol
    if "hi_damage_eol" in df.columns:
        _box(axes[4], "hi_damage_eol", "HI_damage at EOL")

    # 6) HI_cal_eol
    if "hi_cal_eol" in df.columns:
        _box(axes[5], "hi_cal_eol", "HI_cal_v2 at EOL")

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

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fd004_worst20] Using device: {device}")
    print(f"[fd004_worst20] Results dir: {RESULTS_DIR}")

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    # 1) Load summary
    summary = load_encoder_summary(SUMMARY_PATH)

    # 2) Build per-engine EOL metrics + trajectories
    eol_df, trajectories, df_test_raw = build_eol_metrics_df(
        experiment_dir=RESULTS_DIR,
        dataset_name=DATASET,
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

    # 4) Feature pipeline for residuals + HI_phys
    df_test_fe, feature_cols = build_feature_pipeline_for_fd004(
        summary=summary,
        dataset_name=DATASET,
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
        title=f"{RUN_NAME} – Worst 20 engines (RUL + HI)",
        save_path=RESULTS_DIR / "diagnostics_worst20_rul_hi.png",
    )
    plot_rul_hi_trajectories(
        per_unit=per_unit_best,
        title=f"{RUN_NAME} – Best 20 engines (RUL + HI)",
        save_path=RESULTS_DIR / "diagnostics_best20_rul_hi.png",
    )

    # 7) Per-engine summary + basic stats (worst vs rest)
    df_summary = build_engine_level_summary(per_unit_all, eol_df)
    plot_worst_vs_rest_stats(
        df_summary=df_summary,
        worst_unit_ids=worst_units,
        save_path=RESULTS_DIR / "diagnostics_worst_vs_rest_stats.png",
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
            save_path=RESULTS_DIR / "diagnostics_worst20_residual_deltas.png",
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
    main()


