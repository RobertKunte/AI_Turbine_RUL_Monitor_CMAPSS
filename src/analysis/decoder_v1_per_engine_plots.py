from __future__ import annotations

"""
Per-engine trajectory and aggregate diagnostics for Decoder v1 on FD004.

Usage:
    python -m src.analysis.decoder_v1_per_engine_plots \
        --dataset FD004 \
        --run fd004_decoder_v1_from_encoder_v3d \
        --num_per_group 10

This script assumes that:
  - The Decoder v1 run has been trained and saved under:
        results/<dataset>/<run>/
    with:
        decoder_v1_best.pt
        summary_decoder_v1.json
  - The corresponding encoder experiment directory and scaler are stored in
    that summary (as written by src.rul_decoder_training_v1).
"""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

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
from src.eol_full_lstm import build_full_eol_sequences_from_df
from src.feature_safety import remove_rul_leakage
from src.analysis.inference import load_model_from_experiment
from src.models.rul_decoder import RULTrajectoryDecoderV1
from src.rul_decoder_training_v1 import apply_loaded_scaler


@dataclass
class EngineTrajectory:
    unit_id: int
    time_idx: np.ndarray       # [T_engine]
    rul_true: np.ndarray       # [T_engine]
    rul_pred: np.ndarray       # [T_engine]
    hi_phys: np.ndarray        # [T_engine]
    hi_damage: np.ndarray      # [T_engine]


def load_decoder_and_encoder(
    dataset: str,
    run: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, Path, dict]:
    """
    Load frozen encoder and trained Decoder v1 for a given run.

    Returns:
        encoder, decoder, results_dir_run, decoder_summary
    """
    results_dir_run = Path("results") / dataset.lower() / run
    if not results_dir_run.exists():
        raise FileNotFoundError(f"Decoder run directory not found: {results_dir_run}")

    summary_path = results_dir_run / "summary_decoder_v1.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_decoder_v1.json not found at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    encoder_experiment = summary.get("encoder_experiment")
    results_dir_encoder = summary.get("results_dir_encoder")
    if results_dir_encoder is None:
        # Fallback: construct from experiment name
        results_dir_encoder = str(Path("results") / dataset.lower() / encoder_experiment)

    encoder_experiment_dir = Path(results_dir_encoder)
    if not encoder_experiment_dir.exists():
        raise FileNotFoundError(f"Encoder experiment directory not found: {encoder_experiment_dir}")

    # Load encoder via standard inference helper
    encoder, _ = load_model_from_experiment(encoder_experiment_dir, device=device)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load decoder
    decoder_ckpt_path = results_dir_run / "decoder_v1_best.pt"
    if not decoder_ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt_path}")

    ckpt = torch.load(decoder_ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    latent_dim = getattr(encoder, "d_model", None)
    if latent_dim is None:
        raise RuntimeError("Encoder is missing attribute d_model; expected Transformer encoder.")

    decoder = RULTrajectoryDecoderV1(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()

    return encoder, decoder, results_dir_run, summary


def prepare_fd004_test_full_sequences(
    encoder_experiment: str,
    dataset: str,
    past_len: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    np.ndarray,
]:
    """
    Rebuild the ms+DT v2 + residual + twin feature pipeline for the TEST split
    of FD004 (or given dataset), and build full sliding-window sequences
    (one window per time step per engine).

    Returns:
        X_full_test:          [N, past_len, F]
        y_full_test:          [N]
        unit_ids_full_test:   [N]
        cond_ids_full_test:   [N]
        health_phys_seq_test: [N, past_len] or None
        y_test_true:          [num_engines] numpy array (EOL RUL from NASA loader)
    """
    summary_path = Path("results") / dataset.lower() / encoder_experiment / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Encoder summary.json not found at {summary_path}")

    with open(summary_path, "r") as f:
        summary_cfg = json.load(f)

    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # ------------------------------------------------------------------
    # Add synthetic RUL column to TEST set for full sliding-window targets.
    # y_test_true[i] is the (capped) RUL at the last observed test cycle
    # for engine i+1. We reconstruct a simple per-cycle RUL by counting
    # backwards from that value.
    # ------------------------------------------------------------------
    df_test = df_test.copy()
    max_rul_cfg = float(summary_cfg.get("max_rul", 125.0))
    unit_ids_test_unique = np.sort(df_test["UnitNumber"].unique())

    if len(unit_ids_test_unique) != len(y_test_true):
        # Fallback: create a mapping by UnitNumber -> index (NASA style: idx = unit_id - 1)
        print(
            f"[decoder_v1] WARNING: Length mismatch between y_test_true "
            f"({len(y_test_true)}) and unique test units ({len(unit_ids_test_unique)}). "
            "Using UnitNumber-1 as index mapping."
        )

    for i, uid in enumerate(unit_ids_test_unique):
        # Map engine index to y_test_true index
        if len(y_test_true) == len(unit_ids_test_unique):
            rul_end = float(y_test_true[i])
        else:
            idx = int(uid) - 1
            if 0 <= idx < len(y_test_true):
                rul_end = float(y_test_true[idx])
            else:
                rul_end = float(y_test_true[-1])

        df_u = df_test[df_test["UnitNumber"] == uid].sort_values("TimeInCycles")
        L_u = len(df_u)
        # RUL at step k (0..L-1) within test segment: rul_end + (L-1-k)
        rul_seq = rul_end + (L_u - 1 - np.arange(L_u, dtype=np.float32))
        # Optional capping at max_rul_cfg for consistency
        rul_seq = np.minimum(rul_seq, max_rul_cfg)
        df_test.loc[df_u.index, "RUL"] = rul_seq

    # Physics & feature configs (mirror rul_decoder_training_v1 / run_experiments)
    name_lower = encoder_experiment.lower()
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
    df_train_fe = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test_fe = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        df_train_fe = build_condition_features(
            df_train_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )
        df_test_fe = build_condition_features(
            df_test_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin + residuals
    if use_twin_features:
        df_train_fe, twin_model = create_twin_features(
            df_train_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test_fe = twin_model.add_twin_and_residuals(df_test_fe)

    # 4) Temporal / multi-scale features
    df_train_fe = create_all_features(
        df_train_fe,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )
    df_test_fe = create_all_features(
        df_test_fe,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    # Ensure HI_phys_v3 is present on test set as well (for plotting)
    from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

    if "HI_phys_v3" not in df_test_fe.columns:
        hi_v3_test = compute_hi_phys_v3_from_residuals(
            df_test_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_test_fe["HI_phys_v3"] = hi_v3_test

    # Build feature columns (same as run_experiments)
    feature_cols = [
        c
        for c in df_train_fe.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid", "HI_phys_v2", "HI_phys_v3"]
    ]

    # Build full sliding-window sequences from TEST data
    result_test = build_full_eol_sequences_from_df(
        df=df_test_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=summary_cfg.get("max_rul", 125.0),
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    X_full_test, y_full_test, unit_ids_full_test, cond_ids_full_test = result_test[:4]
    health_phys_seq_test = result_test[4] if len(result_test) > 4 else None

    return (
        X_full_test,
        y_full_test,
        unit_ids_full_test,
        cond_ids_full_test,
        health_phys_seq_test,
        np.asarray(y_test_true, dtype=np.float32),
    )


def build_engine_trajectories(
    encoder: nn.Module,
    decoder: nn.Module,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    unit_ids_full: torch.Tensor,
    cond_ids_full: torch.Tensor,
    health_phys_seq_full: torch.Tensor | None,
    device: torch.device,
) -> Tuple[Dict[int, EngineTrajectory], Dict[int, float], Dict[int, int], Dict[int, float], np.ndarray, np.ndarray]:
    """
    Build per-engine trajectories and per-sample trajectory errors.

    Returns:
        engine_trajs:    unit_id -> EngineTrajectory
        errors_eol:      unit_id -> EOL error (pred - true) based on minimal-RUL window
        cond_ids_eol:    unit_id -> ConditionID at EOL
        hi_phys_eol:     unit_id -> HI_phys_v3 at EOL
        all_life_frac:   [N_total_samples] life fraction per sample
        all_errors_traj: [N_total_samples] trajectory error per sample
    """
    encoder.eval()
    decoder.eval()

    unit_ids_np = unit_ids_full.cpu().numpy()
    y_full_np = y_full.cpu().numpy()
    cond_ids_np = cond_ids_full.cpu().numpy()

    # Prepare arrays to hold per-sample predictions and HI values
    N = X_full.shape[0]
    rul_true_all = np.zeros(N, dtype=np.float32)
    rul_pred_all = np.zeros(N, dtype=np.float32)
    hi_phys_all = np.zeros(N, dtype=np.float32)
    hi_damage_all = np.zeros(N, dtype=np.float32)

    # Batched forward pass over all windows
    batch_size = 512
    num_batches = int(np.ceil(N / batch_size))
    idx_offset = 0

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            X_batch = X_full[start:end].to(device)
            y_batch = y_full[start:end].to(device)
            cond_ids_batch = cond_ids_full[start:end].to(device)

            if health_phys_seq_full is not None:
                hi_phys_seq_batch = health_phys_seq_full[start:end].to(device)
            else:
                hi_phys_seq_batch = torch.ones(
                    X_batch.size(0), X_batch.size(1), device=device, dtype=X_batch.dtype
                )

            # Encoder -> latent + damage-based HI sequence
            z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                X_batch,
                cond_ids=cond_ids_batch,
                cond_vec=None,
            )

            # Decoder RUL trajectory
            rul_seq_pred = decoder(z_seq, hi_phys_seq_batch, hi_damage_seq)
            eol_pred = rul_seq_pred[:, -1]  # Use last timestep prediction for window

            # HI values at last timestep
            hi_phys_last = hi_phys_seq_batch[:, -1]
            if hi_damage_seq.dim() == 3:
                hi_damage_last = hi_damage_seq[:, -1, 0]
            else:
                hi_damage_last = hi_damage_seq[:, -1]

            idx_slice = slice(start, end)
            rul_true_all[idx_slice] = y_batch.cpu().numpy()
            rul_pred_all[idx_slice] = eol_pred.cpu().numpy()
            hi_phys_all[idx_slice] = hi_phys_last.cpu().numpy()
            hi_damage_all[idx_slice] = hi_damage_last.cpu().numpy()

    # Build per-engine trajectories
    engine_trajs: Dict[int, EngineTrajectory] = {}
    errors_eol: Dict[int, float] = {}
    cond_ids_eol: Dict[int, int] = {}
    hi_phys_eol: Dict[int, float] = {}
    all_life_frac: List[float] = []
    all_errors_traj: List[float] = []

    unique_units = np.unique(unit_ids_np)
    for uid in unique_units:
        mask = unit_ids_np == uid
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue

        # Ensure chronological order (build_full_eol_sequences_from_df already does it,
        # but sort by index for safety)
        idxs_sorted = np.sort(idxs)

        rul_true_u = rul_true_all[idxs_sorted]
        rul_pred_u = rul_pred_all[idxs_sorted]
        hi_phys_u = hi_phys_all[idxs_sorted]
        hi_damage_u = hi_damage_all[idxs_sorted]

        T_u = len(idxs_sorted)
        time_idx_u = np.arange(T_u, dtype=np.float32)

        # Normalized life fraction for this unit
        life_frac_u = (time_idx_u + 1.0) / float(T_u)
        err_traj_u = rul_pred_u - rul_true_u
        all_life_frac.extend(life_frac_u.tolist())
        all_errors_traj.extend(err_traj_u.tolist())

        # EOL window = minimal true RUL (should correspond to last sample)
        eol_local_idx = int(np.argmin(rul_true_u))
        err_eol = float(rul_pred_u[eol_local_idx] - rul_true_u[eol_local_idx])
        errors_eol[int(uid)] = err_eol
        cond_ids_eol[int(uid)] = int(cond_ids_np[idxs_sorted[eol_local_idx]])
        hi_phys_eol[int(uid)] = float(hi_phys_u[eol_local_idx])

        engine_trajs[int(uid)] = EngineTrajectory(
            unit_id=int(uid),
            time_idx=time_idx_u,
            rul_true=rul_true_u,
            rul_pred=rul_pred_u,
            hi_phys=hi_phys_u,
            hi_damage=hi_damage_u,
        )

    return (
        engine_trajs,
        errors_eol,
        cond_ids_eol,
        hi_phys_eol,
        np.asarray(all_life_frac, dtype=np.float32),
        np.asarray(all_errors_traj, dtype=np.float32),
    )


def select_engine_groups(
    errors_eol: Dict[int, float],
    num_per_group: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Select unit_ids for worst / medium / best groups based on |EOL error|.
    """
    unit_ids = np.array(sorted(errors_eol.keys()), dtype=int)
    abs_err = np.array([abs(errors_eol[u]) for u in unit_ids], dtype=np.float32)

    # Worst: largest absolute error
    worst_idx = np.argsort(-abs_err)[:num_per_group]
    worst_units = unit_ids[worst_idx].tolist()

    # Best: smallest absolute error
    best_idx = np.argsort(abs_err)[:num_per_group]
    best_units = unit_ids[best_idx].tolist()

    # Medium: around median
    N = len(unit_ids)
    median_pos = N // 2
    half = num_per_group // 2
    start = max(0, median_pos - half)
    end = min(N, start + num_per_group)
    start = max(0, end - num_per_group)
    mid_units = unit_ids[start:end].tolist()

    return worst_units, mid_units, best_units


def plot_engine_group(
    group_name: str,
    unit_ids: List[int],
    engine_trajs: Dict[int, EngineTrajectory],
    errors_eol: Dict[int, float],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    if not unit_ids:
        print(f"[decoder_v1] No units for group '{group_name}', skipping plot.")
        return

    n = len(unit_ids)
    nrows = min(5, n)
    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, uid in enumerate(unit_ids):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        traj = engine_trajs.get(uid)
        if traj is None:
            ax.set_visible(False)
            continue

        t = traj.time_idx

        # Left axis: RUL true vs predicted
        ax.plot(t, traj.rul_true, "k--", label="RUL true")
        ax.plot(t, traj.rul_pred, "b-", label="RUL pred")
        ax.set_xlabel("window index")
        ax.set_ylabel("RUL [cycles]")

        # Right axis: HI_phys and HI_damage
        ax2 = ax.twinx()
        ax2.plot(t, traj.hi_phys, color="green", linestyle="-", label="HI_phys_v3")
        ax2.plot(t, traj.hi_damage, color="orange", linestyle="-.", label="HI_damage")
        ax2.set_ylabel("Health Index")
        ax2.set_ylim(0.0, 1.05)

        # Title with EOL error
        err = errors_eol.get(uid, 0.0)
        ax.set_title(f"Unit {uid}, EOL error = {err:.1f} cycles")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(f"Decoder v1 – {group_name} engines (by |EOL error|)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[decoder_v1] Saved {group_name} engine trajectories to {save_path}")


def plot_error_vs_condition(
    errors_eol: Dict[int, float],
    cond_ids_eol: Dict[int, int],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    units = sorted(errors_eol.keys())
    errs = np.array([errors_eol[u] for u in units], dtype=np.float32)
    conds = np.array([cond_ids_eol[u] for u in units], dtype=np.int32)

    plt.figure(figsize=(6, 4))
    # Scatter; x is discrete ConditionID
    plt.scatter(conds, errs, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("ConditionID")
    plt.ylabel("EOL error (pred - true) [cycles]")
    plt.title("Decoder v1 – EOL Error vs ConditionID (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[decoder_v1] Saved error-vs-condition plot to {save_path}")


def plot_error_vs_hi_eol(
    errors_eol: Dict[int, float],
    hi_phys_eol: Dict[int, float],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    units = sorted(errors_eol.keys())
    errs = np.array([errors_eol[u] for u in units], dtype=np.float32)
    hi = np.array([hi_phys_eol[u] for u in units], dtype=np.float32)

    plt.figure(figsize=(6, 4))
    plt.scatter(hi, errs, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("HI_phys_v3 at EOL")
    plt.ylabel("EOL error (pred - true) [cycles]")
    plt.title("Decoder v1 – EOL Error vs HI_phys_v3 at EOL (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[decoder_v1] Saved error-vs-HI_phys_v3(EOL) plot to {save_path}")


def plot_error_vs_life_fraction(
    all_life_frac: np.ndarray,
    all_errors_traj: np.ndarray,
    save_path: Path,
    num_bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    # Bin errors by life fraction
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_err = np.zeros(num_bins, dtype=np.float32)
    std_err = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (all_life_frac >= bins[i]) & (all_life_frac < bins[i + 1])
        if not np.any(mask):
            mean_err[i] = 0.0
            std_err[i] = 0.0
        else:
            vals = all_errors_traj[mask]
            mean_err[i] = float(np.mean(vals))
            std_err[i] = float(np.std(vals))

    plt.figure(figsize=(6, 4))
    plt.errorbar(bin_centers, mean_err, yerr=std_err, fmt="o-", capsize=4)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("Normalized life fraction")
    plt.ylabel("Mean trajectory error (pred - true) [cycles]")
    plt.title("Decoder v1 – Trajectory Error vs Normalized Life Fraction (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[decoder_v1] Saved error-vs-life-fraction plot to {save_path}")


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset
    run = args.run
    num_per_group = args.num_per_group

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[decoder_v1] Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Load encoder + decoder
    # ------------------------------------------------------------------
    encoder, decoder, results_dir_run, summary = load_decoder_and_encoder(dataset, run, device)

    # Determine encoder experiment and scaler path
    encoder_experiment = summary.get("encoder_experiment")
    if encoder_experiment is None:
        raise RuntimeError("encoder_experiment missing from decoder summary.")
    encoder_experiment_dir = Path(summary.get("results_dir_encoder", "")) or (
        Path("results") / dataset.lower() / encoder_experiment
    )
    if not encoder_experiment_dir.exists():
        encoder_experiment_dir = Path("results") / dataset.lower() / encoder_experiment

    scaler_path = encoder_experiment_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    import pickle

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[decoder_v1] Loaded scaler from {scaler_path}")

    # ------------------------------------------------------------------
    # 2) Rebuild full TEST sliding windows and apply scaler
    # ------------------------------------------------------------------
    past_len = 30
    (
        X_full_test,
        y_full_test,
        unit_ids_full_test,
        cond_ids_full_test,
        health_phys_seq_test,
        y_test_true_eol,
    ) = prepare_fd004_test_full_sequences(
        encoder_experiment=encoder_experiment,
        dataset=dataset,
        past_len=past_len,
    )

    # Scale test features using encoder scaler (condition-wise or global)
    X_full_test_scaled = apply_loaded_scaler(X_full_test, cond_ids_full_test, scaler)

    # ------------------------------------------------------------------
    # 3) Build per-engine trajectories and EOL errors
    # ------------------------------------------------------------------
    (
        engine_trajs,
        errors_eol,
        cond_ids_eol,
        hi_phys_eol,
        all_life_frac,
        all_errors_traj,
    ) = build_engine_trajectories(
        encoder=encoder,
        decoder=decoder,
        X_full=X_full_test_scaled,
        y_full=y_full_test,
        unit_ids_full=unit_ids_full_test,
        cond_ids_full=cond_ids_full_test,
        health_phys_seq_full=health_phys_seq_test,
        device=device,
    )

    # ------------------------------------------------------------------
    # 4) Select worst / medium / best groups by |EOL error|
    # ------------------------------------------------------------------
    worst_units, mid_units, best_units = select_engine_groups(errors_eol, num_per_group)

    # ------------------------------------------------------------------
    # 5) Plot per-engine trajectories for each group
    # ------------------------------------------------------------------
    plot_engine_group(
        group_name=f"{num_per_group} worst",
        unit_ids=worst_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        save_path=results_dir_run / "per_engine_worst.png",
    )
    plot_engine_group(
        group_name=f"{num_per_group} medium",
        unit_ids=mid_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        save_path=results_dir_run / "per_engine_medium.png",
    )
    plot_engine_group(
        group_name=f"{num_per_group} best",
        unit_ids=best_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        save_path=results_dir_run / "per_engine_best.png",
    )

    # ------------------------------------------------------------------
    # 6) Aggregate diagnostics: error vs ConditionID / HI_phys(EOL) / life fraction
    # ------------------------------------------------------------------
    plot_error_vs_condition(
        errors_eol=errors_eol,
        cond_ids_eol=cond_ids_eol,
        save_path=results_dir_run / "error_vs_condition_decoder_v1.png",
    )
    plot_error_vs_hi_eol(
        errors_eol=errors_eol,
        hi_phys_eol=hi_phys_eol,
        save_path=results_dir_run / "error_vs_hi_eol_decoder_v1.png",
    )
    plot_error_vs_life_fraction(
        all_life_frac=all_life_frac,
        all_errors_traj=all_errors_traj,
        save_path=results_dir_run / "error_vs_life_fraction_decoder_v1.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-engine diagnostics for Decoder v1 on FD004.")
    parser.add_argument("--dataset", type=str, default="FD004")
    parser.add_argument("--run", type=str, default="fd004_decoder_v1_from_encoder_v3d")
    parser.add_argument("--num_per_group", type=int, default=10)
    args = parser.parse_args()

    main(args)


