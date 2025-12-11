from __future__ import annotations

"""
Per-engine trajectory and aggregate diagnostics for RUL decoders (v2 by default).

Usage:
    python -m src.analysis.decoder_per_engine_plots \
        --dataset FD004 \
        --run fd004_decoder_v2_from_encoder_v3d \
        --num_per_group 10

This script assumes that:
  - The decoder run has been trained and saved under:
        results/<dataset>/<run>/
    with:
        decoder_v2_best.pt
        summary_decoder_v2.json
  - The corresponding encoder experiment directory and scaler are stored in
    that summary (as written by src.rul_decoder_training_v2.train_rul_decoder_v2).

It can be extended to support decoder v1 runs as well; for now the primary
target is Decoder v2 with HI_phys_v3 + HI_cal_v1 + HI_damage.
"""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.analysis.inference import load_model_from_experiment
from src.analysis.hi_calibration import load_hi_calibrator, calibrate_hi_array
from src.analysis.decoder_v1_per_engine_plots import (
    prepare_fd004_test_full_sequences,
)
from src.models.rul_decoder import RULTrajectoryDecoderV2, RULTrajectoryDecoderV3
from src.rul_decoder_training_v1 import apply_loaded_scaler


@dataclass
class EngineTrajectoryV2:
    unit_id: int
    time_idx: np.ndarray       # [T_engine]
    rul_true: np.ndarray       # [T_engine]
    rul_pred: np.ndarray       # [T_engine]
    hi_phys: np.ndarray        # [T_engine]
    hi_cal: np.ndarray         # [T_engine]
        hi_damage: np.ndarray      # [T_engine]


# Slope windows used for Decoder v3 (must match training)
SLOPE_WINDOW_SIZES: Tuple[int, ...] = (1, 3, 5)


def compute_slope_features_for_v3(
    hi_phys_seq: torch.Tensor,
    hi_cal2_seq: torch.Tensor,
    hi_damage_seq: torch.Tensor,
    window_sizes: Tuple[int, ...] = SLOPE_WINDOW_SIZES,
) -> torch.Tensor:
    """
    Compute local slopes/deltas for HI_phys, HI_cal2, HI_damage at multiple scales.

    Mirrors the helper used in Decoder v3 training so that the decoder receives
    the same slope feature layout at inference time.
    """
    signals = [hi_phys_seq, hi_cal2_seq, hi_damage_seq]
    B, T = hi_phys_seq.shape
    slope_feats: List[torch.Tensor] = []

    for sig in signals:
        for w in window_sizes:
            if w <= 0:
                continue
            pad = sig[:, :1].expand(-1, w)  # [B, w]
            padded = torch.cat([pad, sig], dim=1)  # [B, T + w]
            prev = padded[:, :-w]  # [B, T]
            curr = padded[:, w:]   # [B, T]
            delta = (curr - prev) / float(w)
            slope_feats.append(delta.unsqueeze(-1))

    return torch.cat(slope_feats, dim=-1)  # [B, T, S]


def load_encoder_and_decoder_v2(
    dataset: str,
    run: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, Path, Dict[str, Any]]:
    """
    Load frozen encoder and trained Decoder (v2 or v3) for a given run.

    Returns:
        encoder, decoder, results_dir_run, decoder_summary
    """
    results_dir_run = Path("results") / dataset.lower() / run
    if not results_dir_run.exists():
        raise FileNotFoundError(f"Decoder run directory not found: {results_dir_run}")

    summary_v2 = results_dir_run / "summary_decoder_v2.json"
    summary_v3 = results_dir_run / "summary_decoder_v3.json"
    summary_v1 = results_dir_run / "summary_decoder_v1.json"

    if summary_v2.exists():
        decoder_version = "v2"
        summary_path = summary_v2
    elif summary_v3.exists():
        decoder_version = "v3"
        summary_path = summary_v3
    elif summary_v1.exists():
        raise RuntimeError(
            f"Decoder v1 summary found at {summary_v1}. "
            f"Please use 'python -m src.analysis.decoder_v1_per_engine_plots' for this run."
        )
    else:
        raise FileNotFoundError(
            f"Neither summary_decoder_v2.json nor summary_decoder_v3.json found in {results_dir_run}"
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    encoder_experiment = summary.get("encoder_experiment")
    if encoder_experiment is None:
        raise RuntimeError("encoder_experiment missing from decoder summary.")

    encoder_experiment_dir = Path("results") / dataset.lower() / encoder_experiment
    if not encoder_experiment_dir.exists():
        raise FileNotFoundError(f"Encoder experiment directory not found: {encoder_experiment_dir}")

    # Load encoder via standard inference helper
    encoder, _ = load_model_from_experiment(encoder_experiment_dir, device=device)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Determine latent_dim from encoder
    latent_dim = getattr(encoder, "d_model", None)
    if latent_dim is None:
        raise RuntimeError("Encoder is missing attribute d_model; expected Transformer encoder.")

    # Instantiate appropriate decoder
    if decoder_version == "v2":
        decoder = RULTrajectoryDecoderV2(
            latent_dim=latent_dim,
            hi_feature_dim=3,
            hidden_dim=summary.get("training", {}).get("decoder_hidden_dim", 128),
            num_layers=summary.get("training", {}).get("decoder_num_layers", 2),
            dropout=summary.get("training", {}).get("decoder_dropout", 0.1),
            use_zone_weights=True,
        ).to(device)
        decoder_ckpt_path = results_dir_run / "decoder_v2_best.pt"
    else:  # decoder_version == "v3"
        decoder = RULTrajectoryDecoderV3(
            latent_dim=latent_dim,
            hi_feature_dim=4,
            slope_feature_dim=len(SLOPE_WINDOW_SIZES) * 3,
            hidden_dim=summary.get("training", {}).get("decoder_hidden_dim", 128),
            num_layers=summary.get("training", {}).get("decoder_num_layers", 2),
            dropout=summary.get("training", {}).get("decoder_dropout", 0.1),
        ).to(device)
        decoder_ckpt_path = results_dir_run / "decoder_v3_best.pt"

    if not decoder_ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt_path}")

    ckpt = torch.load(decoder_ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    decoder.load_state_dict(state_dict)
    decoder.eval()

    # Store version info in summary for downstream logic
    summary.setdefault("decoder_version", decoder_version)

    return encoder, decoder, results_dir_run, summary


def build_engine_trajectories_v2(
    encoder: nn.Module,
    decoder: nn.Module,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    unit_ids_full: torch.Tensor,
    cond_ids_full: torch.Tensor,
    hi_phys_seq_full: torch.Tensor,
    hi_cal_seq_full: torch.Tensor,
    device: torch.device,
) -> Tuple[
    Dict[int, EngineTrajectoryV2],
    Dict[int, float],
    Dict[int, int],
    Dict[int, float],
    Dict[int, float],
    np.ndarray,
    np.ndarray,
]:
    """
    Build per-engine trajectories and per-sample trajectory errors for Decoder v2.

    Returns:
        engine_trajs:    unit_id -> EngineTrajectoryV2
        errors_eol:      unit_id -> EOL error (pred - true) based on minimal-RUL window
        cond_ids_eol:    unit_id -> ConditionID at EOL
        hi_phys_eol:     unit_id -> HI_phys_v3 at EOL
        hi_cal_eol:      unit_id -> HI_cal_v1 at EOL
        all_life_frac:   [N_total_samples] life fraction per sample
        all_errors_traj: [N_total_samples] trajectory error per sample
    """
    encoder.eval()
    decoder.eval()

    unit_ids_np = unit_ids_full.cpu().numpy()
    y_full_np = y_full.cpu().numpy()
    cond_ids_np = cond_ids_full.cpu().numpy()

    N = X_full.shape[0]
    rul_true_all = np.zeros(N, dtype=np.float32)
    rul_pred_all = np.zeros(N, dtype=np.float32)
    hi_phys_all = np.zeros(N, dtype=np.float32)
    hi_cal_all = np.zeros(N, dtype=np.float32)
    hi_damage_all = np.zeros(N, dtype=np.float32)

    batch_size = 512
    num_batches = int(np.ceil(N / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            X_batch = X_full[start:end].to(device)
            y_batch = y_full[start:end].to(device)
            cond_ids_batch = cond_ids_full[start:end].to(device)

            hi_phys_seq_batch = hi_phys_seq_full[start:end].to(device)
            hi_cal_seq_batch = hi_cal_seq_full[start:end].to(device)

            # Encoder -> latent + damage-based HI sequence
            z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                X_batch,
                cond_ids=cond_ids_batch,
                cond_vec=None,
            )

            if hi_damage_seq.dim() == 3:
                hi_damage_seq_use = hi_damage_seq.squeeze(-1)
            else:
                hi_damage_seq_use = hi_damage_seq

            # Decoder RUL trajectory
            rul_seq_pred = decoder(z_seq, hi_phys_seq_batch, hi_cal_seq_batch, hi_damage_seq_use)
            eol_pred = rul_seq_pred[:, -1]

            # HI values at last timestep of window
            hi_phys_last = hi_phys_seq_batch[:, -1]
            hi_cal_last = hi_cal_seq_batch[:, -1]
            hi_damage_last = hi_damage_seq_use[:, -1]

            idx_slice = slice(start, end)
            rul_true_all[idx_slice] = y_batch.cpu().numpy()
            rul_pred_all[idx_slice] = eol_pred.cpu().numpy()
            hi_phys_all[idx_slice] = hi_phys_last.cpu().numpy()
            hi_cal_all[idx_slice] = hi_cal_last.cpu().numpy()
            hi_damage_all[idx_slice] = hi_damage_last.cpu().numpy()

    # Build per-engine trajectories
    engine_trajs: Dict[int, EngineTrajectoryV2] = {}
    errors_eol: Dict[int, float] = {}
    cond_ids_eol: Dict[int, int] = {}
    hi_phys_eol: Dict[int, float] = {}
    hi_cal_eol: Dict[int, float] = {}
    all_life_frac: List[float] = []
    all_errors_traj: List[float] = []

    unique_units = np.unique(unit_ids_np)
    for uid in unique_units:
        mask = unit_ids_np == uid
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue

        idxs_sorted = np.sort(idxs)

        rul_true_u = rul_true_all[idxs_sorted]
        rul_pred_u = rul_pred_all[idxs_sorted]
        hi_phys_u = hi_phys_all[idxs_sorted]
        hi_cal_u = hi_cal_all[idxs_sorted]
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
        hi_cal_eol[int(uid)] = float(hi_cal_u[eol_local_idx])

        engine_trajs[int(uid)] = EngineTrajectoryV2(
            unit_id=int(uid),
            time_idx=time_idx_u,
            rul_true=rul_true_u,
            rul_pred=rul_pred_u,
            hi_phys=hi_phys_u,
            hi_cal=hi_cal_u,
            hi_damage=hi_damage_u,
        )

    return (
        engine_trajs,
        errors_eol,
        cond_ids_eol,
        hi_phys_eol,
        hi_cal_eol,
        np.asarray(all_life_frac, dtype=np.float32),
        np.asarray(all_errors_traj, dtype=np.float32),
    )


def build_engine_trajectories_v3(
    encoder: nn.Module,
    decoder: nn.Module,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    unit_ids_full: torch.Tensor,
    cond_ids_full: torch.Tensor,
    hi_phys_seq_full: torch.Tensor,
    hi_cal1_seq_full: torch.Tensor,
    device: torch.device,
) -> Tuple[
    Dict[int, EngineTrajectoryV2],
    Dict[int, float],
    Dict[int, int],
    Dict[int, float],
    Dict[int, float],
    np.ndarray,
    np.ndarray,
]:
    """
    Build per-engine trajectories and trajectory errors for Decoder v3.

    This mirrors build_engine_trajectories_v2 but additionally computes
    slope features and passes them to the v3 decoder.
    """
    encoder.eval()
    decoder.eval()

    unit_ids_np = unit_ids_full.cpu().numpy()
    y_full_np = y_full.cpu().numpy()
    cond_ids_np = cond_ids_full.cpu().numpy()

    N = X_full.shape[0]
    rul_true_all = np.zeros(N, dtype=np.float32)
    rul_pred_all = np.zeros(N, dtype=np.float32)
    hi_phys_all = np.zeros(N, dtype=np.float32)
    hi_cal_all = np.zeros(N, dtype=np.float32)
    hi_damage_all = np.zeros(N, dtype=np.float32)

    batch_size = 512
    num_batches = int(np.ceil(N / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            X_batch = X_full[start:end].to(device)
            y_batch = y_full[start:end].to(device)
            cond_ids_batch = cond_ids_full[start:end].to(device)

            hi_phys_seq_batch = hi_phys_seq_full[start:end].to(device)
            hi_cal1_seq_batch = hi_cal1_seq_full[start:end].to(device)
            hi_cal2_seq_batch = 1.0 - hi_cal1_seq_batch

            # Encoder -> latent + damage-based HI sequence
            z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                X_batch,
                cond_ids=cond_ids_batch,
                cond_vec=None,
            )

            if hi_damage_seq.dim() == 3:
                hi_damage_seq_use = hi_damage_seq.squeeze(-1)
            else:
                hi_damage_seq_use = hi_damage_seq

            slope_feats = compute_slope_features_for_v3(
                hi_phys_seq_batch, hi_cal2_seq_batch, hi_damage_seq_use, SLOPE_WINDOW_SIZES
            )

            # Decoder v3 RUL trajectory
            rul_seq_pred, _ = decoder(
                z_seq=z_seq,
                hi_phys_seq=hi_phys_seq_batch,
                hi_cal1_seq=hi_cal1_seq_batch,
                hi_cal2_seq=hi_cal2_seq_batch,
                hi_damage_seq=hi_damage_seq_use,
                slope_feats=slope_feats,
            )
            eol_pred = rul_seq_pred[:, -1]

            # HI values at last timestep of window
            hi_phys_last = hi_phys_seq_batch[:, -1]
            hi_cal_last = hi_cal1_seq_batch[:, -1]
            hi_damage_last = hi_damage_seq_use[:, -1]

            idx_slice = slice(start, end)
            rul_true_all[idx_slice] = y_batch.cpu().numpy()
            rul_pred_all[idx_slice] = eol_pred.cpu().numpy()
            hi_phys_all[idx_slice] = hi_phys_last.cpu().numpy()
            hi_cal_all[idx_slice] = hi_cal_last.cpu().numpy()
            hi_damage_all[idx_slice] = hi_damage_last.cpu().numpy()

    # Build per-engine trajectories (same structure as v2)
    engine_trajs: Dict[int, EngineTrajectoryV2] = {}
    errors_eol: Dict[int, float] = {}
    cond_ids_eol: Dict[int, int] = {}
    hi_phys_eol: Dict[int, float] = {}
    hi_cal_eol: Dict[int, float] = {}
    all_life_frac: List[float] = []
    all_errors_traj: List[float] = []

    unique_units = np.unique(unit_ids_np)
    for uid in unique_units:
        mask = unit_ids_np == uid
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue

        idxs_sorted = np.sort(idxs)

        rul_true_u = rul_true_all[idxs_sorted]
        rul_pred_u = rul_pred_all[idxs_sorted]
        hi_phys_u = hi_phys_all[idxs_sorted]
        hi_cal_u = hi_cal_all[idxs_sorted]
        hi_damage_u = hi_damage_all[idxs_sorted]

        T_u = len(idxs_sorted)
        time_idx_u = np.arange(T_u, dtype=np.float32)

        life_frac_u = (time_idx_u + 1.0) / float(T_u)
        err_traj_u = rul_pred_u - rul_true_u
        all_life_frac.extend(life_frac_u.tolist())
        all_errors_traj.extend(err_traj_u.tolist())

        eol_local_idx = int(np.argmin(rul_true_u))
        err_eol = float(rul_pred_u[eol_local_idx] - rul_true_u[eol_local_idx])
        errors_eol[int(uid)] = err_eol
        cond_ids_eol[int(uid)] = int(cond_ids_np[idxs_sorted[eol_local_idx]])
        hi_phys_eol[int(uid)] = float(hi_phys_u[eol_local_idx])
        hi_cal_eol[int(uid)] = float(hi_cal_u[eol_local_idx])

        engine_trajs[int(uid)] = EngineTrajectoryV2(
            unit_id=int(uid),
            time_idx=time_idx_u,
            rul_true=rul_true_u,
            rul_pred=rul_pred_u,
            hi_phys=hi_phys_u,
            hi_cal=hi_cal_u,
            hi_damage=hi_damage_u,
        )

    return (
        engine_trajs,
        errors_eol,
        cond_ids_eol,
        hi_phys_eol,
        hi_cal_eol,
        np.asarray(all_life_frac, dtype=np.float32),
        np.asarray(all_errors_traj, dtype=np.float32),
    )


def select_engine_groups_v2(
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


def plot_engine_group_v2(
    group_name: str,
    unit_ids: List[int],
    engine_trajs: Dict[int, EngineTrajectoryV2],
    errors_eol: Dict[int, float],
    cond_ids_eol: Dict[int, int],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    if not unit_ids:
        print(f"[decoder_v2] No units for group '{group_name}', skipping plot.")
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

        # Right axis: HI_phys, HI_cal, HI_damage
        ax2 = ax.twinx()
        ax2.plot(t, traj.hi_phys, color="green", linestyle="-", label="HI_phys_v3")
        ax2.plot(t, traj.hi_cal, color="orange", linestyle="-.", label="HI_cal_v1")
        ax2.plot(t, traj.hi_damage, color="red", linestyle=":", label="HI_damage")
        ax2.set_ylabel("Health Index")
        ax2.set_ylim(0.0, 1.05)

        # Title with EOL error and ConditionID
        err = errors_eol.get(uid, 0.0)
        cond = cond_ids_eol.get(uid, -1)
        ax.set_title(f"Unit {uid}, EOL error = {err:.1f} cycles, CondID = {cond}")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(f"Decoder v2 – {group_name} engines (by |EOL error|)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[decoder_v2] Saved {group_name} engine trajectories to {save_path}")


def plot_error_vs_condition_v2(
    errors_eol: Dict[int, float],
    cond_ids_eol: Dict[int, int],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    units = sorted(errors_eol.keys())
    errs = np.array([errors_eol[u] for u in units], dtype=np.float32)
    conds = np.array([cond_ids_eol[u] for u in units], dtype=np.int32)

    plt.figure(figsize=(6, 4))
    plt.scatter(conds, errs, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("ConditionID")
    plt.ylabel("EOL error (pred - true) [cycles]")
    plt.title("Decoder v2 – EOL Error vs ConditionID (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[decoder_v2] Saved error-vs-condition plot to {save_path}")


def plot_error_vs_hi_eol_v2(
    errors_eol: Dict[int, float],
    hi_phys_eol: Dict[int, float],
    hi_cal_eol: Dict[int, float],
    save_path_phys: Path,
    save_path_cal: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    units = sorted(errors_eol.keys())
    errs = np.array([errors_eol[u] for u in units], dtype=np.float32)
    hi_phys = np.array([hi_phys_eol[u] for u in units], dtype=np.float32)
    hi_cal = np.array([hi_cal_eol[u] for u in units], dtype=np.float32)

    # Error vs HI_phys
    plt.figure(figsize=(6, 4))
    plt.scatter(hi_phys, errs, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("HI_phys_v3 at EOL")
    plt.ylabel("EOL error (pred - true) [cycles]")
    plt.title("Decoder v2 – EOL Error vs HI_phys_v3 at EOL (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path_phys.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_phys, dpi=150)
    plt.close()
    print(f"[decoder_v2] Saved error-vs-HI_phys_v3(EOL) plot to {save_path_phys}")

    # Error vs HI_cal
    plt.figure(figsize=(6, 4))
    plt.scatter(hi_cal, errs, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("HI_cal_v1 at EOL")
    plt.ylabel("EOL error (pred - true) [cycles]")
    plt.title("Decoder v2 – EOL Error vs HI_cal_v1 at EOL (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_cal, dpi=150)
    plt.close()
    print(f"[decoder_v2] Saved error-vs-HI_cal_v1(EOL) plot to {save_path_cal}")


def plot_error_vs_life_fraction_v2(
    all_life_frac: np.ndarray,
    all_errors_traj: np.ndarray,
    save_path: Path,
    num_bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

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
    plt.title("Decoder v2 – Trajectory Error vs Normalized Life Fraction (FD004)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[decoder_v2] Saved error-vs-life-fraction plot to {save_path}")


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset
    run = args.run
    num_per_group = args.num_per_group

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[decoder_v2] Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Load encoder + decoder (v2 or v3)
    # ------------------------------------------------------------------
    encoder, decoder, results_dir_run, summary = load_encoder_and_decoder_v2(dataset, run, device)

    encoder_experiment = summary.get("encoder_experiment")
    if encoder_experiment is None:
        raise RuntimeError("encoder_experiment missing from decoder summary.")

    decoder_version = summary.get("decoder_version", summary.get("model_type", "decoder_v2"))

    # ------------------------------------------------------------------
    # 2) Prepare full TEST sliding windows (features + HI_phys_v3)
    # ------------------------------------------------------------------
    past_len = int(summary.get("training", {}).get("past_len", 30) or 30)
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

    # Scale test features using encoder scaler
    encoder_experiment_dir = Path("results") / dataset.lower() / encoder_experiment
    scaler_path = encoder_experiment_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    import pickle

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[decoder_v2] Loaded scaler from {scaler_path}")

    X_full_test_scaled = apply_loaded_scaler(X_full_test, cond_ids_full_test, scaler)

    # ------------------------------------------------------------------
    # 3) HI_phys_v3 and HI_cal_v1 on test windows
    # ------------------------------------------------------------------
    if health_phys_seq_test is None:
        print(
            "[decoder_v2] Warning: health_phys_seq_test is None – using ones as neutral HI_phys for test."
        )
        hi_phys_seq_test = torch.ones(
            X_full_test_scaled.size(0),
            X_full_test_scaled.size(1),
            dtype=X_full_test_scaled.dtype,
        )
    else:
        hi_phys_seq_test = health_phys_seq_test

    # Load HI calibrator and compute HI_cal_v1 on test windows
    calibrator_path = (
        encoder_experiment_dir
        / f"hi_calibrator_{dataset}.pkl"
    )
    if calibrator_path.exists():
        print(f"[decoder_v2] Loading HI calibrator from {calibrator_path}")
        calibrator = load_hi_calibrator(calibrator_path)
        hi_phys_np = hi_phys_seq_test.cpu().numpy()
        hi_cal_np = calibrate_hi_array(hi_phys_np, calibrator)
        hi_cal_seq_test = torch.from_numpy(hi_cal_np).to(dtype=hi_phys_seq_test.dtype)
        print(
            f"[decoder_v2] HI_cal_v1 stats (test windows): "
            f"min={hi_cal_np.min():.4f}, max={hi_cal_np.max():.4f}, mean={hi_cal_np.mean():.4f}"
        )
    else:
        print(
            f"[decoder_v2] WARNING: HI calibrator not found at {calibrator_path}. "
            f"Falling back to HI_cal = HI_phys on test."
        )
        hi_cal_seq_test = hi_phys_seq_test.clone()

    # ------------------------------------------------------------------
    # 4) Build per-engine trajectories and EOL errors
    # ------------------------------------------------------------------
    if str(decoder_version).endswith("v3") or decoder_version == "decoder_v3":
        (
            engine_trajs,
            errors_eol,
            cond_ids_eol,
            hi_phys_eol,
            hi_cal_eol,
            all_life_frac,
            all_errors_traj,
        ) = build_engine_trajectories_v3(
            encoder=encoder,
            decoder=decoder,
            X_full=X_full_test_scaled,
            y_full=y_full_test,
            unit_ids_full=unit_ids_full_test,
            cond_ids_full=cond_ids_full_test,
            hi_phys_seq_full=hi_phys_seq_test,
            hi_cal1_seq_full=hi_cal_seq_test,
            device=device,
        )
    else:
        (
            engine_trajs,
            errors_eol,
            cond_ids_eol,
            hi_phys_eol,
            hi_cal_eol,
            all_life_frac,
            all_errors_traj,
        ) = build_engine_trajectories_v2(
            encoder=encoder,
            decoder=decoder,
            X_full=X_full_test_scaled,
            y_full=y_full_test,
            unit_ids_full=unit_ids_full_test,
            cond_ids_full=cond_ids_full_test,
            hi_phys_seq_full=hi_phys_seq_test,
            hi_cal_seq_full=hi_cal_seq_test,
            device=device,
        )

    # ------------------------------------------------------------------
    # 5) Select worst / medium / best groups by |EOL error|
    # ------------------------------------------------------------------
    worst_units, mid_units, best_units = select_engine_groups_v2(errors_eol, num_per_group)

    # ------------------------------------------------------------------
    # 6) Plot per-engine trajectories
    # ------------------------------------------------------------------
    plot_engine_group_v2(
        group_name=f"{num_per_group} worst",
        unit_ids=worst_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        cond_ids_eol=cond_ids_eol,
        save_path=results_dir_run / "per_engine_worst_v2.png",
    )
    plot_engine_group_v2(
        group_name=f"{num_per_group} medium",
        unit_ids=mid_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        cond_ids_eol=cond_ids_eol,
        save_path=results_dir_run / "per_engine_medium_v2.png",
    )
    plot_engine_group_v2(
        group_name=f"{num_per_group} best",
        unit_ids=best_units,
        engine_trajs=engine_trajs,
        errors_eol=errors_eol,
        cond_ids_eol=cond_ids_eol,
        save_path=results_dir_run / "per_engine_best_v2.png",
    )

    # ------------------------------------------------------------------
    # 7) Aggregate diagnostics
    # ------------------------------------------------------------------
    plot_error_vs_condition_v2(
        errors_eol=errors_eol,
        cond_ids_eol=cond_ids_eol,
        save_path=results_dir_run / "error_vs_condition_decoder_v2.png",
    )
    plot_error_vs_hi_eol_v2(
        errors_eol=errors_eol,
        hi_phys_eol=hi_phys_eol,
        hi_cal_eol=hi_cal_eol,
        save_path_phys=results_dir_run / "error_vs_hi_phys_eol_decoder_v2.png",
        save_path_cal=results_dir_run / "error_vs_hi_cal_eol_decoder_v2.png",
    )
    plot_error_vs_life_fraction_v2(
        all_life_frac=all_life_frac,
        all_errors_traj=all_errors_traj,
        save_path=results_dir_run / "error_vs_life_fraction_decoder_v2.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-engine diagnostics for RUL decoders (v2) on FD004.")
    parser.add_argument("--dataset", type=str, default="FD004")
    parser.add_argument("--run", type=str, default="fd004_decoder_v2_from_encoder_v3d")
    parser.add_argument("--num_per_group", type=int, default=10)
    args = parser.parse_args()

    main(args)


