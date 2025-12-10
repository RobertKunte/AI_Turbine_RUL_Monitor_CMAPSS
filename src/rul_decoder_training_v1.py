from __future__ import annotations

"""
RUL Trajectory Decoder v1 training script.

This script trains a lightweight GRU-based decoder (RULTrajectoryDecoderV1)
on top of a **frozen** EOLFullTransformerEncoder that was trained in the
FD004 ms_dt_v2 damage_v3d_delta_two_phase experiment.

The decoder consumes:
  - latent encoder sequence z_seq      [B, T, D]
  - physics HI_phys_v3 sequence       [B, T]  (from the data pipeline)
  - learned damage HI sequence (v3d)  [B, T]  (from CumulativeDamageHead)

and predicts a full RUL trajectory over the same T steps.

Results are stored under:
  results/fd004/decoder_v1_from_encoder_v3d/
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analysis.inference import load_model_from_experiment
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
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
    SequenceDatasetWithUnits,
)
from src.metrics import compute_eol_errors_and_nasa
from src.models.rul_decoder import RULTrajectoryDecoderV1


EXPERIMENT_NAME_V3D = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
DATASET_NAME = "FD004"
ENCODER_EXPERIMENT_DIR = (
    Path("results") / "fd004" / EXPERIMENT_NAME_V3D
)
ENCODER_CHECKPOINT = (
    ENCODER_EXPERIMENT_DIR
    / f"eol_full_lstm_best_{EXPERIMENT_NAME_V3D}.pt"
)
SCALER_PATH = ENCODER_EXPERIMENT_DIR / "scaler.pkl"

DECODER_RESULTS_DIR = Path("results") / "fd004" / "decoder_v1_from_encoder_v3d"


def build_rul_seq_from_last(rul_last: torch.Tensor, T: int) -> torch.Tensor:
    """
    Reconstruct a simple RUL trajectory by counting backwards from the
    last-step RUL (which is already capped if needed).

    Args:
        rul_last: [B] tensor of RUL at the last timestep of each window.
        T:        sequence length (past_len, e.g. 30).

    Returns:
        rul_seq: [B, T] tensor where
            rul_seq[:, -1] == rul_last
            rul_seq[:, j] = rul_last + (T - 1 - j)
    """
    if rul_last.dim() != 1:
        rul_last = rul_last.view(-1)
    B = rul_last.shape[0]
    device = rul_last.device
    # Offsets: [T-1, ..., 1, 0]
    steps = torch.arange(T - 1, -1, -1, device=device, dtype=rul_last.dtype).unsqueeze(0)  # [1, T]
    rul_seq = rul_last.unsqueeze(1) + steps  # [B, T]
    return rul_seq


def apply_loaded_scaler(
    X: torch.Tensor,
    cond_ids: torch.Tensor,
    scaler,
) -> torch.Tensor:
    """
    Apply a pre-fitted scaler (global or condition-wise) to a 3D tensor X.

    Args:
        X:        [N, T, F] unscaled features
        cond_ids: [N] condition IDs per sample
        scaler:   StandardScaler or Dict[int, StandardScaler]

    Returns:
        X_scaled: [N, T, F]
    """
    N, T, F = X.shape

    # Condition-wise scaling (dict[int, StandardScaler])
    if isinstance(scaler, dict):
        X_scaled_list = []
        cond_ids_np = cond_ids.cpu().numpy()
        if not scaler:
            # Degenerate case: no scalers stored – fall back to identity.
            return X.clone()

        # Fallback scaler in case a condition ID is missing
        fallback_scaler = next(iter(scaler.values()))

        for i in range(N):
            cid = int(cond_ids_np[i])
            sc = scaler.get(cid, fallback_scaler)
            x_i = X[i].cpu().numpy()  # [T, F]
            x_scaled = sc.transform(x_i)  # [T, F]
            X_scaled_list.append(torch.from_numpy(x_scaled))

        X_scaled = torch.stack(X_scaled_list, dim=0)
        return X_scaled.to(X.device, dtype=X.dtype)

    # Global StandardScaler
    X_flat = X.cpu().numpy().reshape(-1, F)  # [N*T, F]
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = torch.from_numpy(X_scaled_flat.reshape(N, T, F)).to(X.device, dtype=X.dtype)
    return X_scaled


def make_engine_split_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    unit_ids: torch.Tensor,
    cond_ids: torch.Tensor,
    health_phys_seq: torch.Tensor | None,
    batch_size: int,
    engine_train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val loaders using an engine-based split, mirroring
    create_full_dataloaders but reusing already-scaled features.
    """
    unique_units = torch.unique(unit_ids)
    n_units = len(unique_units)
    n_train_units = int(n_units * engine_train_ratio)
    if n_units > 1 and n_train_units == 0:
        n_train_units = 1

    # Shuffle engine IDs deterministically
    gen = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(n_units, generator=gen)
    train_unit_ids = unique_units[perm[:n_train_units]]
    val_unit_ids = unique_units[perm[n_train_units:]]

    train_mask = torch.isin(unit_ids, train_unit_ids)
    val_mask = torch.isin(unit_ids, val_unit_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    cond_ids_train = cond_ids[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    cond_ids_val = cond_ids[val_mask]

    if health_phys_seq is not None:
        health_train = health_phys_seq[train_mask]
        health_val = health_phys_seq[val_mask]
    else:
        health_train = None
        health_val = None

    train_unit_ids_samples = unit_ids[train_mask]
    val_unit_ids_samples = unit_ids[val_mask]

    train_ds = SequenceDatasetWithUnits(
        X_train, y_train, train_unit_ids_samples, cond_ids_train, health_phys_seq=health_train
    )
    val_ds = SequenceDatasetWithUnits(
        X_val, y_val, val_unit_ids_samples, cond_ids_val, health_phys_seq=health_val
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("============================================================")
    print("[decoder_v1] Engine-based split for decoder training")
    print("============================================================")
    print(f"Total units: {n_units}")
    print(f"Train units: {len(train_unit_ids)}, Val units: {len(val_unit_ids)}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("============================================================")

    return train_loader, val_loader


def prepare_fd004_ms_dt_v3d_data(
    device: torch.device,
    past_len: int = 30,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    Dict,
]:
    """
    Rebuild the FD004 feature pipeline used in the v3d encoder experiment
    and return:

      - X_full:            [N, past_len, F] train sequences (unscaled)
      - y_full:            [N]              last-step RUL targets
      - unit_ids_full:     [N]
      - cond_ids_full:     [N]
      - health_phys_seq:   [N, past_len] or None (HI_phys_v3)
      - X_test:            [N_test, past_len, F] test sequences (unscaled)
      - unit_ids_test:     [N_test]
      - cond_ids_test:     [N_test]
      - feature_cols:      list of feature column names (encoder input)
      - summary_cfg:       summary.json config from the encoder experiment
    """
    print("============================================================")
    print("[decoder_v1] Preparing FD004 data (ms_dt_v2 + residual + twin)")
    print("============================================================")

    # Load original experiment summary to reconstruct feature/physics config
    summary_path = ENCODER_EXPERIMENT_DIR / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")

    with open(summary_path, "r") as f:
        summary_cfg = json.load(f)

    # Load raw CMAPSS data (same helper as core experiments)
    df_train, df_test, y_test_true = load_cmapps_subset(
        DATASET_NAME,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # ------------------------------------------------------------------
    # Physics & feature configs (mirroring run_experiments.py)
    # ------------------------------------------------------------------
    name_lower = EXPERIMENT_NAME_V3D.lower()
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
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        print("  Using continuous condition vector features (Cond_*)")
        df_train = build_condition_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )
        df_test = build_condition_features(
            df_test,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin + residuals
    if use_twin_features:
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len})")
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Temporal / multi-scale features
    df_train = create_all_features(
        df_train,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )
    df_test = create_all_features(
        df_test,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    # 5) HI_phys_v3 column must already exist in df_train for v3d runs
    if "HI_phys_v3" not in df_train.columns:
        raise RuntimeError(
            "[decoder_v1] Expected HI_phys_v3 in df_train. "
            "Please ensure the v3d encoder experiment was run with HI_phys_v3 computation."
        )

    # ------------------------------------------------------------------
    # Feature column selection (mirrors run_experiments.py)
    # ------------------------------------------------------------------
    feature_cols = [
        c
        for c in df_train.columns
        if c
        not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]

    # Remove RUL leakage (same helper as in run_experiments)
    from src.feature_safety import remove_rul_leakage

    feature_cols, _ = remove_rul_leakage(feature_cols)
    # Never feed HI_* targets as input features
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid", "HI_phys_v2", "HI_phys_v3"]
    ]

    print(f"[decoder_v1] Using {len(feature_cols)} features for encoder/decoder input.")

    # ------------------------------------------------------------------
    # Build full-trajectory sliding windows (train) and test sequences
    # ------------------------------------------------------------------
    X_full, y_full, unit_ids_full, cond_ids_full, health_phys_seq_full = build_full_eol_sequences_from_df(
        df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=summary_cfg.get("max_rul", 125.0),
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )

    X_test, unit_ids_test, cond_ids_test = build_test_sequences_from_df(
        df_test,
        feature_cols=feature_cols,
        past_len=past_len,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        cond_col="ConditionID",
    )

    # y_test_true from load_cmapps_subset is EOL RUL per engine
    y_test_true_arr = np.asarray(y_test_true, dtype=np.float32)
    y_test_true_tensor = torch.from_numpy(y_test_true_arr)

    # Move tensors to desired device later; keep on CPU for now
    return (
        X_full,
        y_full,
        unit_ids_full,
        cond_ids_full,
        health_phys_seq_full,
        X_test,
        unit_ids_test,
        cond_ids_test,
        feature_cols,
        {
            "summary_cfg": summary_cfg,
            "y_test_true": y_test_true_tensor,
        },
    )


def evaluate_decoder_eol(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Evaluate decoder on a loader of sliding windows by aggregating per-engine
    EOL predictions (one value per engine: the window with minimal RUL).
    """
    encoder.eval()
    decoder.eval()

    y_true_eol: Dict[int, float] = {}
    y_pred_eol: Dict[int, float] = {}

    with torch.no_grad():
        for batch in loader:
            # Robust batch unpacking (SequenceDatasetWithUnits)
            health_phys_seq_batch = None
            if len(batch) == 5:
                X_batch, y_batch, unit_ids_batch, cond_ids_batch, health_phys_seq_batch = batch
            elif len(batch) == 4:
                X_batch, y_batch, unit_ids_batch, cond_ids_batch = batch
            elif len(batch) == 3:
                X_batch, y_batch, unit_ids_batch = batch
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)
            else:
                X_batch, y_batch = batch
                unit_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            unit_ids_batch = unit_ids_batch.to(device)
            cond_ids_batch = cond_ids_batch.to(device)

            if health_phys_seq_batch is not None:
                hi_phys_batch = health_phys_seq_batch.to(device)
            else:
                hi_phys_batch = None

            z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                X_batch,
                cond_ids=cond_ids_batch,
                cond_vec=None,
            )

            if hi_phys_batch is None:
                hi_phys_batch = torch.zeros_like(hi_damage_seq, device=device)

            rul_pred_seq = decoder(z_seq, hi_phys_batch, hi_damage_seq)  # [B, T]
            eol_pred = rul_pred_seq[:, -1]

            # Aggregate per engine: choose sample with minimal true RUL
            for i in range(y_batch.shape[0]):
                uid = int(unit_ids_batch[i].item())
                true_rul = float(y_batch[i].item())
                pred_rul = float(eol_pred[i].item())

                if uid not in y_true_eol or true_rul < y_true_eol[uid]:
                    y_true_eol[uid] = true_rul
                    y_pred_eol[uid] = pred_rul

    if not y_true_eol:
        raise RuntimeError("[decoder_v1] No EOL samples collected for evaluation.")

    y_true_arr = np.array([y_true_eol[k] for k in sorted(y_true_eol.keys())], dtype=np.float32)
    y_pred_arr = np.array([y_pred_eol[k] for k in sorted(y_pred_eol.keys())], dtype=np.float32)

    metrics = compute_eol_errors_and_nasa(y_true_arr, y_pred_arr, max_rul=125.0)
    return metrics


def evaluate_decoder_on_test(
    encoder: nn.Module,
    decoder: nn.Module,
    X_test: torch.Tensor,
    y_test_true: torch.Tensor,
    unit_ids_test: torch.Tensor,
    cond_ids_test: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Evaluate decoder on FD004 test set (one sequence per engine).
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        X_batch = X_test.to(device)
        cond_ids_batch = cond_ids_test.to(device)

        # HI_phys_v3 is not present for test sequences; feed zeros for hi_phys_seq
        z_seq, _, hi_damage_seq = encoder.encode_with_hi(
            X_batch,
            cond_ids=cond_ids_batch,
            cond_vec=None,
        )
        hi_phys_batch = torch.zeros_like(hi_damage_seq, device=device)

        rul_pred_seq = decoder(z_seq, hi_phys_batch, hi_damage_seq)  # [B, T]
        eol_pred = rul_pred_seq[:, -1]  # [B]

    y_true_eol = y_test_true.cpu().numpy().astype(np.float32)
    y_pred_eol = eol_pred.cpu().numpy().astype(np.float32)
    metrics = compute_eol_errors_and_nasa(y_true_eol, y_pred_eol, max_rul=125.0)
    return metrics


def train_rul_decoder_v1(
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 256,
) -> None:
    # Resolve device
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    print(f"[decoder_v1] Using device: {device_t}")

    DECODER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Prepare data + load original scaler
    # ------------------------------------------------------------------
    (
        X_full,
        y_full,
        unit_ids_full,
        cond_ids_full,
        health_phys_seq_full,
        X_test,
        unit_ids_test,
        cond_ids_test,
        feature_cols,
        extra_info,
    ) = prepare_fd004_ms_dt_v3d_data(device_t)

    # Load scaler from original encoder experiment
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"[decoder_v1] Loaded scaler from {SCALER_PATH}")

    # Scale features using the loaded scaler (condition-wise or global)
    X_full_scaled = apply_loaded_scaler(X_full, cond_ids_full, scaler)
    X_test_scaled = apply_loaded_scaler(X_test, cond_ids_test, scaler)

    # Move tensors to desired device lazily during training/inference
    y_test_true = extra_info["y_test_true"]

    # ------------------------------------------------------------------
    # 2) Build train/val loaders (engine-based split)
    # ------------------------------------------------------------------
    train_loader, val_loader = make_engine_split_loaders(
        X_full_scaled,
        y_full,
        unit_ids_full,
        cond_ids_full,
        health_phys_seq_full,
        batch_size=batch_size,
        engine_train_ratio=0.8,
        random_seed=42,
    )

    # ------------------------------------------------------------------
    # 3) Load frozen encoder from v3d experiment
    # ------------------------------------------------------------------
    if not ENCODER_CHECKPOINT.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found at {ENCODER_CHECKPOINT}")

    print(f"[decoder_v1] Loading encoder from {ENCODER_EXPERIMENT_DIR}")
    encoder, encoder_cfg = load_model_from_experiment(
        ENCODER_EXPERIMENT_DIR,
        device=device_t,
    )

    # Ensure we have a Transformer encoder with damage head
    encoder.to(device_t)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Set cond_feature_indices so the encoder can reconstruct Cond_* sequences
    cond_feature_indices = [i for i, c in enumerate(feature_cols) if c.startswith("Cond_")]
    if getattr(encoder, "use_cond_encoder", False) and getattr(encoder, "cond_in_dim", 0) > 0:
        if cond_feature_indices and len(cond_feature_indices) == getattr(encoder, "cond_in_dim", 0):
            encoder.cond_feature_indices = cond_feature_indices
            print(
                f"[decoder_v1] Set encoder.cond_feature_indices "
                f"({len(cond_feature_indices)} Cond_* features)"
            )
        else:
            print(
                "[decoder_v1] Warning: cond_feature_indices length does not match cond_in_dim; "
                "continuous condition encoder will not use explicit Cond_* features."
            )

    # ------------------------------------------------------------------
    # 4) Instantiate decoder
    # ------------------------------------------------------------------
    latent_dim = getattr(encoder, "d_model", None)
    if latent_dim is None:
        raise RuntimeError("Encoder is missing attribute d_model (expected Transformer encoder).")

    decoder = RULTrajectoryDecoderV1(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ).to(device_t)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # ------------------------------------------------------------------
    # 5) Training loop
    # ------------------------------------------------------------------
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        decoder.train()
        train_losses = []

        for batch in train_loader:
            health_phys_seq_batch = None
            if len(batch) == 5:
                X_batch, y_batch, unit_ids_batch, cond_ids_batch, health_phys_seq_batch = batch
            elif len(batch) == 4:
                X_batch, y_batch, unit_ids_batch, cond_ids_batch = batch
            elif len(batch) == 3:
                X_batch, y_batch, unit_ids_batch = batch
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)
            else:
                X_batch, y_batch = batch
                unit_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64)

            X_batch = X_batch.to(device_t)
            y_batch = y_batch.to(device_t)
            cond_ids_batch = cond_ids_batch.to(device_t)

            if health_phys_seq_batch is not None:
                hi_phys_batch = health_phys_seq_batch.to(device_t)
            else:
                # If physics HI is unavailable, use ones (healthy) as a neutral baseline
                hi_phys_batch = torch.ones(
                    X_batch.size(0), X_batch.size(1), device=device_t, dtype=X_batch.dtype
                )

            optimizer.zero_grad()

            with torch.no_grad():
                z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                    X_batch,
                    cond_ids=cond_ids_batch,
                    cond_vec=None,
                )

            # Build RUL trajectory target
            T = X_batch.size(1)
            rul_target_seq = build_rul_seq_from_last(y_batch, T)  # [B, T]

            # Decoder forward
            rul_pred_seq = decoder(z_seq, hi_phys_batch, hi_damage_seq)  # [B, T]

            # Loss components
            traj_loss = torch.mean((rul_pred_seq - rul_target_seq) ** 2)
            eol_pred = rul_pred_seq[:, -1]
            eol_loss = torch.mean((eol_pred - y_batch) ** 2)

            diffs = rul_pred_seq[:, 1:] - rul_pred_seq[:, :-1]
            mono_violation = torch.relu(-diffs)  # penalize decreasing RUL
            mono_loss = mono_violation.mean()

            loss = traj_loss + 0.5 * eol_loss + 0.1 * mono_loss
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else math.nan

        # ------------------------------------------------------------------
        # Validation (EOL metrics)
        # ------------------------------------------------------------------
        val_metrics = evaluate_decoder_eol(encoder, decoder, val_loader, device_t)
        val_rmse = float(val_metrics["rmse"])
        scheduler.step(val_rmse)

        print(
            f"[decoder_v1][Epoch {epoch:03d}] "
            f"train_loss={mean_train_loss:.4f}, "
            f"val_rmse={val_rmse:.3f}, "
            f"val_mae={val_metrics['mae']:.3f}, "
            f"val_bias={val_metrics['bias']:.3f}, "
            f"val_r2={val_metrics['r2']:.3f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = decoder.state_dict()

    if best_state is None:
        raise RuntimeError("[decoder_v1] Training finished without any valid epochs.")

    # Restore best decoder state
    decoder.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 6) Final evaluation on validation + test
    # ------------------------------------------------------------------
    final_val_metrics = evaluate_decoder_eol(encoder, decoder, val_loader, device_t)
    final_test_metrics = evaluate_decoder_on_test(
        encoder,
        decoder,
        X_test_scaled.to(device_t),
        y_test_true.to(device_t),
        unit_ids_test,
        cond_ids_test,
        device_t,
    )

    # ------------------------------------------------------------------
    # 7) Save decoder + summary + simple plots
    # ------------------------------------------------------------------
    DECODER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    decoder_ckpt_path = DECODER_RESULTS_DIR / "decoder_v1_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "meta": {
                "encoder_experiment": EXPERIMENT_NAME_V3D,
                "dataset": DATASET_NAME,
                "latent_dim": latent_dim,
            },
        },
        decoder_ckpt_path,
    )
    print(f"[decoder_v1] Saved best decoder checkpoint to {decoder_ckpt_path}")

    summary = {
        "dataset": DATASET_NAME,
        "encoder_experiment": EXPERIMENT_NAME_V3D,
        "results_dir_encoder": str(ENCODER_EXPERIMENT_DIR),
        "decoder_results_dir": str(DECODER_RESULTS_DIR),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "best_val_rmse": best_val_rmse,
        },
        "val_metrics": {
            "rmse": final_val_metrics["rmse"],
            "mae": final_val_metrics["mae"],
            "bias": final_val_metrics["bias"],
            "r2": final_val_metrics["r2"],
            "nasa_mean": final_val_metrics["nasa_mean"],
            "nasa_sum": final_val_metrics["nasa_sum"],
            "num_engines": final_val_metrics["num_engines"],
        },
        "test_metrics": {
            "rmse": final_test_metrics["rmse"],
            "mae": final_test_metrics["mae"],
            "bias": final_test_metrics["bias"],
            "r2": final_test_metrics["r2"],
            "nasa_mean": final_test_metrics["nasa_mean"],
            "nasa_sum": final_test_metrics["nasa_sum"],
            "num_engines": final_test_metrics["num_engines"],
        },
    }

    summary_path = DECODER_RESULTS_DIR / "summary_decoder_v1.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[decoder_v1] Saved summary to {summary_path}")

    # Simple plots (error histogram + true-vs-pred) using matplotlib
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]

        # Error histogram (test)
        errors = (
            np.asarray(final_test_metrics["errors"], dtype=np.float32)
            if "errors" in final_test_metrics
            else None
        )
        if errors is not None:
            plt.figure(figsize=(6, 4))
            plt.hist(errors, bins=30, alpha=0.8, color="tab:blue")
            plt.xlabel("Prediction Error (pred - true) [cycles]")
            plt.ylabel("Count")
            plt.title("Decoder v1: Test EOL Error Histogram")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = DECODER_RESULTS_DIR / "error_hist_decoder_v1.png"
            plt.savefig(hist_path)
            plt.close()
            print(f"[decoder_v1] Saved error histogram to {hist_path}")

        # True vs predicted (test)
        # We re-construct y_true/y_pred arrays from metrics dict
        # if available; otherwise recompute quickly.
        if "errors" in final_test_metrics:
            # errors = y_pred - y_true
            errs = np.asarray(final_test_metrics["errors"], dtype=np.float32)
            # NASA helper returns capped predictions; we approximate y_true from rmse info
            # but for plotting we can recompute quickly:
            # NOTE: For simplicity, recompute using compute_eol_errors_and_nasa if needed.
            pass

        # Minimal recomputation for plotting
        # Note: we already have y_true_eol and y_pred_eol as numpy arrays above.
        # Recompute here for clarity.
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            Xb = X_test_scaled.to(device_t)
            cb = cond_ids_test.to(device_t)
            z_seq_b, _, hi_dmg_b = encoder.encode_with_hi(Xb, cond_ids=cb, cond_vec=None)
            hi_phys_b = torch.zeros_like(hi_dmg_b, device=device_t)
            rul_seq_b = decoder(z_seq_b, hi_phys_b, hi_dmg_b)
            eol_pred_b = rul_seq_b[:, -1]

        y_true_plot = y_test_true.cpu().numpy().astype(np.float32)
        y_pred_plot = eol_pred_b.cpu().numpy().astype(np.float32)

        plt.figure(figsize=(5, 5))
        plt.scatter(y_true_plot, y_pred_plot, s=12, alpha=0.7, edgecolor="none")
        max_val = float(max(y_true_plot.max(), y_pred_plot.max(), 1.0))
        plt.plot([0, max_val], [0, max_val], "k--", linewidth=1.0)
        plt.xlabel("True RUL (cycles)")
        plt.ylabel("Predicted RUL (cycles)")
        plt.title("Decoder v1: True vs Predicted RUL (Test EOL)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = DECODER_RESULTS_DIR / "true_vs_pred_decoder_v1.png"
        plt.savefig(scatter_path)
        plt.close()
        print(f"[decoder_v1] Saved true-vs-pred scatter to {scatter_path}")
    except ImportError:
        print("[decoder_v1] matplotlib not available; skipping plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL Trajectory Decoder v1 on FD004/v3d encoder.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    args = parser.parse_args()

    train_rul_decoder_v1(device=args.device, epochs=args.epochs, batch_size=args.batch_size)


