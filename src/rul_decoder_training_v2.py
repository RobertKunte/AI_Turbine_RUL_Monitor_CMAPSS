from __future__ import annotations

"""
RUL Trajectory Decoder v2 training script (FD004, encoder v3d).

Decoder v2 improves over Decoder v1 by:
  - using richer inputs: latent encoder sequence + HI_phys_v3 + HI_cal_v1 + HI_damage,
  - putting stronger emphasis on the full RUL trajectory (not just EOL),
  - adding monotonicity and smoothness regularisation terms,
  - optionally using zone-based weighting over life fraction (later timesteps get more weight).

The encoder is a frozen EOLFullTransformerEncoder (v3d), trained beforehand.
This script does NOT train the encoder.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.analysis.inference import load_model_from_experiment
from src.analysis.hi_calibration import load_hi_calibrator, calibrate_hi_array
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
)
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
from src.feature_safety import remove_rul_leakage
from src.metrics import compute_eol_errors_and_nasa
from src.models.rul_decoder import RULTrajectoryDecoderV2


DATASET_NAME = "FD004"
DEFAULT_ENCODER_EXPERIMENT_V3D = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"


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
    steps = torch.arange(T - 1, -1, -1, device=device, dtype=rul_last.dtype).unsqueeze(0)  # [1, T]
    rul_seq = rul_last.unsqueeze(1) + steps  # [B, T]
    return rul_seq


@dataclass
class DecoderV2Batch:
    x: torch.Tensor          # [B, T, F]
    rul_last: torch.Tensor   # [B]
    unit_ids: torch.Tensor   # [B]
    cond_ids: torch.Tensor   # [B]
    hi_phys: torch.Tensor    # [B, T]
    hi_cal: torch.Tensor     # [B, T]


class DecoderV2Dataset(Dataset):
    """
    Simple dataset for Decoder v2 with physics + calibrated HI sequences.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_last: torch.Tensor,
        unit_ids: torch.Tensor,
        cond_ids: torch.Tensor,
        hi_phys_seq: torch.Tensor,
        hi_cal_seq: torch.Tensor,
    ) -> None:
        assert X.shape[0] == y_last.shape[0] == unit_ids.shape[0] == cond_ids.shape[0]
        assert hi_phys_seq.shape == hi_cal_seq.shape == (X.shape[0], X.shape[1])

        self.X = X
        self.y_last = y_last
        self.unit_ids = unit_ids
        self.cond_ids = cond_ids
        self.hi_phys_seq = hi_phys_seq
        self.hi_cal_seq = hi_cal_seq

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.X[idx],
            self.y_last[idx],
            self.unit_ids[idx],
            self.cond_ids[idx],
            self.hi_phys_seq[idx],
            self.hi_cal_seq[idx],
        )


def make_engine_split_loaders_v2(
    X: torch.Tensor,
    y_last: torch.Tensor,
    unit_ids: torch.Tensor,
    cond_ids: torch.Tensor,
    hi_phys_seq: torch.Tensor,
    hi_cal_seq: torch.Tensor,
    batch_size: int,
    engine_train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val loaders using an engine-based split, mirroring the logic
    used in Decoder v1 but with additional HI channels.
    """
    unique_units = torch.unique(unit_ids)
    n_units = len(unique_units)
    n_train_units = int(n_units * engine_train_ratio)
    if n_units > 1 and n_train_units == 0:
        n_train_units = 1

    gen = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(n_units, generator=gen)
    train_unit_ids = unique_units[perm[:n_train_units]]
    val_unit_ids = unique_units[perm[n_train_units:]]

    train_mask = torch.isin(unit_ids, train_unit_ids)
    val_mask = torch.isin(unit_ids, val_unit_ids)

    def _subset(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return t[mask] if t is not None else t

    X_train = _subset(X, train_mask)
    y_train = _subset(y_last, train_mask)
    unit_ids_train = _subset(unit_ids, train_mask)
    cond_ids_train = _subset(cond_ids, train_mask)
    hi_phys_train = _subset(hi_phys_seq, train_mask)
    hi_cal_train = _subset(hi_cal_seq, train_mask)

    X_val = _subset(X, val_mask)
    y_val = _subset(y_last, val_mask)
    unit_ids_val = _subset(unit_ids, val_mask)
    cond_ids_val = _subset(cond_ids, val_mask)
    hi_phys_val = _subset(hi_phys_seq, val_mask)
    hi_cal_val = _subset(hi_cal_seq, val_mask)

    train_ds = DecoderV2Dataset(
        X_train,
        y_train,
        unit_ids_train,
        cond_ids_train,
        hi_phys_train,
        hi_cal_train,
    )
    val_ds = DecoderV2Dataset(
        X_val,
        y_val,
        unit_ids_val,
        cond_ids_val,
        hi_phys_val,
        hi_cal_val,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("============================================================")
    print("[decoder_v2] Engine-based split for decoder training")
    print("============================================================")
    print(f"Total units: {n_units}")
    print(f"Train units: {len(train_unit_ids)}, Val units: {len(val_unit_ids)}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("============================================================")

    return train_loader, val_loader


def _prepare_fd004_ms_dt_encoder_data_with_hi(
    encoder_experiment: str,
    dataset_name: str = DATASET_NAME,
    past_len: int = 30,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    Dict[str, Any],
]:
    """
    Rebuild the FD004 ms_dt_v2 + residual + twin pipeline exactly as used in the
    encoder experiment, returning TRAIN + TEST sequences and HI_phys_v3 sequences.

    This mirrors src.rul_decoder_training_v1.prepare_fd004_ms_dt_encoder_data
    but is kept local here to avoid circular imports.
    """
    summary_path = Path("results") / dataset_name.lower() / encoder_experiment / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")

    with open(summary_path, "r") as f:
        summary_cfg = json.load(f)

    # Raw CMAPSS data
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Physics & feature configs (mirroring run_experiments and decoder v1)
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
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        print("  Using continuous condition vector features (Cond_*) [decoder_v2]")
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
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len}) [decoder_v2]")
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

    # 5) HI_phys_v3 on TRAIN (for encoder and calibrator training)
    from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

    if "HI_phys_v3" not in df_train.columns:
        print("  [decoder_v2] HI_phys_v3 not found in df_train – computing via residual pipeline.")
        hi_v3_series = compute_hi_phys_v3_from_residuals(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_train["HI_phys_v3"] = hi_v3_series
        print(
            f"  HI_phys_v3 stats (decoder_v2 train): "
            f"min={hi_v3_series.min():.4f}, "
            f"max={hi_v3_series.max():.4f}, "
            f"mean={hi_v3_series.mean():.4f}"
        )

    # Note: For TEST we do not strictly need HI_phys_v3 for decoder training; test
    # evaluation below uses zeros/ones as neutral baseline. If needed, we can
    # mirror the above block for df_test.

    # Feature columns (same as run_experiments / decoder_v1)
    feature_cols = [
        c
        for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid", "HI_phys_v2", "HI_phys_v3"]
    ]

    print(f"[decoder_v2] Using {len(feature_cols)} features for encoder/decoder input.")

    # Build full-trajectory sliding windows (train) and test sequences
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

    y_test_true_arr = np.asarray(y_test_true, dtype=np.float32)
    y_test_true_tensor = torch.from_numpy(y_test_true_arr)

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


def train_rul_decoder_v2(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Train RUL Trajectory Decoder v2 on FD004/v3d encoder.

    Args:
        config: Experiment config (from experiment_configs.get_fd004_decoder_v2_from_encoder_v3d_config)
        device: torch.device

    Returns:
        summary dict with training/validation/test metrics.
    """
    dataset_name = config.get("dataset", DATASET_NAME)
    assert dataset_name == DATASET_NAME, "Decoder v2 currently implemented for FD004 only."

    experiment_name = config.get("experiment_name", "fd004_decoder_v2_from_encoder_v3d")
    encoder_experiment = config.get(
        "encoder_experiment",
        DEFAULT_ENCODER_EXPERIMENT_V3D,
    )

    training_params = config.get("training_params", {})
    num_epochs = int(training_params.get("num_epochs", 60))
    batch_size = int(training_params.get("batch_size", 256))
    engine_train_ratio = float(training_params.get("engine_train_ratio", 0.8))
    random_seed = int(training_params.get("random_seed", 42))

    # Loss weights
    w_traj = float(config.get("w_traj", 1.0))
    w_eol = float(config.get("w_eol", 0.2))
    w_mono = float(config.get("w_mono", 0.1))
    w_smooth = float(config.get("w_smooth", 0.01))

    past_len = int(config.get("past_len", 30))
    max_rul = float(config.get("max_rul", 125.0))

    print("============================================================")
    print(f"[decoder_v2] Training Decoder v2 for experiment '{experiment_name}'")
    print(f"  Dataset: {dataset_name}")
    print(f"  Encoder experiment: {encoder_experiment}")
    print(f"  Device: {device}")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1) Prepare data (mirrors encoder pipeline + Decoder v1)
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
    ) = _prepare_fd004_ms_dt_encoder_data_with_hi(
        encoder_experiment=encoder_experiment,
        dataset_name=dataset_name,
        past_len=past_len,
    )

    # 2) Load scaler from original encoder experiment and scale features
    encoder_experiment_dir = Path("results") / dataset_name.lower() / encoder_experiment
    scaler_path = encoder_experiment_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    import pickle

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[decoder_v2] Loaded scaler from {scaler_path}")

    # Apply scaler (global or condition-wise)
    from src.rul_decoder_training_v1 import apply_loaded_scaler

    X_full_scaled = apply_loaded_scaler(X_full, cond_ids_full, scaler)
    X_test_scaled = apply_loaded_scaler(X_test, cond_ids_test, scaler)

    # ------------------------------------------------------------------
    # 3) HI_phys and calibrated HI_cal sequences
    # ------------------------------------------------------------------
    if health_phys_seq_full is None:
        print("[decoder_v2] Warning: health_phys_seq_full is None – using ones as neutral HI_phys.")
        hi_phys_seq_full = torch.ones(
            X_full_scaled.size(0),
            X_full_scaled.size(1),
            dtype=X_full_scaled.dtype,
        )
    else:
        hi_phys_seq_full = health_phys_seq_full

    # Load calibrator (if available) and compute HI_cal_v1
    hi_calibrator_path_cfg = config.get("hi_calibrator_path", None)
    if hi_calibrator_path_cfg is None:
        hi_calibrator_path = (
            Path("results")
            / dataset_name.lower()
            / encoder_experiment
            / f"hi_calibrator_{dataset_name}.pkl"
        )
    else:
        hi_calibrator_path = Path(hi_calibrator_path_cfg)

    if hi_calibrator_path.exists():
        print(f"[decoder_v2] Loading HI calibrator from {hi_calibrator_path}")
        calibrator = load_hi_calibrator(hi_calibrator_path)
        hi_phys_np = hi_phys_seq_full.cpu().numpy()
        hi_cal_np = calibrate_hi_array(hi_phys_np, calibrator)
        hi_cal_seq_full = torch.from_numpy(hi_cal_np).to(dtype=hi_phys_seq_full.dtype)
        print(
            f"[decoder_v2] HI_cal_v1 stats (train windows): "
            f"min={hi_cal_np.min():.4f}, max={hi_cal_np.max():.4f}, mean={hi_cal_np.mean():.4f}"
        )
    else:
        print(
            f"[decoder_v2] WARNING: HI calibrator not found at {hi_calibrator_path}. "
            f"Falling back to HI_cal = HI_phys."
        )
        hi_cal_seq_full = hi_phys_seq_full.clone()

    # ------------------------------------------------------------------
    # 4) Build engine-based train/val loaders
    # ------------------------------------------------------------------
    train_loader, val_loader = make_engine_split_loaders_v2(
        X_full_scaled,
        y_full,
        unit_ids_full,
        cond_ids_full,
        hi_phys_seq_full,
        hi_cal_seq_full,
        batch_size=batch_size,
        engine_train_ratio=engine_train_ratio,
        random_seed=random_seed,
    )

    # ------------------------------------------------------------------
    # 5) Load frozen encoder
    # ------------------------------------------------------------------
    encoder_checkpoint = (
        encoder_experiment_dir
        / f"eol_full_lstm_best_{encoder_experiment}.pt"
    )
    if not encoder_checkpoint.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found at {encoder_checkpoint}")

    print(f"[decoder_v2] Loading encoder from {encoder_experiment_dir}")
    encoder, encoder_cfg = load_model_from_experiment(
        encoder_experiment_dir,
        device=device,
    )
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Ensure encoder.cond_feature_indices are set if needed
    cond_feature_indices = [i for i, c in enumerate(feature_cols) if c.startswith("Cond_")]
    if getattr(encoder, "use_cond_encoder", False) and getattr(encoder, "cond_in_dim", 0) > 0:
        if cond_feature_indices and len(cond_feature_indices) == getattr(encoder, "cond_in_dim", 0):
            encoder.cond_feature_indices = cond_feature_indices
            print(
                f"[decoder_v2] Set encoder.cond_feature_indices "
                f"({len(cond_feature_indices)} Cond_* features)"
            )
        else:
            print(
                "[decoder_v2] Warning: cond_feature_indices length does not match cond_in_dim; "
                "continuous condition encoder will not use explicit Cond_* features."
            )

    # ------------------------------------------------------------------
    # 6) Instantiate decoder v2
    # ------------------------------------------------------------------
    latent_dim = getattr(encoder, "d_model", None)
    if latent_dim is None:
        raise RuntimeError("Encoder is missing attribute d_model (expected Transformer encoder).")

    decoder_hidden_dim = int(config.get("decoder_hidden_dim", 128))
    decoder_num_layers = int(config.get("decoder_num_layers", 2))
    decoder_dropout = float(config.get("decoder_dropout", 0.1))

    decoder = RULTrajectoryDecoderV2(
        latent_dim=latent_dim,
        hi_feature_dim=3,
        hidden_dim=decoder_hidden_dim,
        num_layers=decoder_num_layers,
        dropout=decoder_dropout,
        use_zone_weights=True,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # ------------------------------------------------------------------
    # 7) Training loop
    # ------------------------------------------------------------------
    # Store results under the experiment_name directory so that analysis scripts
    # can refer to runs via their experiment name (e.g. fd004_decoder_v2_from_encoder_v3d).
    decoder_results_subdir = experiment_name
    decoder_results_dir = Path("results") / dataset_name.lower() / decoder_results_subdir
    decoder_results_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        decoder.train()
        train_losses = []

        for batch in train_loader:
            (
                X_batch,
                y_last_batch,
                unit_ids_batch,
                cond_ids_batch,
                hi_phys_batch,
                hi_cal_batch,
            ) = batch

            X_batch = X_batch.to(device)
            y_last_batch = y_last_batch.to(device)
            cond_ids_batch = cond_ids_batch.to(device)
            hi_phys_batch = hi_phys_batch.to(device)
            hi_cal_batch = hi_cal_batch.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                    X_batch,
                    cond_ids=cond_ids_batch,
                    cond_vec=None,
                )

            if hi_damage_seq.dim() == 3:
                hi_damage_seq_use = hi_damage_seq.squeeze(-1)
            else:
                hi_damage_seq_use = hi_damage_seq

            T = X_batch.size(1)
            rul_target_seq = build_rul_seq_from_last(y_last_batch, T)  # [B, T]

            # Decoder forward
            rul_pred_seq = decoder(
                z_seq,
                hi_phys_batch,
                hi_cal_batch,
                hi_damage_seq_use,
            )  # [B, T]

            # Loss components
            # 1) Trajectory loss with zone-based weighting (later steps higher weight)
            time_idx = torch.arange(T, device=device, dtype=rul_pred_seq.dtype)  # [T]
            zone_weights = (time_idx + 1.0) / float(T)  # [T]
            zone_weights = zone_weights.unsqueeze(0).expand_as(rul_pred_seq)  # [B, T]
            traj_loss_weighted = ((rul_pred_seq - rul_target_seq) ** 2 * zone_weights).mean()

            # 2) EOL loss (last timestep)
            eol_loss = F.mse_loss(rul_pred_seq[:, -1], rul_target_seq[:, -1])

            # 3) Monotonicity penalty (RUL should be non-increasing over time)
            diff = rul_pred_seq[:, 1:] - rul_pred_seq[:, :-1]  # [B, T-1]
            mono_violation = F.relu(diff)  # penalize increases
            mono_loss = mono_violation.mean()

            # 4) Smoothness penalty (second differences)
            diff1 = rul_pred_seq[:, 1:] - rul_pred_seq[:, :-1]
            diff2 = diff1[:, 1:] - diff1[:, :-1]
            smooth_loss = (diff2 ** 2).mean()

            loss = (
                w_traj * traj_loss_weighted
                + w_eol * eol_loss
                + w_mono * mono_loss
                + w_smooth * smooth_loss
            )
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ------------------------------------------------------------------
        # Validation (EOL metrics)
        # ------------------------------------------------------------------
        decoder.eval()
        y_true_eol_list = []
        y_pred_eol_list = []

        with torch.no_grad():
            for batch in val_loader:
                (
                    X_batch,
                    y_last_batch,
                    unit_ids_batch,
                    cond_ids_batch,
                    hi_phys_batch,
                    hi_cal_batch,
                ) = batch

                X_batch = X_batch.to(device)
                y_last_batch = y_last_batch.to(device)
                cond_ids_batch = cond_ids_batch.to(device)
                hi_phys_batch = hi_phys_batch.to(device)
                hi_cal_batch = hi_cal_batch.to(device)

                z_seq, _, hi_damage_seq = encoder.encode_with_hi(
                    X_batch,
                    cond_ids=cond_ids_batch,
                    cond_vec=None,
                )
                if hi_damage_seq.dim() == 3:
                    hi_damage_seq_use = hi_damage_seq.squeeze(-1)
                else:
                    hi_damage_seq_use = hi_damage_seq

                T = X_batch.size(1)
                rul_target_seq = build_rul_seq_from_last(y_last_batch, T)

                rul_pred_seq = decoder(
                    z_seq,
                    hi_phys_batch,
                    hi_cal_batch,
                    hi_damage_seq_use,
                )

                y_true_eol_list.append(rul_target_seq[:, -1].cpu().numpy())
                y_pred_eol_list.append(rul_pred_seq[:, -1].cpu().numpy())

        y_true_eol = np.concatenate(y_true_eol_list, axis=0) if y_true_eol_list else np.zeros(0, dtype=np.float32)
        y_pred_eol = np.concatenate(y_pred_eol_list, axis=0) if y_pred_eol_list else np.zeros(0, dtype=np.float32)

        if y_true_eol.size == 0:
            raise RuntimeError("[decoder_v2] No validation samples collected.")

        val_metrics = compute_eol_errors_and_nasa(y_true_eol, y_pred_eol, max_rul=max_rul)
        val_rmse = float(val_metrics["rmse"])
        scheduler.step(val_rmse)

        print(
            f"[decoder_v2][Epoch {epoch:03d}] "
            f"train_loss={mean_train_loss:.4f}, "
            f"val_rmse={val_rmse:.3f}, "
            f"val_mae={val_metrics['mae']:.3f}, "
            f"val_bias={val_metrics['bias']:.3f}, "
            f"val_r2={val_metrics['r2']:.3f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = decoder.state_dict()
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("[decoder_v2] Training finished without any valid epochs.")

    # Restore best decoder state
    decoder.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 8) Final evaluation on validation + (optional) test
    # ------------------------------------------------------------------
    final_val_metrics = val_metrics  # from last epoch with scheduler.step

    # Simple test evaluation: use zeros/ones for HI inputs (no test HI_phys available here)
    decoder.eval()
    with torch.no_grad():
        Xb_test = X_test_scaled.to(device)
        cond_ids_test_b = cond_ids_test.to(device)

        z_seq_test, _, hi_damage_seq_test = encoder.encode_with_hi(
            Xb_test,
            cond_ids=cond_ids_test_b,
            cond_vec=None,
        )

        hi_phys_test = torch.ones_like(hi_damage_seq_test, device=device)
        hi_cal_test = hi_phys_test.clone()

        if hi_damage_seq_test.dim() == 3:
            hi_damage_test_use = hi_damage_seq_test.squeeze(-1)
        else:
            hi_damage_test_use = hi_damage_seq_test

        T_test = Xb_test.size(1)
        y_test_true = extra_info["y_test_true"].to(device)
        rul_target_seq_test = build_rul_seq_from_last(y_test_true, T_test)

        rul_pred_seq_test = decoder(
            z_seq_test,
            hi_phys_test,
            hi_cal_test,
            hi_damage_test_use,
        )
        y_true_eol_test = rul_target_seq_test[:, -1].cpu().numpy().astype(np.float32)
        y_pred_eol_test = rul_pred_seq_test[:, -1].cpu().numpy().astype(np.float32)

    test_metrics = compute_eol_errors_and_nasa(y_true_eol_test, y_pred_eol_test, max_rul=max_rul)

    # ------------------------------------------------------------------
    # 9) Save decoder + summary
    # ------------------------------------------------------------------
    decoder_results_dir.mkdir(parents=True, exist_ok=True)

    decoder_ckpt_path = decoder_results_dir / "decoder_v2_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "meta": {
                "encoder_experiment": encoder_experiment,
                "dataset": dataset_name,
                "latent_dim": latent_dim,
                "encoder_type": "eol_full_transformer_encoder_v3d",
                "decoder_type": "rul_trajectory_decoder_v2",
            },
        },
        decoder_ckpt_path,
    )
    print(f"[decoder_v2] Saved best decoder checkpoint to {decoder_ckpt_path}")

    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model_type": "decoder_v2",
        "encoder_experiment": encoder_experiment,
        "decoder_results_dir": str(decoder_results_dir),
        "training": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "w_traj": w_traj,
            "w_eol": w_eol,
            "w_mono": w_mono,
            "w_smooth": w_smooth,
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
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "bias": test_metrics["bias"],
            "r2": test_metrics["r2"],
            "nasa_mean": test_metrics["nasa_mean"],
            "nasa_sum": test_metrics["nasa_sum"],
            "num_engines": test_metrics["num_engines"],
        },
    }

    summary_path = decoder_results_dir / "summary_decoder_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[decoder_v2] Saved summary to {summary_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RUL Trajectory Decoder v2 on FD004/v3d encoder.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    if args.device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Minimal inline config for standalone runs; run_experiments.py will provide a richer config.
    cfg: Dict[str, Any] = {
        "experiment_name": "fd004_decoder_v2_from_encoder_v3d",
        "dataset": "FD004",
        "encoder_experiment": DEFAULT_ENCODER_EXPERIMENT_V3D,
        "hi_calibrator_path": str(
            Path("results")
            / "fd004"
            / DEFAULT_ENCODER_EXPERIMENT_V3D
            / "hi_calibrator_FD004.pkl"
        ),
        "past_len": 30,
        "max_rul": 125.0,
        "training_params": {
            "num_epochs": 60,
            "batch_size": 256,
            "engine_train_ratio": 0.8,
            "random_seed": 42,
        },
        "decoder_hidden_dim": 128,
        "decoder_num_layers": 2,
        "decoder_dropout": 0.1,
        "w_traj": 1.0,
        "w_eol": 0.2,
        "w_mono": 0.1,
        "w_smooth": 0.01,
    }

    train_rul_decoder_v2(cfg, device_t)


