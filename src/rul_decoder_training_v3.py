from __future__ import annotations

"""
RUL Trajectory Decoder v3 training script (FD004, encoder v3d).

Decoder v3 extends decoder v2 by:
  - using richer health inputs:
      * HI_phys_v3 (physics-based HI)
      * HI_cal_v1 (global calibrated HI)
      * HI_cal_v2 = 1 - HI_cal_v1
      * HI_damage (encoder damage head)
      * multi-scale slope features of HI_phys_v3, HI_cal_v2, HI_damage
  - explicitly predicting a degradation-rate sequence as an auxiliary head.
  - using a composite loss:
      * trajectory-weighted RUL MSE
      * EOL MSE
      * monotonicity penalty on RUL
      * smoothness penalty on RUL curvature
      * slope-consistency loss (predicted vs true |ΔRUL|).
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.analysis.hi_calibration import (
    load_hi_calibrator,
    calibrate_hi_array,
    hi_cal_v2_from_v1,
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
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
    SequenceDatasetWithUnits,
)
from src.feature_safety import remove_rul_leakage
from src.metrics import compute_eol_errors_and_nasa
from src.models.rul_decoder import RULTrajectoryDecoderV3
from src.rul_decoder_training_v1 import apply_loaded_scaler
from src.analysis.inference import load_model_from_experiment


DATASET_NAME = "FD004"
DEFAULT_ENCODER_EXPERIMENT_V3D = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"

# Window sizes used for slope features (must stay in sync with decoder.slope_feature_dim)
SLOPE_WINDOW_SIZES: Tuple[int, ...] = (1, 3, 5)


def build_rul_seq_from_last(rul_last: torch.Tensor, T: int) -> torch.Tensor:
    """
    Reconstruct a simple RUL trajectory by counting backwards from the
    last-step RUL (which is already capped if needed).
    """
    if rul_last.dim() != 1:
        rul_last = rul_last.view(-1)
    device = rul_last.device
    steps = torch.arange(T - 1, -1, -1, device=device, dtype=rul_last.dtype).unsqueeze(0)
    return rul_last.unsqueeze(1) + steps  # [B, T]


def compute_slope_features(
    hi_phys_seq: torch.Tensor,
    hi_cal2_seq: torch.Tensor,
    hi_damage_seq: torch.Tensor,
    window_sizes: Tuple[int, ...] = (1, 3, 5),
) -> torch.Tensor:
    """
    Compute local slopes/deltas for HI_phys, HI_cal2, HI_damage at multiple scales.

    For each signal x and window size w:
      slope[t] = (x[t] - x[t-w]) / w for t >= w, and 0 for t < w.

    Returns:
        slopes: [B, T, S] where S = len(window_sizes) * 3.
    """
    signals = [hi_phys_seq, hi_cal2_seq, hi_damage_seq]
    B, T = hi_phys_seq.shape
    slope_feats = []

    for sig in signals:
        for w in window_sizes:
            if w <= 0:
                continue
            # pad on the left with the first value so we can index safely
            pad = sig[:, :1].expand(-1, w)  # [B, w]
            padded = torch.cat([pad, sig], dim=1)  # [B, T + w]
            prev = padded[:, :-w]  # [B, T]
            curr = padded[:, w:]   # [B, T]
            delta = (curr - prev) / float(w)  # [B, T]
            slope_feats.append(delta.unsqueeze(-1))

    return torch.cat(slope_feats, dim=-1)  # [B, T, S]


class DecoderV3Dataset(Dataset):
    """
    Dataset for Decoder v3: stores X, last-step RUL, IDs, and HI_phys/HI_cal1 sequences.
    HI_cal2 and slopes are computed on the fly in the training loop.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_last: torch.Tensor,
        unit_ids: torch.Tensor,
        cond_ids: torch.Tensor,
        hi_phys_seq: torch.Tensor,
        hi_cal1_seq: torch.Tensor,
    ) -> None:
        assert X.shape[0] == y_last.shape[0] == unit_ids.shape[0] == cond_ids.shape[0]
        assert hi_phys_seq.shape == hi_cal1_seq.shape == (X.shape[0], X.shape[1])

        self.X = X
        self.y_last = y_last
        self.unit_ids = unit_ids
        self.cond_ids = cond_ids
        self.hi_phys_seq = hi_phys_seq
        self.hi_cal1_seq = hi_cal1_seq

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.X[idx],
            self.y_last[idx],
            self.unit_ids[idx],
            self.cond_ids[idx],
            self.hi_phys_seq[idx],
            self.hi_cal1_seq[idx],
        )


def make_engine_split_loaders_v3(
    X: torch.Tensor,
    y_last: torch.Tensor,
    unit_ids: torch.Tensor,
    cond_ids: torch.Tensor,
    hi_phys_seq: torch.Tensor,
    hi_cal1_seq: torch.Tensor,
    batch_size: int,
    engine_train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Engine-based split (train/val) for Decoder v3, analogous to v2.
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
    hi_cal1_train = _subset(hi_cal1_seq, train_mask)

    X_val = _subset(X, val_mask)
    y_val = _subset(y_last, val_mask)
    unit_ids_val = _subset(unit_ids, val_mask)
    cond_ids_val = _subset(cond_ids, val_mask)
    hi_phys_val = _subset(hi_phys_seq, val_mask)
    hi_cal1_val = _subset(hi_cal1_seq, val_mask)

    train_ds = DecoderV3Dataset(
        X_train,
        y_train,
        unit_ids_train,
        cond_ids_train,
        hi_phys_train,
        hi_cal1_train,
    )
    val_ds = DecoderV3Dataset(
        X_val,
        y_val,
        unit_ids_val,
        cond_ids_val,
        hi_phys_val,
        hi_cal1_val,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("============================================================")
    print("[decoder_v3] Engine-based split for decoder training")
    print("============================================================")
    print(f"Total units: {n_units}")
    print(f"Train units: {len(train_unit_ids)}, Val units: {len(val_unit_ids)}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("============================================================")

    return train_loader, val_loader


def _prepare_fd004_ms_dt_encoder_data_with_hi_phys(
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

    This is similar to the helper used in Decoder v2, but defined locally to keep
    Decoder v3 self-contained.
    """
    import json

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
        print("  Using continuous condition vector features (Cond_*) [decoder_v3]")
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
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len}) [decoder_v3]")
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Temporal / multiscale features
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

    # 5) HI_phys_v3 on TRAIN
    from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

    if "HI_phys_v3" not in df_train.columns:
        print("  [decoder_v3] HI_phys_v3 not found in df_train – computing via residual pipeline.")
        hi_v3_series = compute_hi_phys_v3_from_residuals(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_train["HI_phys_v3"] = hi_v3_series
        print(
            f"  HI_phys_v3 stats (decoder_v3 train): "
            f"min={hi_v3_series.min():.4f}, "
            f"max={hi_v3_series.max():.4f}, "
            f"mean={hi_v3_series.mean():.4f}"
        )

    # For consistency, also ensure HI_phys_v3 on test (useful for diagnostics/eval later)
    if "HI_phys_v3" not in df_test.columns:
        hi_v3_test = compute_hi_phys_v3_from_residuals(
            df_test,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_test["HI_phys_v3"] = hi_v3_test

    # Feature columns (encoder input, no HI_* targets)
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

    print(f"[decoder_v3] Using {len(feature_cols)} features for encoder/decoder input.")

    # Build full-trajectory sliding windows (train)
    X_full, y_full, unit_ids_full, cond_ids_full, hi_phys_seq_full = build_full_eol_sequences_from_df(
        df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=summary_cfg.get("max_rul", 125.0),
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )

    # Build TEST EOL windows (for simple final evaluation)
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
        hi_phys_seq_full,
        X_test,
        unit_ids_test,
        cond_ids_test,
        feature_cols,
        {
            "summary_cfg": summary_cfg,
            "y_test_true": y_test_true_tensor,
            "df_test": df_test,
        },
    )


def train_rul_decoder_v3(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Train RUL Trajectory Decoder v3 on FD004/v3d encoder.
    """
    dataset_name = config.get("dataset", DATASET_NAME)
    assert dataset_name == DATASET_NAME, "Decoder v3 currently implemented for FD004 only."

    experiment_name = config.get("experiment_name", "fd004_decoder_v3_from_encoder_v3d")
    encoder_experiment = config.get(
        "encoder_experiment",
        DEFAULT_ENCODER_EXPERIMENT_V3D,
    )

    training_params = config.get("training_params", {})
    num_epochs = int(training_params.get("num_epochs", 80))
    batch_size = int(training_params.get("batch_size", 256))
    engine_train_ratio = float(training_params.get("engine_train_ratio", 0.8))
    random_seed = int(training_params.get("random_seed", 42))

    # Loss weights
    w_traj = float(config.get("w_traj", 1.0))
    w_eol = float(config.get("w_eol", 0.2))
    w_mono = float(config.get("w_mono", 0.1))
    w_smooth = float(config.get("w_smooth", 0.01))
    w_slope = float(config.get("w_slope", 0.2))

    past_len = int(config.get("past_len", 30))
    max_rul = float(config.get("max_rul", 125.0))

    print("============================================================")
    print(f"[decoder_v3] Training Decoder v3 for experiment '{experiment_name}'")
    print(f"  Dataset: {dataset_name}")
    print(f"  Encoder experiment: {encoder_experiment}")
    print(f"  Device: {device}")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1) Prepare data (mirrors encoder pipeline)
    # ------------------------------------------------------------------
    (
        X_full,
        y_full,
        unit_ids_full,
        cond_ids_full,
        hi_phys_seq_full,
        X_test,
        unit_ids_test,
        cond_ids_test,
        feature_cols,
        extra_info,
    ) = _prepare_fd004_ms_dt_encoder_data_with_hi_phys(
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

    print(f"[decoder_v3] Loaded scaler from {scaler_path}")

    X_full_scaled = apply_loaded_scaler(X_full, cond_ids_full, scaler)
    X_test_scaled = apply_loaded_scaler(X_test, cond_ids_test, scaler)

    # ------------------------------------------------------------------
    # 3) HI_phys and calibrated HI_cal sequences (TRAIN)
    # ------------------------------------------------------------------
    if hi_phys_seq_full is None:
        print("[decoder_v3] Warning: hi_phys_seq_full is None – using ones as neutral HI_phys.")
        hi_phys_seq_full = torch.ones(
            X_full_scaled.size(0),
            X_full_scaled.size(1),
            dtype=X_full_scaled.dtype,
        )

    # Calibrator path
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
        print(f"[decoder_v3] Loading HI calibrator from {hi_calibrator_path}")
        calibrator = load_hi_calibrator(hi_calibrator_path)
        hi_phys_np = hi_phys_seq_full.cpu().numpy()
        hi_cal1_np = calibrate_hi_array(hi_phys_np, calibrator)
        hi_cal1_seq_full = torch.from_numpy(hi_cal1_np).to(dtype=hi_phys_seq_full.dtype)
        hi_cal2_np = hi_cal_v2_from_v1(hi_cal1_np)
        hi_cal2_seq_full = torch.from_numpy(hi_cal2_np).to(dtype=hi_phys_seq_full.dtype)
        print(
            f"[decoder_v3] HI_cal_v1 stats (train windows): "
            f"min={hi_cal1_np.min():.4f}, max={hi_cal1_np.max():.4f}, mean={hi_cal1_np.mean():.4f}"
        )
    else:
        print(
            f"[decoder_v3] WARNING: HI calibrator not found at {hi_calibrator_path}. "
            f"Falling back to HI_cal_v1 = HI_phys and HI_cal_v2 = 1 - HI_cal_v1."
        )
        hi_cal1_seq_full = hi_phys_seq_full.clone()
        hi_cal2_seq_full = 1.0 - hi_cal1_seq_full

    # ------------------------------------------------------------------
    # 4) Build engine-based train/val loaders for v3
    # ------------------------------------------------------------------
    train_loader, val_loader = make_engine_split_loaders_v3(
        X_full_scaled,
        y_full,
        unit_ids_full,
        cond_ids_full,
        hi_phys_seq_full,
        hi_cal1_seq_full,
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

    print(f"[decoder_v3] Loading encoder from {encoder_experiment_dir}")
    encoder, _ = load_model_from_experiment(
        encoder_experiment_dir,
        device=device,
    )
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    cond_feature_indices = [i for i, c in enumerate(feature_cols) if c.startswith("Cond_")]
    if getattr(encoder, "use_cond_encoder", False) and getattr(encoder, "cond_in_dim", 0) > 0:
        if cond_feature_indices and len(cond_feature_indices) == getattr(encoder, "cond_in_dim", 0):
            encoder.cond_feature_indices = cond_feature_indices
            print(
                f"[decoder_v3] Set encoder.cond_feature_indices "
                f"({len(cond_feature_indices)} Cond_* features)"
            )
        else:
            print(
                "[decoder_v3] Warning: cond_feature_indices length does not match cond_in_dim; "
                "continuous condition encoder will not use explicit Cond_* features."
            )

    # ------------------------------------------------------------------
    # 6) Instantiate decoder v3
    # ------------------------------------------------------------------
    latent_dim = getattr(encoder, "d_model", None)
    if latent_dim is None:
        raise RuntimeError("Encoder is missing attribute d_model (expected Transformer encoder).")

    decoder_hidden_dim = int(config.get("decoder_hidden_dim", 128))
    decoder_num_layers = int(config.get("decoder_num_layers", 2))
    decoder_dropout = float(config.get("decoder_dropout", 0.1))

    decoder = RULTrajectoryDecoderV3(
        latent_dim=latent_dim,
        hi_feature_dim=4,
        slope_feature_dim=len(SLOPE_WINDOW_SIZES) * 3,  # 3 signals * len(window_sizes)
        hidden_dim=decoder_hidden_dim,
        num_layers=decoder_num_layers,
        dropout=decoder_dropout,
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
    decoder_results_dir = Path("results") / dataset_name.lower() / experiment_name
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
                hi_cal1_batch,
            ) = batch

            X_batch = X_batch.to(device)
            y_last_batch = y_last_batch.to(device)
            cond_ids_batch = cond_ids_batch.to(device)
            hi_phys_batch = hi_phys_batch.to(device)
            hi_cal1_batch = hi_cal1_batch.to(device)

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
            rul_seq_true = build_rul_seq_from_last(y_last_batch, T)  # [B, T]

            hi_cal2_batch = 1.0 - hi_cal1_batch

            # Compute slope features
            slope_feats = compute_slope_features(
                hi_phys_batch, hi_cal2_batch, hi_damage_seq_use, SLOPE_WINDOW_SIZES
            )  # [B, T, S]

            # Forward pass
            rul_seq_pred, degr_rate_pred = decoder(
                z_seq=z_seq,
                hi_phys_seq=hi_phys_batch,
                hi_cal1_seq=hi_cal1_batch,
                hi_cal2_seq=hi_cal2_batch,
                hi_damage_seq=hi_damage_seq_use,
                slope_feats=slope_feats,
            )

            # Loss components
            time_idx = torch.arange(T, device=device, dtype=rul_seq_true.dtype)  # [T]
            zone_weights = (time_idx + 1.0) / float(T)  # [T]
            zone_weights = zone_weights.unsqueeze(0).expand_as(rul_seq_true)  # [B, T]
            traj_loss = ((rul_seq_pred - rul_seq_true) ** 2 * zone_weights).mean()

            eol_loss = F.mse_loss(rul_seq_pred[:, -1], rul_seq_true[:, -1])

            diff_rul = rul_seq_pred[:, 1:] - rul_seq_pred[:, :-1]
            mono_violation = torch.relu(diff_rul)
            mono_loss = mono_violation.mean()

            diff1 = rul_seq_pred[:, 1:] - rul_seq_pred[:, :-1]
            diff2 = diff1[:, 1:] - diff1[:, :-1]
            smooth_loss = (diff2 ** 2).mean()

            # True degradation rate (positive when RUL decreases)
            delta_rul = rul_seq_true[:, 1:] - rul_seq_true[:, :-1]
            true_degr = -delta_rul
            true_degr = F.pad(true_degr, (1, 0), mode="replicate")  # [B, T]

            slope_loss = F.mse_loss(degr_rate_pred, true_degr)

            loss = (
                w_traj * traj_loss
                + w_eol * eol_loss
                + w_mono * mono_loss
                + w_smooth * smooth_loss
                + w_slope * slope_loss
            )

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ------------------------------------------------------------------
        # 8) Validation (EOL metrics)
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
                    hi_cal1_batch,
                ) = batch

                X_batch = X_batch.to(device)
                y_last_batch = y_last_batch.to(device)
                cond_ids_batch = cond_ids_batch.to(device)
                hi_phys_batch = hi_phys_batch.to(device)
                hi_cal1_batch = hi_cal1_batch.to(device)

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
                rul_seq_true = build_rul_seq_from_last(y_last_batch, T)

                hi_cal2_batch = 1.0 - hi_cal1_batch
                slope_feats = compute_slope_features(
                    hi_phys_batch, hi_cal2_batch, hi_damage_seq_use, SLOPE_WINDOW_SIZES
                )

                rul_seq_pred, _ = decoder(
                    z_seq=z_seq,
                    hi_phys_seq=hi_phys_batch,
                    hi_cal1_seq=hi_cal1_batch,
                    hi_cal2_seq=hi_cal2_batch,
                    hi_damage_seq=hi_damage_seq_use,
                    slope_feats=slope_feats,
                )

                y_true_eol_list.append(rul_seq_true[:, -1].cpu().numpy())
                y_pred_eol_list.append(rul_seq_pred[:, -1].cpu().numpy())

        y_true_eol = np.concatenate(y_true_eol_list, axis=0) if y_true_eol_list else np.zeros(0, dtype=np.float32)
        y_pred_eol = np.concatenate(y_pred_eol_list, axis=0) if y_pred_eol_list else np.zeros(0, dtype=np.float32)

        if y_true_eol.size == 0:
            raise RuntimeError("[decoder_v3] No validation samples collected.")

        val_metrics = compute_eol_errors_and_nasa(y_true_eol, y_pred_eol, max_rul=max_rul)
        val_rmse = float(val_metrics["rmse"])
        scheduler.step(val_rmse)

        print(
            f"[decoder_v3][Epoch {epoch:03d}] "
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
        raise RuntimeError("[decoder_v3] Training finished without any valid epochs.")

    # Restore best decoder state
    decoder.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 9) Simple TEST evaluation (EOL metrics, neutral HI inputs)
    # ------------------------------------------------------------------
    decoder.eval()
    with torch.no_grad():
        Xb_test = X_test_scaled.to(device)
        cond_ids_test_b = cond_ids_test.to(device)

        z_seq_test, _, hi_damage_seq_test = encoder.encode_with_hi(
            Xb_test,
            cond_ids=cond_ids_test_b,
            cond_vec=None,
        )

        hi_phys_test = torch.ones_like(z_seq_test[:, :, 0], device=device)
        hi_cal1_test = hi_phys_test.clone()
        hi_cal2_test = 1.0 - hi_cal1_test

        if hi_damage_seq_test.dim() == 3:
            hi_damage_test_use = hi_damage_seq_test.squeeze(-1)
        else:
            hi_damage_test_use = hi_damage_seq_test

        slope_feats_test = compute_slope_features(
            hi_phys_test, hi_cal2_test, hi_damage_test_use, SLOPE_WINDOW_SIZES
        )

        T_test = Xb_test.size(1)
        y_test_true = extra_info["y_test_true"].to(device)
        rul_seq_true_test = build_rul_seq_from_last(y_test_true, T_test)

        rul_pred_seq_test, _ = decoder(
            z_seq=z_seq_test,
            hi_phys_seq=hi_phys_test,
            hi_cal1_seq=hi_cal1_test,
            hi_cal2_seq=hi_cal2_test,
            hi_damage_seq=hi_damage_test_use,
            slope_feats=slope_feats_test,
        )
        y_true_eol_test = rul_seq_true_test[:, -1].cpu().numpy().astype(np.float32)
        y_pred_eol_test = rul_pred_seq_test[:, -1].cpu().numpy().astype(np.float32)

    test_metrics = compute_eol_errors_and_nasa(y_true_eol_test, y_pred_eol_test, max_rul=max_rul)

    # ------------------------------------------------------------------
    # 10) Save decoder + summary
    # ------------------------------------------------------------------
    decoder_ckpt_path = decoder_results_dir / "decoder_v3_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "meta": {
                "encoder_experiment": encoder_experiment,
                "dataset": dataset_name,
                "latent_dim": latent_dim,
                "encoder_type": "eol_full_transformer_encoder_v3d",
                "decoder_type": "rul_trajectory_decoder_v3",
            },
        },
        decoder_ckpt_path,
    )
    print(f"[decoder_v3] Saved best decoder checkpoint to {decoder_ckpt_path}")

    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model_type": "decoder_v3",
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
            "w_slope": w_slope,
        },
        "val_metrics": {
            "rmse": val_metrics["rmse"],
            "mae": val_metrics["mae"],
            "bias": val_metrics["bias"],
            "r2": val_metrics["r2"],
            "nasa_mean": val_metrics["nasa_mean"],
            "nasa_sum": val_metrics["nasa_sum"],
            "num_engines": val_metrics["num_engines"],
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

    summary_path = decoder_results_dir / "summary_decoder_v3.json"
    with open(summary_path, "w") as f:
        import json as _json

        _json.dump(summary, f, indent=2)
    print(f"[decoder_v3] Saved summary to {summary_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RUL Trajectory Decoder v3 on FD004/v3d encoder.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    if args.device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cfg: Dict[str, Any] = {
        "experiment_name": "fd004_decoder_v3_from_encoder_v3d",
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
            "num_epochs": 80,
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
        "w_slope": 0.2,
    }

    train_rul_decoder_v3(cfg, device_t)


