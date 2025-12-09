"""
Diagnostics script for TransformerWorldModelV1 on FD004.

This script:
- Rebuilds the FD004 feature pipeline exactly as in run_experiments.py
  for the experiment fd004_transformer_worldmodel_v1 (ms+DT + condition vector).
- Loads the trained TransformerWorldModelV1 checkpoint.
- Builds single-unit seq2seq samples (past window + future horizon).
- Runs autoregressive rollouts of sensors, HI and RUL.
- Visualises:
    * Multi-step sensor rollouts vs. ground truth
    * HI_future vs. ground truth HI
    * RUL_future vs. ground truth RUL
  and prints simple per-horizon error summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

# Ensure project root (containing 'src') is on sys.path, regardless of CWD
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from src.feature_safety import remove_rul_leakage
from src.models.transformer_world_model_v1 import TransformerWorldModelV1
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.experiment_configs import get_experiment_by_name


# -----------------------------------------------------------------------------
# 1. Data + Feature Pipeline (mirrors run_experiments for fd004_transformer_worldmodel_v1)
# -----------------------------------------------------------------------------


def load_fd004_data(max_rul: int = 125) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load FD004 raw data (train + test + true test RUL).
    Uses the same loader as run_experiments.py.
    """
    df_train, df_test, y_test_true = load_cmapps_subset(
        "FD004",
        max_rul=None,      # keep raw RUL in train; we'll clip manually where needed
        clip_train=False,
        clip_test=True,
    )
    return df_train, df_test, y_test_true


def _add_rul_to_df(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """
    Add per-cycle RUL and RUL_raw columns to a DataFrame (train or test),
    consistent with load_cmapps_subset's logic for training data.
    """
    df = df.copy()
    df_max_time = (
        df.groupby("UnitNumber")["TimeInCycles"]
        .max()
        .reset_index()
        .rename(columns={"TimeInCycles": "MaxTime"})
    )
    df = df.merge(df_max_time, on="UnitNumber", how="left")
    df["RUL_raw"] = df["MaxTime"] - df["TimeInCycles"]
    df["RUL"] = np.minimum(df["RUL_raw"], max_rul)
    return df


def build_features_for_fd004_worldmodel(
    experiment_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], dict, int, object, List[str]]:
    """
    Rebuild the feature pipeline for fd004_transformer_worldmodel_v1,
    mirroring run_experiments.py for this experiment.

    Returns:
        df_train_fe: training DataFrame with engineered features
        df_test_fe:  test DataFrame with engineered features
        feature_cols: ordered list of feature columns
        config: experiment config dict
        max_rul: maximum RUL used for clipping / HI mapping
    """
    config = get_experiment_by_name(experiment_name)
    dataset_name = config["dataset"]

    assert dataset_name == "FD004", f"Expected FD004, got {dataset_name}"

    world_model_params = config.get("world_model_params", {})
    max_rul = int(world_model_params.get("max_rul", 125))

    print(f"[Diagnostics] Loading {dataset_name} data for {experiment_name} ...")
    df_train, df_test, _ = load_fd004_data(max_rul=max_rul)

    # Add RUL to TEST as well (TRAIN already has RUL from loader)
    df_test = _add_rul_to_df(df_test, max_rul=max_rul)

    # ------------------------------
    # Physics / residual / ms+DT config (mirrors run_experiments.py)
    # ------------------------------
    name_lower = experiment_name.lower()
    is_phase4_residual = (
        (("phase4" in name_lower) or ("phase5" in name_lower)) and "residual" in name_lower
    ) or ("residual" in name_lower) or ("resid" in name_lower)

    from src.config import ResidualFeatureConfig

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

    features_cfg = config.get("features", {})
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

    phys_opts = config.get("phys_features", {})
    use_phys_condition_vec = phys_opts.get("use_condition_vector", False)
    use_twin_features = phys_opts.get(
        "use_twin_features",
        phys_opts.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_opts.get("twin_baseline_len", 30)
    condition_vector_version = phys_opts.get("condition_vector_version", 2)

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
        df_train, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config
    )
    df_test = create_all_features(
        df_test, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config
    )

    feature_cols = [
        c
        for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    print(f"[Diagnostics] Using {len(feature_cols)} features for model input.")

    # ------------------------------------------------------------------
    # Sensor scaler: match the normalization used in training.
    # Wir skalieren nur die 21 Basis-Sensoren, die auch der World-Model-Head
    # vorhersagt (Sensor1..Sensor21), nicht alle abgeleiteten Sensor-Features.
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler

    all_sensor_cols = [c for c in feature_cols if c.startswith("Sensor")]
    # Begrenze auf die ersten 21 "rohen" Sensoren (Sensor1..Sensor21)
    sensor_cols = all_sensor_cols[:21]
    if not sensor_cols:
        sensor_scaler = None
    else:
        sensor_scaler = StandardScaler()
        sensor_values_train = df_train[sensor_cols].to_numpy(dtype=np.float32, copy=True)
        sensor_scaler.fit(sensor_values_train)

        df_train[sensor_cols] = sensor_scaler.transform(sensor_values_train)
        df_test[sensor_cols] = sensor_scaler.transform(
            df_test[sensor_cols].to_numpy(dtype=np.float32, copy=True)
        )

    return df_train, df_test, feature_cols, config, max_rul, sensor_scaler, sensor_cols


# -----------------------------------------------------------------------------
# 2. Load world model + config
# -----------------------------------------------------------------------------


def load_world_model_and_config(
    feature_cols: List[str],
    config: dict,
    experiment_dir: Path | None = None,
) -> Tuple[TransformerWorldModelV1, torch.device]:
    """
    Rebuild encoder + TransformerWorldModelV1 and load the best checkpoint.
    """
    experiment_name = config["experiment_name"]
    dataset_name = config.get("dataset", "FD004").lower()

    if experiment_dir is None:
        experiment_dir = Path("results") / dataset_name / experiment_name
    else:
        experiment_dir = Path(experiment_dir)

    ckpt_name = f"transformer_world_model_v1_best_{experiment_name}.pt"
    ckpt_path = experiment_dir / ckpt_name
    summary_path = experiment_dir / "summary.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not summary_path.exists():
        print(f"[Diagnostics] Warning: summary.json not found at {summary_path}")
        summary = {}
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model_params = config.get("world_model_params", {})
    horizon = int(world_model_params.get("future_horizon", world_model_params.get("horizon", 20)))

    input_dim = len(feature_cols)
    encoder_kwargs = config["encoder_kwargs"]

    n_heads = encoder_kwargs.get("n_heads", encoder_kwargs.get("nhead", 4))

    encoder = EOLFullTransformerEncoder(
        input_dim=input_dim,
        d_model=encoder_kwargs.get("d_model", 64),
        num_layers=encoder_kwargs.get("num_layers", 3),
        n_heads=n_heads,
        dim_feedforward=encoder_kwargs.get("dim_feedforward", None),
        dropout=encoder_kwargs.get("dropout", 0.1),
        use_condition_embedding=True,
        num_conditions=7,
        cond_emb_dim=4,
        max_seq_len=300,
    )

    # These match train_transformer_world_model_v1 defaults
    num_sensors_out = 21
    cond_dim = 9

    # World-model specific flags from config (if present)
    decoder_hidden_dim = world_model_params.get("decoder_hidden_dim", 256)
    num_layers_decoder = world_model_params.get("num_layers_decoder", 1)
    target_mode = world_model_params.get("target_mode", "sensors")
    init_from_rul_hi = world_model_params.get("init_from_rul_hi", False)

    # Dynamic latent world-model flags (must match training config to avoid state_dict mismatches)
    use_latent_history = world_model_params.get("use_latent_history", False)
    use_hi_anchor = world_model_params.get("use_hi_anchor", False)
    use_future_conds = world_model_params.get("use_future_conds", False)

    world_model = TransformerWorldModelV1(
        encoder=encoder,
        input_dim=input_dim,
        num_sensors_out=num_sensors_out,
        cond_dim=cond_dim,
        future_horizon=horizon,
        decoder_hidden_dim=decoder_hidden_dim,
        num_layers_decoder=num_layers_decoder,
        dropout=encoder_kwargs.get("dropout", 0.1),
        predict_hi=True,
        predict_rul=True,
        target_mode=target_mode,
        init_from_rul_hi=init_from_rul_hi,
        use_latent_history=use_latent_history,
        use_hi_anchor=use_hi_anchor,
        use_future_conds=use_future_conds,
    )

    state = torch.load(ckpt_path, map_location=device)
    world_model.load_state_dict(state["model_state_dict"])
    world_model.to(device)
    world_model.eval()

    print(f"[Diagnostics] Loaded world model from {ckpt_path}")

    return world_model, device


# -----------------------------------------------------------------------------
# 3. Build single-unit seq2seq sample
# -----------------------------------------------------------------------------


def build_unit_seq2seq_sample(
    df: pd.DataFrame,
    unit_id: int,
    feature_cols: List[str],
    sensor_cols: List[str],
    past_len: int,
    horizon: int,
    max_rul: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build a single (X_past, Y_future_sens, Y_future_rul, Y_future_hi, cond_id)
    sample for one unit. Uses a central anchor within the trajectory to ensure
    past_len and horizon fit. ConditionID is assumed constant per unit and
    is returned as an int for use with condition embeddings.
    """
    df_unit = df[df["UnitNumber"] == unit_id].sort_values("TimeInCycles")
    n_cycles = len(df_unit)

    if n_cycles < past_len + horizon:
        raise ValueError(f"Unit {unit_id} too short for past_len={past_len}, horizon={horizon}")

    # Choose an anchor within the valid range such that we always have
    # `past_len` steps before and `horizon` steps after.
    valid_start = past_len
    valid_end = n_cycles - horizon
    if valid_end <= valid_start:
        # Degenerate but should be caught by n_cycles check above
        anchor = valid_start
    else:
        anchor = (valid_start + valid_end) // 2

    start_past = anchor - past_len
    end_past = anchor
    start_future = end_past
    end_future = end_past + horizon

    window_past = df_unit.iloc[start_past:end_past]
    window_future = df_unit.iloc[start_future:end_future]

    X_past = window_past[feature_cols].values.astype(np.float32)
    Y_sens = window_future[sensor_cols].values.astype(np.float32)
    future_cycles = window_future["TimeInCycles"].values.astype(np.float32)

    # Build future RUL + HI using same physics mapping as in training
    future_rul = window_future["RUL"].values.astype(np.float32)
    future_rul = np.clip(future_rul, 0.0, float(max_rul))

    MAX_VISIBLE_RUL = float(max_rul)
    hi_linear = np.clip(future_rul / MAX_VISIBLE_RUL, 0.0, 1.0)
    future_hi = np.where(future_rul > MAX_VISIBLE_RUL, 1.0, hi_linear).astype(np.float32)

    # ConditionID (constant per unit)
    if "ConditionID" not in df_unit.columns:
        raise KeyError("Expected 'ConditionID' column in dataframe for diagnostics.")
    cond_id = int(df_unit["ConditionID"].iloc[0])

    return X_past, Y_sens, future_rul, future_hi, future_cycles, cond_id


# -----------------------------------------------------------------------------
# 4. Rollout + plotting
# -----------------------------------------------------------------------------


def plot_world_model_rollout_for_unit(
    world_model: TransformerWorldModelV1,
    device: torch.device,
    df_test: pd.DataFrame,
    unit_id: int,
    feature_cols: List[str],
    sensor_cols: List[str],
    past_len: int,
    horizon: int,
    max_rul: int,
    out_dir: Path,
    sensor_scaler: object | None = None,
    num_sensors_to_plot: int = 3,
) -> None:
    """
    Build a single-unit seq2seq sample, run autoregressive rollout, and plot
    sensors / HI / RUL vs. ground truth. Also prints simple per-horizon errors.
    """
    try:
        X_past_np, Y_sens_np, Y_rul_np, Y_hi_np, future_cycles, cond_id = build_unit_seq2seq_sample(
            df_test,
            unit_id=unit_id,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
            past_len=past_len,
            horizon=horizon,
            max_rul=max_rul,
        )
    except ValueError as e:
        # Unit too short or invalid window configuration – skip gracefully.
        print(f"[Diagnostics] Skipping unit {unit_id}: {e}")
        return

    X_past = torch.from_numpy(X_past_np).unsqueeze(0).to(device)  # (1, T_in, F)
    # For now we mirror training: continuous condition vector is zeros.
    cond_vec = torch.zeros((1, world_model.cond_dim), dtype=torch.float32, device=device)
    cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

    with torch.no_grad():
        pred_sensors, pred_hi, pred_rul = world_model(
            past_seq=X_past,
            cond_vec=cond_vec,
            cond_ids=cond_ids,
            future_horizon=horizon,
            teacher_forcing_targets=None,  # pure autoregressive rollout
        )

    pred_sensors_np = pred_sensors.squeeze(0).cpu().numpy()  # (H, num_sensors_out)
    pred_hi_np = (
        pred_hi.squeeze(0).squeeze(-1).cpu().numpy() if pred_hi is not None else None
    )  # (H,)
    pred_rul_np = (
        pred_rul.squeeze(0).squeeze(-1).cpu().numpy() if pred_rul is not None else None
    )  # (H,)

    # X-Achse: echte Zyklusindizes des Forecast-Fensters
    t = future_cycles

    # --- Simple error summaries pro Zyklus im Forecast-Fenster ---
    # De-normalize sensors for plotting if a scaler is available
    if sensor_scaler is not None:
        Y_sens_denorm = sensor_scaler.inverse_transform(Y_sens_np)
        pred_sens_denorm = sensor_scaler.inverse_transform(pred_sensors_np)
        sensor_mse_per_step = np.mean((pred_sens_denorm - Y_sens_denorm) ** 2, axis=1)
    else:
        Y_sens_denorm = Y_sens_np
        pred_sens_denorm = pred_sensors_np
        sensor_mse_per_step = np.mean((pred_sensors_np[:, : Y_sens_np.shape[1]] - Y_sens_np) ** 2, axis=1)

    print(f"\n[Diagnostics] Unit {unit_id} – sensor MSE per step (first 5): {sensor_mse_per_step[:5]}")

    if pred_hi_np is not None:
        hi_mse_per_step = (pred_hi_np - Y_hi_np) ** 2
        print(f"[Diagnostics] Unit {unit_id} – HI MSE per step (first 5): {hi_mse_per_step[:5]}")

    if pred_rul_np is not None:
        # Model predicts normalized RUL ∈ [0,1]; de-normalize for metrics/plots.
        pred_rul_denorm = pred_rul_np * float(max_rul)
        rul_mae_per_step = np.abs(pred_rul_denorm - Y_rul_np)
        print(f"[Diagnostics] Unit {unit_id} – RUL MAE per step (first 5): {rul_mae_per_step[:5]}")

    # 1) Sensor plots
    plt.figure(figsize=(12, 6))
    num_plots = min(num_sensors_to_plot, pred_sens_denorm.shape[1], len(sensor_cols))
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(t, Y_sens_denorm[:, i], label=f"{sensor_cols[i]} – true")
        plt.plot(t, pred_sens_denorm[:, i], linestyle="--", label=f"{sensor_cols[i]} – pred")
        plt.legend()
        plt.xlabel("Forecast step")
        plt.ylabel("Sensor value")
    plt.suptitle(f"FD004 World Model V1 – Sensors – Unit {unit_id}")
    plt.tight_layout()
    sensors_path = out_dir / f"unit_{unit_id}_sensors.png"
    plt.savefig(sensors_path, dpi=200)
    plt.show()

    # 2) HI plot
    if pred_hi_np is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(t, Y_hi_np, label="HI true")
        plt.plot(t, pred_hi_np, linestyle="--", label="HI pred")
        plt.title(f"FD004 World Model V1 – HI – Unit {unit_id}")
        plt.xlabel("Forecast step")
        plt.ylabel("HI")
        plt.legend()
        plt.tight_layout()
        hi_path = out_dir / f"unit_{unit_id}_hi.png"
        plt.savefig(hi_path, dpi=200)
        plt.show()

    # 3) RUL plot
    if pred_rul_np is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(t, Y_rul_np, label="RUL true")
        plt.plot(t, pred_rul_denorm, linestyle="--", label="RUL pred")
        plt.title(f"FD004 World Model V1 – RUL – Unit {unit_id}")
        plt.xlabel("Forecast step")
        plt.ylabel("RUL [cycles]")
        plt.legend()
        plt.tight_layout()
        rul_path = out_dir / f"unit_{unit_id}_rul.png"
        plt.savefig(rul_path, dpi=200)
        plt.show()


# -----------------------------------------------------------------------------
# 5. EOL RUL scatter + error histogram
# -----------------------------------------------------------------------------


def build_eol_rul_predictions(
    world_model: TransformerWorldModelV1,
    device: torch.device,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    past_len: int,
    horizon: int,
    max_rul: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build EOL-style RUL predictions for each test engine by:
    - taking the last past_len window before EOL,
    - running a free rollout of length `horizon`,
    - taking the last RUL prediction in the horizon as EOL-like forecast,
    - comparing against the true RUL at the last observed cycle.
    """
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    if "RUL" not in df_test.columns:
        print("[Diagnostics] Warning: no 'RUL' column in df_test; cannot build EOL RUL plots.")
        return np.array([]), np.array([])

    for unit_id, df_unit in df_test.groupby("UnitNumber"):
        df_u = df_unit.sort_values("TimeInCycles")
        feats = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
        n = len(feats)
        if n < 1:
            continue

        # Build past window ending at EOL (last observed cycle)
        if n < past_len:
            pad = np.tile(feats[0:1], (past_len - n, 1))
            feats_past = np.vstack([pad, feats])
        else:
            feats_past = feats[-past_len:]

        X_past = torch.from_numpy(feats_past).unsqueeze(0).to(device)  # (1, past_len, F)
        cond_vec = torch.zeros((1, world_model.cond_dim), dtype=torch.float32, device=device)
        cond_id = int(df_u["ConditionID"].iloc[0])
        cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

        # Current RUL/HI at EOL
        curr_rul = float(df_u["RUL"].iloc[-1])
        curr_rul = max(0.0, min(curr_rul, float(max_rul)))
        curr_rul_norm = curr_rul / float(max_rul) if max_rul > 0 else 0.0
        curr_hi = max(0.0, min(curr_rul_norm, 1.0))
        current_rul = torch.tensor([curr_rul_norm], dtype=torch.float32, device=device)
        current_hi = torch.tensor([curr_hi], dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, pred_rul = world_model(
                past_seq=X_past,
                cond_vec=cond_vec,
                cond_ids=cond_ids,
                future_horizon=horizon,
                teacher_forcing_targets=None,
                current_rul=current_rul,
                current_hi=current_hi,
            )

        if pred_rul is None:
            continue

        # Take last horizon step, de-normalize to cycles
        pred_rul_last_norm = float(pred_rul[0, -1, 0].cpu().item())
        pred_rul_last = max(0.0, pred_rul_last_norm * float(max_rul))

        y_true_all.append(curr_rul)
        y_pred_all.append(pred_rul_last)

    if not y_true_all:
        return np.array([]), np.array([])

    return np.array(y_true_all, dtype=np.float32), np.array(y_pred_all, dtype=np.float32)


# -----------------------------------------------------------------------------
# 5a. Encoder anchor scatter (HI/RUL) for the underlying ms+DT encoder
# -----------------------------------------------------------------------------


def build_encoder_anchor_predictions(
    world_model: TransformerWorldModelV1,
    device: torch.device,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    past_len: int,
    max_rul: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build encoder-level HI/RUL anchors for each test engine.

    For each engine:
      - take the last past_len window before EOL (last observed cycle),
      - run the encoder-only forward pass,
      - extract RUL_anchor (cycles) and HI_anchor (in [0,1]),
      - compare against true RUL/HI at EOL.
    """
    y_true_rul_all: List[float] = []
    y_pred_rul_all: List[float] = []
    hi_true_all: List[float] = []
    hi_pred_all: List[float] = []

    if "RUL" not in df_test.columns:
        print("[Diagnostics] Warning: no 'RUL' column in df_test; cannot build encoder anchor plots.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    encoder = world_model.encoder

    for unit_id, df_unit in df_test.groupby("UnitNumber"):
        df_u = df_unit.sort_values("TimeInCycles")
        feats = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
        n = len(feats)
        if n < 1:
            continue

        # Build past window ending at EOL (last observed cycle)
        if n < past_len:
            pad = np.tile(feats[0:1], (past_len - n, 1))
            feats_past = np.vstack([pad, feats])
        else:
            feats_past = feats[-past_len:]

        X_past = torch.from_numpy(feats_past).unsqueeze(0).to(device)  # (1, past_len, F)
        cond_id = int(df_u["ConditionID"].iloc[0])
        cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

        # True RUL/HI at EOL
        true_rul = float(df_u["RUL"].iloc[-1])
        true_rul = max(0.0, min(true_rul, float(max_rul)))
        true_hi = max(0.0, min(true_rul / max_rul if max_rul > 0 else 0.0, 1.0))

        with torch.no_grad():
            # EOLFullTransformerEncoder forward: (rul_pred, health_last, health_seq)
            rul_anchor_raw, hi_anchor_last, _ = encoder(X_past, cond_ids)

        # Clamp / normalize anchors to match training conventions
        rul_anchor = float(torch.clamp(rul_anchor_raw, 0.0, float(max_rul))[0].cpu().item())
        hi_anchor = float(torch.clamp(torch.sigmoid(hi_anchor_last[0]), 0.0, 1.0).cpu().item())

        y_true_rul_all.append(true_rul)
        y_pred_rul_all.append(rul_anchor)
        hi_true_all.append(true_hi)
        hi_pred_all.append(hi_anchor)

    if not y_true_rul_all:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.array(hi_true_all, dtype=np.float32),
        np.array(hi_pred_all, dtype=np.float32),
        np.array(y_true_rul_all, dtype=np.float32),
        np.array(y_pred_rul_all, dtype=np.float32),
    )


def plot_encoder_anchor_scatter(
    hi_true: np.ndarray,
    hi_pred: np.ndarray,
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot encoder anchor quality: HI_true vs HI_pred and RUL_true vs RUL_pred."""
    if hi_true.size == 0 or rul_true.size == 0:
        print("[Diagnostics] No encoder anchor predictions to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: RUL (cycles)
    plt.figure(figsize=(5, 5))
    plt.scatter(rul_true, rul_pred, s=8, alpha=0.5)
    plt.xlabel("True RUL at EOL [cycles]")
    plt.ylabel("Encoder RUL anchor [cycles]")
    plt.title("Encoder anchor: RUL_true vs RUL_pred (ms+DT encoder)")
    plt.grid(True)
    rmin, rmax = float(rul_true.min()), float(rul_true.max())
    plt.plot([rmin, rmax], [rmin, rmax], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_rul_anchor_scatter.png", dpi=150)
    plt.close()

    # Scatter: HI (0–1)
    plt.figure(figsize=(5, 5))
    plt.scatter(hi_true, hi_pred, s=8, alpha=0.5)
    plt.xlabel("True HI (RUL/max_rul)")
    plt.ylabel("Encoder HI anchor")
    plt.title("Encoder anchor: HI_true vs HI_pred (ms+DT encoder)")
    plt.grid(True)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_hi_anchor_scatter.png", dpi=150)
    plt.close()


def plot_eol_rul_scatter_and_error_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot true vs predicted EOL RUL scatter and error histogram."""
    if y_true.size == 0 or y_pred.size == 0:
        print("[Diagnostics] No EOL RUL predictions to plot.")
        return

    # Scatter: true vs pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "k--", label="Ideal")
    plt.xlabel("True RUL at EOL [cycles]")
    plt.ylabel("Predicted RUL at EOL [cycles]")
    plt.title("WorldModel V1 – EOL RUL: True vs Pred")
    plt.legend()
    plt.grid(alpha=0.3)
    scatter_path = out_dir / "eol_rul_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.show()

    # Error histogram
    errors = y_pred - y_true
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=40, alpha=0.8, color="tab:blue", edgecolor="black")
    plt.axvline(0.0, color="k", linestyle="--", label="Zero error")
    plt.xlabel("Prediction error (Predicted RUL - True RUL) [cycles]")
    plt.ylabel("Count")
    plt.title("WorldModel V1 – EOL RUL Error Histogram")
    plt.legend()
    plt.grid(alpha=0.3)
    hist_path = out_dir / "eol_rul_error_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.show()


# -----------------------------------------------------------------------------
# 6. Sliding HI/RUL trajectories over full engine life
# -----------------------------------------------------------------------------


def build_full_hi_rul_trajectory_for_unit(
    world_model: TransformerWorldModelV1,
    device: torch.device,
    df_unit: pd.DataFrame,
    feature_cols: List[str],
    past_len: int,
    horizon: int,
    max_rul: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window HI and RUL trajectories for a single unit over its
    full lifetime. For each cycle t where we have at least `past_len` history
    and `horizon` future steps, we:

    - take past window [t-past_len, t)
    - use true RUL/HI at step t as "current" (for decoder init)
    - run a free rollout of length `horizon`
    - take the first predicted HI/RUL timestep as the model's estimate at t.

    Returns:
        cycles:       (T_valid,) cycle indices where predictions are available
        hi_true:      (T_valid,) true HI at those cycles
        hi_pred:      (T_valid,) predicted HI at those cycles
        rul_true:     (T_valid,) true RUL (cycles) at those cycles
        rul_pred:     (T_valid,) predicted RUL (cycles) at those cycles
    """
    df_u = df_unit.sort_values("TimeInCycles").reset_index(drop=True)
    cycles_full = df_u["TimeInCycles"].to_numpy(dtype=np.float32)
    rul_full = df_u["RUL"].clip(lower=0.0, upper=max_rul).to_numpy(dtype=np.float32)

    T = len(df_u)
    # We can predict from t = past_len .. T - horizon - 1 to ensure enough future
    t_start = past_len
    t_end = T - horizon
    if t_end <= t_start:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    cycles = []
    hi_true_list = []
    hi_pred_list = []
    rul_true_list = []
    rul_pred_list = []

    for t in range(t_start, t_end):
        past = df_u.iloc[t - past_len : t]
        future = df_u.iloc[t : t + horizon]

        X_past = past[feature_cols].to_numpy(dtype=np.float32)
        X_past_t = torch.from_numpy(X_past).unsqueeze(0).to(device)  # (1, past_len, F)

        cond_vec = torch.zeros((1, world_model.cond_dim), dtype=torch.float32, device=device)
        cond_id = int(past["ConditionID"].iloc[-1])
        cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

        # True RUL/HI at cycle t
        true_rul_t = float(rul_full[t])
        true_hi_t = float(np.clip(true_rul_t / max_rul, 0.0, 1.0))

        # Normalized current RUL/HI for decoder init
        curr_rul_norm = true_rul_t / max_rul if max_rul > 0 else 0.0
        current_rul = torch.tensor([curr_rul_norm], dtype=torch.float32, device=device)
        current_hi = torch.tensor([true_hi_t], dtype=torch.float32, device=device)

        with torch.no_grad():
            _, pred_hi, pred_rul = world_model(
                past_seq=X_past_t,
                cond_vec=cond_vec,
                cond_ids=cond_ids,
                future_horizon=horizon,
                teacher_forcing_targets=None,
                current_rul=current_rul,
                current_hi=current_hi,
            )

        if pred_hi is None or pred_rul is None:
            continue

        # First predicted step as estimate at cycle t
        hi_t_pred = float(pred_hi[0, 0, 0].cpu().item())
        rul_t_pred_norm = float(pred_rul[0, 0, 0].cpu().item())
        rul_t_pred = max(0.0, rul_t_pred_norm * max_rul)

        cycles.append(cycles_full[t])
        hi_true_list.append(true_hi_t)
        hi_pred_list.append(hi_t_pred)
        rul_true_list.append(true_rul_t)
        rul_pred_list.append(rul_t_pred)

    if not cycles:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.array(cycles, dtype=np.float32),
        np.array(hi_true_list, dtype=np.float32),
        np.array(hi_pred_list, dtype=np.float32),
        np.array(rul_true_list, dtype=np.float32),
        np.array(rul_pred_list, dtype=np.float32),
    )


def plot_full_hi_rul_trajectory_for_unit(
    unit_id: int,
    cycles: np.ndarray,
    hi_true: np.ndarray,
    hi_pred: np.ndarray,
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot full-life HI and RUL trajectories for a single engine."""
    if cycles.size == 0:
        print(f"[Diagnostics] No full trajectory for unit {unit_id}.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # HI subplot
    ax_hi = axes[0]
    ax_hi.plot(cycles, hi_true, label="HI true", color="tab:blue")
    ax_hi.plot(cycles, hi_pred, "--", label="HI pred", color="tab:red")
    ax_hi.set_ylabel("HI")
    ax_hi.set_title(f"WorldModel V1 – Full HI trajectory – Unit {unit_id}")
    ax_hi.set_ylim([-0.05, 1.05])
    ax_hi.grid(alpha=0.3)
    ax_hi.legend()

    # RUL subplot
    ax_r = axes[1]
    ax_r.plot(cycles, rul_true, label="RUL true", color="tab:green")
    ax_r.plot(cycles, rul_pred, "--", label="RUL pred", color="tab:orange")
    ax_r.set_xlabel("Cycle")
    ax_r.set_ylabel("RUL [cycles]")
    ax_r.set_title(f"WorldModel V1 – Full RUL trajectory – Unit {unit_id}")
    ax_r.grid(alpha=0.3)
    ax_r.legend()

    plt.tight_layout()
    path = out_dir / f"unit_{unit_id}_hi_rul_full.png"
    plt.savefig(path, dpi=200)
    plt.show()


# -----------------------------------------------------------------------------
# 7. Main entry point
# -----------------------------------------------------------------------------


def main() -> None:
    # Optional: experiment name as first CLI argument, default to base V1 experiment
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = "fd004_transformer_worldmodel_v1"

    # Guard: this script is designed for TransformerWorldModelV1 experiments.
    # For pure encoder runs (e.g. fd004_transformer_encoder_ms_dt_v2) there is
    # no world-model checkpoint, so we exit with a clear message instead of
    # raising a FileNotFoundError.
    if "worldmodel" not in experiment_name.lower():
        print(
            f"[WorldModelDiagnostics] Experiment '{experiment_name}' is an encoder-only run "
            "(no TransformerWorldModelV1 checkpoint expected)."
        )
        print(
            "[WorldModelDiagnostics] Use 'FD004_encoder_anchor_diagnostics.py' for encoder V1/V2 "
            "anchor plots and RUL/HI diagnostics, or run a dedicated world-model experiment "
            "such as 'fd004_transformer_worldmodel_v1'."
        )
        return

    # 1) Build features
    df_train_fe, df_test_fe, feature_cols, config, max_rul, sensor_scaler, sensor_cols = build_features_for_fd004_worldmodel(
        experiment_name
    )

    # 2) Load world model
    world_model, device = load_world_model_and_config(
        feature_cols=feature_cols,
        config=config,
    )

    world_model.eval()

    # 3) Sensor columns (first 21 Sensor* columns used in training)
    if not sensor_cols:
        sensor_cols = [c for c in feature_cols if c.startswith("Sensor")][:21]
    world_model_params = config.get("world_model_params", {})
    horizon = int(world_model_params.get("future_horizon", world_model_params.get("horizon", 20)))
    past_len = int(world_model_params.get("past_len", 30))

    # Output directory based on experiment
    dataset_name = config.get("dataset", "FD004").lower()
    base_dir = Path("results") / dataset_name / experiment_name
    out_dir = base_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick units for visual inspection, ensuring they are long enough
    # to support the requested past_len + horizon (after feature engineering).
    unit_lengths = (
        df_test_fe.groupby("UnitNumber")["TimeInCycles"]
        .size()
        .to_dict()
    )
    valid_unit_ids = [
        uid for uid, n in unit_lengths.items() if n >= past_len + horizon
    ]
    if not valid_unit_ids:
        raise RuntimeError(
            f"No test units have at least past_len + horizon = {past_len + horizon} "
            "cycles after feature engineering."
        )

    # Prefer "degraded" engines: solche mit kleinem minimalem RUL im Testset.
    if "RUL" in df_test_fe.columns:
        rul_stats = (
            df_test_fe.groupby("UnitNumber")["RUL"]
            .min()
            .reset_index()
            .sort_values("RUL")  # kleinste RUL = stärkste Degradation
        )
        degraded_candidates = rul_stats["UnitNumber"].tolist()
        degraded_unit_ids = [
            int(uid) for uid in degraded_candidates if int(uid) in valid_unit_ids
        ]
        # Nimm bis zu 10 degradierte Units
        unit_ids = degraded_unit_ids[:10] if degraded_unit_ids else sorted(valid_unit_ids)[:10]
        print(f"[Diagnostics] Selected degraded units: {unit_ids}")
    else:
        # Fallback: einfach die ersten 10 gültigen Units
        unit_ids = sorted(valid_unit_ids)[:10]
        print(f"[Diagnostics] Selected units (no RUL column available): {unit_ids}")

    # Per-engine HI/RUL trajectories for selected (degraded) units
    for uid in unit_ids:
        plot_world_model_rollout_for_unit(
            world_model=world_model,
            device=device,
            df_test=df_test_fe,
            unit_id=uid,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
            past_len=past_len,
            horizon=horizon,
            max_rul=max_rul,
            out_dir=out_dir,
            sensor_scaler=sensor_scaler,
        )
        # Full sliding HI/RUL trajectories over the entire engine life
        df_unit = df_test_fe[df_test_fe["UnitNumber"] == uid]
        cycles, hi_true, hi_pred, rul_true, rul_pred = build_full_hi_rul_trajectory_for_unit(
            world_model=world_model,
            device=device,
            df_unit=df_unit,
            feature_cols=feature_cols,
            past_len=past_len,
            horizon=horizon,
            max_rul=max_rul,
        )
        plot_full_hi_rul_trajectory_for_unit(
            unit_id=uid,
            cycles=cycles,
            hi_true=hi_true,
            hi_pred=hi_pred,
            rul_true=rul_true,
            rul_pred=rul_pred,
            out_dir=out_dir,
        )

    # Global EOL RUL scatter + error histogram across all test engines
    y_true_eol, y_pred_eol = build_eol_rul_predictions(
        world_model=world_model,
        device=device,
        df_test=df_test_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        horizon=horizon,
        max_rul=max_rul,
    )
    plot_eol_rul_scatter_and_error_hist(
        y_true=y_true_eol,
        y_pred=y_pred_eol,
        out_dir=out_dir,
    )

    # 5a) Encoder anchor scatter (HI/RUL) for the underlying ms+DT encoder
    hi_true_anchor, hi_pred_anchor, rul_true_anchor, rul_pred_anchor = build_encoder_anchor_predictions(
        world_model=world_model,
        device=device,
        df_test=df_test_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
    )
    plot_encoder_anchor_scatter(
        hi_true=hi_true_anchor,
        hi_pred=hi_pred_anchor,
        rul_true=rul_true_anchor,
        rul_pred=rul_pred_anchor,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()


