"""
FD004 Physics-Informed State Encoder V3 Diagnostics
---------------------------------------------------

This script:
- Rebuilds the FD004 ms+DT feature pipeline (same as encoder ms_dt_v1).
- Builds sliding-window sequences (full-life) with the same feature_cols.
- Recomputes physics-informed HI labels from:
    - residual norms (Resid_*),
    - EGT_Drift,
    - Effizienz_HPC_Proxy.
- Loads the trained TransformerStateEncoderV3_Physics checkpoint.
- Produces:
    * HI_true_physics vs HI_pred scatter
    * RUL_norm_true vs RUL_norm_pred scatter
    * Example HI trajectories over cycles for a few degraded engines

Run from project root:
    python notebooks/FD004_state_encoder_v3_physics_diagnostics.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

# Ensure project root (containing 'src') is on sys.path, regardless of CWD
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
    group_feature_columns,
)
from src.data.physics_hi import add_physics_hi_v2
from src.config import ResidualFeatureConfig
from src.feature_safety import remove_rul_leakage
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
)
from src.experiment_configs import get_experiment_by_name
from src.models.transformer_state_encoder_v3_physics import TransformerStateEncoderV3_Physics
from src.state_encoder_training_v3_physics import build_physics_hi_from_components
from src.metrics import compute_eol_errors_and_nasa


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_feature_pipeline(cfg) -> Tuple[FeatureConfig, PhysicsFeatureConfig, dict]:
    """
    Mirror the feature / physics configuration used in run_experiments.py
    for fd004_transformer_encoder_ms_dt_v1.
    """
    experiment_name = cfg["experiment_name"]
    name_lower = experiment_name.lower()
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

    # High-level feature block configuration (multiscale)
    features_cfg = cfg.get("features", {})
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

    phys_opts = cfg.get("phys_features", {})
    return feature_config, physics_config, phys_opts


def _add_rul_to_df(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """
    Add per-cycle RUL and RUL_raw columns to a DataFrame (train or test),
    without relying on a pre-existing 'MaxTime' column.
    """
    df = df.copy()
    # Compute per-engine maximum cycle and derive RUL from it
    max_time = df.groupby("UnitNumber")["TimeInCycles"].transform("max")
    df["RUL_raw"] = max_time - df["TimeInCycles"]
    df["RUL"] = np.minimum(df["RUL_raw"], max_rul)
    return df


def prepare_fd004_msdt_data(run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int, np.ndarray]:
    """
    Load FD004 train/test data and apply the same feature pipeline as for the
    given state-encoder experiment.

    Args:
        run_name: Name of the experiment, e.g.
            "fd004_transformer_state_encoder_v3_physics"
            "fd004_state_encoder_v3_physics_A1_ms_only"
            "fd004_state_encoder_v3_physics_B1_msdt"

    Returns:
        df_train_fe, df_test_fe, feature_cols, max_rul, y_test_true
    """
    # Use the SAME config as for the specified physics state encoder experiment
    # so that feature engineering matches the training run.
    cfg = get_experiment_by_name(run_name)
    dataset = cfg["dataset"]
    assert dataset == "FD004", f"Expected dataset FD004, got {dataset}"

    data_cfg = cfg.get("data", {})
    max_rul = int(data_cfg.get("max_rul", 125))
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Add RUL to train (ensure we have capped RUL)
    df_train = _add_rul_to_df(df_train, max_rul=max_rul)

    feature_config, physics_config, phys_opts = _build_feature_pipeline(cfg)

    # 1) Physics features – mirror run_experiments.py for BOTH train and test
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector (Cond_*)
    use_phys_condition_vec = phys_opts.get("use_condition_vector", False)
    condition_vector_version = phys_opts.get("condition_vector_version", 2)
    if use_phys_condition_vec:
        print("  Using continuous condition vector features (Cond_*)")
        df_train = build_condition_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )
        # IMPORTANT: apply same condition-vector transformation to TEST so that
        # Cond_* columns exist for both train and test (otherwise feature
        # counts diverge in diagnostics).
        df_test = build_condition_features(
            df_test,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin residuals
    use_twin_features = phys_opts.get(
        "use_twin_features",
        phys_opts.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_opts.get("twin_baseline_len", 30)
    if use_twin_features:
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len})")
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        # Apply the same twin model to TEST data, just like in run_experiments.py
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Multi-scale temporal features
    df_train_fe = create_all_features(
        df_train,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    df_test_fe = create_all_features(
        df_test,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    # 5) Physics-based HI_v2 + Hybrid HI_target – mirror run_experiments.py:
    #    add_physics_hi_v2 AFTER all engineered features, and keep all helper
    #    channels so that the encoder sees the same 36x-dimensional space as
    #    during training.
    max_rul_cfg = data_cfg.get("max_rul", 125)
    df_train_fe, hi_scalers = add_physics_hi_v2(
        df_train_fe,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        cond_col="ConditionID",
        is_training=True,
        max_rul=float(max_rul_cfg),
        rul_col="RUL",
        alpha_hybrid=0.7,
    )
    # Apply the same HI_v2 / damage scaling to the test set without per-unit
    # rescaling of the tail (as in training).
    df_test_fe = add_physics_hi_v2(
        df_test_fe,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        cond_col="ConditionID",
        is_training=False,
        max_rul=float(max_rul_cfg),
        rul_col="RUL",
        alpha_hybrid=0.7,
        global_scalers=hi_scalers,
    )

    feature_cols = [
        c
        for c in df_train_fe.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    # Never feed HI_* target columns as input features; they are used as
    # supervised targets for the state encoder. All other helper columns
    # (damage channels, HI_phys_v2, etc.) remain as raw inputs.
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid"]
    ]
    print(f"[StateEncoderPhysicsDiag] Using {len(feature_cols)} features for ms+DT input.")

    return df_train_fe, df_test_fe, feature_cols, max_rul, y_test_true


def load_state_encoder_v3_physics(
    input_dim: int,
    cond_in_dim: int,
    checkpoint_dir: Path,
    experiment_name: str,
    model_cfg: Optional[dict] = None,
) -> TransformerStateEncoderV3_Physics:
    """
    Instantiate and load the TransformerStateEncoderV3_Physics from checkpoint.

    The checkpoint file is expected to follow the naming convention used in
    train_state_encoder_v3_physics:
        <experiment_name>_state_encoder_v3_physics.pt
    """
    ckpt_path = checkpoint_dir / f"{experiment_name}_state_encoder_v3_physics.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Explicitly load full checkpoint object (not weights_only) – safe here because
    # the file is produced by our own training code.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)

    # Use model configuration from experiment config when available so that
    # the architecture (including optional damage head) matches training.
    if model_cfg is None:
        d_model = 96
        num_layers = 3
        num_heads = 4
        dim_feedforward = 256
        dropout = 0.1
        use_damage_head = False
        L_ref = 300.0
        alpha_base = 0.1
    else:
        d_model = int(model_cfg.get("d_model", 96))
        num_layers = int(model_cfg.get("num_layers", 3))
        num_heads = int(model_cfg.get("num_heads", 4))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        dropout = float(model_cfg.get("dropout", 0.1))
        use_damage_head = bool(model_cfg.get("use_damage_head", False))
        L_ref = float(model_cfg.get("L_ref", 300.0))
        alpha_base = float(model_cfg.get("alpha_base", 0.1))

    model = TransformerStateEncoderV3_Physics(
        input_dim=input_dim,
        cond_in_dim=cond_in_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_damage_head=use_damage_head,
        L_ref=L_ref,
        alpha_base=alpha_base,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[StateEncoderPhysicsDiag] Loaded model from {ckpt_path}")
    return model


def run_diagnostics(run_name: str = "fd004_transformer_state_encoder_v3_physics") -> None:
    # 0) Load experiment config to inspect model options (e.g. damage head)
    exp_cfg = get_experiment_by_name(run_name)
    model_cfg = exp_cfg.get("model", {})

    # 1) Build features and sequences (train + test)
    df_train_fe, df_test_fe, feature_cols, max_rul, y_test_true = prepare_fd004_msdt_data(run_name)

    past_len = 30
    X_full, y_rul_full, unit_ids_full, _ = build_full_eol_sequences_from_df(
        df=df_train_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )

    # 2) Compute physics-based HI labels (same as training):
    #    Prefer HI_target_hybrid / HI_phys_v2 if present, otherwise fall back
    #    to legacy residual/EGT/HPC-based HI from components.
    groups = group_feature_columns(feature_cols)
    residual_cols = groups.get("residual", [])
    cond_cols = groups.get("cond", [])
    residual_idxs = [feature_cols.index(c) for c in residual_cols]
    cond_feature_idxs = [feature_cols.index(c) for c in cond_cols]

    unit_col = "UnitNumber"
    cycle_col = "TimeInCycles"

    if "HI_target_hybrid" in df_train_fe.columns:
        hi_col = "HI_target_hybrid"
    elif "HI_phys_v2" in df_train_fe.columns:
        hi_col = "HI_phys_v2"
    else:
        hi_col = None

    if hi_col is not None:
        hi_values: List[float] = []
        units_df = df_train_fe[unit_col].unique()
        for uid in units_df:
            df_u = (
                df_train_fe[df_train_fe[unit_col] == uid]
                .sort_values(cycle_col)
                .reset_index(drop=True)
            )
            if len(df_u) < past_len:
                continue
            hi_seq = df_u[hi_col].to_numpy(dtype=np.float32)
            for i in range(past_len - 1, len(df_u)):
                hi_values.append(float(hi_seq[i]))

        if len(hi_values) != X_full.shape[0]:
            raise RuntimeError(
                "[StateEncoderPhysicsDiag] Mismatch between number of HI labels "
                f"from '{hi_col}' and number of sequences: {len(hi_values)} vs {X_full.shape[0]}"
            )
        hi_phys = torch.from_numpy(np.array(hi_values, dtype=np.float32))

        # For later per-engine trajectories we still want residual / physics
        # indices, so we keep the proxies below for plotting if needed.
        hpc_col = "Effizienz_HPC_Proxy"
        egt_col = "EGT_Drift"
        if hpc_col not in feature_cols or egt_col not in feature_cols:
            raise KeyError(
                f"Required physics columns '{hpc_col}' and '{egt_col}' not found in feature_cols."
            )
        hpc_idx = feature_cols.index(hpc_col)
        egt_idx = feature_cols.index(egt_col)
    else:
        # Legacy fallback: build HI from residual norm + EGT_Drift + Effizienz_HPC_Proxy
        hpc_col = "Effizienz_HPC_Proxy"
        egt_col = "EGT_Drift"
        if hpc_col not in feature_cols or egt_col not in feature_cols:
            raise KeyError(
                f"Required physics columns '{hpc_col}' and '{egt_col}' not found in feature_cols."
            )
        hpc_idx = feature_cols.index(hpc_col)
        egt_idx = feature_cols.index(egt_col)

        X_last = X_full[:, -1, :]  # [N,F]
        if residual_idxs:
            resid_last = X_last[:, residual_idxs]  # [N,R]
            resid_norm = torch.sqrt(torch.mean(resid_last ** 2, dim=-1)).cpu().numpy()
        else:
            resid_norm = np.zeros(X_last.shape[0], dtype=np.float32)

        egt_drift = X_last[:, egt_idx].cpu().numpy()
        hpc_eff = X_last[:, hpc_idx].cpu().numpy()

        hi_phys_np = build_physics_hi_from_components(
            resid_norm=resid_norm,
            egt_drift=egt_drift,
            hpc_eff_proxy=hpc_eff,
            w1=0.5,
            w2=0.3,
            w3=0.2,
        )
        hi_phys = torch.from_numpy(hi_phys_np)

    # Normalised RUL
    y_rul_norm = y_rul_full / max_rul

    # 3) Build cond_seq for model
    if cond_feature_idxs:
        cond_seq = X_full[:, :, cond_feature_idxs]  # [N,T,C]
        cond_in_dim = len(cond_feature_idxs)
    else:
        cond_seq = None
        cond_in_dim = 0

    # 4) Load model (match training result_dir / experiment_name)
    checkpoint_dir = Path("results") / "fd004" / run_name
    model = load_state_encoder_v3_physics(
        input_dim=len(feature_cols),
        cond_in_dim=cond_in_dim,
        checkpoint_dir=checkpoint_dir,
        experiment_name=run_name,
        model_cfg=model_cfg,
    )

    # 5) Run model on all train sequences (last step for each sequence)
    X_full_dev = X_full.to(device)
    cond_seq_dev = cond_seq.to(device) if cond_seq is not None else None

    with torch.no_grad():
        outputs = model(X_full_dev, cond_seq=cond_seq_dev, return_dict=True)

    hi_raw = outputs["hi_raw"]
    rul_raw = outputs["rul_raw"]
    hi_seq_phys_pred = outputs.get("hi_seq_phys", None)

    hi_pred_scalar = torch.sigmoid(hi_raw.view(-1)).cpu().numpy()
    rul_pred = torch.sigmoid(rul_raw.view(-1)).cpu().numpy()

    hi_true = hi_phys.numpy()
    rul_true = y_rul_norm.numpy()

    # 6) Create diagnostics directory
    out_dir = checkpoint_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: HI_true_physics vs HI_pred (scalar head)
    plt.figure(figsize=(5, 5))
    plt.scatter(hi_true, hi_pred_scalar, s=8, alpha=0.4)
    plt.xlabel("HI true (physics-informed)")
    plt.ylabel("HI pred (state encoder)")
    plt.title("State Encoder V3 Physics – HI_true vs HI_pred")
    plt.grid(True)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "hi_true_vs_hi_pred.png", dpi=150)
    plt.close()

    # Scatter: RUL_norm_true vs RUL_norm_pred
    plt.figure(figsize=(5, 5))
    plt.scatter(rul_true, rul_pred, s=8, alpha=0.4)
    plt.xlabel("RUL_norm true")
    plt.ylabel("RUL_norm pred")
    plt.title("State Encoder V3 Physics – RUL_norm_true vs RUL_norm_pred")
    plt.grid(True)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "rul_norm_true_vs_pred.png", dpi=150)
    plt.close()

    # 7) Compute EOL metrics on FD004 TEST set (RUL) using last window per engine
    print("[StateEncoderPhysicsDiag] Computing EOL RUL metrics on TEST set...")
    X_test, unit_ids_test, _ = build_test_sequences_from_df(
        df_test_fe,
        feature_cols=feature_cols,
        past_len=30,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    # Build cond_seq for test
    if cond_feature_idxs:
        cond_seq_test = X_test[:, :, cond_feature_idxs]  # [N,H,C] but here H=past_len
    else:
        cond_seq_test = None

    X_test_dev = X_test.to(device)
    cond_test_dev = cond_seq_test.to(device) if cond_seq_test is not None else None

    with torch.no_grad():
        hi_raw_test, rul_raw_test, _ = model(X_test_dev, cond_seq=cond_test_dev)

    rul_pred_norm_test = torch.sigmoid(rul_raw_test.view(-1)).cpu().numpy()
    rul_pred_eol = rul_pred_norm_test * float(max_rul)
    # True EOL RUL from loader (capped)
    y_true_eol = np.minimum(y_test_true.astype(np.float32), float(max_rul))

    eol_metrics = compute_eol_errors_and_nasa(y_true_eol, rul_pred_eol, max_rul=float(max_rul))

    print("[StateEncoderPhysicsDiag] EOL RUL metrics on TEST set:")
    print(f"  RMSE:      {eol_metrics['rmse']:.2f} cycles")
    print(f"  MAE:       {eol_metrics['mae']:.2f} cycles")
    print(f"  Bias:      {eol_metrics['bias']:.2f} cycles")
    print(f"  R²:        {eol_metrics['r2']:.4f}")
    print(f"  NASA mean: {eol_metrics['nasa_mean']:.2f}")
    print(f"  NASA sum:  {eol_metrics['nasa_sum']:.2f}")

    # Optionally save metrics to JSON (convert numpy types to Python scalars/lists)
    try:
        import json

        def _to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        serializable_metrics = {k: _to_serializable(v) for k, v in eol_metrics.items()}
        with open(out_dir / "eol_metrics_test.json", "w") as f:
            json.dump(serializable_metrics, f, indent=2)
    except Exception as e:
        print(f"[StateEncoderPhysicsDiag] Warning: could not save eol_metrics_test.json: {e}")

    # 8) Example HI trajectories across engine life
    # For a few degraded engines, plot HI_true / HI_pred over time (using one
    # window per cycle where possible). For runs with the cumulative damage
    # head enabled, we use the HI_phys(t) trajectory; otherwise we fall back
    # to the scalar HI head and legacy physics HI construction.
    print("[StateEncoderPhysicsDiag] Building example HI trajectories...")
    df = df_train_fe.sort_values(["UnitNumber", "TimeInCycles"]).reset_index(drop=True)
    units = sorted(df["UnitNumber"].unique().tolist())

    # Choose a few units with strongest degradation (small min RUL)
    rul_stats = (
        df.groupby("UnitNumber")["RUL"]
        .min()
        .reset_index()
        .sort_values("RUL")
    )
    degraded_units = rul_stats["UnitNumber"].tolist()[:5]

    for uid in degraded_units:
        df_u = df[df["UnitNumber"] == uid].reset_index(drop=True)
        T_u = len(df_u)
        if T_u < past_len:
            continue

        cycles = []
        hi_true_traj = []
        hi_pred_traj = []

        # Slide one window per cycle (where we have enough history)
        for t in range(past_len - 1, T_u):
            window = df_u.iloc[t - past_len + 1 : t + 1]
            x_win = window[feature_cols].to_numpy(dtype=np.float32)  # [past_len,F]
            x_win_t = torch.from_numpy(x_win).unsqueeze(0).to(device)

            # True HI from DataFrame if available (HI_target_hybrid / HI_phys_v2),
            # otherwise compute legacy physics-based HI from components.
            if hi_col is not None:
                hi_phys_t = float(window[hi_col].iloc[-1])
            else:
                x_last = x_win[-1, :]
                if residual_idxs:
                    resid_last_np = x_last[residual_idxs]
                    resid_norm_t = float(np.sqrt(np.mean(resid_last_np ** 2)))
                else:
                    resid_norm_t = 0.0
                egt_t = float(x_last[egt_idx])
                hpc_t = float(x_last[hpc_idx])
                hi_phys_t = build_physics_hi_from_components(
                    np.array([resid_norm_t], dtype=np.float32),
                    np.array([egt_t], dtype=np.float32),
                    np.array([hpc_t], dtype=np.float32),
                    w1=0.5,
                    w2=0.3,
                    w3=0.2,
                )[0]

            # cond_seq for this window
            if cond_feature_idxs:
                cond_win = x_win[:, cond_feature_idxs]  # [past_len,C]
                cond_win_t = torch.from_numpy(cond_win).unsqueeze(0).to(device)
            else:
                cond_win_t = None

            with torch.no_grad():
                outputs_t = model(x_win_t, cond_seq=cond_win_t, return_dict=True)
            hi_seq_phys_t = outputs_t.get("hi_seq_phys", None)
            hi_raw_t = outputs_t["hi_raw"]

            if hi_seq_phys_t is not None and bool(model_cfg.get("use_damage_head", False)):
                # Use last timestep of the physical HI trajectory from damage head
                hi_pred_t = float(hi_seq_phys_t[0, -1].detach().cpu().item())
            else:
                # Fallback: scalar HI head
                hi_pred_t = float(torch.sigmoid(hi_raw_t.view(-1))[0].cpu().item())

            cycles.append(float(window["TimeInCycles"].iloc[-1]))
            hi_true_traj.append(hi_phys_t)
            hi_pred_traj.append(hi_pred_t)

        if not cycles:
            continue

        plt.figure(figsize=(8, 4))
        plt.plot(cycles, hi_true_traj, label="HI true (physics)", color="tab:blue")
        plt.plot(cycles, hi_pred_traj, "--", label="HI pred", color="tab:red")
        plt.xlabel("Cycle")
        plt.ylabel("HI")
        plt.ylim([-0.05, 1.05])
        plt.title(f"State Encoder V3 Physics – HI trajectory – Unit {uid}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"unit_{uid}_hi_trajectory.png", dpi=150)
        plt.close()

    print(f"[StateEncoderPhysicsDiag] Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostics for FD004 Transformer State Encoder V3 Physics"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="fd004_transformer_state_encoder_v3_physics",
        help=(
            "Experiment name / run directory under results/fd004/, e.g. "
            "'fd004_state_encoder_v3_physics_A1_ms_only', "
            "'fd004_state_encoder_v3_physics_B1_msdt', etc."
        ),
    )
    args = parser.parse_args()

    run_diagnostics(run_name=args.run_name)


