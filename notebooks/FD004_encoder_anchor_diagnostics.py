"""
FD004 Encoder Anchor Diagnostics
--------------------------------

Small standalone script to check how well the FD004 ms+DT Transformer encoder
(`fd004_transformer_encoder_ms_dt_v1`) produces "anchors" for:

    - current Health Index  HI_current
    - current Remaining Useful Life RUL_current

The script:
1) Rebuilds the same feature pipeline as in `fd004_transformer_encoder_ms_dt_v1`
   (physics + condition vector + digital twin + multiscale).
2) Loads the trained `EOLFullTransformerEncoder` checkpoint.
3) Builds one EOL-style window per engine from the FD004 TEST set.
4) Runs the encoder on these windows to obtain RUL and HI anchors.
5) Compares against ground-truth RUL/HI in simple scatter plots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root (containing 'src') is on sys.path, regardless of CWD
import sys
from pathlib import Path

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
from src.config import ResidualFeatureConfig, USE_CONDITION_WISE_SCALING
from src.feature_safety import remove_rul_leakage
from src.eol_full_lstm import build_test_sequences_from_df
from src.experiment_configs import get_experiment_by_name
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.analysis.inference import rebuild_scaler_from_training_data


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


def prepare_fd004_msdt_data(experiment_name: str):
    """
    Load FD004 train/test data and apply the same ms+DT feature pipeline as
    in `fd004_transformer_encoder_ms_dt_v1`.
    """
    cfg = get_experiment_by_name(experiment_name)
    dataset = cfg["dataset"]
    assert dataset == "FD004", f"Expected dataset FD004, got {dataset}"

    max_rul = int(cfg.get("max_rul", 125))
    df_train, df_test, y_test_true = load_cmapps_subset(
        fd_id=dataset,
        max_rul=max_rul,
        clip_train=True,
        clip_test=True,
    )

    feature_config, physics_config, phys_opts = _build_feature_pipeline(cfg)

    # 1) Physics features
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
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Multi-scale temporal features
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

    feature_cols = [
        c
        for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    print(f"[EncoderDiagnostics] Using {len(feature_cols)} features for ms+DT encoder input.")

    return cfg, df_train, df_test, y_test_true, feature_cols, max_rul


def load_msdt_encoder(cfg, input_dim: int, num_conditions: int) -> EOLFullTransformerEncoder:
    """
    Instantiate and load the EOLFullTransformerEncoder for ms_dt_v1.
    """
    exp_name = cfg["experiment_name"]
    exp_dir = Path("results") / "fd004" / exp_name

    enc_kwargs = cfg["encoder_kwargs"]
    d_model = enc_kwargs["d_model"]
    num_layers = enc_kwargs["num_layers"]
    n_heads = enc_kwargs["n_heads"]
    dim_feedforward = enc_kwargs.get("dim_feedforward", None)
    dropout = enc_kwargs.get("dropout", 0.1)

    phase2 = cfg["phase_2_params"]
    use_condition_embedding = phase2.get("use_condition_embedding", True)
    cond_emb_dim = phase2.get("cond_emb_dim", 4 if use_condition_embedding else 0)

    model = EOLFullTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_condition_embedding=use_condition_embedding,
        num_conditions=num_conditions if use_condition_embedding else 1,
        cond_emb_dim=cond_emb_dim,
        max_seq_len=300,
    )

    # Find checkpoint (prefer "eol_full_lstm_best_*.pt" if present)
    ckpt_path = exp_dir / "eol_full_lstm_best_fd004_transformer_encoder_ms_dt_v1.pt"
    if not ckpt_path.exists():
        # Fallback: any .pt file in the directory
        pt_files = list(exp_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt checkpoint found in {exp_dir}")
        ckpt_path = pt_files[0]

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[EncoderDiagnostics] Loaded encoder from {ckpt_path}")
    print(f"[EncoderDiagnostics] Missing keys: {missing}")
    print(f"[EncoderDiagnostics] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def evaluate_encoder_anchors_fd004(experiment_name: str = "fd004_transformer_encoder_ms_dt_v1") -> None:
    cfg, df_train, df_test, y_test_true, feature_cols, max_rul = prepare_fd004_msdt_data(experiment_name)

    # Build scaler exactly as in training (condition-wise or global)
    use_cond_scaling = USE_CONDITION_WISE_SCALING
    scaler = rebuild_scaler_from_training_data(
        df_train=df_train,
        feature_cols=feature_cols,
        use_condition_wise_scaling=use_cond_scaling,
    )

    # Build one test window per engine (last past_len cycles)
    past_len = 30
    X_test, unit_ids_test, cond_ids_test = build_test_sequences_from_df(
        df_test,
        feature_cols=feature_cols,
        past_len=past_len,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )

    # Apply scaling
    X_test_np = X_test.numpy()
    B, T, F = X_test_np.shape

    if isinstance(scaler, dict):
        X_scaled_list = []
        for i in range(B):
            cond_id = int(cond_ids_test[i])
            if cond_id in scaler:
                x_sample = X_test_np[i]
                x_scaled = scaler[cond_id].transform(x_sample)
                X_scaled_list.append(torch.from_numpy(x_scaled))
            else:
                # Fallback: use first available scaler
                first_scaler = list(scaler.values())[0]
                x_scaled = first_scaler.transform(X_test_np[i])
                X_scaled_list.append(torch.from_numpy(x_scaled))
        X_test_scaled = torch.stack(X_scaled_list).float()
    else:
        X_test_flat = X_test_np.reshape(-1, F)
        X_test_scaled_flat = scaler.transform(X_test_flat)
        X_test_scaled = torch.from_numpy(X_test_scaled_flat.reshape(B, T, F)).float()

    # True RUL per engine (capped as in evaluation)
    y_true_rul = np.minimum(y_test_true.astype(np.float32), max_rul)
    hi_true = np.clip(y_true_rul / float(max_rul), 0.0, 1.0)

    # Derive input_dim and num_conditions for encoder instantiation
    input_dim = X_test_scaled.shape[-1]
    num_conditions = int(cond_ids_test.max().item()) + 1 if cond_ids_test.numel() > 0 else 1
    encoder = load_msdt_encoder(cfg, input_dim=input_dim, num_conditions=num_conditions)

    # Run encoder on all test windows
    X_test_scaled = X_test_scaled.to(device)
    cond_ids_test_tensor = cond_ids_test.to(device)

    with torch.no_grad():
        rul_pred, hi_pred, _ = encoder(X_test_scaled, cond_ids_test_tensor)

    rul_pred_np = rul_pred.cpu().numpy().astype(np.float32)
    # clamp predictions to valid range
    rul_pred_np = np.minimum(np.maximum(rul_pred_np, 0.0), max_rul)
    hi_pred_np = hi_pred.cpu().numpy().astype(np.float32)  # already in [0,1] for default head

    # Output directory: use the same directory as the encoder model checkpoint
    exp_name = cfg["experiment_name"]
    out_dir = Path("results") / "fd004" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: RUL
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_rul, rul_pred_np, s=8, alpha=0.5)
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("Encoder anchor: RUL_true vs RUL_pred (FD004 ms+DT)")
    plt.grid(True)
    rmin, rmax = y_true_rul.min(), y_true_rul.max()
    plt.plot([rmin, rmax], [rmin, rmax], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_rul_anchor_scatter.png", dpi=150)
    plt.close()

    # Scatter: HI
    plt.figure(figsize=(5, 5))
    plt.scatter(hi_true, hi_pred_np, s=8, alpha=0.5)
    plt.xlabel("True HI (RUL/max_rul)")
    plt.ylabel("Predicted HI (encoder health_last)")
    plt.title("Encoder anchor: HI_true vs HI_pred (FD004 ms+DT)")
    plt.grid(True)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_hi_anchor_scatter.png", dpi=150)
    plt.close()

    print(f"[EncoderDiagnostics] Saved scatter plots to {out_dir}")


if __name__ == "__main__":
    # Optional CLI argument: experiment name (default: fd004_transformer_encoder_ms_dt_v1)
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        exp_name = "fd004_transformer_encoder_ms_dt_v1"
    evaluate_encoder_anchors_fd004(exp_name)


