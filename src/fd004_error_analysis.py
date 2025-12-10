"""
FD004 Error Analysis Module.

This module provides focused diagnostic analysis for FD004 (and FD002) experiments,
including per-engine EOL error decomposition, condition-wise summaries, and
worst-engine trajectory plotting.

Designed to work with Phase 3 UniversalEncoderV1 experiments and earlier phases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
)
from src.feature_safety import remove_rul_leakage
from src.eol_full_lstm import (
    build_test_sequences_from_df,
    build_full_eol_sequences_from_df,
    nasa_phm_score,
    EOLFullLSTMWithHealth,
)
from src.eval_utils import forward_rul_only
from src.models.universal_encoder_v1 import (
    RULHIUniversalModel,
    UniversalEncoderV1,
    UniversalEncoderV2,
    RULHIUniversalModelV2,
)


def compute_per_engine_eol_metrics(
    true_rul_eol: np.ndarray,
    pred_rul_eol: np.ndarray,
    unit_ids: np.ndarray,
    cond_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute per-engine EOL metrics including NASA scores.
    
    Args:
        true_rul_eol: True RUL at EOL for each engine [N]
        pred_rul_eol: Predicted RUL at EOL for each engine [N]
        unit_ids: Unit IDs for each engine [N]
        cond_ids: Optional condition IDs for each engine [N]
    
    Returns:
        DataFrame with columns: unit_id, condition_id, true_rul_eol, pred_rul_eol,
        error, abs_error, nasa_score, is_underestimation, is_overestimation
    """
    errors = pred_rul_eol - true_rul_eol
    abs_errors = np.abs(errors)
    
    # Compute NASA PHM08 score per engine
    nasa_scores = np.array([
        nasa_phm_score(np.array([true_rul_eol[i]]), np.array([pred_rul_eol[i]]))
        for i in range(len(true_rul_eol))
    ])
    
    df = pd.DataFrame({
        "unit_id": unit_ids,
        "condition_id": cond_ids if cond_ids is not None else np.full(len(unit_ids), np.nan),
        "true_rul_eol": true_rul_eol,
        "pred_rul_eol": pred_rul_eol,
        "error": errors,
        "abs_error": abs_errors,
        "nasa_score": nasa_scores,
        "is_underestimation": errors < 0,  # pred < true (unsafe, late prediction)
        "is_overestimation": errors > 0,    # pred > true (conservative, early prediction)
    })
    
    return df


def summarize_by_condition(
    df_per_engine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group per-engine metrics by condition_id and compute aggregates.
    
    Args:
        df_per_engine: DataFrame from compute_per_engine_eol_metrics()
    
    Returns:
        DataFrame with one row per condition containing:
        condition_id, num_engines, rmse_eol, mae_eol, bias_eol, nasa_mean, nasa_sum
    """
    if "condition_id" not in df_per_engine.columns or df_per_engine["condition_id"].isna().all():
        # No condition IDs available
        return pd.DataFrame({
            "condition_id": [None],
            "num_engines": [len(df_per_engine)],
            "rmse_eol": [np.sqrt(np.mean(df_per_engine["error"] ** 2))],
            "mae_eol": [df_per_engine["abs_error"].mean()],
            "bias_eol": [df_per_engine["error"].mean()],
            "nasa_mean": [df_per_engine["nasa_score"].mean()],
            "nasa_sum": [df_per_engine["nasa_score"].sum()],
        })
    
    # Group by condition_id
    grouped = df_per_engine.groupby("condition_id", dropna=False)
    
    summary = pd.DataFrame({
        "condition_id": grouped["condition_id"].first(),
        "num_engines": grouped.size(),
        "rmse_eol": grouped.apply(lambda g: np.sqrt(np.mean(g["error"] ** 2))),
        "mae_eol": grouped["abs_error"].mean(),
        "bias_eol": grouped["error"].mean(),
        "nasa_mean": grouped["nasa_score"].mean(),
        "nasa_sum": grouped["nasa_score"].sum(),
    }).reset_index(drop=True)
    
    return summary


def select_worst_engines(
    df_per_engine: pd.DataFrame,
    top_k_by_nasa: int = 5,
    top_k_by_underestimation: int = 5,
) -> Dict[str, List[int]]:
    """
    Select worst engines by NASA score and underestimation.
    
    Args:
        df_per_engine: DataFrame from compute_per_engine_eol_metrics()
        top_k_by_nasa: Number of top engines by NASA score
        top_k_by_underestimation: Number of top engines by underestimation (most negative error)
    
    Returns:
        Dictionary with keys:
        - 'top_by_nasa': list of unit_ids with highest NASA scores
        - 'top_by_underestimation': list of unit_ids with most negative errors
    """
    # Top by NASA score (worst = highest)
    top_nasa = df_per_engine.nlargest(top_k_by_nasa, "nasa_score")["unit_id"].tolist()
    
    # Top by underestimation (worst = most negative error)
    top_underest = df_per_engine.nsmallest(top_k_by_underestimation, "error")["unit_id"].tolist()
    
    return {
        "top_by_nasa": top_nasa,
        "top_by_underestimation": top_underest,
    }


def load_model_from_experiment(
    experiment_dir: Path,
    device: torch.device | str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    Load model and config from experiment directory.
    
    Reuses logic from src/analysis/inference.py but kept here for self-contained module.
    
    Args:
        experiment_dir: Path to experiment directory
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Configuration dictionary
    """
    # Load summary.json for config
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {experiment_dir}")
    
    with open(summary_path, "r") as f:
        config = json.load(f)
    
    # Find model checkpoint
    model_files = list(experiment_dir.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt model file found in {experiment_dir}")
    
    # Prefer files with "best" in name
    best_models = [f for f in model_files if "best" in f.name.lower()]
    if best_models:
        model_path = best_models[0]
    else:
        model_path = model_files[0]
    
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Determine encoder type and create model
    encoder_type = config.get("encoder_type", "lstm")
    dataset_name = config.get("dataset", "FD004")
    
    # Get input dimension from checkpoint
    if encoder_type == "universal_v2":
        # For UniversalEncoderV2, infer from checkpoint
        use_cond_fusion = config.get("use_condition_embedding", False)
        cond_emb_dim = config.get("cond_emb_dim", 4) if use_cond_fusion else 0
        
        # Try to infer input_dim from CNN first layer
        if "encoder.cnn_branches.0.0.weight" in state_dict:
            input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
            print(f"Inferred input_dim={input_dim} from UniversalEncoderV2 CNN checkpoint")
        else:
            input_dim = config.get("input_dim", 245)
            print(f"Warning: Could not infer input_dim from checkpoint, using {input_dim} from config")
        
        d_model = config.get("d_model", 64)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", None)
        dropout = config.get("dropout", 0.1)
        kernel_sizes = config.get("kernel_sizes", [3, 5, 9])
        seq_encoder_type = config.get("seq_encoder_type", "transformer")
        use_layer_norm = config.get("use_layer_norm", True)
        num_conditions = config.get("num_conditions", None) if use_cond_fusion else None
        
        encoder = UniversalEncoderV2(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim if use_cond_fusion else 0,
            max_seq_len=300,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
        )
        
        model = RULHIUniversalModelV2(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    elif encoder_type == "universal_v1":
        # For UniversalEncoderV1, infer from checkpoint
        use_cond_fusion = config.get("use_condition_embedding", False)
        cond_emb_dim = config.get("cond_emb_dim", 4) if use_cond_fusion else 0
        
        # Try to infer input_dim from CNN first layer
        # CNN branches: Conv1d(input_dim, cnn_channels, kernel_size)
        if "encoder.cnn_branches.0.0.weight" in state_dict:
            # First CNN branch first layer: [cnn_channels, input_dim, kernel_size]
            input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
            print(f"Inferred input_dim={input_dim} from UniversalEncoderV1 CNN checkpoint")
        else:
            # Fallback to config
            input_dim = config.get("input_dim", 245)  # Typical feature count
            print(f"Warning: Could not infer input_dim from checkpoint, using {input_dim} from config")
        
        d_model = config.get("d_model", 48)
        cnn_channels = config.get("cnn_channels", None)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", 256)
        dropout = config.get("dropout", 0.1)
        num_conditions = config.get("num_conditions", None) if use_cond_fusion else None
        
        encoder = UniversalEncoderV1(
            input_dim=input_dim,
            d_model=d_model,
            cnn_channels=cnn_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim if use_cond_fusion else 0,
            max_seq_len=300,
        )
        
        model = RULHIUniversalModel(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    else:
        # LSTM or Transformer (EOLFullLSTMWithHealth)
        # Infer parameters from checkpoint
        use_condition_embedding = config.get("use_condition_embedding", False)
        num_conditions = config.get("num_conditions", 1) if use_condition_embedding else 1
        cond_emb_dim = config.get("cond_emb_dim", 4) if use_condition_embedding else 0
        
        # Infer input_dim
        if encoder_type == "transformer" and "encoder.input_proj.weight" in state_dict:
            encoder_input_dim = state_dict["encoder.input_proj.weight"].shape[1]
            input_dim = encoder_input_dim - cond_emb_dim if use_condition_embedding else encoder_input_dim
        else:
            input_dim = config.get("input_dim", 245)
        
        if encoder_type == "lstm":
            hidden_dim = config.get("hidden_dim", 50)
        else:
            hidden_dim = config.get("d_model", 48)
        
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)
        transformer_nhead = config.get("nhead", 4) if encoder_type == "transformer" else 4
        transformer_dim_feedforward = config.get("dim_feedforward", 256) if encoder_type == "transformer" else 256
        
        model = EOLFullLSTMWithHealth(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            lambda_health=config.get("health_loss_weight", 0.35),
            use_condition_embedding=use_condition_embedding,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim,
            encoder_type=encoder_type,
            transformer_nhead=transformer_nhead,
            transformer_dim_feedforward=transformer_dim_feedforward,
        )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)  # Use strict=False for flexibility
    model.to(device)
    model.eval()
    
    return model, config


def plot_engine_trajectories(
    df_full_test: pd.DataFrame,
    model: nn.Module,
    scaler,  # StandardScaler or Dict[int, StandardScaler]
    unit_ids: List[int],
    feature_cols: List[str],
    past_len: int,
    save_path: Path,
    device: torch.device | str = "cpu",
    max_rul: int = 125,
) -> None:
    """
    Plot RUL trajectories for selected engines.
    
    For each engine, builds rolling sequences and runs inference at each timestep,
    then plots true vs predicted RUL over time.
    
    Args:
        df_full_test: Full test DataFrame with all cycles
        model: Trained model
        scaler: Fitted scaler (or dict of scalers per condition)
        unit_ids: List of unit IDs to plot
        feature_cols: Feature column names
        past_len: Sequence length
        save_path: Directory to save plots
        device: Device to run inference on
        max_rul: Maximum RUL for capping
    """
    model.eval()
    use_cond_emb = getattr(model, "use_condition_embedding", False)
    
    for unit_id in unit_ids:
        df_engine = df_full_test[df_full_test["UnitNumber"] == unit_id].sort_values("TimeInCycles")
        
        if len(df_engine) < past_len:
            print(f"Warning: Engine {unit_id} has only {len(df_engine)} cycles, skipping")
            continue
        
        cycles = df_engine["TimeInCycles"].values
        true_rul = df_engine["RUL"].values if "RUL" in df_engine.columns else None
        
        # Build sequences for each timestep (rolling window)
        pred_rul_list = []
        
        with torch.no_grad():
            for i in range(past_len - 1, len(df_engine)):
                # Get window [i - past_len + 1 : i + 1]
                window_df = df_engine.iloc[i - past_len + 1 : i + 1]
                window_features = window_df[feature_cols].values.astype(np.float32)
                
                # Apply scaling
                if isinstance(scaler, dict):
                    cond_id = int(window_df["ConditionID"].iloc[0]) if "ConditionID" in window_df.columns else 0
                    if cond_id in scaler:
                        window_features = scaler[cond_id].transform(window_features)
                    else:
                        window_features = list(scaler.values())[0].transform(window_features)
                else:
                    window_features = scaler.transform(window_features)
                
                # Convert to tensor
                X_window = torch.from_numpy(window_features).unsqueeze(0).float().to(device)  # [1, past_len, F]
                
                # Get condition ID if needed
                if use_cond_emb:
                    cond_id_val = int(window_df["ConditionID"].iloc[0]) if "ConditionID" in window_df.columns else 0
                    cond_ids_tensor = torch.tensor([cond_id_val], dtype=torch.int64, device=device)
                else:
                    cond_ids_tensor = None
                
                # Predict
                if use_cond_emb and cond_ids_tensor is not None:
                    rul_pred = forward_rul_only(model, X_window, cond_ids=cond_ids_tensor)
                else:
                    rul_pred = forward_rul_only(model, X_window)
                
                pred_rul_list.append(float(rul_pred.cpu().item()))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot true RUL (if available)
        if true_rul is not None:
            plot_cycles = cycles[past_len - 1:]
            plot_true_rul = true_rul[past_len - 1:]
            ax.plot(plot_cycles, plot_true_rul, 'g-', linewidth=2, label='True RUL', alpha=0.7)
        
        # Plot predicted RUL
        plot_cycles = cycles[past_len - 1:]
        ax.plot(plot_cycles, pred_rul_list, 'r--', linewidth=2, label='Predicted RUL', alpha=0.7)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('RUL [cycles]')
        ax.set_title(f'Engine #{unit_id} - RUL Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max_rul + 10])
        
        plt.tight_layout()
        plot_file = save_path / f"engine_{unit_id}_trajectory.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved trajectory plot for engine {unit_id} to {plot_file}")


def analyze_experiment_fd004(
    experiment_name: str = "fd004_phase3_universal_v1_tuned",
    dataset_name: str = "FD004",
    results_dir: str | Path = Path("results"),
    save_plots: bool = True,
    max_engines_to_plot: int = 5,
    device: str | torch.device = "auto",
) -> Dict:
    """
    High-level error analysis entry point for FD002/FD004 experiments.
    
    Loads model, runs inference on test set, computes per-engine and condition-wise
    metrics, and saves CSV files and optional plots.
    
    Works for both FD002 and FD004 (and can be extended to FD001/FD003).
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Dataset name (FD002 or FD004)
        results_dir: Root results directory
        save_plots: Whether to save trajectory plots for worst engines
        max_engines_to_plot: Maximum number of worst engines to plot
        device: Device to use ("auto", "cpu", "cuda")
    
    Returns:
        Dictionary with key results:
        - n_engines: Total number of test engines
        - global_rmse_eol: Global RMSE at EOL
        - global_mae_eol: Global MAE at EOL
        - global_bias_eol: Global bias at EOL
        - global_nasa_mean: Global NASA mean score
        - conditions_summary_path: Path to condition-wise CSV
        - per_engine_csv_path: Path to per-engine CSV
        - worst_engines_by_nasa: List of unit IDs with worst NASA scores
        - worst_engines_by_underestimation: List of unit IDs with worst underestimation
    """
    results_dir = Path(results_dir)
    experiment_dir = results_dir / dataset_name.lower() / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Analyzing experiment: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    
    # Load model and config
    model, config = load_model_from_experiment(experiment_dir, device)
    
    # Load data
    print("\nLoading data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )
    
    # Feature engineering (same as training)
    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=False,
        use_temporal_on_physics=False,
    )
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=True,
        temporal=TemporalFeatureConfig(
            base_cols=None,
            short_windows=(5, 10),
            long_windows=(30,),
            add_rolling_mean=True,
            add_rolling_std=False,
            add_trend=True,
            add_delta=True,
            delta_lags=(5, 10),
        ),
    )
    
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_train = create_all_features(df_train, "UnitNumber", "TimeInCycles", feature_config, inplace=False)
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_all_features(df_test, "UnitNumber", "TimeInCycles", feature_config, inplace=False)
    
    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    
    # Get sequence parameters
    past_len = config.get("sequence_length", 30)
    max_rul = config.get("max_rul", 125)
    
    # Build test sequences (EOL only - last cycle per engine)
    print("\nBuilding test sequences...")
    X_test, unit_ids_test, cond_ids_test = build_test_sequences_from_df(
        df_test,
        feature_cols=feature_cols,
        past_len=past_len,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    
    # Apply scaling - reconstruct scaler from training data
    # (Scaler is not saved with model, so we need to rebuild it)
    print("\nReconstructing scaler from training data...")
    from sklearn.preprocessing import StandardScaler
    
    # Build training sequences to fit scaler
    result = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    X_train_full, _, unit_ids_train_full, cond_ids_train_full = result[:4]
    health_phys_seq_train_full = result[4] if len(result) > 4 else None
    
    # Use condition-wise scaling (standard for FD002/FD004)
    use_condition_wise = dataset_name in ["FD002", "FD004"]
    
    if use_condition_wise and cond_ids_train_full is not None:
        # Condition-wise scaling
        unique_conds = torch.unique(cond_ids_train_full)
        scaler = {}
        
        for cond_id in unique_conds:
            cond_mask = (cond_ids_train_full == cond_id)
            if not cond_mask.any():
                continue
            
            X_cond = X_train_full[cond_mask]  # [N_cond, past_len, num_features]
            X_cond_flat = X_cond.numpy().reshape(-1, len(feature_cols))
            
            scaler_cond = StandardScaler()
            scaler_cond.fit(X_cond_flat)
            scaler[int(cond_id)] = scaler_cond
        
        # Apply to test data
        X_test_np = X_test.numpy()
        B, T, F = X_test_np.shape
        X_test_scaled_list = []
        
        for i in range(B):
            cond_id = int(cond_ids_test[i]) if cond_ids_test is not None else 0
            if cond_id in scaler:
                x_sample = X_test_np[i]  # [past_len, num_features]
                x_scaled = scaler[cond_id].transform(x_sample)
                X_test_scaled_list.append(torch.from_numpy(x_scaled))
            else:
                # Fallback: use first available scaler
                if scaler:
                    x_sample = X_test_np[i]
                    first_scaler = list(scaler.values())[0]
                    x_scaled = first_scaler.transform(x_sample)
                    X_test_scaled_list.append(torch.from_numpy(x_scaled))
                else:
                    X_test_scaled_list.append(torch.from_numpy(X_test_np[i]))
        
        X_test = torch.stack(X_test_scaled_list).float()
        print(f"Applied condition-wise scaling ({len(scaler)} conditions)")
    else:
        # Global scaling
        X_train_flat = X_train_full.numpy().reshape(-1, len(feature_cols))
        scaler = StandardScaler()
        scaler.fit(X_train_flat)
        
        # Apply to test
        X_test_np = X_test.numpy()
        B, T, F = X_test_np.shape
        X_test_flat = X_test_np.reshape(-1, F)
        X_test_scaled = scaler.transform(X_test_flat)
        X_test = torch.from_numpy(X_test_scaled.reshape(B, T, F)).float()
        print("Applied global scaling")
    
    # Run inference
    print("\nRunning inference...")
    use_cond_emb = getattr(model, "use_condition_embedding", False)
    cond_ids_tensor = cond_ids_test.to(device) if use_cond_emb else None
    
    with torch.no_grad():
        X_test = X_test.to(device)
        pred_rul = forward_rul_only(model, X_test, cond_ids=cond_ids_tensor).cpu().numpy().flatten()
    
    # Cap predictions
    pred_rul = np.clip(pred_rul, 0, max_rul)
    true_rul = np.clip(y_test_true, 0, max_rul)
    
    # Compute per-engine metrics
    print("\nComputing per-engine metrics...")
    df_per_engine = compute_per_engine_eol_metrics(
        true_rul_eol=true_rul,
        pred_rul_eol=pred_rul,
        unit_ids=unit_ids_test.numpy(),
        cond_ids=cond_ids_test.numpy() if cond_ids_test is not None else None,
    )
    
    # Condition-wise summary
    df_condition_summary = summarize_by_condition(df_per_engine)
    
    # Select worst engines
    worst_engines = select_worst_engines(df_per_engine, top_k_by_nasa=max_engines_to_plot)
    
    # Save CSV files
    print("\nSaving results...")
    per_engine_csv = experiment_dir / f"per_engine_{dataset_name.lower()}_eol_metrics.csv"
    df_per_engine.to_csv(per_engine_csv, index=False)
    print(f"Saved per-engine metrics to {per_engine_csv}")
    
    condition_csv = experiment_dir / f"condition_wise_{dataset_name.lower()}_summary.csv"
    df_condition_summary.to_csv(condition_csv, index=False)
    print(f"Saved condition-wise summary to {condition_csv}")
    
    # Plot worst engines if requested
    if save_plots:
        print(f"\nPlotting trajectories for worst engines...")
        engines_to_plot = worst_engines['top_by_nasa'][:max_engines_to_plot]
        
        if scaler is not None:
            plot_engine_trajectories(
                df_full_test=df_test,
                model=model,
                scaler=scaler,
                unit_ids=engines_to_plot,
                feature_cols=feature_cols,
                past_len=past_len,
                save_path=experiment_dir,
                device=device,
                max_rul=max_rul,
            )
        else:
            print("Warning: Scaler not available, skipping trajectory plots")
    
    # Compute global metrics
    global_rmse = np.sqrt(np.mean(df_per_engine["error"] ** 2))
    global_mae = df_per_engine["abs_error"].mean()
    global_bias = df_per_engine["error"].mean()
    global_nasa_mean = df_per_engine["nasa_score"].mean()
    
    results = {
        "n_engines": len(df_per_engine),
        "global_rmse_eol": float(global_rmse),
        "global_mae_eol": float(global_mae),
        "global_bias_eol": float(global_bias),
        "global_nasa_mean": float(global_nasa_mean),
        "conditions_summary_path": str(condition_csv),
        "per_engine_csv_path": str(per_engine_csv),
        "worst_engines_by_nasa": worst_engines["top_by_nasa"],
        "worst_engines_by_underestimation": worst_engines["top_by_underestimation"],
    }
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    results = analyze_experiment_fd004(
        experiment_name="fd004_phase3_universal_v1_tuned",
        dataset_name="FD004",
        results_dir=Path("results"),
        save_plots=True,
        max_engines_to_plot=5,
    )
    
    print("\n" + "=" * 80)
    print("=== FD004 Analysis Summary ===")
    print("=" * 80)
    for k, v in results.items():
        if isinstance(v, list):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")
    print("=" * 80)

