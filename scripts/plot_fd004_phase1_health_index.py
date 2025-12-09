"""
FD004 Phase-1 Health Index Plotting Script

Generates Health Index plots for FD004 Phase-1 experiments:
1. Health Index vs True RUL (scatter)
2. Predicted vs Target Health Index (scatter)
3. Health Index trajectories for sample engines

Example usage:
    python scripts/plot_fd004_phase1_health_index.py --config config/fd004_phase1_rmse.yaml
    python scripts/plot_fd004_phase1_health_index.py --config config/fd004_phase1_nasa.yaml --num_engines 5
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import random

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
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
from src.config import (
    MAX_RUL,
    SEQUENCE_LENGTH,
    HI_RUL_PLATEAU_THRESH,
)
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
    EOLFullLSTMWithHealth,
)
from src.feature_safety import remove_rul_leakage
from sklearn.preprocessing import StandardScaler


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_target_health_index(rul: np.ndarray, max_rul: int = 125, plateau_thresh: int = 80) -> np.ndarray:
    """
    Compute target Health Index from RUL.
    
    Target HI:
    - HI = 1.0 for RUL > plateau_thresh (early life)
    - HI = linear decay from 1.0 to 0.0 for RUL in [0, plateau_thresh]
    """
    rul_capped = np.minimum(rul, max_rul)
    hi = np.ones_like(rul_capped, dtype=float)
    
    # Linear decay in late-life region
    late_mask = rul_capped <= plateau_thresh
    hi[late_mask] = rul_capped[late_mask] / plateau_thresh
    
    return np.clip(hi, 0.0, 1.0)


def load_model_and_scaler(config: dict, input_dim: int, device: torch.device) -> tuple:
    """Load trained model from checkpoint."""
    exp_name = config['experiment_name']
    results_dir = Path(config['results']['base_dir']) / exp_name
    checkpoint_path = results_dir / f"eol_full_lstm_best_{exp_name}.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config values for architecture (they should match the checkpoint)
    model = EOLFullLSTMWithHealth(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        lambda_health=config['multi_task']['health_loss_weight'],
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"Model loaded: input_dim={input_dim}, hidden_dim={config['model']['hidden_dim']}, num_layers={config['model']['num_layers']}")
    
    return model


def prepare_data_and_features(config: dict):
    """Prepare training and test data with feature engineering."""
    dataset = config['dataset']
    
    print(f"\n[1] Loading {dataset} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=config['data']['clip_train'],
        clip_test=config['data']['clip_test'],
    )
    
    # Feature Engineering (same as training)
    physics_config = PhysicsFeatureConfig(
        use_core=config['features']['physics']['use_core'],
        use_extended=config['features']['physics']['use_extended'],
        use_residuals=config['features']['physics']['use_residuals'],
        use_temporal_on_physics=config['features']['physics']['use_temporal_on_physics'],
    )
    
    temporal_config = TemporalFeatureConfig(
        base_cols=None,
        short_windows=tuple(config['features']['temporal']['short_windows']),
        long_windows=tuple(config['features']['temporal']['long_windows']),
        add_rolling_mean=config['features']['temporal']['add_rolling_mean'],
        add_rolling_std=config['features']['temporal']['add_rolling_std'],
        add_trend=config['features']['temporal']['add_trend'],
        add_delta=config['features']['temporal']['add_delta'],
        delta_lags=tuple(config['features']['temporal']['delta_lags']),
    )
    
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=config['features']['temporal']['add_temporal_features'],
        temporal=temporal_config,
    )
    
    # Apply feature engineering
    df_train = create_physical_features(
        df_train,
        physics_config=physics_config,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    df_train = create_all_features(
        df_train,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        config=feature_config,
        inplace=False,
    )
    
    df_test = create_physical_features(
        df_test,
        physics_config=physics_config,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    df_test = create_all_features(
        df_test,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        config=feature_config,
        inplace=False,
    )
    
    # Feature columns
    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, leaked = remove_rul_leakage(feature_cols)
    
    return df_train, df_test, y_test_true, feature_cols


def run_inference_on_val_data(
    model: EOLFullLSTMWithHealth,
    df_train: pd.DataFrame,
    feature_cols: list,
    scaler: StandardScaler | dict,
    config: dict,
    device: torch.device,
) -> dict:
    """Run inference on validation data to get RUL and Health Index predictions."""
    print("\n[2] Running inference on validation data...")
    
    # Build sequences from training data (we'll use the validation split)
    X_full, y_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=config['data']['past_len'],
        max_rul=config['data']['max_rul'],
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    
    # Split into train/val (same as training)
    from src.eol_full_lstm import create_full_dataloaders
    train_loader, val_loader, _, train_unit_ids, val_unit_ids = create_full_dataloaders(
        X=X_full,
        y=y_full,
        unit_ids=unit_ids_full,
        cond_ids=cond_ids_full,
        batch_size=256,
        engine_train_ratio=config['training']['engine_train_ratio'],
        shuffle_engines=config['training']['shuffle_engines'],
        random_seed=config['training']['random_seed'],
        use_condition_wise_scaling=config['data']['use_condition_wise_scaling'],
    )
    
    # Collect predictions
    model.eval()
    all_rul_pred = []
    all_health_pred = []
    all_health_seq = []
    all_rul_true = []
    all_unit_ids = []
    all_cond_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch, unit_ids_batch, cond_ids_batch = batch
            X_batch = X_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            if isinstance(outputs, tuple):
                rul_pred, health_pred, health_seq = outputs
            else:
                rul_pred = outputs
                health_pred = None
                health_seq = None
            
            all_rul_pred.append(rul_pred.cpu().numpy())
            all_rul_true.append(y_batch.numpy())
            all_unit_ids.append(unit_ids_batch.numpy())
            all_cond_ids.append(cond_ids_batch.numpy())
            
            if health_pred is not None:
                all_health_pred.append(health_pred.cpu().numpy())
            if health_seq is not None:
                all_health_seq.append(health_seq.cpu().numpy())
    
    rul_pred = np.concatenate(all_rul_pred)
    rul_true = np.concatenate(all_rul_true)
    unit_ids = np.concatenate(all_unit_ids)
    cond_ids = np.concatenate(all_cond_ids)
    
    health_pred = np.concatenate(all_health_pred) if all_health_pred else None
    health_seq = np.concatenate(all_health_seq) if all_health_seq else None
    
    return {
        'rul_pred': rul_pred,
        'rul_true': rul_true,
        'health_pred': health_pred,
        'health_seq': health_seq,
        'unit_ids': unit_ids,
        'cond_ids': cond_ids,
    }


def plot_health_index_vs_rul(
    rul_true: np.ndarray,
    health_pred: np.ndarray,
    save_path: Path,
    exp_name: str,
):
    """Plot 1: Health Index vs True RUL (scatter)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(rul_true, health_pred, alpha=0.3, s=5)
    ax.set_xlabel("True RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted Health Index", fontsize=12)
    ax.set_title(f"{exp_name}: Health Index vs True RUL", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 125])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_predicted_vs_target_hi(
    hi_target: np.ndarray,
    health_pred: np.ndarray,
    save_path: Path,
    exp_name: str,
):
    """Plot 2: Predicted vs Target Health Index (scatter)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(hi_target, health_pred, alpha=0.3, s=5)
    max_hi = max(hi_target.max(), health_pred.max())
    ax.plot([0, max_hi], [0, max_hi], 'r--', linewidth=2, label="Ideal")
    ax.set_xlabel("Target Health Index (from RUL)", fontsize=12)
    ax.set_ylabel("Predicted Health Index", fontsize=12)
    ax.set_title(f"{exp_name}: Predicted vs Target Health Index", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_engine_trajectories(
    df_train: pd.DataFrame,
    health_seq: np.ndarray,
    rul_true: np.ndarray,
    unit_ids: np.ndarray,
    save_path: Path,
    exp_name: str,
    num_engines: int = 5,
):
    """Plot 3: Health Index trajectories for sample engines."""
    unique_units = np.unique(unit_ids)
    n_sample = min(num_engines, len(unique_units))
    np.random.seed(42)
    sample_units = np.random.choice(unique_units, size=n_sample, replace=False)
    
    fig, axes = plt.subplots(n_sample, 1, figsize=(14, 3 * n_sample))
    if n_sample == 1:
        axes = [axes]
    
    for idx, unit_id in enumerate(sample_units):
        ax = axes[idx]
        mask = unit_ids == unit_id
        
        # Get cycle numbers
        df_unit = df_train[df_train['UnitNumber'] == unit_id].sort_values('TimeInCycles')
        cycles = df_unit['TimeInCycles'].values
        
        # Align predictions with cycles
        if len(cycles) >= SEQUENCE_LENGTH:
            cycles_for_pred = cycles[SEQUENCE_LENGTH-1:]
            health_traj = health_seq[mask][:len(cycles_for_pred)]
            rul_traj = rul_true[mask][:len(cycles_for_pred)]
        else:
            cycles_for_pred = cycles
            health_traj = health_seq[mask][:len(cycles_for_pred)]
            rul_traj = rul_true[mask][:len(cycles_for_pred)]
        
        # Compute target HI
        hi_target_traj = compute_target_health_index(
            rul_traj,
            max_rul=MAX_RUL,
            plateau_thresh=int(HI_RUL_PLATEAU_THRESH),
        )
        
        # Plot Health Index
        ax.plot(cycles_for_pred, health_traj, 'b-', label='Predicted HI', linewidth=2, alpha=0.7)
        ax.plot(cycles_for_pred, hi_target_traj, 'r--', label='Target HI (from RUL)', linewidth=2, alpha=0.7)
        ax.set_xlabel("Cycle Number", fontsize=11)
        ax.set_ylabel("Health Index", fontsize=11)
        ax.set_title(f"Engine {int(unit_id)}: Health Index Trajectory", fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add RUL on right axis
        ax2 = ax.twinx()
        ax2.plot(cycles_for_pred, rul_traj, 'g:', label='True RUL', linewidth=1, alpha=0.5)
        ax2.set_ylabel("RUL (cycles)", color='green', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Health Index for FD004 Phase-1 experiments")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num_engines",
        type=int,
        default=5,
        help="Number of sample engines for trajectory plots (default: 5)",
    )
    args = parser.parse_args()
    
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Load config
    config = load_config(args.config)
    exp_name = config['experiment_name']
    
    print("=" * 80)
    print(f"FD004 Phase-1 Health Index Plotting: {exp_name}")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    df_train, df_test, y_test_true, feature_cols = prepare_data_and_features(config)
    
    # Load model (need input_dim first)
    model = load_model_and_scaler(config, len(feature_cols), device)
    
    # Run inference on validation data
    results = run_inference_on_val_data(
        model, df_train, feature_cols, None, config, device
    )
    
    if results['health_pred'] is None:
        raise ValueError("Model does not return health predictions. Ensure multi_task.enabled=true in config.")
    
    # Compute target Health Index
    hi_target = compute_target_health_index(
        results['rul_true'],
        max_rul=config['data']['max_rul'],
        plateau_thresh=int(config['multi_task']['hi_plateau_threshold']),
    )
    
    # Create output directory
    output_dir = Path(config['results']['base_dir']) / "phase1" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n[3] Generating plots...")
    
    # Plot 1: HI vs RUL
    plot_health_index_vs_rul(
        results['rul_true'],
        results['health_pred'],
        output_dir / f"{exp_name}_hi_vs_rul.png",
        exp_name,
    )
    
    # Plot 2: Predicted vs Target HI
    plot_predicted_vs_target_hi(
        hi_target,
        results['health_pred'],
        output_dir / f"{exp_name}_pred_vs_target_hi.png",
        exp_name,
    )
    
    # Plot 3: Engine trajectories
    if results['health_seq'] is not None:
        plot_engine_trajectories(
            df_train,
            results['health_seq'].squeeze(-1) if results['health_seq'].ndim == 3 else results['health_seq'],
            results['rul_true'],
            results['unit_ids'],
            output_dir / f"{exp_name}_engine_trajectories.png",
            exp_name,
            num_engines=args.num_engines,
        )
    else:
        # Fallback: use health_pred if health_seq not available
        print("Warning: health_seq not available, using health_pred for trajectories")
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

