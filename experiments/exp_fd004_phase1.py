"""
FD004 Phase-1 Training Script

This script loads a YAML configuration file and trains the FD004 Phase-1 Health-Head model.

Example usage:
    python -m experiments.exp_fd004_phase1 --config config/fd004_phase1_rmse.yaml
    python -m experiments.exp_fd004_phase1 --config config/fd004_phase1_nasa.yaml
"""

import os
import sys
import random
import argparse
import yaml
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

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
    HIDDEN_SIZE,
    NUM_LAYERS,
)
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTMWithHealth,
    train_eol_full_lstm,
)
from src.feature_safety import remove_rul_leakage


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training_from_config(config_path: Path):
    """Train FD004 Phase-1 model from YAML config."""
    print("=" * 80)
    print(f"FD004 Phase-1 Training from Config: {config_path}")
    print("=" * 80)
    
    # Load config
    config = load_config(config_path)
    exp_name = config['experiment_name']
    dataset = config['dataset']
    
    # Reproduzierbarkeit
    random_seed = config['training']['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===================================================================
    # 1. Daten laden
    # ===================================================================
    print(f"\n[1] Loading {dataset} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=config['data']['clip_train'],
        clip_test=config['data']['clip_test'],
    )
    
    # Feature Engineering
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
    print(f"Using {len(feature_cols)} features for model input.")
    
    # ===================================================================
    # 2. Sequences bauen
    # ===================================================================
    print("\n[2] Building full-trajectory sequences...")
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
    
    # ===================================================================
    # 3. Dataloaders
    # ===================================================================
    print("\n[3] Creating dataloaders...")
    train_loader, val_loader, scaler, train_unit_ids, val_unit_ids = create_full_dataloaders(
        X=X_full,
        y=y_full,
        unit_ids=unit_ids_full,
        cond_ids=cond_ids_full,
        batch_size=config['training']['batch_size'],
        engine_train_ratio=config['training']['engine_train_ratio'],
        shuffle_engines=config['training']['shuffle_engines'],
        random_seed=random_seed,
        use_condition_wise_scaling=config['data']['use_condition_wise_scaling'],
    )
    
    # ===================================================================
    # 4. Modell initialisieren
    # ===================================================================
    print("\n[4] Initializing model...")
    model = EOLFullLSTMWithHealth(
        input_dim=X_full.shape[-1],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        lambda_health=config['multi_task']['health_loss_weight'],
    )
    model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ===================================================================
    # 5. Training
    # ===================================================================
    print("\n[5] Training model...")
    results_dir = Path(config['results']['base_dir']) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model, history = train_eol_full_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        device=device,
        results_dir=results_dir,
        run_name=exp_name,
        use_mixed_precision=config['training']['use_mixed_precision'],
        use_health_head=config['multi_task']['enabled'],
        max_rul=float(config['data']['max_rul']),
        tau=config['multi_task']['rul_beta'],
        lambda_health=config['multi_task']['health_loss_weight'],
        hi_condition_calib_weight=config['multi_task']['hi_condition_calib_weight'],
        hi_plateau_threshold=config['multi_task']['hi_plateau_threshold'],
        hi_mono_late_weight=config['multi_task']['mono_late_weight'],
        hi_mono_global_weight=config['multi_task']['mono_global_weight'],
    )
    
    print("\n" + "=" * 80)
    print(f"Training completed. Best model saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FD004 Phase-1 model from YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    run_training_from_config(args.config)

