"""
FD004 Phase-1 Health-Head: Cross-Dataset Baseline

This script applies the best Phase-1 Health-Head hyperparameters from FD004
to all datasets (FD001, FD002, FD003, FD004) and generates a unified results CSV.

Usage:
    python -m experiments.exp_phase1_health_head_all_datasets
"""

import os
import sys
import random
import csv
from pathlib import Path
from typing import Dict, Any

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
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
    HI_RUL_PLATEAU_THRESH,
)
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTMWithHealth,
    train_eol_full_lstm,
    evaluate_eol_full_lstm,
    evaluate_on_test_data,
)
from src.feature_safety import remove_rul_leakage


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_phase1_for_dataset(
    dataset_name: str,
    base_config: dict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train Phase-1 Health-Head model for a single dataset.
    
    Args:
        dataset_name: Dataset name (FD001, FD002, FD003, or FD004)
        base_config: Base configuration dictionary
        device: torch device
        
    Returns:
        Dictionary with validation and test metrics
    """
    print("\n" + "=" * 80)
    print(f"Running Phase-1 Health-Head for {dataset_name}")
    print("=" * 80)
    
    # Override dataset in config
    config = base_config.copy()
    config['dataset'] = dataset_name
    
    # Set experiment name
    experiment_name = f"{dataset_name.lower()}_phase1_baseline"
    config['experiment_name'] = experiment_name
    
    # Reproduzierbarkeit
    random_seed = int(config['training']['random_seed'])
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ===================================================================
    # 1. Daten laden
    # ===================================================================
    print(f"\n[1] Loading {dataset_name} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
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
        past_len=int(config['data']['past_len']),
        max_rul=int(config['data']['max_rul']),
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
        batch_size=int(config['training']['batch_size']),
        engine_train_ratio=float(config['training']['engine_train_ratio']),
        shuffle_engines=bool(config['training']['shuffle_engines']),
        random_seed=random_seed,
        use_condition_wise_scaling=bool(config['data']['use_condition_wise_scaling']),
    )
    
    # ===================================================================
    # 4. Modell initialisieren
    # ===================================================================
    print("\n[4] Initializing model...")
    model = EOLFullLSTMWithHealth(
        input_dim=X_full.shape[-1],
        hidden_dim=int(config['model']['hidden_dim']),
        num_layers=int(config['model']['num_layers']),
        dropout=float(config['model']['dropout']),
        bidirectional=bool(config['model']['bidirectional']),
        lambda_health=float(config['multi_task']['health_loss_weight']),
    )
    model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ===================================================================
    # 5. Training
    # ===================================================================
    print("\n[5] Training model...")
    results_dir = Path(config['results']['base_dir']) / dataset_name.lower() / config['results']['phase_tag']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model, history = train_eol_full_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(config['training']['num_epochs']),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        patience=int(config['training']['patience']),
        device=device,
        results_dir=results_dir,
        run_name=experiment_name,
        use_mixed_precision=bool(config['training']['use_mixed_precision']),
        use_health_head=bool(config['multi_task']['enabled']),
        max_rul=float(config['data']['max_rul']),
        tau=float(config['multi_task']['rul_beta']),
        lambda_health=float(config['multi_task']['health_loss_weight']),
        hi_condition_calib_weight=float(config['multi_task']['hi_condition_calib_weight']),
        hi_plateau_threshold=float(config['multi_task']['hi_plateau_threshold']),
        hi_mono_late_weight=float(config['multi_task']['mono_late_weight']),
        hi_mono_global_weight=float(config['multi_task']['mono_global_weight']),
    )
    
    # ===================================================================
    # 6. Validation Metriken
    # ===================================================================
    print("\n[6] Evaluating on validation set...")
    val_metrics = evaluate_eol_full_lstm(
        model=model,
        val_loader=val_loader,
        device=device,
    )
    
    # ===================================================================
    # 7. Test Metriken
    # ===================================================================
    print("\n[7] Evaluating on test set...")
    test_metrics = evaluate_on_test_data(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler=scaler,
        past_len=int(config['data']['past_len']),
        max_rul=int(config['data']['max_rul']),
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        device=device,
    )
    
    # Extract metrics
    val_rmse = val_metrics["pointwise"]["rmse"]
    val_mae = val_metrics["pointwise"]["mae"]
    val_bias = val_metrics["pointwise"]["bias"]
    val_r2 = val_metrics["pointwise"]["r2"]
    val_nasa_mean = val_metrics["nasa_pointwise"]["score_mean"]
    
    test_rmse = test_metrics["pointwise"]["rmse"]
    test_mae = test_metrics["pointwise"]["mae"]
    test_bias = test_metrics["pointwise"]["bias"]
    test_r2 = test_metrics["pointwise"]["r2"]
    test_nasa_mean = test_metrics["nasa_pointwise"]["score_mean"]
    
    print("\n" + "=" * 80)
    print(f"[{dataset_name}] Results Summary")
    print("=" * 80)
    print("Validation Metrics:")
    print(f"  RMSE: {val_rmse:.4f} cycles")
    print(f"  MAE: {val_mae:.4f} cycles")
    print(f"  Bias: {val_bias:.4f} cycles")
    print(f"  R²: {val_r2:.4f}")
    print(f"  NASA (mean): {val_nasa_mean:.4f}")
    print("\nTest Metrics:")
    print(f"  RMSE: {test_rmse:.4f} cycles")
    print(f"  MAE: {test_mae:.4f} cycles")
    print(f"  Bias: {test_bias:.4f} cycles")
    print(f"  R²: {test_r2:.4f}")
    print(f"  NASA (mean): {test_nasa_mean:.4f}")
    print("=" * 80)
    
    return {
        "dataset": dataset_name,
        "experiment_name": experiment_name,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_bias": val_bias,
        "val_r2": val_r2,
        "val_nasa_mean": val_nasa_mean,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_bias": test_bias,
        "test_r2": test_r2,
        "test_nasa_mean": test_nasa_mean,
    }


def main():
    """Main function to run Phase-1 Health-Head baseline for all datasets."""
    print("=" * 80)
    print("FD004 Phase-1 Health-Head: Cross-Dataset Baseline")
    print("=" * 80)
    print("\nThis script applies the best Phase-1 Health-Head hyperparameters")
    print("from FD004 to all datasets (FD001, FD002, FD003, FD004).")
    print("=" * 80)
    
    # Load base config
    config_path = Path("config/health_head_phase1/base_phase1_health_head.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    base_config = load_config(config_path)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Datasets
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    
    # Results directory
    results_dir = Path("results/health_index")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Output CSV
    output_csv = results_dir / "phase1_cross_dataset_baseline.csv"
    
    # Field names for CSV
    fieldnames = [
        "dataset",
        "rul_beta",
        "health_loss_weight",
        "mono_late_weight",
        "mono_global_weight",
        "hi_condition_calib_weight",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_r2",
        "val_nasa_mean",
        "test_rmse",
        "test_mae",
        "test_bias",
        "test_r2",
        "test_nasa_mean",
    ]
    
    # Run experiments and collect results
    all_results = []
    
    for dataset in datasets:
        try:
            result = run_phase1_for_dataset(dataset, base_config, device)
            
            # Add hyperparameters to result
            result.update({
                "rul_beta": base_config['multi_task']['rul_beta'],
                "health_loss_weight": base_config['multi_task']['health_loss_weight'],
                "mono_late_weight": base_config['multi_task']['mono_late_weight'],
                "mono_global_weight": base_config['multi_task']['mono_global_weight'],
                "hi_condition_calib_weight": base_config['multi_task']['hi_condition_calib_weight'],
            })
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write CSV
    print("\n" + "=" * 80)
    print("Writing results to CSV...")
    print("=" * 80)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {key: result.get(key, '') for key in fieldnames}
            writer.writerow(row)
    
    print(f"\n✓ Cross-dataset baseline written to: {output_csv}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print("\n| Dataset | Test RMSE | Test MAE | Test Bias | Test R² | Test NASA Mean |")
    print("|---------|-----------|----------|-----------|---------|----------------|")
    for result in all_results:
        print(
            f"| {result['dataset']:7} | "
            f"{result['test_rmse']:9.2f} | "
            f"{result['test_mae']:8.2f} | "
            f"{result['test_bias']:9.2f} | "
            f"{result['test_r2']:7.4f} | "
            f"{result['test_nasa_mean']:14.2f} |"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

