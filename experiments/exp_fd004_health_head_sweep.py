"""
FD004 Phase 1 Hyperparameter Sweep: Health Head Stabilization

Dieses Skript führt eine systematische Hyperparameter-Sweep über FD004 durch, um:
- Stabilere Monotonie zu erreichen
- Condition-Drift zu reduzieren
- Bessere HI-Kurven zu generieren

Phase 1 Sweep-Parameter:
- health_loss_weight (λ) ∈ [0.25, 0.30, 0.35, 0.40]
- mono_late_weight ∈ [0.03, 0.05]
- mono_global_weight ∈ [0.003, 0.005]

Fixe Werte:
- rul_beta = 45.0
- hi_condition_calib_weight = 0.0

Total: 4 × 2 × 2 = 16 Experimente
"""

import os
import sys
import random
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# Reproduzierbarkeit sicherstellen
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    FeatureConfig,
    TemporalFeatureConfig,
)
from src.config import (
    GLOBAL_FEATURE_COLS,
    PhysicsFeatureConfig,
    MAX_RUL,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    HI_RUL_PLATEAU_THRESH,
    HI_EOL_THRESH,
    HI_MONO_WEIGHT,
    HI_EOL_WEIGHT,
    HI_GLOBAL_MONO_WEIGHT,
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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def run_fd004_experiment(
    rul_beta: float,
    hi_condition_calib_weight: float,
    health_loss_weight: float,
    mono_late_weight: float,
    mono_global_weight: float,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Trainiert das Multi-Task-FD004-Modell mit gegebenen Hyperparametern
    und gibt ein Dict mit wichtigen Metriken zurück.

    Args:
        rul_beta: RUL-Gewichtung τ (fix = 45.0)
        hi_condition_calib_weight: Weight für Condition-Calibration-Loss (fix = 0.0)
        health_loss_weight: Trade-off λ zwischen RUL und Health Loss
        mono_late_weight: Weight für Late-Monotonicity-Loss
        mono_global_weight: Weight für Global-Monotonicity-Loss
        experiment_name: Name des Experiments (für Checkpoints)

    Returns:
        Dictionary mit Metriken:
        - val_rmse, val_mae, val_bias, val_r2, val_nasa_mean
        - test_rmse, test_mae, test_bias, test_r2, test_nasa_mean
        - val_mono_late, val_mono_global (falls vorhanden)
    """
    print("\n" + "=" * 80)
    print(f"[EXPERIMENT] {experiment_name}")
    print("=" * 80)
    print(f"  rul_beta (τ): {rul_beta}")
    print(f"  hi_condition_calib_weight: {hi_condition_calib_weight}")
    print(f"  health_loss_weight (λ): {health_loss_weight}")
    print(f"  mono_late_weight: {mono_late_weight}")
    print(f"  mono_global_weight: {mono_global_weight}")
    print("=" * 80)

    # ===================================================================
    # 1. Daten laden: FD004
    # ===================================================================
    print("\n[1] Loading FD004 data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        "FD004",
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Feature Engineering
    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=False,
        use_temporal_on_physics=False,
    )
    
    # Feature configuration for temporal features
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=True,
        temporal=TemporalFeatureConfig(
            base_cols=None,  # Auto-infer
            short_windows=(5, 10),
            long_windows=(30,),
            add_rolling_mean=True,
            add_rolling_std=False,
            add_trend=True,
            add_delta=True,
            delta_lags=(5, 10),
        ),
    )
    
    # Apply feature engineering to training data
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
    
    # Apply SAME feature engineering to test data
    print("\n[1.5] Applying feature engineering to test data...")
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
        c
        for c in df_train.columns
        if c
        not in [
            "UnitNumber",
            "TimeInCycles",
            "RUL",
            "RUL_raw",
            "MaxTime",
            "ConditionID",
        ]
    ]
    feature_cols, leaked = remove_rul_leakage(feature_cols)
    print(f"Using {len(feature_cols)} features for model input.")

    # ===================================================================
    # 2. Full-Trajectory Sequenzen bauen
    # ===================================================================
    print("\n[2] Building full-trajectory sequences...")
    X_full, y_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=SEQUENCE_LENGTH,
        max_rul=MAX_RUL,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )

    # ===================================================================
    # 3. Dataloaders erstellen (Condition-wise Scaler)
    # ===================================================================
    print("\n[3] Creating dataloaders with condition-wise scaling...")
    train_loader, val_loader, scaler, train_unit_ids, val_unit_ids = create_full_dataloaders(
        X=X_full,
        y=y_full,
        unit_ids=unit_ids_full,
        cond_ids=cond_ids_full,
        batch_size=256,  # Standard batch size
        engine_train_ratio=0.8,
        shuffle_engines=True,
        random_seed=RANDOM_SEED,
        use_condition_wise_scaling=True,  # Condition-wise scaling aktiviert
    )

    # ===================================================================
    # 4. Modell initialisieren
    # ===================================================================
    print("\n[4] Initializing model...")
    model = EOLFullLSTMWithHealth(
        input_dim=X_full.shape[-1],
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        bidirectional=False,
        lambda_health=health_loss_weight,  # Health loss weight
    )
    model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ===================================================================
    # 5. Training
    # ===================================================================
    print("\n[5] Training model...")
    results_dir = Path("results/health_index/fd004") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_eol_full_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=80,
        lr=1e-4,
        weight_decay=1e-4,
        patience=8,
        device=device,
        results_dir=results_dir,
        run_name=experiment_name,
        use_mixed_precision=True,
        use_health_head=True,
        max_rul=float(MAX_RUL),
        tau=rul_beta,  # RUL-Gewichtung
        lambda_health=health_loss_weight,  # Health Loss Weight
        hi_condition_calib_weight=hi_condition_calib_weight,
        hi_plateau_threshold=float(HI_RUL_PLATEAU_THRESH),
        hi_mono_late_weight=mono_late_weight,  # Late monotonicity weight
        hi_mono_global_weight=mono_global_weight,  # Global monotonicity weight
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

    val_rmse = val_metrics["pointwise"]["rmse"]
    val_mae = val_metrics["pointwise"]["mae"]
    val_bias = val_metrics["pointwise"]["bias"]
    val_r2 = val_metrics["pointwise"]["r2"]
    val_nasa_mean = val_metrics["nasa_pointwise"]["score_mean"]

    # EOL Metriken (falls verfügbar)
    if "eol" in val_metrics:
        val_rmse_eol = val_metrics["eol"]["rmse"]
        val_mae_eol = val_metrics["eol"]["mae"]
        val_bias_eol = val_metrics["eol"]["bias"]
        val_nasa_mean_eol = val_metrics["eol"]["nasa_score_mean"]
    else:
        val_rmse_eol = val_rmse
        val_mae_eol = val_mae
        val_bias_eol = val_bias
        val_nasa_mean_eol = val_nasa_mean

    # Monotonie-Losses aus History (falls vorhanden)
    val_mono_late = None
    val_mono_global = None
    if "val_mono_late" in history and len(history["val_mono_late"]) > 0:
        val_mono_late = history["val_mono_late"][-1]  # Letzter Epoch
    if "val_mono_global" in history and len(history["val_mono_global"]) > 0:
        val_mono_global = history["val_mono_global"][-1]  # Letzter Epoch

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
        past_len=SEQUENCE_LENGTH,
        max_rul=MAX_RUL,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        device=device,
    )

    test_rmse = test_metrics["pointwise"]["rmse"]
    test_mae = test_metrics["pointwise"]["mae"]
    test_bias = test_metrics["pointwise"]["bias"]
    test_r2 = test_metrics["pointwise"]["r2"]
    test_nasa_mean = test_metrics["nasa_pointwise"]["score_mean"]

    # EOL Metriken (falls verfügbar)
    if "eol" in test_metrics:
        test_rmse_eol = test_metrics["eol"]["rmse"]
        test_mae_eol = test_metrics["eol"]["mae"]
        test_bias_eol = test_metrics["eol"]["bias"]
        test_nasa_mean_eol = test_metrics["eol"]["nasa_score_mean"]
    else:
        test_rmse_eol = test_rmse
        test_mae_eol = test_mae
        test_bias_eol = test_bias
        test_nasa_mean_eol = test_nasa_mean

    # ===================================================================
    # 8. Zusammenfassung
    # ===================================================================
    print("\n" + "=" * 80)
    print(f"[EXPERIMENT SUMMARY] {experiment_name}")
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
    if val_mono_late is not None:
        print(f"\nMonotonicity Losses (val):")
        print(f"  mono_late: {val_mono_late:.6f}")
        if val_mono_global is not None:
            print(f"  mono_global: {val_mono_global:.6f}")
    print("=" * 80)

    # Return dictionary mit allen Metriken
    return {
        "experiment_name": experiment_name,
        "rul_beta": rul_beta,
        "hi_condition_calib_weight": hi_condition_calib_weight,
        "health_loss_weight": health_loss_weight,
        "mono_late_weight": mono_late_weight,
        "mono_global_weight": mono_global_weight,
        # Validation metrics
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_bias": val_bias,
        "val_r2": val_r2,
        "val_nasa_mean": val_nasa_mean,
        "val_rmse_eol": val_rmse_eol,
        "val_mae_eol": val_mae_eol,
        "val_bias_eol": val_bias_eol,
        "val_nasa_mean_eol": val_nasa_mean_eol,
        # Test metrics
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_bias": test_bias,
        "test_r2": test_r2,
        "test_nasa_mean": test_nasa_mean,
        "test_rmse_eol": test_rmse_eol,
        "test_mae_eol": test_mae_eol,
        "test_bias_eol": test_bias_eol,
        "test_nasa_mean_eol": test_nasa_mean_eol,
        # Monotonicity losses
        "val_mono_late": val_mono_late if val_mono_late is not None else np.nan,
        "val_mono_global": val_mono_global if val_mono_global is not None else np.nan,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("FD004 Phase 1 Hyperparameter Sweep: Health Head Stabilization")
    print("=" * 80)
    print("Sweep Parameters:")
    print("  health_loss_weight (λ) ∈ [0.25, 0.30, 0.35, 0.40]")
    print("  mono_late_weight ∈ [0.03, 0.05]")
    print("  mono_global_weight ∈ [0.003, 0.005]")
    print("\nFixed Parameters:")
    print("  rul_beta = 45.0")
    print("  hi_condition_calib_weight = 0.0")
    print("=" * 80)
    print(f"Total experiments: 4 × 2 × 2 = 16\n")

    # Phase 1: Grid Search über alle Kombinationen
    health_loss_weights = [0.25, 0.30, 0.35, 0.40]
    mono_late_weights = [0.03, 0.05]
    mono_global_weights = [0.003, 0.005]
    
    # Fixe Werte
    rul_beta = 45.0
    hi_condition_calib_weight = 0.0
    
    # Erzeuge alle Kombinationen
    experiments = []
    exp_idx = 1
    for lambda_health in health_loss_weights:
        for mono_late in mono_late_weights:
            for mono_global in mono_global_weights:
                exp_name = f"fd004_phase1_l{lambda_health:.2f}_ml{mono_late:.3f}_mg{mono_global:.4f}"
                experiments.append({
                    "rul_beta": rul_beta,
                    "hi_condition_calib_weight": hi_condition_calib_weight,
                    "health_loss_weight": lambda_health,
                    "mono_late_weight": mono_late,
                    "mono_global_weight": mono_global,
                    "experiment_name": exp_name,
                })
                exp_idx += 1

    # Führe alle Experimente durch
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"RUN {i}/{len(experiments)}")
        print(f"{'='*80}")
        
        result = run_fd004_experiment(**exp_config)
        results.append(result)

    # ===================================================================
    # Ergebnisse in DataFrame zusammenfassen
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY - All Runs")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    # Spalten-Reihenfolge für bessere Lesbarkeit
    column_order = [
        "experiment_name",
        "rul_beta",
        "hi_condition_calib_weight",
        "health_loss_weight",
        "mono_late_weight",
        "mono_global_weight",
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
        "val_mono_late",
        "val_mono_global",
    ]

    # Nur vorhandene Spalten verwenden
    available_columns = [col for col in column_order if col in df_results.columns]
    df_results = df_results[available_columns]

    # Tabelle ausgeben
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print("\n" + df_results.to_string(index=False))
    print("\n" + "=" * 80)

    # ===================================================================
    # CSV speichern
    # ===================================================================
    output_dir = Path("results/health_index/fd004")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hparam_sweep_fd004_phase1.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print("=" * 80)

