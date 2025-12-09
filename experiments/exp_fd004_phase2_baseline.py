"""
FD004 Phase-2 Baseline Experiment

This script runs a baseline Phase-2 experiment for FD004 with:
- Condition embeddings (7 conditions for FD004)
- HI smoothness loss (optional, can be disabled)
- LSTM encoder (default, can be switched to Transformer)

Usage:
    python -m experiments.exp_fd004_phase2_baseline
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

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


def main():
    print("=" * 80)
    print("FD004 Phase-2 Baseline Experiment")
    print("=" * 80)
    print("\nPhase-2 Features:")
    print("  - Condition embeddings (7 conditions for FD004)")
    print("  - HI smoothness loss (optional)")
    print("  - LSTM encoder (default)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # ===================================================================
    # Configuration (Phase-2 Baseline)
    # ===================================================================
    dataset_name = "FD004"
    experiment_name = "fd004_phase2_transformer"
    
    # Phase-1 hyperparameters (from best RMSE-optimized config)
    rul_beta = 45.0
    health_loss_weight = 0.35
    mono_late_weight = 0.03
    mono_global_weight = 0.003
    hi_condition_calib_weight = 0.0
    
    # Phase-2 hyperparameters
    use_condition_embedding = True
    cond_emb_dim = 4  # Dimension of condition embeddings
    smooth_hi_weight = 0.02  # HI smoothness loss weight (0.0 to disable)
    smooth_hi_plateau_threshold = 80.0  # RUL threshold for smoothness masking
    
    # Encoder configuration
    encoder_type = "lstm"  # "lstm" or "transformer"
    hidden_dim = 50
    num_layers = 2
    dropout = 0.1
    bidirectional = False
    
    # Training configuration
    num_epochs = 80
    batch_size = 256
    learning_rate = 0.0001
    weight_decay = 0.0001
    patience = 8
    use_mixed_precision = True
    random_seed = 42
    engine_train_ratio = 0.8
    shuffle_engines = True
    
    # Data configuration
    past_len = 30
    max_rul = 125
    use_condition_wise_scaling = True
    clip_train = False
    clip_test = True
    
    # Feature configuration
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
    
    # ===================================================================
    # Load Data
    # ===================================================================
    print(f"\n[1] Loading {dataset_name} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=clip_train,
        clip_test=clip_test,
    )
    
    # Feature Engineering
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_train = create_all_features(df_train, "UnitNumber", "TimeInCycles", feature_config, inplace=False)
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_all_features(df_test, "UnitNumber", "TimeInCycles", feature_config, inplace=False)
    
    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    print(f"Using {len(feature_cols)} features for model input.")
    
    # ===================================================================
    # Build Sequences and Dataloaders
    # ===================================================================
    print("\n[2] Building full-trajectory sequences...")
    X_full, y_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    
    # Determine number of unique conditions
    unique_conditions = torch.unique(cond_ids_full).cpu().numpy()
    num_conditions = len(unique_conditions)
    print(f"Found {num_conditions} unique conditions: {unique_conditions}")
    
    print("\n[3] Creating dataloaders...")
    train_loader, val_loader, scaler, _, _ = create_full_dataloaders(
        X=X_full,
        y=y_full,
        unit_ids=unit_ids_full,
        cond_ids=cond_ids_full,
        batch_size=batch_size,
        engine_train_ratio=engine_train_ratio,
        shuffle_engines=shuffle_engines,
        random_seed=random_seed,
        use_condition_wise_scaling=use_condition_wise_scaling,
    )
    
    # ===================================================================
    # Initialize Model
    # ===================================================================
    print("\n[4] Initializing model...")
    model = EOLFullLSTMWithHealth(
        input_dim=X_full.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        lambda_health=health_loss_weight,
        # Phase 2: Condition embeddings
        use_condition_embedding=use_condition_embedding,
        num_conditions=num_conditions,
        cond_emb_dim=cond_emb_dim,
        # Phase 2: Encoder type
        encoder_type=encoder_type,
        transformer_nhead=4,  # Only used if encoder_type="transformer"
        transformer_dim_feedforward=256,  # Only used if encoder_type="transformer"
    )
    model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    if use_condition_embedding:
        print(f"  - Condition embeddings: {num_conditions} conditions, {cond_emb_dim}D embeddings")
    print(f"  - Encoder type: {encoder_type}")
    
    # ===================================================================
    # Training
    # ===================================================================
    print("\n[5] Training model...")
    results_dir = Path("results/health_index/fd004/phase2") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model, history = train_eol_full_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        results_dir=results_dir,
        run_name=experiment_name,
        use_mixed_precision=use_mixed_precision,
        use_health_head=True,
        max_rul=max_rul,
        tau=rul_beta,
        lambda_health=health_loss_weight,
        hi_condition_calib_weight=hi_condition_calib_weight,
        hi_plateau_threshold=HI_RUL_PLATEAU_THRESH,
        hi_mono_late_weight=mono_late_weight,
        hi_mono_global_weight=mono_global_weight,
        # Phase 2: Smoothness loss
        smooth_hi_weight=smooth_hi_weight,
        smooth_hi_plateau_threshold=smooth_hi_plateau_threshold,
        # Phase 2: Condition embedding flag
        use_condition_embedding=use_condition_embedding,
    )
    
    # ===================================================================
    # Evaluation
    # ===================================================================
    print("\n[6] Evaluating on validation set...")
    val_metrics = evaluate_eol_full_lstm(
        model=model,
        val_loader=val_loader,
        device=device,
    )
    
    print("\n[7] Evaluating on test set...")
    test_metrics = evaluate_on_test_data(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler=scaler,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        device=device,
    )
    
    # ===================================================================
    # Save Results Summary
    # ===================================================================
    print("\n[8] Saving results summary...")
    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "phase": "phase2_baseline",
        "use_condition_embedding": use_condition_embedding,
        "cond_emb_dim": cond_emb_dim if use_condition_embedding else None,
        "smooth_hi_weight": smooth_hi_weight,
        "encoder_type": encoder_type,
        "rul_beta": rul_beta,
        "health_loss_weight": health_loss_weight,
        "mono_late_weight": mono_late_weight,
        "mono_global_weight": mono_global_weight,
        "val_rmse": val_metrics["pointwise"]["rmse"],
        "val_mae": val_metrics["pointwise"]["mae"],
        "val_bias": val_metrics["pointwise"]["bias"],
        "val_r2": val_metrics["pointwise"]["r2"],
        "val_nasa_mean": val_metrics["nasa_pointwise"]["score_mean"],
        "test_rmse": test_metrics["pointwise"]["rmse"],
        "test_mae": test_metrics["pointwise"]["mae"],
        "test_bias": test_metrics["pointwise"]["bias"],
        "test_r2": test_metrics["pointwise"]["r2"],
        "test_nasa_mean": test_metrics["nasa_pointwise"]["score_mean"],
    }
    
    summary_path = results_dir / "summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("Phase-2 Baseline Experiment Complete")
    print("=" * 80)
    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_metrics['pointwise']['rmse']:.2f} cycles")
    print(f"  MAE: {test_metrics['pointwise']['mae']:.2f} cycles")
    print(f"  Bias: {test_metrics['pointwise']['bias']:.2f} cycles")
    print(f"  R²: {test_metrics['pointwise']['r2']:.4f}")
    print(f"  NASA Mean: {test_metrics['nasa_pointwise']['score_mean']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

