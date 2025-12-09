"""
Ablation runner for EOL-LSTM training across FD001-FD004.

Runs systematic feature ablation studies and logs results to CSV.
"""

import os
import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    get_feature_columns,
    add_temporal_window_features,
)
from src.feature_config import FeatureGroupsConfig, TemporalWindowConfig
from src.feature_safety import remove_rul_leakage
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTM,
    train_eol_full_lstm,
    evaluate_eol_full_lstm,
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================================================================
# Ablation Configurations
# ===================================================================

ABLATIONS = [
    {
        "name": "baseline_physics_only",
        "cfg": FeatureGroupsConfig(
            use_settings=True,
            use_sensors=True,
            use_physics_core=True,
            use_temporal_windows=False,
            use_condition_id=False,
        ),
    },
    {
        "name": "physics_plus_temporal_short",
        "cfg": FeatureGroupsConfig(
            use_settings=True,
            use_sensors=True,
            use_physics_core=True,
            use_temporal_windows=True,
            use_condition_id=False,
            temporal_cfg=TemporalWindowConfig(
                short_windows=(5, 10),
                long_windows=(),
                include_derivatives=True,
                include_rolling_stats=True,
                include_deltas=True,
            ),
        ),
    },
    {
        "name": "physics_plus_temporal_short_long",
        "cfg": FeatureGroupsConfig(
            use_settings=True,
            use_sensors=True,
            use_physics_core=True,
            use_temporal_windows=True,
            use_condition_id=False,
            temporal_cfg=TemporalWindowConfig(
                short_windows=(5, 10),
                long_windows=(20, 30),
                include_derivatives=True,
                include_rolling_stats=True,
                include_deltas=True,
            ),
        ),
    },
    {
        "name": "no_physics_temporal_only",
        "cfg": FeatureGroupsConfig(
            use_settings=True,
            use_sensors=True,
            use_physics_core=False,
            use_temporal_windows=True,
            use_condition_id=False,
            temporal_cfg=TemporalWindowConfig(
                short_windows=(5, 10),
                long_windows=(20, 30),
                include_derivatives=True,
                include_rolling_stats=True,
                include_deltas=True,
            ),
        ),
    },
    {
        "name": "full_plus_condition_id",
        "cfg": FeatureGroupsConfig(
            use_settings=True,
            use_sensors=True,
            use_physics_core=True,
            use_temporal_windows=True,
            use_condition_id=True,
            temporal_cfg=TemporalWindowConfig(
                short_windows=(5, 10),
                long_windows=(20, 30),
                include_derivatives=True,
                include_rolling_stats=True,
                include_deltas=True,
            ),
        ),
    },
]

# Training config (same for all ablations)
TRAINING_CONFIG = {
    "past_len": 30,
    "max_rul": 125,
    "batch_size": 256,
    "engine_train_ratio": 0.8,
    "num_epochs": 80,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "bidirectional": False,
}

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def run_ablation():
    """Main ablation runner function."""
    records = []

    print("=" * 80)
    print("EOL-LSTM Ablation Study")
    print("=" * 80)
    print(f"Datasets: {DATASETS}")
    print(f"Ablations: {len(ABLATIONS)}")
    print(f"Total runs: {len(DATASETS) * len(ABLATIONS)}")
    print("=" * 80)

    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")

        # Load data
        print(f"\n[{dataset_name}] Loading data...")
        df_train, _, _ = load_cmapps_subset(
            dataset_name,
            max_rul=None,
            clip_train=False,
            clip_test=True,
        )

        # Add FD_ID
        fd_id_map = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
        df_train["FD_ID"] = fd_id_map[dataset_name]

        # Apply physics features (always, as base)
        df_train = create_physical_features(df_train)

        for abl in ABLATIONS:
            abl_name = abl["name"]
            cfg = abl["cfg"]

            print(f"\n[{dataset_name}] Ablation: {abl_name}")
            print("-" * 80)

            # Apply temporal window features if needed
            df_abl = df_train.copy()
            if cfg.use_temporal_windows:
                print(f"  Adding temporal window features...")
                df_abl = add_temporal_window_features(
                    df_abl,
                    unit_col="UnitNumber",
                    cycle_col="TimeInCycles",
                    temporal_cfg=cfg.temporal_cfg,
                )

            # Get feature columns
            feature_cols = get_feature_columns(df_abl, cfg)
            feature_cols, leaked = remove_rul_leakage(feature_cols)

            if leaked:
                print(f"  ⚠️  Removed RUL leakage: {leaked}")

            print(f"  Using {len(feature_cols)} features")

            # Build sequences
            X_full, y_full, unit_ids_full = build_full_eol_sequences_from_df(
                df=df_abl,
                feature_cols=feature_cols,
                past_len=TRAINING_CONFIG["past_len"],
                max_rul=TRAINING_CONFIG["max_rul"],
                unit_col="UnitNumber",
                cycle_col="TimeInCycles",
                rul_col="RUL",
            )

            # Create dataloaders
            train_loader, val_loader, scaler, _, _ = create_full_dataloaders(
                X=X_full,
                y=y_full,
                unit_ids=unit_ids_full,
                batch_size=TRAINING_CONFIG["batch_size"],
                engine_train_ratio=TRAINING_CONFIG["engine_train_ratio"],
                shuffle_engines=True,
                random_seed=42,
            )

            # Initialize model
            model = EOLFullLSTM(
                input_dim=len(feature_cols),
                hidden_dim=TRAINING_CONFIG["hidden_dim"],
                num_layers=TRAINING_CONFIG["num_layers"],
                dropout=TRAINING_CONFIG["dropout"],
                bidirectional=TRAINING_CONFIG["bidirectional"],
            )

            # Train
            print(f"  Training...")
            model, history = train_eol_full_lstm(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=TRAINING_CONFIG["num_epochs"],
                lr=TRAINING_CONFIG["lr"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                patience=TRAINING_CONFIG["patience"],
                device=device,
                results_dir=f"../results/ablation/{dataset_name.lower()}/{abl_name}",
                run_name=f"{dataset_name.lower()}_{abl_name}",
            )

            # Evaluate
            print(f"  Evaluating...")
            metrics = evaluate_eol_full_lstm(
                model=model,
                val_loader=val_loader,
                device=device,
            )

            # Extract metrics
            m_point = metrics["pointwise"]
            m_nasa_point = metrics["nasa_pointwise"]

            # Build record
            record = {
                "dataset": dataset_name,
                "ablation_name": abl_name,
                "num_features": len(feature_cols),
                "rmse_point": m_point["rmse"],
                "mae_point": m_point["mae"],
                "r2_point": m_point["r2"],
                "nasa_point_mean": m_nasa_point["score_mean"],
                "use_settings": cfg.use_settings,
                "use_sensors": cfg.use_sensors,
                "use_physics_core": cfg.use_physics_core,
                "use_temporal_windows": cfg.use_temporal_windows,
                "use_condition_id": cfg.use_condition_id,
                "short_windows": (
                    list(cfg.temporal_cfg.short_windows) if cfg.use_temporal_windows else []
                ),
                "long_windows": (
                    list(cfg.temporal_cfg.long_windows) if cfg.use_temporal_windows else []
                ),
            }

            # Add EOL metrics if available
            if "eol" in metrics:
                m_eol = metrics["eol"]
                record.update({
                    "rmse_eol": m_eol["rmse"],
                    "mae_eol": m_eol["mae"],
                    "nasa_eol_sum": m_eol["nasa_score_sum"],
                    "nasa_eol_mean": m_eol["nasa_score_mean"],
                })
            else:
                record.update({
                    "rmse_eol": None,
                    "mae_eol": None,
                    "nasa_eol_sum": None,
                    "nasa_eol_mean": None,
                })

            records.append(record)
            print(f"  ✓ Completed: RMSE_point={m_point['rmse']:.4f}, RMSE_eol={record.get('rmse_eol', 'N/A')}")

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    df_rec = pd.DataFrame.from_records(records)
    out_dir = Path("results") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eol_lstm_ablation_metrics.csv"
    df_rec.to_csv(out_path, index=False)

    print(f"\nResults saved to: {out_path}")
    print(f"Total records: {len(records)}")
    print("\nSummary:")
    print(df_rec[["dataset", "ablation_name", "rmse_point", "rmse_eol", "num_features"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_ablation()

