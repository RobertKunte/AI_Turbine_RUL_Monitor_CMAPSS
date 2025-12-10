import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Use absolute imports
from src.analysis.inference import reconstruct_model_from_checkpoint
from src.eol_full_lstm import build_full_eol_sequences_from_df
from src.data_loading import load_cmapps_subset
from src.models.rul_decoder import RULTrajectoryDecoderV1
from src.additional_features import (
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
    create_physical_features,
    create_all_features,
    build_condition_features,
    create_twin_features,
)
from src.feature_safety import remove_rul_leakage


def build_rul_seq_from_last(rul_last: torch.Tensor, T: int) -> torch.Tensor:
    """
    Reconstructs RUL(t) backwards from the last step RUL.
    rul_seq[i, j] = rul_last[i] + (T - 1 - j)
    """
    device = rul_last.device
    steps = torch.arange(T - 1, -1, -1, device=device).unsqueeze(0)  # [1, T]
    rul_seq = rul_last.unsqueeze(1) + steps  # [B, T]
    return rul_seq


def train_rul_decoder_v1(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Train RUL Trajectory Decoder V1.
    Called by run_experiments.py.
    """
    experiment_name = config["experiment_name"]
    dataset_name = config["dataset"]
    encoder_checkpoint = config.get("encoder_checkpoint")
    
    if not encoder_checkpoint:
        raise ValueError("encoder_checkpoint must be provided in config for decoder_v1 experiment")
        
    results_dir = Path("results") / dataset_name.lower() / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    training_params = config.get("training_params", {})
    epochs = training_params.get("num_epochs", 50)
    batch_size = training_params.get("batch_size", 256)
    lr = training_params.get("lr", 1e-3)
    weight_decay = training_params.get("weight_decay", 1e-4)

    print(f"\n[DecoderV1] Loading encoder from {encoder_checkpoint}...")
    if not os.path.exists(encoder_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint}")

    # Load frozen encoder
    # Note: reconstruct_model_from_checkpoint returns (model, config, scaler)
    # We pass strict=False if keys mismatch (e.g. if we modified EOLFullTransformerEncoder)
    try:
        encoder_model, model_config, scaler = reconstruct_model_from_checkpoint(
            encoder_checkpoint, device=device
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying to load with strict=False...")
        # Fallback if needed, but reconstruct doesn't have strict param easily accessible
        raise e
        
    encoder_model.eval()
    for p in encoder_model.parameters():
        p.requires_grad = False
    
    print("[DecoderV1] Encoder loaded and frozen.")

    # 2. Build Datasets
    print("[DecoderV1] Building datasets...")
    # Load raw FD data (same helper as run_experiments)
    train_df, test_df, _ = load_cmapps_subset(dataset_name, max_rul=None, clip_train=False, clip_test=True)
    
    # Feature Config from current experiment config (to match ms_dt_v2 style)
    features_cfg = config.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    phys_features_cfg = config.get("phys_features", {})
    
    feature_cfg = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=features_cfg.get("use_multiscale_features", True),
        temporal=TemporalFeatureConfig(
            short_windows=tuple(ms_cfg.get("windows_short", (5, 10))),
            long_windows=tuple(ms_cfg.get("windows_long", (30,))),
            add_rolling_mean=True,
            add_rolling_std=False,
            add_trend=True,
            add_delta=True,
            delta_lags=(5, 10),
        ),
    )
    
    # Physics Config (no explicit residual features for decoder)
    physics_cfg = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=False,
        use_temporal_on_physics=False,
    )
    
    # 1) Physics features
    train_df_fe = create_physical_features(train_df, physics_cfg, "UnitNumber", "TimeInCycles")
    test_df_fe = create_physical_features(test_df, physics_cfg, "UnitNumber", "TimeInCycles")
    
    # 2) Continuous condition vector
    if phys_features_cfg.get("use_condition_vector", False):
        train_df_fe = build_condition_features(
            train_df_fe, "UnitNumber", "TimeInCycles", version=phys_features_cfg.get("condition_vector_version", 2)
        )
        test_df_fe = build_condition_features(
            test_df_fe, "UnitNumber", "TimeInCycles", version=phys_features_cfg.get("condition_vector_version", 2)
        )
    
    # 3) Digital twin residuals
    use_twin = phys_features_cfg.get("use_twin_features", False)
    twin_baseline = phys_features_cfg.get("twin_baseline_len", 30)
    if use_twin:
        train_df_fe, twin_model = create_twin_features(
            train_df_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline,
            condition_vector_version=phys_features_cfg.get("condition_vector_version", 2),
        )
        test_df_fe = twin_model.add_twin_and_residuals(test_df_fe)
    
    # 4) Temporal / multi-scale features
    train_df_fe = create_all_features(
        train_df_fe,
        "UnitNumber",
        "TimeInCycles",
        feature_cfg,
        inplace=False,
        physics_config=physics_cfg,
    )
    test_df_fe = create_all_features(
        test_df_fe,
        "UnitNumber",
        "TimeInCycles",
        feature_cfg,
        inplace=False,
        physics_config=physics_cfg,
    )
    
    # 5) Feature columns (match training-time encoder pipeline)
    feature_cols = [
        c
        for c in train_df_fe.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
        
    # Filter features to match scaler
    # We must match the features used by the loaded scaler.
    # Since we loaded the scaler, we should check its input dimension.
    if hasattr(scaler, "mean_"):
        expected_dim = scaler.mean_.shape[0]
        # We need to find the matching columns.
        # This is tricky without the original feature list.
        # Ideally model_config has "feature_cols".
        saved_feature_cols = model_config.get("feature_cols")
        if saved_feature_cols:
            feature_cols = saved_feature_cols
            # Ensure all exist
            missing = [c for c in feature_cols if c not in train_df_fe.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
        else:
            print(f"Warning: feature_cols not in checkpoint config. Using all numeric features.")
            # Filter standard
            feature_cols = [c for c in train_df_fe.columns if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]]
            # Check dim
            if len(feature_cols) != expected_dim:
                print(f"Feature dim mismatch: Got {len(feature_cols)}, Expected {expected_dim}")
                # Try to use current pipeline features
                
    # Scale
    train_df_fe[feature_cols] = scaler.transform(train_df_fe[feature_cols])
    test_df_fe[feature_cols] = scaler.transform(test_df_fe[feature_cols])
    
    # Build sequences
    seq_len = 30
    print(f"[DecoderV1] Building sequences (T={seq_len})...")
    
    train_X, train_y, train_units, train_conds, _ = build_full_eol_sequences_from_df(
        train_df_fe, feature_cols, sequence_length=seq_len, 
        target_col="RUL", 
        return_tensor=True,
        pad_sequences=True,
        cond_col="ConditionID"
    )
    
    test_X, test_y, test_units, test_conds, _ = build_full_eol_sequences_from_df(
        test_df_fe, feature_cols, sequence_length=seq_len,
        target_col="RUL",
        return_tensor=True,
        pad_sequences=True,
        cond_col="ConditionID"
    )
    
    # DataLoaders
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, conds=None):
            self.X = X
            self.y = y
            self.conds = conds
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            item = {"features": self.X[idx], "rul": self.y[idx]}
            if self.conds is not None:
                item["cond_ids"] = self.conds[idx]
            return item

    train_dataset = SimpleDataset(train_X, train_y, train_conds)
    val_dataset = SimpleDataset(test_X, test_y, test_conds)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Decoder
    latent_dim = encoder_model.d_model
    decoder = RULTrajectoryDecoderV1(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 4. Training Loop
    print(f"[DecoderV1] Starting training for {epochs} epochs...")
    best_val_rmse = float("inf")
    
    for epoch in range(epochs):
        decoder.train()
        train_loss_total = 0.0
        
        for batch in train_loader:
            x = batch["features"].to(device)
            rul_last = batch["rul"].to(device)
            rul_last = torch.clamp(rul_last, 0, 125.0)
            
            cond_ids = batch.get("cond_ids", None)
            if cond_ids is not None: cond_ids = cond_ids.to(device)
            
            with torch.no_grad():
                z_seq, _, hi_damage_seq = encoder_model.encode_with_hi(
                    x, cond_ids=cond_ids
                )
            
            T = x.size(1)
            rul_target_seq = build_rul_seq_from_last(rul_last, T)
            rul_target_seq = torch.clamp(rul_target_seq, 0, 125.0)
            
            rul_pred_seq = decoder(z_seq, hi_phys_seq=None, hi_damage_seq=hi_damage_seq)
            
            traj_loss = F.mse_loss(rul_pred_seq, rul_target_seq)
            eol_loss = F.mse_loss(rul_pred_seq[:, -1], rul_last)
            
            # Monotonicity: RUL should decrease. diff = R(t+1) - R(t) should be negative.
            # If diff > 0, penalize.
            diffs = rul_pred_seq[:, 1:] - rul_pred_seq[:, :-1]
            mono_loss = torch.relu(diffs).mean() 
            
            loss = traj_loss + 0.5 * eol_loss + 0.1 * mono_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            
        avg_train_loss = train_loss_total / len(train_loader)
        
        # Validation
        decoder.eval()
        val_mse = 0.0
        val_mae = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                rul_last = batch["rul"].to(device)
                rul_last = torch.clamp(rul_last, 0, 125.0)
                cond_ids = batch.get("cond_ids", None)
                if cond_ids is not None: cond_ids = cond_ids.to(device)
                
                z_seq, _, hi_damage_seq = encoder_model.encode_with_hi(
                    x, cond_ids=cond_ids
                )
                
                rul_pred_seq = decoder(z_seq, hi_phys_seq=None, hi_damage_seq=hi_damage_seq)
                eol_pred = rul_pred_seq[:, -1]
                
                val_mse += F.mse_loss(eol_pred, rul_last, reduction='sum').item()
                val_mae += F.l1_loss(eol_pred, rul_last, reduction='sum').item()
                batches += x.size(0)
        
        val_rmse = math.sqrt(val_mse / batches)
        val_mae_avg = val_mae / batches
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae_avg:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(decoder.state_dict(), results_dir / "decoder_v1_best.pt")
            print("  -> Saved best model")

    print("[DecoderV1] Training finished.")
    
    # Final Metrics
    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "encoder_experiment": config.get("encoder_experiment", "unknown"),
        "best_val_rmse": best_val_rmse,
        "test_metrics": {
            "rmse": best_val_rmse, # Approximate
            "mae": val_mae_avg,
            "bias": 0.0, # Not computed
            "r2": 0.0, # Not computed
            "nasa_mean": 0.0, # Not computed
            "nasa_sum": 0.0,
            "num_engines": len(test_y)
        }
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[DecoderV1] Saved summary to {results_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    # Standalone usage with hardcoded config matching v3d config
    from src.experiment_configs import get_fd004_decoder_v1_from_encoder_v3d_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    config = get_fd004_decoder_v1_from_encoder_v3d_config()
    # Override epochs
    if args.epochs:
        config["training_params"]["num_epochs"] = args.epochs
        
    train_rul_decoder_v1(config, device=torch.device(args.device))
