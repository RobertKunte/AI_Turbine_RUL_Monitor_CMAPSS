"""
Ablation Study Training System for World Model v3.

This module provides flexible training for architecture variants (A1-A6)
with configurable hyperparameters (horizon, EOL weight, residual, fusion mode).
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as exc:
    raise ImportError("PyTorch is required for training routines.") from exc

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError("scikit-learn is required for preprocessing.") from exc

from src.world_model_training import (
    build_world_model_dataset_with_cond_ids,
    compute_trajectory_step_weights,
    WorldModelTrainingConfig,
    build_seq2seq_samples_from_df,
)
from src.models.world_model_v3_variants import create_world_model_v3_variant
from src.loss import compute_monotonicity_loss
from src.training_utils import compute_global_trend_loss
from src.metrics import compute_eol_errors_and_nasa


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""
    variant: str  # "A1", "A2", ..., "A6"
    horizon: int  # 20, 40, 60
    eol_loss_weight: float  # 1, 5, 10
    use_residuals: bool  # True/False
    fusion_mode: str  # "early" or "late"
    
    # Architecture parameters
    d_model: int = 96
    num_layers: int = 3
    nhead: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1
    kernel_sizes: List[int] = None
    seq_encoder_type: str = "transformer"
    decoder_num_layers: int = 2
    
    # Training parameters
    batch_size: int = 256
    num_epochs: int = 80
    lr: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 10
    engine_train_ratio: float = 0.8
    random_seed: int = 42
    max_rul: int = 125
    past_len: int = 30
    
    # Loss weights (variant-specific defaults)
    traj_loss_weight: float = 1.0
    hi_loss_weight: float = 2.0
    mono_late_weight: float = 0.1
    mono_global_weight: float = 0.1
    
    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 9]
        
        # Adjust loss weights based on variant
        if self.variant == "A2":  # No trajectory
            self.traj_loss_weight = 0.0
        elif self.variant == "A3":  # No HI
            self.hi_loss_weight = 0.0
        elif self.variant == "A4":  # No monotonicity
            self.mono_late_weight = 0.0
            self.mono_global_weight = 0.0
        elif self.variant == "A5":  # EOL-only
            self.traj_loss_weight = 0.0
            self.hi_loss_weight = 0.0
        elif self.variant == "A6":  # Traj-only
            self.eol_loss_weight = 0.0
            self.hi_loss_weight = 0.0


def train_ablation_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    dataset_name: str,
    ablation_config: AblationConfig,
    results_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train a single ablation experiment.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        y_test_true: True RUL at EOL for test engines
        feature_cols: Feature column names
        dataset_name: Dataset name (e.g., "FD002")
        ablation_config: Ablation configuration
        results_dir: Directory to save results
        device: PyTorch device
    
    Returns:
        Summary dictionary with metrics
    """
    import random
    import torch
    
    # Set random seeds
    random.seed(ablation_config.random_seed)
    np.random.seed(ablation_config.random_seed)
    torch.manual_seed(ablation_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ablation_config.random_seed)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "variant": ablation_config.variant,
        "horizon": ablation_config.horizon,
        "eol_loss_weight": ablation_config.eol_loss_weight,
        "use_residuals": ablation_config.use_residuals,
        "fusion_mode": ablation_config.fusion_mode,
        "d_model": ablation_config.d_model,
        "num_layers": ablation_config.num_layers,
        "nhead": ablation_config.nhead,
        "dim_feedforward": ablation_config.dim_feedforward,
        "dropout": ablation_config.dropout,
        "kernel_sizes": ablation_config.kernel_sizes,
        "seq_encoder_type": ablation_config.seq_encoder_type,
        "decoder_num_layers": ablation_config.decoder_num_layers,
        "batch_size": ablation_config.batch_size,
        "num_epochs": ablation_config.num_epochs,
        "lr": ablation_config.lr,
        "weight_decay": ablation_config.weight_decay,
        "patience": ablation_config.patience,
        "engine_train_ratio": ablation_config.engine_train_ratio,
        "random_seed": ablation_config.random_seed,
        "max_rul": ablation_config.max_rul,
        "past_len": ablation_config.past_len,
        "traj_loss_weight": ablation_config.traj_loss_weight,
        "hi_loss_weight": ablation_config.hi_loss_weight,
        "mono_late_weight": ablation_config.mono_late_weight,
        "mono_global_weight": ablation_config.mono_global_weight,
    }
    
    with open(results_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training Ablation Experiment: {ablation_config.variant}")
    print(f"  Horizon: {ablation_config.horizon}")
    print(f"  EOL Weight: {ablation_config.eol_loss_weight}")
    print(f"  Residuals: {ablation_config.use_residuals}")
    print(f"  Fusion: {ablation_config.fusion_mode}")
    print(f"{'='*80}")
    
    # Ensure RUL is capped in training DataFrame (same as world_model_training_v3.py)
    if "RUL" in df_train.columns:
        df_train = df_train.copy()
        df_train["RUL"] = np.minimum(df_train["RUL"], ablation_config.max_rul)
    
    # Build dataset - use build_world_model_dataset_with_cond_ids (same as world_model_training_v3.py)
    print("\n[1] Building world model dataset...")
    X_train, Y_train, cond_ids_train = build_world_model_dataset_with_cond_ids(
        df=df_train,
        feature_cols=feature_cols,
        target_col="RUL",
        past_len=ablation_config.past_len,
        horizon=ablation_config.horizon,
        unit_col="UnitNumber",
        cond_col="ConditionID",
    )
    
    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Input shape: {X_train.shape}, Target shape: {Y_train.shape}")
    
    # Determine number of conditions
    unique_conditions = torch.unique(cond_ids_train).cpu().numpy()
    num_conditions = len(unique_conditions)
    print(f"  Found {num_conditions} unique conditions: {unique_conditions}")
    
    # Engine-based split (random split on samples, not engines - simpler for ablation)
    # Note: For proper engine-based split, we'd need unit_ids, but for ablation this is acceptable
    n_total = len(X_train)
    n_val = int((1 - ablation_config.engine_train_ratio) * n_total)
    n_train = n_total - n_val
    
    # Use random seed for reproducibility
    import random
    import torch
    generator = torch.Generator().manual_seed(ablation_config.random_seed)
    indices = torch.randperm(n_total, generator=generator)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_split = X_train[train_indices]
    Y_train_split = Y_train[train_indices]
    cond_ids_train_split = cond_ids_train[train_indices]
    
    X_val = X_train[val_indices]
    Y_val = Y_train[val_indices]
    cond_ids_val = cond_ids_train[val_indices]
    
    print(f"  Train samples: {len(X_train_split)}, Val samples: {len(X_val)}")
    
    # Condition-wise scaling
    print("\n[2] Applying condition-wise feature scaling...")
    scaler_dict = {}
    X_train_scaled = X_train_split.clone()
    X_val_scaled = X_val.clone()
    
    X_train_np = X_train_split.numpy()
    X_val_np = X_val.numpy()
    cond_ids_train_np = cond_ids_train_split.numpy()
    cond_ids_val_np = cond_ids_val.numpy()
    
    for cond in unique_conditions:
        cond = int(cond)
        train_mask_cond = (cond_ids_train_np == cond)
        val_mask_cond = (cond_ids_val_np == cond)
        
        scaler = StandardScaler()
        X_train_cond_flat = X_train_np[train_mask_cond].reshape(-1, X_train_np.shape[-1])
        scaler.fit(X_train_cond_flat)
        scaler_dict[cond] = scaler
        
        if train_mask_cond.any():
            X_train_scaled[train_mask_cond] = torch.tensor(
                scaler.transform(X_train_cond_flat).reshape(-1, ablation_config.past_len, len(feature_cols)),
                dtype=torch.float32
            )
        
        if val_mask_cond.any():
            X_val_cond_flat = X_val_np[val_mask_cond].reshape(-1, X_val_np.shape[-1])
            X_val_scaled[val_mask_cond] = torch.tensor(
                scaler.transform(X_val_cond_flat).reshape(-1, ablation_config.past_len, len(feature_cols)),
                dtype=torch.float32
            )
    
    # Save scaler
    import pickle
    with open(results_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_dict, f)
    
    # Create dataloaders
    print("\n[3] Creating dataloaders...")
    train_dataset = TensorDataset(X_train_scaled, Y_train_split, cond_ids_train_split)
    val_dataset = TensorDataset(X_val_scaled, Y_val, cond_ids_val)
    
    train_loader = DataLoader(train_dataset, batch_size=ablation_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ablation_config.batch_size, shuffle=False)
    
    # Initialize model
    print("\n[4] Initializing model...")
    model = create_world_model_v3_variant(
        variant=ablation_config.variant,
        input_size=len(feature_cols),
        d_model=ablation_config.d_model,
        num_layers=ablation_config.num_layers,
        nhead=ablation_config.nhead,
        dim_feedforward=ablation_config.dim_feedforward,
        dropout=ablation_config.dropout,
        num_conditions=num_conditions if num_conditions > 1 else None,
        cond_emb_dim=4,
        kernel_sizes=ablation_config.kernel_sizes,
        seq_encoder_type=ablation_config.seq_encoder_type,
        use_layer_norm=True,
        max_seq_len=300,
        decoder_num_layers=ablation_config.decoder_num_layers,
        horizon=ablation_config.horizon,
        fusion_mode=ablation_config.fusion_mode,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Variant: {ablation_config.variant}")
    
    # Training loop
    print("\n[5] Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=ablation_config.lr, weight_decay=ablation_config.weight_decay)
    mse_loss = nn.MSELoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_traj_loss": [],
        "val_eol_loss": [],
        "val_hi_loss": [],
        "val_mono_loss": [],
    }
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = results_dir / "best_model.pt"
    
    # Trajectory step weights (if needed)
    traj_step_weights = None
    if ablation_config.traj_loss_weight > 0:
        traj_step_weights = compute_trajectory_step_weights(ablation_config.horizon)
        traj_step_weights = traj_step_weights.to(device)
    
    for epoch in range(ablation_config.num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        n_train_samples = 0
        
        for X_batch, Y_batch, cond_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                teacher_forcing_ratio=0.5,
                horizon=ablation_config.horizon,
                cond_ids=cond_batch if num_conditions > 1 else None,
            )
            
            traj_pred = outputs["traj"]  # (B, H, 1)
            eol_pred = outputs["eol"].squeeze(-1)  # (B,)
            hi_pred = outputs["hi"].squeeze(-1)  # (B,)
            
            # Targets
            target_traj = Y_batch  # (B, H, 1)
            target_eol = Y_batch[:, 0, 0]  # (B,)
            target_hi = torch.clamp(1.0 - (target_eol / ablation_config.max_rul), 0.0, 1.0)  # (B,)
            
            # Compute losses (only if corresponding head exists)
            loss_traj = torch.tensor(0.0, device=device)
            loss_eol = torch.tensor(0.0, device=device)
            loss_hi = torch.tensor(0.0, device=device)
            loss_mono = torch.tensor(0.0, device=device)
            
            # Trajectory loss
            if ablation_config.traj_loss_weight > 0 and traj_pred.numel() > 0:
                if traj_step_weights is not None:
                    squared_errors = (traj_pred - target_traj) ** 2
                    weighted_errors = squared_errors.squeeze(-1) * traj_step_weights.unsqueeze(0)
                    loss_traj = weighted_errors.mean()
                else:
                    loss_traj = mse_loss(traj_pred, target_traj)
            
            # EOL loss
            if ablation_config.eol_loss_weight > 0:
                loss_eol = mse_loss(eol_pred, target_eol)
            
            # HI loss
            if ablation_config.hi_loss_weight > 0:
                loss_hi = mse_loss(hi_pred, target_hi)
            
            # Monotonicity loss (only if trajectory and HI exist)
            if (ablation_config.mono_late_weight > 0 or ablation_config.mono_global_weight > 0) and \
               ablation_config.traj_loss_weight > 0 and ablation_config.hi_loss_weight > 0:
                hi_traj = traj_pred.squeeze(-1)  # (B, H)
                rul_traj = target_traj.squeeze(-1)  # (B, H)
                
                mono_late_raw, _ = compute_monotonicity_loss(
                    pred_hi=hi_traj,
                    rul=rul_traj,
                    beta=60.0,
                    weight=1.0,
                )
                mono_late = ablation_config.mono_late_weight * mono_late_raw
                
                mono_global_raw = compute_global_trend_loss(hi_traj)
                mono_global = ablation_config.mono_global_weight * mono_global_raw
                
                loss_mono = mono_late + mono_global
            
            # Weighted total loss
            total_loss = (
                ablation_config.traj_loss_weight * loss_traj +
                ablation_config.eol_loss_weight * loss_eol +
                ablation_config.hi_loss_weight * loss_hi +
                loss_mono
            )
            
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += total_loss.item()
            n_train_samples += X_batch.size(0)
        
        avg_train_loss = running_train_loss / n_train_samples if n_train_samples > 0 else 0.0
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_traj = 0.0
        running_val_eol = 0.0
        running_val_hi = 0.0
        running_val_mono = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for X_batch, Y_batch, cond_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                cond_batch = cond_batch.to(device)
                
                outputs = model(
                    encoder_inputs=X_batch,
                    decoder_targets=Y_batch,
                    teacher_forcing_ratio=0.0,
                    horizon=ablation_config.horizon,
                    cond_ids=cond_batch if num_conditions > 1 else None,
                )
                
                traj_pred = outputs["traj"]
                eol_pred = outputs["eol"].squeeze(-1)
                hi_pred = outputs["hi"].squeeze(-1)
                
                target_traj = Y_batch
                target_eol = Y_batch[:, 0, 0]
                target_hi = torch.clamp(1.0 - (target_eol / ablation_config.max_rul), 0.0, 1.0)
                
                val_traj = torch.tensor(0.0, device=device)
                val_eol = torch.tensor(0.0, device=device)
                val_hi = torch.tensor(0.0, device=device)
                val_mono = torch.tensor(0.0, device=device)
                
                if ablation_config.traj_loss_weight > 0 and traj_pred.numel() > 0:
                    val_traj = mse_loss(traj_pred, target_traj)
                
                if ablation_config.eol_loss_weight > 0:
                    val_eol = mse_loss(eol_pred, target_eol)
                
                if ablation_config.hi_loss_weight > 0:
                    val_hi = mse_loss(hi_pred, target_hi)
                
                if (ablation_config.mono_late_weight > 0 or ablation_config.mono_global_weight > 0) and \
                   ablation_config.traj_loss_weight > 0 and ablation_config.hi_loss_weight > 0:
                    hi_traj = traj_pred.squeeze(-1)
                    rul_traj = target_traj.squeeze(-1)
                    mono_late_raw, _ = compute_monotonicity_loss(pred_hi=hi_traj, rul=rul_traj, beta=60.0, weight=1.0)
                    mono_late = ablation_config.mono_late_weight * mono_late_raw
                    mono_global_raw = compute_global_trend_loss(hi_traj)
                    mono_global = ablation_config.mono_global_weight * mono_global_raw
                    val_mono = mono_late + mono_global
                
                val_loss = (
                    ablation_config.traj_loss_weight * val_traj +
                    ablation_config.eol_loss_weight * val_eol +
                    ablation_config.hi_loss_weight * val_hi +
                    val_mono
                )
                
                running_val_loss += val_loss.item()
                running_val_traj += val_traj.item()
                running_val_eol += val_eol.item()
                running_val_hi += val_hi.item()
                running_val_mono += val_mono.item()
                n_val_samples += X_batch.size(0)
        
        avg_val_loss = running_val_loss / n_val_samples if n_val_samples > 0 else 0.0
        avg_val_traj = running_val_traj / n_val_samples if n_val_samples > 0 else 0.0
        avg_val_eol = running_val_eol / n_val_samples if n_val_samples > 0 else 0.0
        avg_val_hi = running_val_hi / n_val_samples if n_val_samples > 0 else 0.0
        avg_val_mono = running_val_mono / n_val_samples if n_val_samples > 0 else 0.0
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_traj_loss"].append(avg_val_traj)
        history["val_eol_loss"].append(avg_val_eol)
        history["val_hi_loss"].append(avg_val_hi)
        history["val_mono_loss"].append(avg_val_mono)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{ablation_config.num_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"val_eol={avg_val_eol:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": config_dict,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= ablation_config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val_loss={checkpoint['val_loss']:.4f}")
    
    # Evaluate on test set
    print("\n[6] Evaluating on test set...")
    test_metrics = evaluate_ablation_model(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler_dict=scaler_dict,
        past_len=ablation_config.past_len,
        max_rul=ablation_config.max_rul,
        num_conditions=num_conditions,
        horizon=ablation_config.horizon,
        device=device,
    )
    
    # Save results
    summary = {
        "experiment_name": results_dir.name,
        "dataset": dataset_name,
        "variant": ablation_config.variant,
        "horizon": ablation_config.horizon,
        "eol_loss_weight": ablation_config.eol_loss_weight,
        "use_residuals": ablation_config.use_residuals,
        "fusion_mode": ablation_config.fusion_mode,
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
        "best_epoch": checkpoint["epoch"] + 1,
        "history": history,
    }
    
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save training curves
    plot_training_curves(history, results_dir / "training_curves.png")
    
    return summary


def evaluate_ablation_model(
    model: nn.Module,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    scaler_dict: Dict[int, StandardScaler],
    past_len: int,
    max_rul: int,
    num_conditions: int,
    horizon: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate ablation model on test set."""
    model.eval()
    
    y_pred_all = []
    y_true_all = []
    unit_ids_list = []
    
    def _build_eol_input_for_unit(df_unit: pd.DataFrame, feature_cols: List[str], past_len: int) -> np.ndarray:
        df_unit = df_unit.sort_values("TimeInCycles")
        feats = df_unit[feature_cols].values.astype(np.float32)
        
        if len(feats) < past_len:
            padding = np.tile(feats[0:1], (past_len - len(feats), 1))
            feats = np.vstack([padding, feats])
        else:
            feats = feats[-past_len:]
        
        return feats
    
    unit_id_to_idx = {i + 1: i for i in range(len(y_test_true))}
    
    with torch.no_grad():
        for unit_id, df_unit in df_test.groupby("UnitNumber"):
            unit_id = int(unit_id)
            
            X_past_np = _build_eol_input_for_unit(df_unit, feature_cols, past_len)
            cond_id = int(df_unit["ConditionID"].iloc[0])
            
            scaler = scaler_dict.get(cond_id, scaler_dict.get(0))
            X_past_scaled = scaler.transform(X_past_np.reshape(-1, len(feature_cols))).reshape(past_len, len(feature_cols))
            
            X_past = torch.tensor(X_past_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            cond_ids = torch.tensor([cond_id], dtype=torch.long).to(device) if num_conditions > 1 else None
            
            outputs = model(
                encoder_inputs=X_past,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
                cond_ids=cond_ids,
            )
            
            pred_rul = float(outputs["eol"][0, 0].cpu().item())
            pred_rul = np.clip(pred_rul, 0.0, max_rul)
            
            y_pred_all.append(pred_rul)
            unit_ids_list.append(unit_id)
            idx = unit_id_to_idx.get(unit_id, unit_id - 1)
            if idx < len(y_test_true):
                y_true_all.append(y_test_true[idx])
            else:
                y_true_all.append(y_test_true[-1])
    
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    
    # Compute metrics
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    nasa_stats = compute_eol_errors_and_nasa(y_true, y_pred, max_rul=max_rul)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "nasa_sum": nasa_stats["nasa_sum"],
        "nasa_mean": nasa_stats["nasa_mean"],
        "num_engines": len(y_true),
    }


def plot_training_curves(history: Dict[str, List[float]], out_path: Path):
    """Plot training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].plot(epochs, history["val_traj_loss"], label="Val Traj", color="green")
    axes[1].plot(epochs, history["val_eol_loss"], label="Val EOL", color="orange")
    axes[1].plot(epochs, history["val_hi_loss"], label="Val HI", color="purple")
    axes[1].plot(epochs, history["val_mono_loss"], label="Val Mono", color="brown")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Component Losses")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

