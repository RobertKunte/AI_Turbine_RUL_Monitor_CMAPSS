"""
Full-Trajectory LSTM for EOL prediction.

This module provides a LSTM-based regressor that processes complete engine
trajectories (not just tail samples) for Remaining Useful Life (RUL) prediction.
Includes NASA PHM08-style evaluation metrics.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import centralized NASA functions from metrics module
from src.metrics import nasa_phm_score, nasa_phm_score_single
from src.additional_features import group_feature_columns
from src.feature_safety import check_feature_dimensions


def log_damage_head_params(model: nn.Module, logger=print) -> int:
    """
    Log all parameters that belong to the damage head:
    - name
    - shape
    - requires_grad flag
    - total number of parameters in damage head

    Matching is done via 'damage' in the parameter name (case-insensitive).
    """
    total = 0
    logger("[DEBUG DamageHead] Listing damage-head parameters:")
    for name, p in model.named_parameters():
        if "damage" in name.lower():
            num = p.numel()
            total += num
            logger(
                f"  {name}: shape={tuple(p.shape)}, "
                f"requires_grad={p.requires_grad}, numel={num}"
            )
    logger(f"[DEBUG DamageHead] Total damage-head parameters: {total}")
    return total


def log_damage_head_gradients(model: nn.Module, logger=print, prefix: str = "[DEBUG DamageHeadGrad]"):
    """
    Log simple gradient statistics for damage-head parameters:
    - mean absolute gradient
    Only logs parameters that have non-None .grad.
    """
    grads = []
    num_params_with_grad = 0
    for name, p in model.named_parameters():
        if "damage" in name.lower() and p.grad is not None:
            g = p.grad.detach()
            num_params_with_grad += 1
            grads.append(g.abs().mean().item())

    if not grads:
        logger(f"{prefix} No gradients found for damage-head parameters!")
    else:
        mean_grad = float(sum(grads) / len(grads))
        logger(
            f"{prefix} num_params_with_grad={num_params_with_grad}, "
            f"mean_abs_grad={mean_grad:.4e}"
        )
    return grads


def build_full_eol_sequences_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    past_len: int = 30,
    max_rul: int = 125,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    rul_col: str = "RUL",
    cond_col: str = "ConditionID",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Baut Sliding-Window-Sequenzen über die KOMPLETTE Lebensdauer jeder Engine.

    Für jede Engine:
      - sortiere nach cycle_col
      - laufe t = past_len .. T
      - X_seq = Features[t-past_len:t, :]
      - y_seq = min(RUL_t, max_rul)
      - health_phys_seq = HI_phys_v2[t-past_len:t] (if available)

    Args:
        df: DataFrame mit mindestens feature_cols, unit_col, cycle_col, rul_col
        feature_cols: Liste der Feature-Spalten
        past_len: Länge des Vergangenheitsfensters
        max_rul: Maximum RUL (wird gecappt, Default: 125)
        unit_col: Name der Unit/Engine-Spalte
        cycle_col: Name der Cycle/Time-Spalte
        rul_col: Name der RUL-Spalte
        cond_col: Name der Condition-Spalte

    Returns:
        X: FloatTensor [N, past_len, num_features] - Input-Sequenzen
        y: FloatTensor [N] - RUL-Targets (gecappt auf max_rul)
        unit_ids: IntTensor [N] - Unit-IDs für jedes Sample
        cond_ids: IntTensor [N] - Condition-IDs für jedes Sample
        health_phys_seq: Optional[FloatTensor [N, past_len]] - HI_phys_v2 sequences (None if not available)
    """
    X_list = []
    y_list = []
    unit_id_list = []
    cond_id_list = []
    health_phys_seq_list = []

    unit_ids = df[unit_col].unique()
    
    # Check which HI_phys column is available (prefer v3 over v2)
    hi_phys_col: Optional[str] = None
    if "HI_phys_v3" in df.columns:
        hi_phys_col = "HI_phys_v3"
    elif "HI_phys_v2" in df.columns:
        hi_phys_col = "HI_phys_v2"

    print("============================================================")
    print("[build_full_eol_sequences_from_df] Summary")
    print("============================================================")
    print(f"Num units: {len(unit_ids)}")
    print(f"Using past_len={past_len}, max_rul={max_rul}")
    print(f"Num feature cols: {len(feature_cols)}")
    # High-level feature grouping for encoder/decoder design and debugging.
    try:
        groups = group_feature_columns(feature_cols)
        print(
            "[Train] Feature groups: "
            f"total={len(feature_cols)}, "
            f"raw={len(groups['raw'])}, "
            f"ms={len(groups['ms'])}, "
            f"residual={len(groups['residual'])}, "
            f"cond={len(groups['cond'])}, "
            f"twin={len(groups['twin'])}"
        )
    except Exception as e:
        # Grouping is purely diagnostic; never break the pipeline.
        print(f"Warning: could not compute feature groups: {e}")

    for uid in unit_ids:
        df_u = (
            df[df[unit_col] == uid]
            .sort_values(cycle_col)
            .reset_index(drop=True)
        )

        if len(df_u) < past_len:
            continue

        # ConditionID is constant per unit; take any row
        cond_id = int(df_u[cond_col].iloc[0]) if cond_col in df_u.columns else 0

        values = df_u[feature_cols].to_numpy(dtype=np.float32)
        rul_values = df_u[rul_col].to_numpy(dtype=np.float32)
        
        # Extract HI_phys (v3 preferred, else v2) if available
        if hi_phys_col is not None and hi_phys_col in df_u.columns:
            hi_phys_values = df_u[hi_phys_col].to_numpy(dtype=np.float32)
        else:
            hi_phys_values = None

        # Sliding window über den gesamten Lebenslauf
        for i in range(past_len - 1, len(df_u)):
            window = values[i - past_len + 1 : i + 1]  # [past_len, num_features]
            rul_i = rul_values[i]
            rul_capped = min(rul_i, max_rul)  # Cap RUL at max_rul

            X_list.append(window)
            y_list.append(rul_capped)
            unit_id_list.append(uid)
            cond_id_list.append(cond_id)
            
            # Extract HI_phys sequence for this window (v3 preferred, else v2)
            if hi_phys_values is not None:
                hi_phys_window = hi_phys_values[i - past_len + 1 : i + 1]  # [past_len]
                health_phys_seq_list.append(hi_phys_window)

    if len(X_list) == 0:
        raise ValueError(
            "[build_full_eol_sequences_from_df] No samples built – "
            "check past_len and data."
        )

    X = torch.from_numpy(np.stack(X_list))  # [N, past_len, num_features]
    y = torch.from_numpy(np.array(y_list, dtype=np.float32))  # [N]
    unit_ids_tensor = torch.from_numpy(np.array(unit_id_list, dtype=np.int32))  # [N]
    cond_ids_tensor = torch.from_numpy(np.array(cond_id_list, dtype=np.int64))  # [N]
    
    # Build health_phys_seq if available
    if health_phys_seq_list and len(health_phys_seq_list) > 0:
        health_phys_seq = torch.from_numpy(np.stack(health_phys_seq_list))  # [N, past_len]
        print(f"X shape: {X.shape}, y shape: {y.shape}, unit_ids shape: {unit_ids_tensor.shape}, cond_ids shape: {cond_ids_tensor.shape}, health_phys_seq shape: {health_phys_seq.shape}")
    else:
        health_phys_seq = None
        print(f"X shape: {X.shape}, y shape: {y.shape}, unit_ids shape: {unit_ids_tensor.shape}, cond_ids shape: {cond_ids_tensor.shape}, health_phys_seq: None")

    return X, y, unit_ids_tensor, cond_ids_tensor, health_phys_seq
    print(
        f"RUL stats (capped at {max_rul}): min={y.min().item():.2f}, "
        f"max={y.max().item():.2f}, mean={y.mean().item():.2f}, "
        f"std={y.std().item():.2f}"
    )
    if cond_col in df.columns:
        unique_conds = torch.unique(cond_ids_tensor)
        print(f"ConditionIDs: {unique_conds.tolist()}")
    print("============================================================")

    return X, y, unit_ids_tensor, cond_ids_tensor


class SequenceDatasetWithUnits:
    """Dataset that includes unit_ids and condition_ids for each sample."""
    def __init__(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        unit_ids: torch.Tensor, 
        cond_ids: torch.Tensor = None,
        health_phys_seq: Optional[torch.Tensor] = None,
    ):
        self.X = X
        self.y = y
        self.unit_ids = unit_ids
        self.cond_ids = cond_ids if cond_ids is not None else torch.zeros(len(unit_ids), dtype=torch.int64)
        self.health_phys_seq = health_phys_seq

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        items = [self.X[idx], self.y[idx], self.unit_ids[idx]]
        if self.cond_ids is not None:
            items.append(self.cond_ids[idx])
        if self.health_phys_seq is not None:
            items.append(self.health_phys_seq[idx])
        return tuple(items)


def build_test_sequences_from_df(
    df_test: pd.DataFrame,
    feature_cols: list[str],
    past_len: int = 30,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    cond_col: str = "ConditionID",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Baut Sequenzen für Test-Daten (nur letzte Sequenz pro Engine).
    
    Test-Daten enthalten nur die letzten Zyklen jeder Engine.
    Wir nehmen nur die letzte Sequenz (past_len Zyklen) pro Engine.
    
    Args:
        df_test: Test DataFrame (ohne RUL-Spalte)
        feature_cols: Liste der Feature-Spalten
        past_len: Länge des Vergangenheitsfensters
        unit_col: Name der Unit/Engine-Spalte
        cycle_col: Name der Cycle/Time-Spalte
        
    Returns:
        X: FloatTensor [N, past_len, num_features] - Input-Sequenzen
        unit_ids: IntTensor [N] - Unit-IDs für jedes Sample
        cond_ids: IntTensor [N] - Condition-IDs für jedes Sample
    """
    X_list = []
    unit_id_list = []
    cond_id_list = []
    
    # Sort unit_ids to ensure consistent ordering (matches y_test_true order from load_cmapps_subset)
    unit_ids = np.sort(df_test[unit_col].unique())
    
    # Filter feature_cols to only include columns that exist in test DataFrame
    available_feature_cols = [col for col in feature_cols if col in df_test.columns]
    missing_cols = [col for col in feature_cols if col not in df_test.columns]
    
    if missing_cols:
        print(f"[WARNING] {len(missing_cols)} feature columns not found in test data: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
        print(f"[WARNING] Using {len(available_feature_cols)} available features instead of {len(feature_cols)}")
    
    if len(available_feature_cols) == 0:
        raise ValueError(
            "[build_test_sequences_from_df] No feature columns available in test DataFrame. "
            f"Requested: {len(feature_cols)}, Available: {len(available_feature_cols)}"
        )
    
    print("============================================================")
    print("[build_test_sequences_from_df] Summary")
    print("============================================================")
    print(f"Num test units: {len(unit_ids)}")
    print(f"Using past_len={past_len}")
    print(f"Num feature cols requested: {len(feature_cols)}")
    print(f"Num feature cols available: {len(available_feature_cols)}")
    
    # Use only available features
    feature_cols = available_feature_cols
    
    for uid in unit_ids:
        df_u = (
            df_test[df_test[unit_col] == uid]
            .sort_values(cycle_col)
            .reset_index(drop=True)
        )
        
        # ConditionID is constant per unit; take any row
        cond_id = int(df_u[cond_col].iloc[0]) if cond_col in df_u.columns else 0
        
        if len(df_u) < past_len:
            # Wenn Engine zu kurz ist, verwende alle verfügbaren Zyklen
            # und padde mit dem letzten Wert
            values = df_u[feature_cols].to_numpy(dtype=np.float32)
            if len(values) > 0:
                # Pad with last value
                padding = np.repeat(values[-1:], past_len - len(values), axis=0)
                window = np.vstack([padding, values])
            else:
                continue
        else:
            # Nimm die letzten past_len Zyklen
            values = df_u[feature_cols].to_numpy(dtype=np.float32)
            window = values[-past_len:]  # [past_len, num_features]
        
        X_list.append(window)
        unit_id_list.append(uid)
        cond_id_list.append(cond_id)
    
    if len(X_list) == 0:
        raise ValueError(
            "[build_test_sequences_from_df] No samples built – "
            "check past_len and test data."
        )
    
    X = torch.from_numpy(np.stack(X_list))  # [N, past_len, num_features]
    unit_ids_tensor = torch.from_numpy(np.array(unit_id_list, dtype=np.int32))  # [N]
    cond_ids_tensor = torch.from_numpy(np.array(cond_id_list, dtype=np.int64))  # [N]

    print(f"X shape: {X.shape}, unit_ids shape: {unit_ids_tensor.shape}, cond_ids shape: {cond_ids_tensor.shape}")
    if cond_col in df_test.columns:
        unique_conds = torch.unique(cond_ids_tensor)
        print(f"ConditionIDs: {unique_conds.tolist()}")
    print("============================================================")

    return X, unit_ids_tensor, cond_ids_tensor


def evaluate_on_test_data(
    model: nn.Module,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: list[str],
    scaler: Optional[StandardScaler | Dict[int, StandardScaler]],
    past_len: int = 30,
    max_rul: int = 125,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates model on test data with true RUL from RUL_FD00x.txt.
    
    Args:
        model: Trained EOLFullLSTM model
        df_test: Test DataFrame (from test_FD00x.txt)
        y_test_true: True RUL values (from RUL_FD00x.txt), one per engine
        feature_cols: List of feature columns
        scaler: Optional StandardScaler or Dict[cond_id, StandardScaler] (fitted on training data)
        past_len: Sequence length
        max_rul: Maximum RUL for capping
        unit_col: Name of unit column
        cycle_col: Name of cycle column
        device: torch.device
        
    Returns:
        Dictionary with test metrics (pointwise, EOL, NASA)
    """
    model.eval()
    model.to(device)
    
    # Filter feature_cols to only include columns that exist in test DataFrame
    # This ensures we don't try to use features that were computed during training
    # but are not available in test data
    available_feature_cols = [col for col in feature_cols if col in df_test.columns]
    missing_cols = [col for col in feature_cols if col not in df_test.columns]
    
    if missing_cols:
        print(f"[WARNING] {len(missing_cols)} feature columns not found in test data: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
        print(f"[WARNING] Using {len(available_feature_cols)} available features instead of {len(feature_cols)}")
        print(f"[WARNING] Missing columns: {missing_cols}")
    
    if len(available_feature_cols) == 0:
        raise ValueError(
            "[evaluate_on_test_data] No feature columns available in test DataFrame. "
            f"Requested: {len(feature_cols)}, Available: {len(available_feature_cols)}"
        )
    
    # Hard safety check: scaler + model vs. feature dimensionality
    # This raises early if diagnostics / inference accidentally use a different
    # feature configuration than the training run (e.g. 244 vs. 295 features).
    check_feature_dimensions(
        feature_cols=available_feature_cols,
        scaler=scaler,
        model=model,
        context="evaluate_on_test_data",
    )
    
    # Build test sequences (one per engine, last past_len cycles)
    X_test, unit_ids_test, cond_ids_test = build_test_sequences_from_df(
        df_test,
        feature_cols=available_feature_cols,  # Use filtered feature list
        past_len=past_len,
        unit_col=unit_col,
        cycle_col=cycle_col,
    )
    
    # Apply scaling if available
    if scaler is not None:
        X_test_np = X_test.numpy()
        B, T, F = X_test_np.shape
        
        if isinstance(scaler, dict):
            # Condition-wise scaling
            X_test_scaled_list = []
            for i in range(B):
                cond_id = int(cond_ids_test[i])
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
        else:
            # Global scaling
            X_test_flat = X_test_np.reshape(-1, F)
            X_test_scaled = scaler.transform(X_test_flat)
            X_test = torch.from_numpy(X_test_scaled.reshape(B, T, F)).float()
    
    # Predict
    from src.eval_utils import forward_rul_only
    
    # Check if model uses condition embeddings
    use_cond_emb = getattr(model, 'use_condition_embedding', False)
    
    with torch.no_grad():
        X_test = X_test.to(device)
        cond_ids_test_tensor = cond_ids_test.to(device) if use_cond_emb and cond_ids_test is not None else None
        # Use helper to extract RUL predictions (handles both single-task and multi-task)
        y_test_pred = forward_rul_only(model, X_test, cond_ids=cond_ids_test_tensor).cpu().numpy()
    
    # Cap predictions at max_rul
    y_test_pred = np.minimum(y_test_pred, max_rul)
    y_test_pred = np.maximum(y_test_pred, 0.0)  # Ensure non-negative
    
    # Cap true RUL at max_rul for consistency
    y_test_true_capped = np.minimum(y_test_true, max_rul)
    
    # Use shared metrics function for consistency with diagnostics
    # This ensures both evaluation and diagnostics use exactly the same formulas
    # Note: Values are already capped above, so pass max_rul=None to avoid double-capping
    from src.metrics import compute_eol_errors_and_nasa
    eol_metrics_dict = compute_eol_errors_and_nasa(
        y_true_eol=y_test_true_capped,
        y_pred_eol=y_test_pred,
        max_rul=None,  # Already capped above, so no need to cap again
    )
    
    # Extract metrics from shared function
    mse = eol_metrics_dict["mse"]
    rmse = eol_metrics_dict["rmse"]
    mae = eol_metrics_dict["mae"]
    bias = eol_metrics_dict["bias"]
    r2 = eol_metrics_dict["r2"]
    nasa_score_sum = eol_metrics_dict["nasa_sum"]
    nasa_score_mean = eol_metrics_dict["nasa_mean"]
    
    # EOL metrics (same as pointwise for test data - one sample per engine)
    rmse_eol = rmse
    mae_eol = mae
    bias_eol = bias
    
    print("============================================================")
    print("[evaluate_on_test_data] Test Metrics")
    print("============================================================")
    print(f"Num test engines: {len(y_test_true_capped)}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f} cycles")
    print(f"MAE: {mae:.4f} cycles")
    print(f"Bias: {bias:.4f} cycles")
    print(f"R²: {r2:.4f}")
    print(f"NASA Score (sum): {nasa_score_sum:.2f}")
    print(f"NASA Score (mean): {nasa_score_mean:.4f}")
    print("============================================================")
    
    return {
        "pointwise": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r2": r2,
        },
        "eol": {
            "rmse": rmse_eol,
            "mae": mae_eol,
            "bias": bias_eol,
            "nasa_score_sum": nasa_score_sum,
            "nasa_score_mean": nasa_score_mean,
            "num_engines": len(y_test_true_capped),
        },
        "nasa_pointwise": {  # Für Konsistenz mit evaluate_eol_full_lstm
            "score_sum": nasa_score_sum,
            "score_mean": nasa_score_mean,
        },
        "nasa_test": {  # Alias für Rückwärtskompatibilität
            "score_sum": nasa_score_sum,
            "score_mean": nasa_score_mean,
        },
        "y_true": y_test_true_capped,
        "y_pred": y_test_pred,
        "unit_ids": unit_ids_test.numpy(),
    }


def create_full_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    unit_ids: torch.Tensor,
    cond_ids: torch.Tensor = None,
    health_phys_seq: Optional[torch.Tensor] = None,
    batch_size: int = 256,
    engine_train_ratio: float = 0.8,
    shuffle_engines: bool = True,
    random_seed: int = 42,
    use_condition_wise_scaling: bool = True,
) -> Tuple[DataLoader, DataLoader, StandardScaler | Dict[int, StandardScaler], torch.Tensor, torch.Tensor]:
    """
    Erstellt Train- und Validation-Dataloader mit engine-basiertem Split.

    - Split auf ENGINE-Ebene: zufällige 80% der Engines -> train, rest -> val
    - Feature-Scaling NUR aus Train (StandardScaler)
    - Liefert train_loader, val_loader, scaler, train_unit_ids, val_unit_ids

    Args:
        X: Input-Sequenzen, shape (N, past_len, F)
        y: RUL-Targets, shape (N,)
        unit_ids: Unit-IDs für jedes Sample, shape (N,)
        cond_ids: Condition-IDs für jedes Sample, shape (N,). If None, defaults to zeros.
        batch_size: Batch-Größe
        engine_train_ratio: Anteil der Engines für Training (Default: 0.8)
        shuffle_engines: Ob Engines zufällig gemischt werden sollen
        random_seed: Random Seed für Reproduzierbarkeit

    Returns:
        train_loader: DataLoader für Training
        val_loader: DataLoader für Validation
        scaler: Fitted StandardScaler (nur auf Train-Daten) oder Dict[cond_id, StandardScaler] bei condition-wise scaling
        train_unit_ids: Unit-IDs der Train-Engines
        val_unit_ids: Unit-IDs der Val-Engines
    """
    from src.config import USE_CONDITION_WISE_SCALING
    
    # Use config default if not explicitly provided
    if use_condition_wise_scaling is None:
        use_condition_wise_scaling = USE_CONDITION_WISE_SCALING
    
    # Default cond_ids to zeros if not provided (backward compatibility)
    if cond_ids is None:
        cond_ids = torch.zeros(len(unit_ids), dtype=torch.int64)
    # Engine-basierter Split
    unique_units = torch.unique(unit_ids)
    n_units = len(unique_units)
    n_train_units = int(n_units * engine_train_ratio)
    if n_units > 1 and n_train_units == 0:
        n_train_units = 1
    n_val_units = n_units - n_train_units

    # Shuffle Engines
    if shuffle_engines:
        gen = torch.Generator().manual_seed(random_seed)
        perm = torch.randperm(n_units, generator=gen)
        train_unit_ids = unique_units[perm[:n_train_units]]
        val_unit_ids = unique_units[perm[n_train_units:]]
    else:
        train_unit_ids = unique_units[:n_train_units]
        val_unit_ids = unique_units[n_train_units:]

    # Masken für Train/Val
    train_mask = torch.isin(unit_ids, train_unit_ids)
    val_mask = torch.isin(unit_ids, val_unit_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    cond_ids_train = cond_ids[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    cond_ids_val = cond_ids[val_mask]
    
    # Split health_phys_seq if available
    health_phys_seq_train = health_phys_seq[train_mask] if health_phys_seq is not None else None
    health_phys_seq_val = health_phys_seq[val_mask] if health_phys_seq is not None else None

    # Feature-Scaling: Fit nur auf Train-Daten
    N_train, past_len, num_features = X_train.shape
    N_val = X_val.shape[0]
    
    if use_condition_wise_scaling:
        # Condition-wise scaling: separate scaler per condition
        unique_conds_train = torch.unique(cond_ids_train)
        scalers = {}  # Dict[cond_id, StandardScaler]
        
        # Fit scalers per condition on train data
        for cond_id in unique_conds_train:
            cond_mask_train = (cond_ids_train == cond_id)
            if not cond_mask_train.any():
                continue
            
            X_cond_train = X_train[cond_mask_train]  # [N_cond, past_len, num_features]
            X_cond_train_flat = X_cond_train.numpy().reshape(-1, num_features)
            
            scaler_cond = StandardScaler()
            scaler_cond.fit(X_cond_train_flat)
            scalers[int(cond_id)] = scaler_cond
        
        # Transform train data condition-wise
        X_train_scaled_list = []
        for i in range(N_train):
            cond_id = int(cond_ids_train[i])
            if cond_id in scalers:
                x_sample = X_train[i].numpy()  # [past_len, num_features]
                x_scaled = scalers[cond_id].transform(x_sample)
                X_train_scaled_list.append(torch.from_numpy(x_scaled))
            else:
                # Fallback: use first available scaler or identity
                if scalers:
                    x_sample = X_train[i].numpy()
                    first_scaler = list(scalers.values())[0]
                    x_scaled = first_scaler.transform(x_sample)
                    X_train_scaled_list.append(torch.from_numpy(x_scaled))
                else:
                    X_train_scaled_list.append(X_train[i])
        
        X_train_scaled = torch.stack(X_train_scaled_list)
        
        # Transform val data condition-wise
        X_val_scaled_list = []
        for i in range(N_val):
            cond_id = int(cond_ids_val[i])
            if cond_id in scalers:
                x_sample = X_val[i].numpy()  # [past_len, num_features]
                x_scaled = scalers[cond_id].transform(x_sample)
                X_val_scaled_list.append(torch.from_numpy(x_scaled))
            else:
                # Fallback: use first available scaler or identity
                if scalers:
                    x_sample = X_val[i].numpy()
                    first_scaler = list(scalers.values())[0]
                    x_scaled = first_scaler.transform(x_sample)
                    X_val_scaled_list.append(torch.from_numpy(x_scaled))
                else:
                    X_val_scaled_list.append(X_val[i])
        
        X_val_scaled = torch.stack(X_val_scaled_list)
        
        scaler = scalers  # Return dict of scalers
        scaling_info = f"Condition-wise StandardScaler ({len(scalers)} conditions)"
    else:
        # Global scaling: single scaler for all data
        X_train_flat = X_train.numpy().reshape(-1, num_features)
        scaler = StandardScaler()
        scaler.fit(X_train_flat)

        # Transform Train
        X_train_scaled = scaler.transform(X_train_flat)
        X_train_scaled = torch.from_numpy(X_train_scaled.reshape(N_train, past_len, num_features))

        # Transform Val (mit gleichem Scaler)
        X_val_flat = X_val.numpy().reshape(-1, num_features)
        X_val_scaled = scaler.transform(X_val_flat)
        X_val_scaled = torch.from_numpy(X_val_scaled.reshape(N_val, past_len, num_features))
        
        scaling_info = "Global StandardScaler (fitted on train only)"

    # Speichere unit_ids für Train/Val
    train_unit_ids_samples = unit_ids[train_mask]
    val_unit_ids_samples = unit_ids[val_mask]

    train_dataset = SequenceDatasetWithUnits(X_train_scaled, y_train, train_unit_ids_samples, cond_ids_train, health_phys_seq=health_phys_seq_train)
    val_dataset = SequenceDatasetWithUnits(X_val_scaled, y_val, val_unit_ids_samples, cond_ids_val, health_phys_seq=health_phys_seq_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("============================================================")
    print("[create_full_dataloaders] Engine-based split")
    print("============================================================")
    print(f"Total units: {n_units}")
    print(f"Train units: {n_train_units}, Val units: {n_val_units}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Feature scaling: {scaling_info}")
    if use_condition_wise_scaling and isinstance(scaler, dict):
        print(f"  Conditions: {sorted(scaler.keys())}")
    print("============================================================")

    return train_loader, val_loader, scaler, train_unit_ids, val_unit_ids


# ===================================================================
# Phase 2: Sequence Encoder (LSTM / Transformer)
# ===================================================================

class SequenceEncoder(nn.Module):
    """
    Flexible sequence encoder supporting LSTM or Transformer.
    
    Phase 2: Encapsulates encoder logic to allow switching between
    LSTM (default) and Transformer architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        encoder_type: str = "lstm",
        nhead: int = 4,
        dim_feedforward: int = 256,
    ):
        """
        Initialize sequence encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (LSTM) or model dimension (Transformer)
            num_layers: Number of layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM (only for LSTM)
            encoder_type: "lstm" or "transformer"
            nhead: Number of attention heads (Transformer only)
            dim_feedforward: Feedforward dimension (Transformer only)
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        if encoder_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
        elif encoder_type == "transformer":
            # Validate that hidden_dim is divisible by nhead
            if hidden_dim % nhead != 0:
                raise ValueError(
                    f"For Transformer encoder, hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead}). "
                    f"Please adjust hidden_dim or nhead (e.g., hidden_dim=48 or nhead=2 for hidden_dim=50)."
                )
            
            # Input projection to model dimension
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = hidden_dim
        elif encoder_type == "universal_v2":
            # Note: universal_v2 is typically used via RULHIUniversalModelV2, not via SequenceEncoder.
            # This branch is provided for compatibility, but it's recommended to use RULHIUniversalModelV2 directly.
            raise ValueError(
                f"encoder_type='universal_v2' is not supported in SequenceEncoder. "
                f"Please use RULHIUniversalModelV2 directly instead of EOLFullLSTMWithHealth with encoder_type='universal_v2'."
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Must be 'lstm', 'transformer', or 'universal_v2'. "
                f"Note: 'universal_v2' should be used via RULHIUniversalModelV2, not via SequenceEncoder."
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
        
        Returns:
            out: [B, T, H] - Full sequence output
            last: [B, H] - Last time step representation
        """
        if self.encoder_type == "lstm":
            out, (h_n, c_n) = self.lstm(x)  # out: [B, T, H * num_directions]
            last = h_n[-1]  # [B, H * num_directions] - last hidden state
            return out, last
        else:  # transformer
            x_proj = self.input_proj(x)  # [B, T, hidden_dim]
            out = self.transformer(x_proj)  # [B, T, hidden_dim]
            last = out[:, -1, :]  # [B, hidden_dim] - last token
            return out, last


class EOLFullLSTM(nn.Module):
    """
    LSTM-basierter Regressor für Full-Trajectory EOL-Prediction.

    Verarbeitet komplette Engine-Trajektorien (nicht nur Tail-Samples)
    und gibt RUL als Skalar pro Sample zurück.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 1),
        )

        # Initialisierung
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, F] - Input sequences

        Returns:
            y_hat: [B] - RUL predictions in cycles
        """
        # LSTM-Output
        out, _ = self.lstm(x)  # out: [B, T, H * num_directions]

        # Verwende Hidden State des letzten Zeitschritts
        last = out[:, -1, :]  # [B, H * num_directions]

        y_hat = self.head(last).squeeze(-1)  # [B]
        return y_hat


class EOLFullLSTMWithHealth(nn.Module):
    """
    Multi-task LSTM model for EOL-RUL prediction + Health Index.
    
    Extends EOLFullLSTM by adding a health head that predicts a continuous
    Health Index (HI ∈ [0,1]) as a monotone proxy derived from RUL during training.
    At inference time, HI depends only on sensor features (no true RUL).
    
    - RUL head: scalar RUL prediction (as before)
    - Health head: scalar HI in [0, 1] via sigmoid activation (standard mode)
    - Damage head: cumulative damage model ensuring HI is monotonically decreasing
                   by construction (damage mode)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        lambda_health: float = 0.3,
        # Phase 2: Condition embeddings
        use_condition_embedding: bool = False,
        num_conditions: int = 1,
        cond_emb_dim: int = 4,
        # Phase 2: Encoder type
        encoder_type: str = "lstm",
        transformer_nhead: int = 4,
        transformer_dim_feedforward: int = 256,
    ):
        """
        Initialize multi-task LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension or Transformer model dimension
            num_layers: Number of layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM (only for LSTM encoder)
            lambda_health: Health loss weight (for reference, not used in model)
            use_condition_embedding: Phase 2: If True, learn condition embeddings
            num_conditions: Phase 2: Number of unique conditions (e.g., 7 for FD002/FD004)
            cond_emb_dim: Phase 2: Dimension of condition embeddings
            encoder_type: Phase 2: "lstm" (default) or "transformer"
            transformer_nhead: Phase 2: Number of attention heads (Transformer only)
            transformer_dim_feedforward: Phase 2: Feedforward dimension (Transformer only)
        """
        from src.config import (
            USE_DAMAGE_HEALTH_HEAD,
            DAMAGE_ALPHA_INIT,
            DAMAGE_SOFTPLUS_BETA,
        )
        
        super().__init__()
        
        # Phase 2: Condition embeddings
        self.use_condition_embedding = use_condition_embedding
        self.num_conditions = num_conditions
        self.cond_emb_dim = cond_emb_dim if use_condition_embedding else 0
        
        if use_condition_embedding:
            self.condition_embedding = nn.Embedding(num_conditions, cond_emb_dim)
            # Adjust input_dim to include condition embeddings
            encoder_input_dim = input_dim + cond_emb_dim
        else:
            encoder_input_dim = input_dim
        
        # Phase 2: Use SequenceEncoder (LSTM or Transformer)
        self.encoder = SequenceEncoder(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            encoder_type=encoder_type,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
        )
        
        encoder_out_dim = self.encoder.output_dim
        
        # Shared feature extraction
        self.shared_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # RUL head (as before)
        self.fc_rul = nn.Linear(encoder_out_dim, 1)
        
        # Health head: outputs logit, will be squashed to [0,1] via sigmoid (fallback)
        self.fc_health = nn.Linear(encoder_out_dim, 1)
        
        # --- Damage-based health head ---
        self.use_damage_health_head = USE_DAMAGE_HEALTH_HEAD
        
        # Maps hidden state h_t -> raw damage rate (will be passed through softplus)
        self.fc_damage = nn.Linear(encoder_out_dim, 1)
        
        # Learnable log(alpha) to keep alpha > 0 via exp(log_alpha)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(DAMAGE_ALPHA_INIT), dtype=torch.float32)
        )
        
        # Store softplus beta for forward pass
        self.damage_softplus_beta = DAMAGE_SOFTPLUS_BETA
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize damage head with small weights for small initial rates
        if self.use_damage_health_head:
            nn.init.xavier_normal_(self.fc_damage.weight, gain=0.1)
            if self.fc_damage.bias is not None:
                nn.init.zeros_(self.fc_damage.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
            cond_ids: [B] - Optional condition IDs (int64). Required if use_condition_embedding=True.
        
        Returns:
            rul_pred: [B] - RUL predictions in cycles (from final time step)
            health_last: [B] - Health Index predictions in [0, 1] at final time step
            health_seq: [B, T, 1] - Health Index predictions over full sequence 
                       (monotonically decreasing by construction if damage head is used)
        """
        # Phase 2: Condition embeddings
        if self.use_condition_embedding:
            if cond_ids is None:
                raise ValueError("cond_ids required when use_condition_embedding=True")
            # cond_ids: [B] -> cond_emb: [B, cond_emb_dim]
            cond_emb = self.condition_embedding(cond_ids)  # [B, cond_emb_dim]
            # Expand to sequence length: [B, cond_emb_dim] -> [B, T, cond_emb_dim]
            cond_emb = cond_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, cond_emb_dim]
            # Concatenate with input features
            x = torch.cat([x, cond_emb], dim=-1)  # [B, T, F + cond_emb_dim]
        
        # Phase 2: Use SequenceEncoder (LSTM or Transformer)
        encoder_out, h_last = self.encoder(x)  # encoder_out: [B, T, H], h_last: [B, H]
        
        # Shared feature extraction for final time step
        shared_features = self.shared_head(h_last)  # [B, H * num_directions]
        
        # RUL head (unchanged)
        rul_logit = self.fc_rul(shared_features)  # [B, 1]
        rul_pred = rul_logit.squeeze(-1)  # [B]
        
        if self.use_damage_health_head:
            # ---- Damage-based Health Index ----
            # Apply shared_head to each time step
            B, T, H = encoder_out.shape
            out_flat = encoder_out.reshape(B * T, H)  # [B*T, H]
            shared_seq = self.shared_head(out_flat)  # [B*T, H]
            shared_seq_reshaped = shared_seq.reshape(B, T, -1)  # [B, T, H]
            
            # 1) Map each hidden state to raw damage rate
            raw_rates = self.fc_damage(shared_seq_reshaped)  # [B, T, 1]
            
            # 2) Enforce non-negative rates via softplus
            if self.damage_softplus_beta != 1.0:
                rates = F.softplus(raw_rates, beta=self.damage_softplus_beta)
            else:
                rates = F.softplus(raw_rates)  # [B, T, 1], >= 0
            
            # 3) Cumulative damage over time
            damage = torch.cumsum(rates, dim=1)  # [B, T, 1], monotonically increasing
            
            # 4) Health Index = exp(-alpha * damage), with alpha > 0
            alpha = torch.exp(self.log_alpha)  # scalar > 0
            health_seq = torch.exp(-alpha * damage)  # [B, T, 1], monotonically decreasing
            
            # 5) Last time-step health
            health_last = health_seq[:, -1, :].squeeze(-1)  # [B]
        else:
            # Fallback: original "standard" health head (sigmoid of last hidden)
            health_logit = self.fc_health(shared_features)  # [B, 1]
            health_last = torch.sigmoid(health_logit).squeeze(-1)  # [B]
            
            # For compatibility, we can broadcast the last value over the sequence
            seq_len = x.size(1)
            health_seq = health_last.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)  # [B, T, 1]
        
        return rul_pred, health_last, health_seq


# NASA functions are now imported from src.metrics at the top of the file
# This ensures consistency across the codebase


def evaluate_eol_full_lstm(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates the Full-Trajectory LSTM model.

    Supports both single-task (EOLFullLSTM) and multi-task (EOLFullLSTMWithHealth) models.
    For multi-task models, only RUL predictions are used for evaluation metrics.

    Sammelt für alle Val-Samples:
      - y_true_all, y_pred_all, unit_ids_all

    Berechnet:
      - Pointwise: MSE, RMSE, MAE, Bias, R²

    EOL/NASA:
      - Für jede Engine: wähle Sample mit minimalem true RUL (letzter Zyklus)
      - RMSE_eol, MAE_eol, Bias_eol
      - NASA-Score_eol (Sum + Mean)

    Args:
        model: Trained EOLFullLSTM or EOLFullLSTMWithHealth model
        val_loader: Validation DataLoader
        device: torch.device

    Returns:
        Dictionary mit allen Metriken
    """
    from src.eval_utils import forward_rul_only
    
    model.eval()
    model.to(device)

    y_true_all = []
    y_pred_all = []
    unit_ids_all = []
    health_pred_all = []  # Optional: store health predictions for diagnostics

    # Check if model uses condition embeddings
    use_cond_emb = getattr(model, 'use_condition_embedding', False)
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle different batch formats:
            # - 5 elements: X, y, unit_ids, cond_ids, health_phys_seq
            # - 4 elements: X, y, unit_ids, cond_ids
            # - 3 elements: X, y, unit_ids
            # - 2 elements: X, y (fallback)
            if len(batch) == 5:
                X_batch, y_batch, unit_ids_batch, cond_ids_batch, _ = batch
                unit_ids_all.append(unit_ids_batch.cpu().numpy() if isinstance(unit_ids_batch, torch.Tensor) else unit_ids_batch)
            elif len(batch) == 4:  # SequenceDatasetWithUnits with cond_ids
                X_batch, y_batch, unit_ids_batch, cond_ids_batch = batch
                unit_ids_all.append(unit_ids_batch.cpu().numpy() if isinstance(unit_ids_batch, torch.Tensor) else unit_ids_batch)
            elif len(batch) == 3:  # SequenceDatasetWithUnits without cond_ids (backward compat)
                X_batch, y_batch, unit_ids_batch = batch
                cond_ids_batch = None
                unit_ids_all.append(unit_ids_batch.cpu().numpy() if isinstance(unit_ids_batch, torch.Tensor) else unit_ids_batch)
            else:  # Fallback für TensorDataset
                X_batch, y_batch = batch
                cond_ids_batch = None
                unit_ids_all.append(None)
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if cond_ids_batch is not None:
                cond_ids_batch = cond_ids_batch.to(device)

            # Use helper to extract RUL predictions (handles both single-task and multi-task)
            preds = forward_rul_only(model, X_batch, cond_ids=cond_ids_batch)
            
            # Optionally store health predictions if model is multi-task
            if use_cond_emb and cond_ids_batch is not None:
                model_output = model(X_batch, cond_ids=cond_ids_batch)
            else:
                model_output = model(X_batch)
            if isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
                # Handle both 2-element (rul, health) and 3-element (rul, health, health_seq) tuples
                health_pred_all.append(model_output[1].cpu().numpy())

            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    
    # Concatenate health predictions if available (multi-task model)
    if len(health_pred_all) > 0:
        health_pred_all_concatenated = np.concatenate(health_pred_all)
    else:
        health_pred_all_concatenated = None

    # Pointwise Metriken
    errors = y_pred_all - y_true_all
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    r2 = 1 - np.sum(errors ** 2) / np.sum((y_true_all - y_true_all.mean()) ** 2)
    r2 = float(r2)

    # EOL-Style Metriken: Für jede Engine den letzten Zyklus (minimales RUL)
    # Check if unit_ids are available and valid
    has_unit_ids = False
    unit_ids_concatenated = None
    
    if len(unit_ids_all) > 0 and unit_ids_all[0] is not None:
        # Filter out None values and check if we have valid arrays
        valid_unit_ids = [uid for uid in unit_ids_all if uid is not None]
        if len(valid_unit_ids) > 0:
            try:
                # Check if arrays are not empty and have proper shape
                if all(isinstance(uid, np.ndarray) and uid.size > 0 for uid in valid_unit_ids):
                    unit_ids_concatenated = np.concatenate(valid_unit_ids)
                    has_unit_ids = True
            except (ValueError, TypeError) as e:
                # If concatenation fails, unit_ids are not available
                has_unit_ids = False
                unit_ids_concatenated = None
    
    if has_unit_ids and unit_ids_concatenated is not None:
        # Für jede Engine: finde Sample mit minimalem true RUL
        unique_units = np.unique(unit_ids_concatenated)
        y_true_eol = []
        y_pred_eol = []
        
        for uid in unique_units:
            mask = unit_ids_concatenated == uid
            y_true_unit = y_true_all[mask]
            y_pred_unit = y_pred_all[mask]
            
            # Finde Index mit minimalem RUL (letzter Zyklus)
            min_rul_idx = np.argmin(y_true_unit)
            y_true_eol.append(y_true_unit[min_rul_idx])
            y_pred_eol.append(y_pred_unit[min_rul_idx])
        
        y_true_eol = np.array(y_true_eol)
        y_pred_eol = np.array(y_pred_eol)
        
        # EOL Metriken
        errors_eol = y_pred_eol - y_true_eol
        rmse_eol = float(np.sqrt(np.mean(errors_eol ** 2)))
        mae_eol = float(np.mean(np.abs(errors_eol)))
        bias_eol = float(np.mean(errors_eol))
        
        # NASA Score (EOL)
        nasa_score_sum_eol = nasa_phm_score(y_true_eol, y_pred_eol)
        nasa_score_mean_eol = nasa_score_sum_eol / len(y_true_eol)
        
        has_eol_metrics = True
    else:
        rmse_eol = None
        mae_eol = None
        bias_eol = None
        nasa_score_sum_eol = None
        nasa_score_mean_eol = None
        has_eol_metrics = False

    print("============================================================")
    print("[evaluate_eol_full_lstm] Pointwise Metrics")
    print("============================================================")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f} cycles")
    print(f"MAE: {mae:.4f} cycles")
    print(f"Bias: {bias:.4f} cycles")
    print(f"R²: {r2:.4f}")
    print("============================================================")
    
    if has_eol_metrics:
        print("[evaluate_eol_full_lstm] EOL/NASA Metrics (per Engine, last cycle)")
        print("============================================================")
        print(f"RMSE_eol: {rmse_eol:.4f} cycles")
        print(f"MAE_eol: {mae_eol:.4f} cycles")
        print(f"Bias_eol: {bias_eol:.4f} cycles")
        print(f"NASA Score (sum): {nasa_score_sum_eol:.2f}")
        print(f"NASA Score (mean): {nasa_score_mean_eol:.4f}")
        print(f"Num engines: {len(y_true_eol)}")
    else:
        print("[evaluate_eol_full_lstm] EOL/NASA Metrics")
        print("============================================================")
        print("[NOTE] EOL metrics require unit_ids per sample.")
        print("[NOTE] Using pointwise metrics as approximation.")
    
    print("============================================================")

    # NASA Score (pointwise)
    nasa_score_sum = nasa_phm_score(y_true_all, y_pred_all)
    nasa_score_mean = nasa_score_sum / len(y_true_all)

    metrics = {
        "pointwise": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r2": r2,
        },
        "nasa_pointwise": {
            "score_sum": nasa_score_sum,
            "score_mean": nasa_score_mean,
        },
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }
    
    # Store unit_ids if available for trajectory plotting
    if has_unit_ids and unit_ids_concatenated is not None:
        metrics["unit_ids"] = unit_ids_concatenated
    
    if has_eol_metrics:
        metrics["eol"] = {
            "rmse": rmse_eol,
            "mae": mae_eol,
            "bias": bias_eol,
            "nasa_score_sum": nasa_score_sum_eol,
            "nasa_score_mean": nasa_score_mean_eol,
            "num_engines": len(y_true_eol),
            "y_true_eol": y_true_eol,
            "y_pred_eol": y_pred_eol,
        }
    
    # Optionally add health diagnostics if available
    if health_pred_all_concatenated is not None:
        metrics["health"] = {
            "mean_health": float(np.mean(health_pred_all_concatenated)),
            "std_health": float(np.std(health_pred_all_concatenated)),
            "min_health": float(np.min(health_pred_all_concatenated)),
            "max_health": float(np.max(health_pred_all_concatenated)),
        }

    return metrics


def train_eol_full_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 80,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: torch.device | str = "cpu",
    results_dir: Path | str = "../results/eol_full_lstm",
    run_name: str = "fd001_fd004",
    use_mixed_precision: bool = True,  # Mixed Precision (FP16) für weniger RAM
    use_health_head: bool = False,  # Multi-task mode: RUL + Health Index
    max_rul: float = 125.0,
    tau: float = 40.0,
    lambda_health: float = 0.3,
    hi_condition_calib_weight: float = 0.0,  # Weight for condition calibration loss
    hi_plateau_threshold: float = 80.0,  # RUL threshold for early-life (condition calibration)
    hi_mono_late_weight: float | None = None,  # Override HI_MONO_WEIGHT if provided
    hi_mono_global_weight: float | None = None,  # Override HI_GLOBAL_MONO_WEIGHT if provided
    # Phase 2: Smoothness loss
    smooth_hi_weight: float = 0.0,  # Phase 2: Weight for HI smoothness loss
    smooth_hi_plateau_threshold: float = 80.0,  # Phase 2: RUL threshold for smoothness masking
    # Phase 2: Condition embedding flag (for determining if cond_ids should be passed)
    use_condition_embedding: bool = False,  # Phase 2: Whether model uses condition embeddings
    # NEW: auxiliary condition reconstruction loss weight (for Transformer encoder V2)
    cond_recon_weight: float = 0.0,
    # NEW: damage-based HI loss weight (for Transformer encoder with cumulative damage head)
    damage_hi_weight: float = 0.0,
    # NEW (v3c): two-phase damage training schedule
    damage_two_phase: bool = False,
    damage_warmup_epochs: int = 0,
    damage_phase1_damage_weight: float | None = None,
    damage_phase2_damage_weight: float | None = None,
    # NEW: Smoothness loss weights for damage increments
    damage_phase1_smooth_weight: float | None = None,
    damage_phase2_smooth_weight: float | None = None,
    # NEW: RUL Trajectory weight (v3d)
    rul_traj_weight: float = 1.0,
    # NEW (v3e): damage alignment weights
    damage_hi_align_start_weight: float = 0.0,
    damage_hi_align_end_weight: float = 0.0,
    # NEW (v4): calibrated HI (HI_cal_v2) supervision and slope regularisation
    hi_cal_weight: float = 0.0,
    hi_cal_mono_weight: float = 0.0,
    hi_cal_slope_weight: float = 0.0,
    hi_calibrator_path: str | None = None,
    # NEW (v5u): optional Gaussian NLL for RUL at last observed cycle (requires predicted sigma)
    rul_nll_weight: float = 0.0,
    rul_nll_min_sigma: float = 1e-3,
    # If True: do NOT let the NLL term move the mean predictor (mu/backbone).
    # This keeps mu behavior closer to the pre-uncertainty baseline while still learning sigma.
    rul_nll_detach_mu: bool = False,
    # NEW (v5q): quantile loss for RUL at last observed cycle (pinball loss)
    rul_quantile_weight: float = 0.0,
    rul_quantiles: Optional[list[float]] = None,
    rul_quantile_cross_weight: float = 0.0,
    rul_quantile_p50_mse_weight: float = 0.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Training-Loop für Full-Trajectory LSTM.

    Supports two modes:
    - Single-task (use_health_head=False): Standard RUL prediction with MSELoss
    - Multi-task (use_health_head=True): RUL + Health Index with weighted multi-task loss

    - Adam (lr, weight_decay)
    - LR-Scheduler (ReduceLROnPlateau auf val_loss)
    - Early Stopping (patience)
    - Speichere bestes Modell als 'eol_full_lstm_best.pt'
    - Logge Train/Val MSE + Val RMSE pro Epoch

    Args:
        model: EOLFullLSTM or EOLFullLSTMWithHealth model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Anzahl Epochen
        lr: Learning Rate
        weight_decay: L2-Regularisierung
        patience: Early Stopping Patience
        device: torch.device
        results_dir: Verzeichnis für Checkpoints und Plots
        run_name: Name des Runs (für Dateinamen)
        use_health_head: If True, use multi-task loss (RUL + Health Index)
        max_rul: Maximum RUL for clamping and health target computation
        tau: Scale parameter for exponential RUL weighting
        lambda_health: Trade-off between RUL loss and health loss

    Returns:
        model: Bestes Modell (geladen)
        history: Dictionary mit Trainings-Verlauf
    """
    from src.loss import multitask_rul_health_loss, compute_monotonicity_loss
    from src.training_utils import (
        compute_condition_calibration_loss,
        compute_global_trend_loss,
    )
    from src.config import (
        HI_RUL_PLATEAU_THRESH,
        HI_EOL_THRESH,
        USE_HI_MONOTONICITY,
        HI_MONO_WEIGHT,
        HI_MONO_RUL_BETA,
        HI_EOL_WEIGHT,
        SEQUENCE_LENGTH,
        HI_CONDITION_CALIB_VAR_ALPHA,
        HI_GLOBAL_MONO_WEIGHT,
    )
    from src.loss import health_smoothness_loss
    from src.models.transformer_eol import EOLFullTransformerEncoder
    
    # Use provided weights or fall back to config defaults
    mono_late_weight = hi_mono_late_weight if hi_mono_late_weight is not None else HI_MONO_WEIGHT
    mono_global_weight = hi_mono_global_weight if hi_mono_global_weight is not None else HI_GLOBAL_MONO_WEIGHT
    # Two-phase damage HI configuration (v3c)
    damage_two_phase = bool(damage_two_phase)
    if damage_phase1_damage_weight is None:
        damage_phase1_damage_weight = damage_hi_weight
    if damage_phase2_damage_weight is None:
        damage_phase2_damage_weight = damage_hi_weight

    # Optional HI_cal_v2 calibrator (v4/v5).
    # Load ONLY if HI_cal supervision is actually enabled (model has HI_cal head)
    # and any HI_cal loss weight is > 0. This avoids hard failures for runs that
    # do not use the calibrator (e.g. quantile-only runs) even if config is mis-set.
    hi_calibrator = None
    use_hi_cal_supervision = (
        isinstance(model, EOLFullTransformerEncoder)
        and bool(getattr(model, "use_hi_cal_head", False))
        and getattr(model, "hi_cal_head", None) is not None
    )
    if use_hi_cal_supervision and (hi_cal_weight > 0.0 or hi_cal_mono_weight > 0.0 or hi_cal_slope_weight > 0.0):
        if hi_calibrator_path is None:
            raise ValueError(
                "HI_cal-related weights > 0 but hi_calibrator_path is None. "
                "Please provide a valid calibrator path in the experiment config."
            )
        from pathlib import Path as _Path
        from src.analysis.hi_calibration import load_hi_calibrator, calibrate_hi_array, hi_cal_v2_from_v1

        cal_path = _Path(hi_calibrator_path)
        if not cal_path.exists():
            raise FileNotFoundError(
                f"HI_calibrator file not found at {cal_path}. "
                "Fit it first via src.analysis.hi_calibration."
            )
        try:
            hi_calibrator = load_hi_calibrator(cal_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to load HI_calibrator for HI_cal_v2 supervision. "
                f"Path: {cal_path}\n"
                "This usually means the file is corrupted/empty (e.g., interrupted write).\n"
                "Fix: delete the calibrator file and refit it via:\n"
                "  python -m src.analysis.hi_calibration --dataset FD004 --encoder_run <BASE_ENCODER_RUN>\n"
                "Then re-run this experiment.\n"
                f"Original error: {e}"
            ) from e
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Debug: log all damage-head parameters (if any)
    log_damage_head_params(model)
    
    # Choose loss function based on mode
    if use_health_head:
        criterion = None  # Will use multitask_rul_health_loss
        print("[train_eol_full_lstm] Multi-task mode: RUL + Health Index")
        print(f"  RUL weighting tau: {tau}")
        print(f"  Health loss weight (lambda): {lambda_health}")
        print(f"  HI plateau threshold: {HI_RUL_PLATEAU_THRESH}")
        print(f"  HI EOL threshold: {HI_EOL_THRESH} cycles")
        print(f"  HI EOL weight: {HI_EOL_WEIGHT}")
        print(f"  HI monotonicity: {USE_HI_MONOTONICITY} (weight: {mono_late_weight}, RUL beta: {HI_MONO_RUL_BETA})")
        print(f"  HI global trend weight: {mono_global_weight}")
        print(f"  HI condition calib weight: {hi_condition_calib_weight}")
        print(f"  Phase 2 - HI smoothness weight: {smooth_hi_weight}")
        print(f"  Phase 2 - Use condition embedding: {use_condition_embedding}")
    else:
        criterion = nn.MSELoss()
        print("[train_eol_full_lstm] Single-task mode: RUL only")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    
    # Mixed Precision Training (FP16) für weniger GPU-Memory
    use_amp = use_mixed_precision and torch.cuda.is_available()
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("[train_eol_full_lstm] Mixed Precision Training (FP16) enabled - saves ~50% GPU memory")
    else:
        scaler = None

    # Init damage smoothness weights
    if damage_phase1_smooth_weight is None:
        damage_phase1_smooth_weight = 0.1
    if damage_phase2_smooth_weight is None:
        damage_phase2_smooth_weight = 0.03

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "lr": [],
    }
    
    # Add health-specific logging if in multi-task mode
    if use_health_head:
        history["train_rul_loss_unscaled"] = []
        history["train_rul_loss"] = []
        history["train_health_loss"] = []
        history["train_eol_health_loss"] = []
        history["train_mono_loss_raw"] = []
        history["train_mono_loss"] = []
        history["val_rul_loss_unscaled"] = []
        history["val_rul_loss"] = []
        history["val_health_loss"] = []
        history["val_eol_health_loss"] = []
        history["val_mono_loss_raw"] = []
        history["val_mono_loss"] = []
        history["val_mean_health"] = []
        history["train_mono_late_raw"] = []
        history["train_mono_late"] = []
        history["train_mono_global_raw"] = []
        history["train_mono_global"] = []
        history["val_mono_late_raw"] = []
        history["val_mono_late"] = []
        history["val_mono_global_raw"] = []
        history["val_mono_global"] = []
        # Phase 2: Smoothness loss history
        history["train_smooth_hi_raw"] = []
        history["train_smooth_hi"] = []
        history["val_smooth_hi_raw"] = []
        history["val_smooth_hi"] = []
        if hi_condition_calib_weight > 0.0:
            history["train_calib_loss"] = []
            history["val_calib_loss"] = []
        # NEW: condition reconstruction loss history (encoder V2)
        history["train_cond_loss"] = []
        history["val_cond_loss"] = []
        # NEW: cumulative damage-based HI loss history (Transformer damage head)
        history["train_damage_hi_loss"] = []
        history["val_damage_hi_loss"] = []

    print("============================================================")
    print("[train_eol_full_lstm] Training Configuration")
    print("============================================================")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(f"Device: {device}")
    print("============================================================")

    first_damage_debug_done = False

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = []
        train_component_losses = {
            "rul_loss_unscaled": [],
            "rul_loss": [],
            "health_loss": [],
            "eol_health_loss": [],
            "mono_late_raw": [],
            "mono_late": [],
            "mono_global_raw": [],
            "mono_global": [],
            "calib_loss": [],
            "smooth_hi_raw": [],  # Phase 2
            "smooth_hi": [],  # Phase 2
            "cond_loss": [],  # NEW: condition reconstruction loss
            "damage_hi_loss": [],  # NEW: cumulative damage-based HI loss
        } if use_health_head else None

        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats:
            # - 5 elements: X, y, unit_ids, cond_ids, health_phys_seq
            # - 4 elements: X, y, unit_ids, cond_ids
            # - 3 elements: X, y, unit_ids
            # - 2 elements: X, y (fallback)
            health_phys_seq_batch = None
            if len(batch) == 5:
                X_batch, y_batch, _, cond_ids_batch, health_phys_seq_batch = batch
            elif len(batch) == 4:  # SequenceDatasetWithUnits with cond_ids
                X_batch, y_batch, _, cond_ids_batch = batch
                cond_ids_batch = cond_ids_batch.to(device)
            elif len(batch) == 3:  # SequenceDatasetWithUnits without cond_ids (backward compat)
                X_batch, y_batch, _ = batch
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64, device=device)
            else:  # Fallback für TensorDataset
                X_batch, y_batch = batch
                cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64, device=device)
            
            # Move health_phys_seq to device if present
            if health_phys_seq_batch is not None:
                health_phys_seq_batch = health_phys_seq_batch.to(device)
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            cond_ids_batch = cond_ids_batch.to(device)

            optimizer.zero_grad()
            
            # Mixed Precision Training (FP16)
            if use_amp and scaler is not None:
                # Use new torch.amp.autocast API to avoid FutureWarning
                with torch.amp.autocast("cuda"):
                    if use_health_head:
                        # Phase 2: Pass cond_ids if condition embedding is enabled
                        # NEW: optional condition reconstruction head (Transformer encoder V2)
                        supports_cond_recon = (
                            hasattr(model, "use_cond_recon_head")
                            and getattr(model, "use_cond_recon_head", False)
                            and hasattr(model, "fc_cond_recon")
                            and getattr(model, "fc_cond_recon") is not None
                            and cond_recon_weight > 0.0
                        )
                        cond_seq_avg = None
                        cond_recon = None
                        rul_sigma = None
                        rul_quantiles = None
                        if use_condition_embedding:
                            if supports_cond_recon:
                                out = model(X_batch, cond_ids=cond_ids_batch, return_aux=True)
                                if isinstance(out, (tuple, list)) and len(out) == 7:
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, rul_sigma, rul_quantiles = out
                                elif isinstance(out, (tuple, list)) and len(out) == 6:
                                    # Older v5u contract: no quantiles
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, rul_sigma = out
                                else:
                                    # Backwards-compatible (older models)
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                            else:
                                out = model(X_batch, cond_ids=cond_ids_batch)
                                if isinstance(out, (tuple, list)) and len(out) == 5:
                                    rul_pred, health_last, health_seq, rul_sigma, rul_quantiles = out
                                elif isinstance(out, (tuple, list)) and len(out) == 4:
                                    # Older v5u contract: no quantiles
                                    rul_pred, health_last, health_seq, rul_sigma = out
                                else:
                                    rul_pred, health_last, health_seq = out
                        else:
                            if supports_cond_recon:
                                out = model(X_batch, return_aux=True)
                                if isinstance(out, (tuple, list)) and len(out) == 7:
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, rul_sigma, rul_quantiles = out
                                elif isinstance(out, (tuple, list)) and len(out) == 6:
                                    # Older v5u contract: no quantiles
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, rul_sigma = out
                                else:
                                    # Backwards-compatible (older models)
                                    rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                            else:
                                out = model(X_batch)
                                if isinstance(out, (tuple, list)) and len(out) == 5:
                                    rul_pred, health_last, health_seq, rul_sigma, rul_quantiles = out
                                elif isinstance(out, (tuple, list)) and len(out) == 4:
                                    # Older v5u contract: no quantiles
                                    rul_pred, health_last, health_seq, rul_sigma = out
                                else:
                                    rul_pred, health_last, health_seq = out
                        
                        # Construct RUL sequence for RUL-weighted monotonicity
                        # y_batch: (batch,) - RUL at last step
                        # seq_len: sequence length (e.g., 30)
                        rul_last = y_batch.view(-1, 1)  # (batch, 1)
                        seq_len = X_batch.size(1)  # int
                        # Construct offsets [seq_len-1, ..., 1, 0], so that last step has offset 0
                        offsets = torch.arange(seq_len - 1, -1, -1, device=y_batch.device).float()  # (seq_len,)
                        rul_seq_batch = rul_last + offsets  # broadcast to (batch, seq_len)
                        rul_seq_batch = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                        
                        # Prepare health_seq and rul_seq for loss computation
                        health_seq_flat = health_seq.squeeze(-1) if health_seq.dim() == 3 else health_seq
                        
                        # Compute multitask loss (without monotonicity, we'll add it separately)
                        mt_loss, loss_components = multitask_rul_health_loss(
                            rul_pred=rul_pred.squeeze(-1) if rul_pred.dim() > 1 else rul_pred,
                            rul_true=y_batch,
                            health_pred=health_last.squeeze(-1) if health_last.dim() > 1 else health_last,
                            max_rul=max_rul,
                            tau=tau,
                            lambda_health=lambda_health,
                            plateau_thresh=float(HI_RUL_PLATEAU_THRESH),
                            hi_eol_thresh=float(HI_EOL_THRESH),
                            hi_eol_weight=HI_EOL_WEIGHT,
                            health_seq=None,  # We'll compute monotonicity separately
                            rul_seq=None,
                            hi_mono_weight=0.0,  # Disable monotonicity in multitask loss
                            hi_mono_rul_beta=HI_MONO_RUL_BETA,
                            return_components=True,
                            rul_traj_weight=rul_traj_weight,
                        )

                        # ------------------------------------------------------------------
                        # Two-phase damage training schedule (v3c / v3d)
                        # Phase 1: focus on damage HI (no RUL/standard HI supervision)
                        # Phase 2: full multi-task (RUL + HI) as before
                        # ------------------------------------------------------------------
                        if damage_two_phase and epoch <= damage_warmup_epochs:
                            # Phase 1 (damage-focused warmup): suppress RUL/HI loss.
                            loss = mt_loss.new_tensor(0.0)
                            damage_weight_eff = damage_phase1_damage_weight
                            smooth_weight_eff = damage_phase1_smooth_weight
                            phase_tag = "P1-Damage"
                        else:
                            # Phase 2 (or no two-phase): use full multitask loss.
                            loss = mt_loss
                            damage_weight_eff = (
                                damage_phase2_damage_weight if damage_two_phase else damage_hi_weight
                            )
                            smooth_weight_eff = damage_phase2_smooth_weight if damage_two_phase else 0.0
                            phase_tag = "P2-Multi" if damage_two_phase else "SinglePhase"

                            # --------------------------------------------------------------
                            # NEW (v5u): Gaussian NLL for RUL at last observed cycle
                            # --------------------------------------------------------------
                            if float(rul_nll_weight) > 0.0:
                                if rul_sigma is None:
                                    raise RuntimeError(
                                        "rul_nll_weight > 0 but model did not return rul_sigma. "
                                        "Enable use_rul_uncertainty_head for this run."
                                    )
                                sigma = torch.clamp(rul_sigma, min=float(rul_nll_min_sigma))
                                mu = (rul_pred.squeeze(-1) if rul_pred.dim() > 1 else rul_pred)
                                if bool(rul_nll_detach_mu):
                                    mu = mu.detach()
                                err = (y_batch - mu)
                                nll = 0.5 * ((err * err) / (sigma * sigma) + 2.0 * torch.log(sigma))
                                loss = loss + float(rul_nll_weight) * nll.mean()

                            # --------------------------------------------------------------
                            # NEW (v5q): Pinball (quantile) loss for RUL at last observed cycle
                            # --------------------------------------------------------------
                            if float(rul_quantile_weight) > 0.0:
                                if rul_quantiles is None or len(rul_quantiles) == 0:
                                    raise RuntimeError(
                                        "rul_quantile_weight > 0 but rul_quantiles is None/empty."
                                    )
                                if rul_quantiles is None:
                                    raise RuntimeError("Internal error: rul_quantiles None after check.")
                                if rul_quantiles_pred is None or (not torch.is_tensor(rul_quantiles_pred)):
                                    raise RuntimeError(
                                        "rul_quantile_weight > 0 but model did not return rul_quantiles."
                                    )

                                q_pred = rul_quantiles_pred  # [B, Q]
                                q_list = [float(q) for q in rul_quantiles]
                                if q_pred.dim() != 2 or q_pred.size(1) != len(q_list):
                                    raise RuntimeError(
                                        f"Quantile dim mismatch: q_pred shape {tuple(q_pred.shape)} "
                                        f"but rul_quantiles has len={len(q_list)}"
                                    )

                                y = y_batch.view(-1, 1)  # [B,1]
                                u = y - q_pred  # [B,Q]
                                qs = torch.tensor(q_list, device=q_pred.device, dtype=q_pred.dtype).view(1, -1)
                                pinball = torch.maximum(qs * u, (qs - 1.0) * u).mean()
                                loss = loss + float(rul_quantile_weight) * pinball

                                # Optional: non-crossing penalty (sorted quantiles should be non-decreasing)
                                if float(rul_quantile_cross_weight) > 0.0 and q_pred.size(1) >= 2:
                                    diffs = q_pred[:, :-1] - q_pred[:, 1:]  # should be <= 0
                                    cross_pen = torch.relu(diffs).mean()
                                    loss = loss + float(rul_quantile_cross_weight) * cross_pen

                                # Optional: keep P50 close via MSE to stabilize (uses q closest to 0.5)
                                if float(rul_quantile_p50_mse_weight) > 0.0:
                                    q_arr = torch.tensor(q_list, device=q_pred.device, dtype=q_pred.dtype)
                                    idx50 = int(torch.argmin(torch.abs(q_arr - 0.5)).item())
                                    p50 = q_pred[:, idx50]
                                    p50_mse = F.mse_loss(p50.view(-1), y_batch.view(-1))
                                    loss = loss + float(rul_quantile_p50_mse_weight) * p50_mse
                        
                        # Late monotonicity loss (RUL <= beta)
                        if USE_HI_MONOTONICITY:
                            mono_late_raw, _ = compute_monotonicity_loss(
                                pred_hi=health_seq_flat,
                                rul=rul_seq_batch,
                                beta=HI_MONO_RUL_BETA,
                                weight=1.0,  # Weight applied outside
                            )
                            mono_late = mono_late_weight * mono_late_raw
                        else:
                            mono_late_raw = torch.tensor(0.0, device=health_seq.device)
                            mono_late = torch.tensor(0.0, device=health_seq.device)
                        
                        # Global trend loss (all cycles)
                        mono_global_raw = compute_global_trend_loss(health_seq_flat)
                        mono_global = mono_global_weight * mono_global_raw
                        
                        # Total monotonicity loss
                        mono_loss_total = mono_late + mono_global
                        loss = loss + mono_loss_total
                        
                        # Phase 2: Smoothness loss
                        smooth_hi_raw = health_smoothness_loss(
                            health_seq=health_seq_flat,
                            rul_seq=rul_seq_batch,
                            smooth_weight=1.0,  # Weight applied outside
                            plateau_rul_threshold=smooth_hi_plateau_threshold,
                        )
                        smooth_hi = smooth_hi_weight * smooth_hi_raw
                        loss = loss + smooth_hi

                        # Optional cumulative damage-based HI loss (Transformer encoder with damage head)
                        damage_hi_loss = torch.tensor(0.0, device=health_seq_flat.device)
                        if (
                            damage_hi_weight > 0.0
                            and hasattr(model, "damage_head")
                            and getattr(model, "damage_head", None) is not None
                        ):
                            # Determine HI target sequence:
                            # - If health_phys_seq_batch is available (damage_v2), use it directly
                            # - Otherwise, compute from RUL (damage_v1 legacy behavior)
                            if health_phys_seq_batch is not None:
                                # damage_v2: use physics-based HI_phys_seq target
                                hi_target_seq = health_phys_seq_batch  # [B, T]
                            else:
                                # damage_v1: compute analytic HI target from RUL sequence
                                rul_seq_clamped = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                                hi_target_seq = torch.ones_like(rul_seq_clamped)
                                mask_eol = rul_seq_clamped <= float(HI_EOL_THRESH)
                                mask_plateau = rul_seq_clamped >= float(hi_plateau_threshold)
                                mask_mid = (~mask_eol) & (~mask_plateau)
                                denom = max(hi_plateau_threshold - float(HI_EOL_THRESH), 1e-6)
                                hi_target_seq[mask_eol] = 0.0
                                hi_target_seq[mask_mid] = (
                                    (rul_seq_clamped[mask_mid] - float(HI_EOL_THRESH)) / denom
                                ).clamp(0.0, 1.0)

                            # Encoder latents for damage head
                            try:
                                enc_seq_dmg, _ = model.encode(
                                    X_batch, cond_ids=cond_ids_batch, return_seq=True
                                )
                            except TypeError:
                                enc_seq_dmg = None

                            if enc_seq_dmg is not None:
                                cond_seq_dmg = None
                                if getattr(model, "use_cond_encoder", False) and getattr(
                                    model, "cond_in_dim", 0
                                ) > 0:
                                    if (
                                        hasattr(model, "cond_feature_indices")
                                        and model.cond_feature_indices is not None
                                        and len(model.cond_feature_indices)
                                        == getattr(model, "cond_in_dim", 0)
                                    ):
                                        cond_seq_dmg = X_batch[
                                            :, :, model.cond_feature_indices
                                        ]

                                hi_seq_damage, _, damage_seq_dbg, delta_damage_dbg = model.damage_head(
                                    enc_seq_dmg, cond_seq=cond_seq_dmg
                                )
                                
                                # Align shapes if necessary (safety check)
                                if hi_seq_damage.shape != hi_target_seq.shape:
                                    T = min(hi_seq_damage.size(1), hi_target_seq.size(1))
                                    hi_seq_damage_ = hi_seq_damage[:, :T]
                                    hi_target_seq_ = hi_target_seq[:, :T]
                                else:
                                    hi_seq_damage_ = hi_seq_damage
                                    hi_target_seq_ = hi_target_seq
                                
                                damage_hi_loss = F.mse_loss(hi_seq_damage_, hi_target_seq_)
                                # Use phase-dependent damage HI weight (v3c)
                                loss = loss + damage_weight_eff * damage_hi_loss

                                # NEW: Smoothness loss on delta_damage (v3d)
                                damage_smooth_raw = torch.zeros((), device=device)
                                if delta_damage_dbg is not None:
                                    d = delta_damage_dbg
                                    if d.size(1) > 1:
                                        diff = d[:, 1:] - d[:, :-1]
                                        damage_smooth_raw = (diff ** 2).mean()
                                
                                if smooth_weight_eff > 0:
                                    loss = loss + smooth_weight_eff * damage_smooth_raw

                                # NEW: Alignment loss at start/end of trajectory (v3e)
                                if (damage_hi_align_start_weight > 0.0 or damage_hi_align_end_weight > 0.0):
                                    T_align = hi_seq_damage_.size(1)
                                    k_align = max(1, T_align // 6)
                                    
                                    if damage_hi_align_start_weight > 0.0:
                                        loss_start = F.mse_loss(
                                            hi_seq_damage_[:, :k_align], 
                                            hi_target_seq_[:, :k_align]
                                        )
                                        loss = loss + damage_hi_align_start_weight * loss_start
                                    
                                    if damage_hi_align_end_weight > 0.0:
                                        loss_end = F.mse_loss(
                                            hi_seq_damage_[:, -k_align:], 
                                            hi_target_seq_[:, -k_align:]
                                        )
                                        loss = loss + damage_hi_align_end_weight * loss_end

                                # ------------------------------------------------------------------
                                # NEW (v4): HI_cal_v2 supervision and slope regularisation
                                # ------------------------------------------------------------------
                                if (
                                    hi_calibrator is not None
                                    and (hi_cal_weight > 0.0 or hi_cal_mono_weight > 0.0 or hi_cal_slope_weight > 0.0)
                                    and isinstance(model, EOLFullTransformerEncoder)
                                    and getattr(model, "use_hi_cal_head", False)
                                ):
                                    # Compute HI_cal_v1/v2 targets from HI_phys_v3 using global calibrator
                                    if health_phys_seq_batch is None:
                                        # Fallback: derive HI_phys-like target from RUL if explicit HI_phys is absent
                                        rul_seq_clamped = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                                        hi_phys_target = torch.ones_like(rul_seq_clamped)
                                        mask_eol = rul_seq_clamped <= float(HI_EOL_THRESH)
                                        mask_plateau = rul_seq_clamped >= float(hi_plateau_threshold)
                                        mask_mid = (~mask_eol) & (~mask_plateau)
                                        denom = max(hi_plateau_threshold - float(HI_EOL_THRESH), 1e-6)
                                        hi_phys_target[mask_eol] = 0.0
                                        hi_phys_target[mask_mid] = (
                                            (rul_seq_clamped[mask_mid] - float(HI_EOL_THRESH)) / denom
                                        ).clamp(0.0, 1.0)
                                    else:
                                        hi_phys_target = health_phys_seq_batch

                                    hi_phys_np = hi_phys_target.detach().cpu().numpy()
                                    from src.analysis.hi_calibration import calibrate_hi_array, hi_cal_v2_from_v1
                                    hi_cal1_np = calibrate_hi_array(hi_phys_np, hi_calibrator)
                                    hi_cal2_np = hi_cal_v2_from_v1(hi_cal1_np)
                                    hi_cal2_target = torch.from_numpy(hi_cal2_np).to(
                                        dtype=hi_phys_target.dtype, device=hi_phys_target.device
                                    )

                                    # Use encoder sequence for HI_cal head (reuse enc_seq_dmg)
                                    hi_cal_seq_pred = model.predict_hi_cal_seq(enc_seq_dmg)  # [B, T]

                                    # (4) Alignment loss HI_cal_v2
                                    hi_cal_loss = F.mse_loss(hi_cal_seq_pred, hi_cal2_target)
                                    loss = loss + hi_cal_weight * hi_cal_loss

                                    # (5) Monotonicity penalty on HI_cal_v2 (should decrease over time)
                                    diff_cal = hi_cal_seq_pred[:, 1:] - hi_cal_seq_pred[:, :-1]
                                    mono_violation_cal = F.relu(diff_cal)
                                    mono_loss_hi_cal = mono_violation_cal.mean()
                                    loss = loss + hi_cal_mono_weight * mono_loss_hi_cal

                                    # (6) Slope-consistency between predicted and target HI_cal_v2 slopes
                                    diff_cal_true = hi_cal2_target[:, 1:] - hi_cal2_target[:, :-1]
                                    slope_loss_hi_cal = F.mse_loss(diff_cal, diff_cal_true)
                                    loss = loss + hi_cal_slope_weight * slope_loss_hi_cal

                                # DEBUG: Inspect damage-head input/targets/preds once on first batch
                                if (
                                    epoch == 1
                                    and batch_idx == 0
                                    and health_phys_seq_batch is not None
                                    and not first_damage_debug_done
                                ):
                                    first_damage_debug_done = True
                                    with torch.no_grad():
                                        std_time = enc_seq_dmg[0].std(dim=0).mean().item()
                                        mean_time = enc_seq_dmg[0].mean().item()
                                        print(
                                            f"[DEBUG DamageHeadInput] enc_seq_dmg[0] "
                                            f"mean={mean_time:.4f}, mean_std_over_time={std_time:.4f}"
                                        )
                                        print(
                                            "[DEBUG damage_head] health_phys_seq_batch[0, :10]:",
                                            health_phys_seq_batch[0, :10]
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        )
                                        print(
                                            "[DEBUG damage_head] hi_target_seq_[0, :10]:",
                                            hi_target_seq_[0, :10]
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        )
                                        print(
                                            "[DEBUG damage_head] hi_seq_damage_[0, :10]:",
                                            hi_seq_damage_[0, :10]
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        )
                                        print(
                                            f"[DEBUG damage_head] damage_hi_loss: "
                                            f"{damage_hi_loss.item():.6e}"
                                        )
                                
                                # ====================================================================
                                # DEBUG: Sanity-Check des Damage-Heads (einmal pro 5 Epochen)
                                # ====================================================================
                                # Note: epoch wird später im Training-Loop verfügbar sein
                                # Dieser Check wird im Training-Loop nach dem Forward gemacht
                        
                        # Condition calibration loss (sequence-based)
                        if hi_condition_calib_weight > 0.0:
                            calib_loss = compute_condition_calibration_loss(
                                health_seq=health_seq_flat,
                                rul_seq=rul_seq_batch,
                                cond_ids=cond_ids_batch,
                                hi_plateau_threshold=hi_plateau_threshold,
                                hi_eol_threshold=float(HI_EOL_THRESH),
                            )
                            loss = loss + hi_condition_calib_weight * calib_loss
                            if train_component_losses is not None:
                                train_component_losses["calib_loss"].append(calib_loss.item())
                        else:
                            if train_component_losses is not None:
                                train_component_losses["calib_loss"].append(0.0)

                        # NEW: condition reconstruction loss (encoder V2)
                        if supports_cond_recon and cond_seq_avg is not None and cond_recon is not None:
                            cond_loss_raw = F.mse_loss(cond_recon, cond_seq_avg)
                            cond_loss = cond_recon_weight * cond_loss_raw
                            loss = loss + cond_loss
                            if train_component_losses is not None:
                                train_component_losses["cond_loss"].append(cond_loss.item())
                        else:
                            if train_component_losses is not None:
                                train_component_losses["cond_loss"].append(0.0)
                        
                        # Store component losses
                        if train_component_losses is not None:
                            train_component_losses["mono_late_raw"].append(mono_late_raw.item())
                            train_component_losses["mono_late"].append(mono_late.item())
                            train_component_losses["mono_global_raw"].append(mono_global_raw.item())
                            train_component_losses["mono_global"].append(mono_global.item())
                            train_component_losses["smooth_hi_raw"].append(smooth_hi_raw.item())
                            train_component_losses["smooth_hi"].append(smooth_hi.item())
                            train_component_losses["damage_hi_loss"].append(damage_hi_loss.item())
                    else:
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                        loss_components = None
                scaler.scale(loss).backward()

                # Debug: log damage-head gradients for first few batches of epoch 1.
                # Hinweis: Hier sind die Gradienten noch skaliert, aber für die
                # Erkennung von "keine Gradienten" vs. "nicht-null" ist das egal.
                if epoch == 1 and batch_idx < 3:
                    log_damage_head_gradients(
                        model,
                        prefix=f"[DEBUG DamageHeadGrad] epoch={epoch} batch={batch_idx} (AMP, scaled)",
                    )

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_health_head:
                    # Phase 2: Pass cond_ids if condition embedding is enabled
                    supports_cond_recon = (
                        hasattr(model, "use_cond_recon_head")
                        and getattr(model, "use_cond_recon_head", False)
                        and hasattr(model, "fc_cond_recon")
                        and getattr(model, "fc_cond_recon") is not None
                        and cond_recon_weight > 0.0
                    )
                    cond_seq_avg = None
                    cond_recon = None
                    # Robust unpacking: newer Transformer variants may append rul_sigma and/or rul_quantiles.
                    if use_condition_embedding:
                        if supports_cond_recon:
                            out = model(X_batch, cond_ids=cond_ids_batch, return_aux=True)
                            if isinstance(out, (tuple, list)) and len(out) == 7:
                                # Fixed return_aux contract: (..., rul_sigma, rul_quantiles)
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _, _ = out
                            elif isinstance(out, (tuple, list)) and len(out) == 6:
                                # Older v5u contract: (..., rul_sigma)
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _ = out
                            else:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                        else:
                            rul_pred, health_last, health_seq = model(X_batch, cond_ids=cond_ids_batch)
                    else:
                        if supports_cond_recon:
                            out = model(X_batch, return_aux=True)
                            if isinstance(out, (tuple, list)) and len(out) == 7:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _, _ = out
                            elif isinstance(out, (tuple, list)) and len(out) == 6:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _ = out
                            else:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                        else:
                            rul_pred, health_last, health_seq = model(X_batch)
                    
                    # Construct RUL sequence for RUL-weighted monotonicity
                    # y_batch: (batch,) - RUL at last step
                    # seq_len: sequence length (e.g., 30)
                    rul_last = y_batch.view(-1, 1)  # (batch, 1)
                    seq_len = X_batch.size(1)  # int
                    # Construct offsets [seq_len-1, ..., 1, 0], so that last step has offset 0
                    offsets = torch.arange(seq_len - 1, -1, -1, device=y_batch.device).float()  # (seq_len,)
                    rul_seq_batch = rul_last + offsets  # broadcast to (batch, seq_len)
                    rul_seq_batch = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                    
                    # Prepare health_seq and rul_seq for loss computation
                    health_seq_flat = health_seq.squeeze(-1) if health_seq.dim() == 3 else health_seq
                    
                    # Compute multitask loss (without monotonicity, we'll add it separately)
                    loss, loss_components = multitask_rul_health_loss(
                        rul_pred=rul_pred.squeeze(-1) if rul_pred.dim() > 1 else rul_pred,
                        rul_true=y_batch,
                        health_pred=health_last.squeeze(-1) if health_last.dim() > 1 else health_last,
                        max_rul=max_rul,
                        tau=tau,
                        lambda_health=lambda_health,
                        plateau_thresh=float(HI_RUL_PLATEAU_THRESH),
                        hi_eol_thresh=float(HI_EOL_THRESH),
                        hi_eol_weight=HI_EOL_WEIGHT,
                        health_seq=None,  # We'll compute monotonicity separately
                        rul_seq=None,
                        hi_mono_weight=0.0,  # Disable monotonicity in multitask loss
                        hi_mono_rul_beta=HI_MONO_RUL_BETA,
                        return_components=True,
                    )
                    
                    # Late monotonicity loss (RUL <= beta)
                    if USE_HI_MONOTONICITY:
                        mono_late_raw, _ = compute_monotonicity_loss(
                            pred_hi=health_seq_flat,
                            rul=rul_seq_batch,
                            beta=HI_MONO_RUL_BETA,
                            weight=1.0,  # Weight applied outside
                        )
                        mono_late = mono_late_weight * mono_late_raw
                    else:
                        mono_late_raw = torch.tensor(0.0, device=health_seq.device)
                        mono_late = torch.tensor(0.0, device=health_seq.device)
                    
                    # Global trend loss (all cycles)
                    mono_global_raw = compute_global_trend_loss(health_seq_flat)
                    mono_global = mono_global_weight * mono_global_raw
                    
                    # Total monotonicity loss
                    mono_loss_total = mono_late + mono_global
                    loss = loss + mono_loss_total
                    
                    # Phase 2: Smoothness loss
                    smooth_hi_raw = health_smoothness_loss(
                        health_seq=health_seq_flat,
                        rul_seq=rul_seq_batch,
                        smooth_weight=1.0,  # Weight applied outside
                        plateau_rul_threshold=smooth_hi_plateau_threshold,
                    )
                    smooth_hi = smooth_hi_weight * smooth_hi_raw
                    loss = loss + smooth_hi

                    # Optional cumulative damage-based HI loss (validation)
                    damage_hi_loss = torch.tensor(0.0, device=health_seq_flat.device)
                    if (
                        damage_hi_weight > 0.0
                        and hasattr(model, "damage_head")
                        and getattr(model, "damage_head", None) is not None
                    ):
                        rul_seq_clamped = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                        hi_target_seq = torch.ones_like(rul_seq_clamped)
                        mask_eol = rul_seq_clamped <= float(HI_EOL_THRESH)
                        mask_plateau = rul_seq_clamped >= float(hi_plateau_threshold)
                        mask_mid = (~mask_eol) & (~mask_plateau)
                        denom = max(hi_plateau_threshold - float(HI_EOL_THRESH), 1e-6)
                        hi_target_seq[mask_eol] = 0.0
                        hi_target_seq[mask_mid] = (
                            (rul_seq_clamped[mask_mid] - float(HI_EOL_THRESH)) / denom
                        ).clamp(0.0, 1.0)

                        try:
                            enc_seq_dmg, _ = model.encode(
                                X_batch, cond_ids=cond_ids_batch, return_seq=True
                            )
                        except TypeError:
                            enc_seq_dmg = None

                        if enc_seq_dmg is not None:
                            cond_seq_dmg = None
                            if getattr(model, "use_cond_encoder", False) and getattr(
                                model, "cond_in_dim", 0
                            ) > 0:
                                if (
                                    hasattr(model, "cond_feature_indices")
                                    and model.cond_feature_indices is not None
                                    and len(model.cond_feature_indices)
                                    == getattr(model, "cond_in_dim", 0)
                                ):
                                    cond_seq_dmg = X_batch[:, :, model.cond_feature_indices]

                            hi_seq_damage, _, _, _ = model.damage_head(
                                enc_seq_dmg, cond_seq=cond_seq_dmg
                            )
                            damage_hi_loss = F.mse_loss(hi_seq_damage, hi_target_seq)
                            loss = loss + damage_hi_weight * damage_hi_loss
                    
                    # Condition calibration loss (sequence-based)
                    if hi_condition_calib_weight > 0.0:
                        calib_loss = compute_condition_calibration_loss(
                            health_seq=health_seq_flat,
                            rul_seq=rul_seq_batch,
                            cond_ids=cond_ids_batch,
                            hi_plateau_threshold=hi_plateau_threshold,
                            hi_eol_threshold=float(HI_EOL_THRESH),
                        )
                        loss = loss + hi_condition_calib_weight * calib_loss
                        if train_component_losses is not None:
                            train_component_losses["calib_loss"].append(calib_loss.item())
                    else:
                        if train_component_losses is not None:
                            train_component_losses["calib_loss"].append(0.0)

                    # NEW: condition reconstruction loss (encoder V2)
                    if supports_cond_recon and cond_seq_avg is not None and cond_recon is not None:
                        cond_loss_raw = F.mse_loss(cond_recon, cond_seq_avg)
                        cond_loss = cond_recon_weight * cond_loss_raw
                        loss = loss + cond_loss
                        if train_component_losses is not None:
                            train_component_losses["cond_loss"].append(cond_loss.item())
                    else:
                        if train_component_losses is not None:
                            train_component_losses["cond_loss"].append(0.0)
                    
                    # Store component losses
                    if train_component_losses is not None:
                        train_component_losses["mono_late_raw"].append(mono_late_raw.item())
                        train_component_losses["mono_late"].append(mono_late.item())
                        train_component_losses["mono_global_raw"].append(mono_global_raw.item())
                        train_component_losses["mono_global"].append(mono_global.item())
                        train_component_losses["smooth_hi_raw"].append(smooth_hi_raw.item())
                        train_component_losses["smooth_hi"].append(smooth_hi.item())
                else:
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss_components = None
                loss.backward()

                # Debug: log damage-head gradients for first few batches of epoch 1
                if epoch == 1 and batch_idx < 3:
                    log_damage_head_gradients(
                        model,
                        prefix=f"[DEBUG DamageHeadGrad] epoch={epoch} batch={batch_idx}",
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())
            
            # Store component losses for logging (if available)
            if use_health_head and loss_components is not None and train_component_losses is not None:
                train_component_losses["rul_loss_unscaled"].append(loss_components.get("rul_loss_unscaled", 0.0))
                train_component_losses["rul_loss"].append(loss_components["rul_loss"])
                train_component_losses["health_loss"].append(loss_components["health_loss"])
                train_component_losses["eol_health_loss"].append(loss_components["eol_health_loss"])
                # Note: mono_late, mono_global, and calib_loss are already appended above

        train_loss = float(np.mean(train_losses))
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Store train component losses if available
        if use_health_head and train_component_losses is not None:
            train_rul_loss_unscaled = float(np.mean(train_component_losses["rul_loss_unscaled"])) if train_component_losses["rul_loss_unscaled"] else 0.0
            train_rul_loss = float(np.mean(train_component_losses["rul_loss"])) if train_component_losses["rul_loss"] else 0.0
            train_health_loss = float(np.mean(train_component_losses["health_loss"])) if train_component_losses["health_loss"] else 0.0
            train_eol_health_loss = float(np.mean(train_component_losses["eol_health_loss"])) if train_component_losses["eol_health_loss"] else 0.0
            train_mono_late_raw = float(np.mean(train_component_losses["mono_late_raw"])) if train_component_losses["mono_late_raw"] else 0.0
            train_mono_late = float(np.mean(train_component_losses["mono_late"])) if train_component_losses["mono_late"] else 0.0
            train_mono_global_raw = float(np.mean(train_component_losses["mono_global_raw"])) if train_component_losses["mono_global_raw"] else 0.0
            train_mono_global = float(np.mean(train_component_losses["mono_global"])) if train_component_losses["mono_global"] else 0.0
            train_calib_loss = float(np.mean(train_component_losses["calib_loss"])) if train_component_losses["calib_loss"] else 0.0
            train_smooth_hi_raw = float(np.mean(train_component_losses["smooth_hi_raw"])) if train_component_losses["smooth_hi_raw"] else 0.0
            train_smooth_hi = float(np.mean(train_component_losses["smooth_hi"])) if train_component_losses["smooth_hi"] else 0.0
            # NEW: condition reconstruction loss (may be all zeros if disabled)
            train_cond_loss = float(np.mean(train_component_losses["cond_loss"])) if train_component_losses["cond_loss"] else 0.0
            # NEW: cumulative damage-based HI loss (may be all zeros if disabled)
            train_damage_hi_loss = float(np.mean(train_component_losses["damage_hi_loss"])) if train_component_losses["damage_hi_loss"] else 0.0

        # Validation
        model.eval()
        val_losses = []
        val_targets = []
        val_preds = []
        val_health_preds = []
        val_rul_losses_unscaled = []
        val_rul_losses = []
        val_health_losses = []
        val_eol_health_losses = []
        val_mono_late_raw = []
        val_mono_late = []
        val_mono_global_raw = []
        val_mono_global = []
        val_smooth_hi_raw = []  # Phase 2
        val_smooth_hi = []  # Phase 2
        val_calib_losses = []
        val_calib_losses_raw = []  # Unweighted calibration loss
        val_cond_losses = []  # NEW: condition reconstruction loss
        val_damage_hi_losses = []  # NEW: cumulative damage-based HI loss

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Handle different batch formats (same as training)
                health_phys_seq_batch = None
                if len(batch) == 5:
                    X_batch, y_batch, _, cond_ids_batch, health_phys_seq_batch = batch
                elif len(batch) == 4:  # SequenceDatasetWithUnits with cond_ids
                    X_batch, y_batch, _, cond_ids_batch = batch
                elif len(batch) == 3:  # SequenceDatasetWithUnits without cond_ids (backward compat)
                    X_batch, y_batch, _ = batch
                    cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64, device=device)
                else:  # Fallback für TensorDataset
                    X_batch, y_batch = batch
                    cond_ids_batch = torch.zeros(len(y_batch), dtype=torch.int64, device=device)
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if cond_ids_batch is not None:
                    cond_ids_batch = cond_ids_batch.to(device)
                # Move health_phys_seq to device if present
                if health_phys_seq_batch is not None:
                    health_phys_seq_batch = health_phys_seq_batch.to(device)

                if use_health_head:
                    # Phase 2: Pass cond_ids if condition embedding is enabled
                    supports_cond_recon = (
                        hasattr(model, "use_cond_recon_head")
                        and getattr(model, "use_cond_recon_head", False)
                        and hasattr(model, "fc_cond_recon")
                        and getattr(model, "fc_cond_recon") is not None
                        and cond_recon_weight > 0.0
                    )
                    cond_seq_avg = None
                    cond_recon = None
                    # Robust unpacking: newer Transformer v5u may append rul_sigma.
                    if use_condition_embedding:
                        if supports_cond_recon:
                            out = model(X_batch, cond_ids=cond_ids_batch, return_aux=True)
                            if isinstance(out, (tuple, list)) and len(out) == 6:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _ = out
                            else:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                        else:
                            rul_pred, health_last, health_seq = model(X_batch, cond_ids=cond_ids_batch)
                    else:
                        if supports_cond_recon:
                            out = model(X_batch, return_aux=True)
                            if isinstance(out, (tuple, list)) and len(out) == 6:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon, _ = out
                            else:
                                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon = out
                        else:
                            rul_pred, health_last, health_seq = model(X_batch)
                    
                    # Construct RUL sequence for RUL-weighted monotonicity
                    rul_last = y_batch.view(-1, 1)  # (batch, 1)
                    seq_len = X_batch.size(1)  # int
                    offsets = torch.arange(seq_len - 1, -1, -1, device=y_batch.device).float()  # (seq_len,)
                    rul_seq_batch = rul_last + offsets  # broadcast to (batch, seq_len)
                    rul_seq_batch = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                    
                    # --- DEBUG: Run monotonicity debug helper on first batch of first epoch ---
                    if epoch == 1 and batch_idx == 0:
                        from src.loss import debug_monotonicity_loss
                        # Use the same tensors that go into the loss function
                        health_seq_for_debug = health_seq.squeeze(-1) if health_seq.dim() == 3 else health_seq
                        debug_monotonicity_loss(
                            health_seq=health_seq_for_debug,
                            rul_seq=rul_seq_batch,
                            beta=HI_MONO_RUL_BETA,
                            max_print=5,
                        )
                    
                    # Prepare health_seq and rul_seq for loss computation
                    health_seq_flat = health_seq.squeeze(-1) if health_seq.dim() == 3 else health_seq
                    
                    # Compute multitask loss (without monotonicity, we'll add it separately)
                    loss, loss_components = multitask_rul_health_loss(
                        rul_pred=rul_pred.squeeze(-1) if rul_pred.dim() > 1 else rul_pred,
                        rul_true=y_batch,
                        health_pred=health_last.squeeze(-1) if health_last.dim() > 1 else health_last,
                        max_rul=max_rul,
                        tau=tau,
                        lambda_health=lambda_health,
                        plateau_thresh=float(HI_RUL_PLATEAU_THRESH),
                        hi_eol_thresh=float(HI_EOL_THRESH),
                        hi_eol_weight=HI_EOL_WEIGHT,
                        health_seq=None,  # We'll compute monotonicity separately
                        rul_seq=None,
                        hi_mono_weight=0.0,  # Disable monotonicity in multitask loss
                        hi_mono_rul_beta=HI_MONO_RUL_BETA,
                        return_components=True,
                    )
                    
                    # Late monotonicity loss (RUL <= beta)
                    if USE_HI_MONOTONICITY:
                        mono_late_raw, _ = compute_monotonicity_loss(
                            pred_hi=health_seq_flat,
                            rul=rul_seq_batch,
                            beta=HI_MONO_RUL_BETA,
                            weight=1.0,  # Weight applied outside
                        )
                        mono_late = mono_late_weight * mono_late_raw
                    else:
                        mono_late_raw = torch.tensor(0.0, device=health_seq.device)
                        mono_late = torch.tensor(0.0, device=health_seq.device)
                    
                    # Global trend loss (all cycles)
                    mono_global_raw = compute_global_trend_loss(health_seq_flat)
                    mono_global = mono_global_weight * mono_global_raw
                    
                    # Total monotonicity loss
                    mono_loss_total = mono_late + mono_global
                    loss = loss + mono_loss_total
                    
                    # Phase 2: Smoothness loss
                    smooth_hi_raw = health_smoothness_loss(
                        health_seq=health_seq_flat,
                        rul_seq=rul_seq_batch,
                        smooth_weight=1.0,  # Weight applied outside
                        plateau_rul_threshold=smooth_hi_plateau_threshold,
                    )
                    smooth_hi = smooth_hi_weight * smooth_hi_raw
                    loss = loss + smooth_hi
                    
                    # Condition calibration loss (sequence-based)
                    if hi_condition_calib_weight > 0.0:
                        calib_loss = compute_condition_calibration_loss(
                            health_seq=health_seq_flat,
                            rul_seq=rul_seq_batch,
                            cond_ids=cond_ids_batch,
                            hi_plateau_threshold=hi_plateau_threshold,
                            hi_eol_threshold=float(HI_EOL_THRESH),
                        )
                        loss = loss + hi_condition_calib_weight * calib_loss
                        val_calib_losses.append(calib_loss.item())
                        val_calib_losses_raw.append(calib_loss.item())  # Unweighted
                    else:
                        val_calib_losses.append(0.0)
                        val_calib_losses_raw.append(0.0)

                    # NEW: condition reconstruction loss (encoder V2)
                    if supports_cond_recon and cond_seq_avg is not None and cond_recon is not None:
                        cond_loss_raw = F.mse_loss(cond_recon, cond_seq_avg)
                        cond_loss = cond_recon_weight * cond_loss_raw
                        loss = loss + cond_loss
                        val_cond_losses.append(cond_loss.item())
                    else:
                        val_cond_losses.append(0.0)
                    
                    val_rul_losses_unscaled.append(loss_components.get("rul_loss_unscaled", 0.0))
                    val_rul_losses.append(loss_components["rul_loss"])
                    val_health_losses.append(loss_components["health_loss"])
                    val_eol_health_losses.append(loss_components["eol_health_loss"])
                    val_mono_late_raw.append(mono_late_raw.item())
                    val_mono_late.append(mono_late.item())
                    val_mono_global_raw.append(mono_global_raw.item())
                    val_mono_global.append(mono_global.item())
                    val_smooth_hi_raw.append(smooth_hi_raw.item())
                    val_smooth_hi.append(smooth_hi.item())
                    
                    # Optional cumulative damage-based HI loss (Transformer encoder with damage head)
                    val_damage_hi_loss_batch = torch.tensor(0.0, device=health_seq_flat.device)
                    if (
                        damage_hi_weight > 0.0
                        and hasattr(model, "damage_head")
                        and getattr(model, "damage_head", None) is not None
                    ):
                        # Determine HI target sequence (same logic as training)
                        if health_phys_seq_batch is not None:
                            hi_target_seq = health_phys_seq_batch  # [B, T]
                        else:
                            rul_seq_clamped = torch.clamp(rul_seq_batch, 0.0, float(max_rul))
                            hi_target_seq = torch.ones_like(rul_seq_clamped)
                            mask_eol = rul_seq_clamped <= float(HI_EOL_THRESH)
                            mask_plateau = rul_seq_clamped >= float(hi_plateau_threshold)
                            mask_mid = (~mask_eol) & (~mask_plateau)
                            denom = max(hi_plateau_threshold - float(HI_EOL_THRESH), 1e-6)
                            hi_target_seq[mask_eol] = 0.0
                            hi_target_seq[mask_mid] = (
                                (rul_seq_clamped[mask_mid] - float(HI_EOL_THRESH)) / denom
                            ).clamp(0.0, 1.0)
                        
                        try:
                            enc_seq_dmg, _ = model.encode(X_batch, cond_ids=cond_ids_batch, return_seq=True)
                        except TypeError:
                            enc_seq_dmg = None
                        
                        if enc_seq_dmg is not None:
                            cond_seq_dmg = None
                            if getattr(model, "use_cond_encoder", False) and getattr(model, "cond_in_dim", 0) > 0:
                                if (
                                    hasattr(model, "cond_feature_indices")
                                    and model.cond_feature_indices is not None
                                    and len(model.cond_feature_indices) == getattr(model, "cond_in_dim", 0)
                                ):
                                    cond_seq_dmg = X_batch[:, :, model.cond_feature_indices]
                            
                            hi_seq_damage, _, _, _ = model.damage_head(enc_seq_dmg, cond_seq=cond_seq_dmg)
                            
                            # Align shapes if necessary
                            if hi_seq_damage.shape != hi_target_seq.shape:
                                T = min(hi_seq_damage.size(1), hi_target_seq.size(1))
                                hi_seq_damage_ = hi_seq_damage[:, :T]
                                hi_target_seq_ = hi_target_seq[:, :T]
                            else:
                                hi_seq_damage_ = hi_seq_damage
                                hi_target_seq_ = hi_target_seq
                            
                            val_damage_hi_loss_batch = F.mse_loss(hi_seq_damage_, hi_target_seq_)
                    
                    val_damage_hi_losses.append(val_damage_hi_loss_batch.item())
                    val_cond_losses.append(val_cond_losses[-1] if val_cond_losses else 0.0)
                    
                    val_preds.append(rul_pred.cpu())
                    val_health_preds.append(health_last.cpu())
                else:
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_preds.append(preds.cpu())

                val_losses.append(loss.item())
                val_targets.append(y_batch.cpu())

        val_loss = float(np.mean(val_losses))
        # Ensure consistent shapes to avoid accidental broadcasting (e.g. [N,1] vs [N])
        val_targets_t = torch.cat(val_targets).view(-1)
        val_preds_t = torch.cat(val_preds).view(-1)
        val_rmse = torch.sqrt(torch.mean((val_preds_t - val_targets_t) ** 2)).item()
        
        # Health-specific metrics
        if use_health_head:
            val_rul_loss_unscaled = float(np.mean(val_rul_losses_unscaled))
            val_rul_loss = float(np.mean(val_rul_losses))
            val_health_loss = float(np.mean(val_health_losses))
            val_eol_health_loss = float(np.mean(val_eol_health_losses))
            val_mono_late_raw = float(np.mean(val_mono_late_raw))
            val_mono_late = float(np.mean(val_mono_late))
            val_mono_global_raw = float(np.mean(val_mono_global_raw))
            val_mono_global = float(np.mean(val_mono_global))
            # Phase 2: Smoothness loss
            val_smooth_hi_raw = float(np.mean(val_smooth_hi_raw)) if val_smooth_hi_raw else 0.0
            val_smooth_hi = float(np.mean(val_smooth_hi)) if val_smooth_hi else 0.0
            val_calib_loss_raw = float(np.mean(val_calib_losses_raw)) if val_calib_losses_raw else 0.0
            val_calib_loss = val_calib_loss_raw * hi_condition_calib_weight if hi_condition_calib_weight > 0.0 else 0.0
            val_cond_loss = float(np.mean(val_cond_losses)) if val_cond_losses else 0.0
            val_health_preds_tensor = torch.cat(val_health_preds)
            val_mean_health = float(val_health_preds_tensor.mean().item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(current_lr)
        
        if use_health_head:
            history["train_rul_loss_unscaled"].append(train_rul_loss_unscaled)
            history["train_rul_loss"].append(train_rul_loss)
            history["train_health_loss"].append(train_health_loss)
            history["train_eol_health_loss"].append(train_eol_health_loss)
            history["train_mono_loss_raw"].append(train_mono_late_raw + train_mono_global_raw)  # Total for backward compat
            history["train_mono_loss"].append(train_mono_late + train_mono_global)  # Total for backward compat
            history["val_rul_loss_unscaled"].append(val_rul_loss_unscaled)
            history["val_rul_loss"].append(val_rul_loss)
            history["val_health_loss"].append(val_health_loss)
            history["val_eol_health_loss"].append(val_eol_health_loss)
            history["val_mono_loss_raw"].append(val_mono_late_raw)  # Keep for backward compat
            history["val_mono_loss"].append(val_mono_late + val_mono_global)  # Total mono loss
            history["val_mean_health"].append(val_mean_health)
            history["train_mono_late_raw"].append(train_mono_late_raw)
            history["train_mono_late"].append(train_mono_late)
            history["train_mono_global_raw"].append(train_mono_global_raw)
            history["train_mono_global"].append(train_mono_global)
            history["val_mono_late_raw"].append(val_mono_late_raw)
            history["val_mono_late"].append(val_mono_late)
            history["val_mono_global_raw"].append(val_mono_global_raw)
            history["val_mono_global"].append(val_mono_global)
            history["train_smooth_hi_raw"].append(train_smooth_hi_raw)
            history["train_smooth_hi"].append(train_smooth_hi)
            history["val_smooth_hi_raw"].append(val_smooth_hi_raw)
            history["val_smooth_hi"].append(val_smooth_hi)
            history["train_damage_hi_loss"].append(train_damage_hi_loss)
            history["val_damage_hi_loss"].append(
                float(np.mean(val_damage_hi_losses)) if val_damage_hi_losses else 0.0
            )
            if hi_condition_calib_weight > 0.0:
                history["train_calib_loss"].append(train_calib_loss)
                history["val_calib_loss"].append(val_calib_loss)
            # Always log condition reconstruction losses (0.0 if disabled)
            history["train_cond_loss"].append(train_cond_loss)
            history["val_cond_loss"].append(val_cond_loss)

        # ====================================================================
        # DEBUG: Loss-Balance Check (pro Epoch)
        # ====================================================================
        if use_health_head and train_component_losses is not None:
            # Log damage_hi_loss explicitly (should already be in train_damage_hi_loss)
            log_damage_hi = train_damage_hi_loss if damage_hi_weight > 0.0 else 0.0

            # Determine current phase and effective damage weight (v3c)
            if damage_two_phase and epoch <= damage_warmup_epochs:
                phase_tag = "P1-Damage"
                damage_weight_eff_epoch = damage_phase1_damage_weight
            elif damage_two_phase:
                phase_tag = "P2-Multi"
                damage_weight_eff_epoch = damage_phase2_damage_weight
            else:
                phase_tag = "SinglePhase"
                damage_weight_eff_epoch = damage_hi_weight

            print(f"\n[DEBUG Loss-Balance] Epoch {epoch+1} ({phase_tag}):")
            print(f"  RUL Loss (scaled):        {train_rul_loss:.4f}")
            print(f"  HI Loss (standard):       {train_health_loss:.4f}")
            if damage_hi_weight > 0.0:
                print(f"  Damage HI Loss (raw):     {log_damage_hi:.4f}")
                print(
                    f"  Damage HI Loss (weighted): "
                    f"{(damage_weight_eff_epoch or 0.0) * log_damage_hi:.4f}"
                )
                print(f"  Damage HI Weight (eff.):  {damage_weight_eff_epoch:.4f}")
            else:
                print(f"  Damage HI Loss:            N/A (damage_hi_weight=0)")
            print(f"  Mono Late (weighted):      {train_mono_late:.4f}")
            print(f"  Mono Global (weighted):    {train_mono_global:.4f}")
            print(f"  Smooth HI (weighted):      {train_smooth_hi:.4f}")
            if hi_condition_calib_weight > 0.0:
                print(f"  Calib Loss (weighted):     {train_calib_loss:.4f}")
            if cond_recon_weight > 0.0:
                print(f"  Cond Recon (weighted):     {train_cond_loss:.4f}")
            print(f"  Total Loss:                {train_loss:.4f}")
        
        # ====================================================================
        # DEBUG: Sanity-Check des Damage-Heads (einmal pro 5 Epochen)
        # ====================================================================
        if (
            use_health_head
            and hasattr(model, "use_cum_damage_head")
            and getattr(model, "use_cum_damage_head", False)
            and hasattr(model, "damage_head")
            and getattr(model, "damage_head", None) is not None
            and epoch % 5 == 0
        ):
            try:
                # Use a sample batch from validation for consistency
                model.eval()
                with torch.no_grad():
                    # Get a sample batch
                    sample_batch = next(iter(val_loader))
                    if len(sample_batch) == 4:
                        X_sample, _, _, cond_ids_sample = sample_batch
                    elif len(sample_batch) == 3:
                        X_sample, _, _ = sample_batch
                        cond_ids_sample = torch.zeros(len(X_sample), dtype=torch.int64, device=device)
                    else:
                        X_sample, _ = sample_batch
                        cond_ids_sample = torch.zeros(len(X_sample), dtype=torch.int64, device=device)
                    
                    X_sample = X_sample[:1].to(device)  # Just one sample
                    cond_ids_sample = cond_ids_sample[:1].to(device)
                    
                    # Get encoder output
                    if hasattr(model, "encode"):
                        enc_seq_sample, _ = model.encode(
                            X_sample, cond_ids=cond_ids_sample, return_seq=True
                        )
                    else:
                        # Fallback: manual encoding
                        x_proj = model.input_proj(X_sample)
                        if getattr(model, "use_condition_embedding", False):
                            cond_emb = model.condition_embedding(cond_ids_sample)
                            cond_up = model.cond_proj(cond_emb)
                            cond_up = cond_up.unsqueeze(1).expand(-1, x_proj.shape[1], -1)
                            x_seq = x_proj + cond_up
                        else:
                            x_seq = x_proj
                        if getattr(model, "use_cond_encoder", False) and hasattr(model, "cond_encoder"):
                            if hasattr(model, "cond_feature_indices") and model.cond_feature_indices is not None:
                                cond_seq_sample = X_sample[:, :, model.cond_feature_indices]
                                cond_emb_seq = model.cond_encoder(cond_seq_sample)
                                x_seq = x_seq + cond_emb_seq
                        x_pos = model.pos_encoding(x_seq)
                        enc_seq_sample = model.transformer(x_pos)
                    
                    # Prepare cond_seq for damage_head
                    cond_seq_for_damage = None
                    if getattr(model, "use_cond_encoder", False):
                        if hasattr(model, "cond_feature_indices") and model.cond_feature_indices is not None:
                            cond_seq_for_damage = X_sample[:, :, model.cond_feature_indices]
                    
                    # Call damage_head
                    hi_seq_dbg, hi_last_dbg, damage_seq_dbg, delta_damage_dbg = model.damage_head(
                        enc_seq_sample, cond_seq=cond_seq_for_damage
                    )
                    
                    hi_seq_min = hi_seq_dbg.min().item()
                    hi_seq_max = hi_seq_dbg.max().item()
                    damage_min = damage_seq_dbg.min().item()
                    damage_max = damage_seq_dbg.max().item()
                    
                    print(
                        f"[DEBUG damage_head] Epoch {epoch+1}: "
                        f"hi_seq in [{hi_seq_min:.3f}, {hi_seq_max:.3f}], "
                        f"damage in [{damage_min:.3f}, {damage_max:.3f}]"
                    )
                
                # Restore training mode
                model.train() if epoch < num_epochs - 1 else model.eval()
            except Exception as e:
                # Silent fail to avoid breaking training
                pass

        # Scheduler
        scheduler.step(val_loss)

        # Best Model Tracking & Early Stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            checkpoint_path = results_dir / f"eol_full_lstm_best_{run_name}.pt"
            # Save checkpoint with metadata for easier loading
            # Extract input_dim and num_conditions from model (works for both EOLFullLSTMWithHealth and RULHIUniversalModelV2)
            input_dim_meta = None
            num_conditions_meta = None
            encoder_type_meta = None
            use_condition_embedding_meta = False
            
            if hasattr(model, "input_dim"):
                input_dim_meta = model.input_dim
            elif hasattr(model, "encoder") and hasattr(model.encoder, "input_dim"):
                input_dim_meta = model.encoder.input_dim
            
            if hasattr(model, "num_conditions"):
                num_conditions_meta = model.num_conditions
            elif hasattr(model, "encoder") and hasattr(model.encoder, "num_conditions"):
                num_conditions_meta = model.encoder.num_conditions
            
            if hasattr(model, "encoder"):
                if hasattr(model.encoder, "encoder_type"):
                    encoder_type_meta = model.encoder.encoder_type
                # For universal encoders, check if condition fusion is used
                if hasattr(model.encoder, "use_condition_fusion"):
                    use_condition_embedding_meta = model.encoder.use_condition_fusion
            
            if hasattr(model, "use_condition_embedding"):
                use_condition_embedding_meta = model.use_condition_embedding
            
            checkpoint = {
                "model_state_dict": best_model_state,
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "meta": {
                    "input_dim": input_dim_meta,
                    "num_conditions": num_conditions_meta,
                    "encoder_type": encoder_type_meta,
                    "use_condition_embedding": use_condition_embedding_meta,
                }
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_no_improve += 1

        if use_health_head:
            calib_str = ""
            if hi_condition_calib_weight > 0.0:
                calib_str = f", calib_raw: {val_calib_loss_raw:.6f}, calib: {val_calib_loss:.6f}"
            print(
                f"[EOL-Full-LSTM] Epoch {epoch}/{num_epochs} - "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"val_RMSE: {val_rmse:.4f}\n"
                f"  Loss components (val): "
                f"rul_unscaled: {val_rul_loss_unscaled:.4f}, rul_scaled: {val_rul_loss:.4f}, "
                f"health: {val_health_loss:.4f}, eol_health: {val_eol_health_loss:.4f}, "
                f"mono_late_raw: {val_mono_late_raw:.6f}, mono_late: {val_mono_late:.6f}, "
                f"mono_global_raw: {val_mono_global_raw:.6f}, mono_global: {val_mono_global:.6f}, "
                f"smooth_hi_raw: {val_smooth_hi_raw:.6f}, smooth_hi: {val_smooth_hi:.6f}{calib_str}, "
                f"mean_health: {val_mean_health:.3f}, lr: {current_lr:.2e}"
            )
        else:
            print(
                f"[EOL-Full-LSTM] Epoch {epoch}/{num_epochs} - "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"val_RMSE: {val_rmse:.4f}, lr: {current_lr:.2e}"
            )
        if val_loss < best_val_loss:
            print(f"  --> New best val_loss: {val_loss:.4f}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print(
                f"[EOL-Full-LSTM] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {epochs_no_improve} epochs)."
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"[EOL-Full-LSTM] Loaded best model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    # Plot training curves
    plot_training_curves(history, results_dir / f"training_curves_{run_name}.png")

    print("============================================================")
    print("[train_eol_full_lstm] Training Complete")
    print("============================================================")
    print(f"Best val_loss: {best_val_loss:.4f} (at epoch {best_epoch})")
    print("============================================================")

    return model, history


def plot_training_curves(history: Dict[str, Any], save_path: Path | str):
    """
    Plottet Trainingskurven.

    Args:
        history: Dictionary mit train_loss, val_loss, val_rmse, lr
        save_path: Pfad zum Speichern des Plots
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Train/Val Loss
    axes[0].plot(epochs, history["train_loss"], label="Train MSE", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val MSE", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Full-Trajectory LSTM Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Val RMSE + LR
    axes[1].plot(epochs, history["val_rmse"], label="Val RMSE", linewidth=2, color="green")
    if "lr" in history:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["lr"], label="Learning Rate", linewidth=1, color="orange", linestyle="--")
        ax2.set_ylabel("Learning Rate", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (cycles)", color="green")
    axes[1].tick_params(axis="y", labelcolor="green")
    axes[1].set_title("Full-Trajectory LSTM Validation RMSE & Learning Rate")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved training curves to {save_path}")
    plt.close()

