"""
Cycle Target Scaler Extraction.

This module provides utilities for extracting per-condition mean/std
for cycle branch targets from the fitted condition-wise scalers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


def extract_cycle_target_stats(
    scaler_dict: Dict[int, any],
    feature_cols: List[str],
    target_indices: List[int],
    num_conditions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract mean/std for cycle targets from condition-wise scalers.
    
    Args:
        scaler_dict: Dict mapping condition_id -> fitted StandardScaler
        feature_cols: List of feature column names (same ordering as scaler)
        target_indices: Indices of cycle targets in feature_cols [T24, T30, P30, T50]
        num_conditions: Number of operating conditions
        
    Returns:
        mean_tensor: (num_conditions, n_targets) mean values
        std_tensor: (num_conditions, n_targets) std values
    """
    n_targets = len(target_indices)
    
    mean_list = []
    std_list = []
    
    for cond_id in range(num_conditions):
        scaler = scaler_dict.get(cond_id, scaler_dict.get(0))
        
        if scaler is None:
            # Fallback to zeros/ones if no scaler
            mean_list.append(np.zeros(n_targets, dtype=np.float32))
            std_list.append(np.ones(n_targets, dtype=np.float32))
            continue
        
        # Extract mean and scale (std) for target indices
        try:
            # StandardScaler stores mean_ and scale_ (scale_ = std for StandardScaler)
            scaler_mean = scaler.mean_
            scaler_std = scaler.scale_
            
            cond_mean = []
            cond_std = []
            for idx in target_indices:
                if idx < len(scaler_mean):
                    cond_mean.append(float(scaler_mean[idx]))
                    cond_std.append(float(scaler_std[idx]))
                else:
                    cond_mean.append(0.0)
                    cond_std.append(1.0)
            
            mean_list.append(np.array(cond_mean, dtype=np.float32))
            std_list.append(np.array(cond_std, dtype=np.float32))
        except AttributeError:
            # Scaler doesn't have mean_/scale_ attributes
            mean_list.append(np.zeros(n_targets, dtype=np.float32))
            std_list.append(np.ones(n_targets, dtype=np.float32))
    
    mean_tensor = torch.tensor(np.stack(mean_list, axis=0), dtype=torch.float32)  # (C, 4)
    std_tensor = torch.tensor(np.stack(std_list, axis=0), dtype=torch.float32)    # (C, 4)
    
    return mean_tensor, std_tensor


def normalize_cycle_tensors(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    cond_ids: torch.Tensor,
    mean_buffer: torch.Tensor,
    std_buffer: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize cycle pred/target using condition-wise stats.
    
    Args:
        pred_raw: Raw cycle predictions (B, T, 4) or (B, 4)
        target_raw: Raw cycle targets (same shape as pred_raw)
        cond_ids: Condition IDs (B,) or (B, T)
        mean_buffer: (num_conditions, 4) mean values
        std_buffer: (num_conditions, 4) std values
        eps: Small value for numerical stability
        
    Returns:
        pred_norm: Normalized predictions
        target_norm: Normalized targets
    """
    is_seq = pred_raw.dim() == 3
    
    # Get mean/std for each batch element based on cond_id
    if cond_ids.dim() == 1:
        # cond_ids: (B,) -> broadcast to (B, 1, 4) or (B, 4)
        mean = mean_buffer[cond_ids]  # (B, 4)
        std = std_buffer[cond_ids]    # (B, 4)
        
        if is_seq:
            mean = mean.unsqueeze(1)  # (B, 1, 4) - broadcasts over T
            std = std.unsqueeze(1)
    else:
        # cond_ids: (B, T) -> need to gather for each timestep
        # This is rare, but handle it
        B, T = cond_ids.shape
        mean = mean_buffer[cond_ids.flatten()].reshape(B, T, -1)  # (B, T, 4)
        std = std_buffer[cond_ids.flatten()].reshape(B, T, -1)
    
    # Normalize
    pred_norm = (pred_raw - mean) / (std + eps)
    target_norm = (target_raw - mean) / (std + eps)
    
    return pred_norm, target_norm
