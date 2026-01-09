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
    
    Handles two scaler_dict formats:
    - Format A (legacy): scaler_dict[cond] = StandardScaler
    - Format B (new):    scaler_dict[cond] = {
          'features': StandardScaler,
          'ops': MinMaxScaler,
          'non_ops_indices': [global indices scaled by 'features'],
          'ops_indices': [global indices scaled by 'ops'],
      }
    
    For Format B, cycle targets come from 'features' scaler, and target_indices
    (global) must be mapped to positions within non_ops_indices.
    
    Args:
        scaler_dict: Dict mapping condition_id -> StandardScaler or dict
        feature_cols: List of feature column names (same ordering as scaler)
        target_indices: Indices of cycle targets in feature_cols [T24, T30, P30, T50]
        num_conditions: Number of operating conditions
        
    Returns:
        mean_tensor: (num_conditions, n_targets) mean values
        std_tensor: (num_conditions, n_targets) std values
        
    Raises:
        ValueError: If scaler format is unrecognized, mapping fails, or stats are invalid
    """
    n_targets = len(target_indices)
    mean_list = []
    std_list = []
    
    # Get target names for error messages
    target_names = [feature_cols[i] if i < len(feature_cols) else f"idx{i}" for i in target_indices]
    
    for cond_id in range(num_conditions):
        scaler_entry = scaler_dict.get(cond_id, scaler_dict.get(0))
        
        if scaler_entry is None:
            raise ValueError(
                f"[extract_cycle_target_stats] No scaler for cond_id={cond_id}. "
                f"scaler_dict keys: {list(scaler_dict.keys())}. "
                f"Cannot proceed - identity fallback has been removed (C1 contract)."
            )
        
        # Detect format and extract scaler + index mapping
        if isinstance(scaler_entry, dict):
            # Format B: new dict format with separate ops/features scalers
            scaler = scaler_entry.get('features')
            non_ops_indices = scaler_entry.get('non_ops_indices')
            ops_indices = scaler_entry.get('ops_indices', [])
            
            if scaler is None:
                raise ValueError(
                    f"[extract_cycle_target_stats] scaler_dict[{cond_id}] is dict "
                    f"but missing 'features' key. Keys present: {list(scaler_entry.keys())}. "
                    f"Expected format: {{'features': StandardScaler, 'non_ops_indices': [...]}}"
                )
            
            if non_ops_indices is None:
                raise ValueError(
                    f"[extract_cycle_target_stats] scaler_dict[{cond_id}] is dict "
                    f"but missing 'non_ops_indices' key. Keys: {list(scaler_entry.keys())}. "
                    f"Cannot map target_indices to feature scaler space."
                )
            
            # Map target_indices (global) -> positions in non_ops_indices (feature scaler space)
            target_positions = []
            for tidx in target_indices:
                if tidx in ops_indices:
                    tname = feature_cols[tidx] if tidx < len(feature_cols) else f"idx{tidx}"
                    raise ValueError(
                        f"[extract_cycle_target_stats] target_index={tidx} ({tname}) "
                        f"is in ops_indices! Cycle targets must NOT be operating settings. "
                        f"ops_indices={ops_indices}, requested targets={target_names}"
                    )
                
                if tidx in non_ops_indices:
                    pos = non_ops_indices.index(tidx)
                    target_positions.append(pos)
                else:
                    tname = feature_cols[tidx] if tidx < len(feature_cols) else f"idx{tidx}"
                    raise ValueError(
                        f"[extract_cycle_target_stats] target_index={tidx} ({tname}) "
                        f"not found in non_ops_indices for cond_id={cond_id}. "
                        f"non_ops_indices has {len(non_ops_indices)} entries. "
                        f"Check that cycle targets are included in feature scaling."
                    )
        else:
            # Format A: legacy StandardScaler directly
            scaler = scaler_entry
            target_positions = target_indices  # Direct mapping
        
        # Extract mean/std from StandardScaler
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            raise ValueError(
                f"[extract_cycle_target_stats] Scaler for cond_id={cond_id} "
                f"has no mean_/scale_ attributes. Type: {type(scaler).__name__}. "
                f"Expected sklearn.preprocessing.StandardScaler."
            )
        
        scaler_mean = scaler.mean_
        scaler_std = scaler.scale_
        
        cond_mean = []
        cond_std = []
        for i, pos in enumerate(target_positions):
            if pos >= len(scaler_mean):
                raise ValueError(
                    f"[extract_cycle_target_stats] Position {pos} out of range "
                    f"for scaler (len={len(scaler_mean)}) at cond_id={cond_id}. "
                    f"Target: {target_names[i]}"
                )
            cond_mean.append(float(scaler_mean[pos]))
            cond_std.append(float(scaler_std[pos]))
        
        mean_list.append(np.array(cond_mean, dtype=np.float32))
        std_list.append(np.array(cond_std, dtype=np.float32))
    
    mean_tensor = torch.tensor(np.stack(mean_list, axis=0), dtype=torch.float32)
    std_tensor = torch.tensor(np.stack(std_list, axis=0), dtype=torch.float32)
    
    # C1 Validation: Ensure non-degenerate stats
    std_min = std_tensor.min().item()
    if std_min < 1e-6:
        raise ValueError(
            f"[extract_cycle_target_stats] Degenerate scaler stats! "
            f"std_min={std_min:.2e} < 1e-6. This indicates a constant feature. "
            f"Check that cycle targets have variance in training data."
        )
    
    # C1 Validation: Detect identity transform (mean≈0, std≈1 across all)
    mean_abs_max = mean_tensor.abs().max().item()
    std_deviation_from_1 = (std_tensor - 1.0).abs().max().item()
    if mean_abs_max < 1e-6 and std_deviation_from_1 < 1e-6:
        raise ValueError(
            f"[extract_cycle_target_stats] Scaler stats are IDENTITY (mean=0, std=1)! "
            f"This indicates scaler was not fitted or wrong scaler was passed. "
            f"mean_tensor={mean_tensor[0].tolist()}, std_tensor={std_tensor[0].tolist()}"
        )
    
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
