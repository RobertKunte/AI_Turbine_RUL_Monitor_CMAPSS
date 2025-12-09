"""
Training utility functions, including condition calibration loss and global trend loss.
"""

from __future__ import annotations

from typing import Tuple
import torch


def compute_condition_calibration_loss(
    health_seq: torch.Tensor,   # [batch, seq_len] or [batch, seq_len, 1]
    rul_seq: torch.Tensor,      # [batch, seq_len]
    cond_ids: torch.Tensor,     # [batch] int64
    hi_plateau_threshold: float = 80.0,
    hi_eol_threshold: float = 25.0,
) -> torch.Tensor:
    """
    Condition calibration loss for Health Index sequences.
    
    Reduces variance in HI means across different operating conditions in:
    - Healthy region (RUL >= hi_plateau_threshold): HI should be similar across conditions
    - EOL region (RUL <= hi_eol_threshold): HI should be similar across conditions (≈ 0)
    
    Args:
        health_seq: Predicted HI sequence, shape [batch, seq_len] or [batch, seq_len, 1]
        rul_seq: True RUL sequence, shape [batch, seq_len]
        cond_ids: Condition IDs per sequence, shape [batch] (0..C-1)
        hi_plateau_threshold: RUL threshold for healthy region
        hi_eol_threshold: RUL threshold for EOL region
    
    Returns:
        Scalar loss tensor
    """
    if health_seq.ndim == 3:
        health_seq = health_seq.squeeze(-1)  # [batch, seq_len]
    
    # Masks for healthy and EOL regions
    healthy_mask = rul_seq >= hi_plateau_threshold  # [batch, seq_len]
    eol_mask = rul_seq <= hi_eol_threshold          # [batch, seq_len]
    
    losses = []
    
    # Expand cond_ids to match rul_seq shape for masking
    cond_ids_expanded = cond_ids.unsqueeze(1).expand_as(rul_seq)  # [batch, seq_len]
    
    for mask, region_name in [(healthy_mask, "healthy"), (eol_mask, "eol")]:
        if mask.sum() == 0:
            continue
        
        # Extract values in this region
        h_vals = health_seq[mask]        # [N]
        c_vals = cond_ids_expanded[mask].float()  # [N]
        
        # Global mean in this region
        global_mean = h_vals.mean()
        
        # Condition-specific means
        unique_conds = cond_ids.unique()
        cond_means = []
        for cid in unique_conds:
            cond_mask = (c_vals == float(cid))
            if cond_mask.sum() == 0:
                continue
            cond_means.append(h_vals[cond_mask].mean())
        
        if len(cond_means) <= 1:
            # Only one condition or no valid conditions - skip variance penalty
            continue
        
        cond_means = torch.stack(cond_means)
        # Variance of condition means around global mean
        var = ((cond_means - global_mean) ** 2).mean()
        losses.append(var)
    
    if not losses:
        return torch.tensor(0.0, device=health_seq.device)
    
    return torch.stack(losses).mean()


def condition_calibration_loss(
    health_pred: torch.Tensor,   # [B] or [B, 1]
    rul_targets: torch.Tensor,   # [B]
    cond_ids: torch.Tensor,      # [B] int64
    plateau_threshold: float = 80.0,
    target_hi: float = 1.0,
    var_alpha: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Condition calibration for Health Index:
    - consider only early-life samples with RUL > plateau_threshold (roughly 'healthy')
    - for each condition, compute mean HI in that region and push it towards target_hi
    - additionally, penalize variance between condition means to align them
    
    Args:
        health_pred: Predicted health index, shape [B] or [B, 1]
        rul_targets: True RUL targets, shape [B]
        cond_ids: Condition IDs, shape [B] (int64)
        plateau_threshold: RUL threshold above which samples are considered "early-life"
        target_hi: Target HI value for early-life (typically 1.0)
        var_alpha: Weight for variance penalty between condition means
    
    Returns:
        Tuple of (raw_mean_loss, total_loss)
        - raw_mean_loss: MSE between condition means and target_hi (before variance penalty)
        - total_loss: Combined loss (mean_loss + var_alpha * var_loss)
    """
    # Reshape if needed
    if health_pred.dim() == 2 and health_pred.size(1) == 1:
        health_pred = health_pred[:, 0]
    rul_targets = rul_targets.view(-1)
    cond_ids = cond_ids.view(-1)
    
    # Early-life mask: RUL above plateau threshold
    mask = rul_targets > plateau_threshold
    if not mask.any():
        zero = health_pred.new_tensor(0.0)
        return zero, zero
    
    h = health_pred[mask]
    c = cond_ids[mask]
    
    # Compute mean HI per condition
    unique_cond = torch.unique(c)
    means = []
    for cond in unique_cond:
        idx = c == cond
        if not idx.any():
            continue
        means.append(h[idx].mean())
    
    if len(means) == 0:
        zero = health_pred.new_tensor(0.0)
        return zero, zero
    
    means_tensor = torch.stack(means)  # [num_conditions]
    
    # 1) Pull each condition mean towards target_hi
    mean_loss = ((means_tensor - target_hi) ** 2).mean()
    
    # 2) Reduce variance between condition means
    if means_tensor.numel() > 1:
        var_loss = means_tensor.var(unbiased=False)
    else:
        var_loss = means_tensor.new_tensor(0.0)
    
    total_loss = mean_loss + var_alpha * var_loss
    
    return mean_loss.detach(), total_loss


def compute_global_trend_loss(health_seq: torch.Tensor) -> torch.Tensor:
    """
    Global trend loss: penalizes increases in Health Index over the entire lifecycle.
    
    This is a softer version of monotonicity loss that applies to all cycles,
    not just the late-RUL region. It helps prevent HI "waves" in the middle lifecycle.
    
    Args:
        health_seq: Predicted HI sequence, shape [batch, seq_len] or [batch, seq_len, 1]
    
    Returns:
        Scalar loss tensor (mean of positive differences)
    """
    if health_seq.ndim == 3:
        health_seq = health_seq.squeeze(-1)  # [batch, seq_len]
    
    # Differences: HI(t+1) - HI(t)
    diffs = health_seq[:, 1:] - health_seq[:, :-1]  # [batch, seq_len-1]
    violations = torch.relu(diffs)  # Only increases > 0
    
    if violations.numel() == 0:
        return torch.tensor(0.0, device=health_seq.device)
    
    return violations.mean()

