"""
Quantile Loss Functions for RUL Prediction.

Implements pinball loss and quantile crossing penalty.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pinball loss for quantile regression.
    
    Args:
        y_true: (B,) or (B, T) - Ground truth RUL values
        y_pred: (B, Q) or (B, T, Q) - Predicted quantiles
            where Q = len(quantiles), ordered as [q10, q50, q90]
        quantiles: (Q,) - Quantile values (e.g., [0.1, 0.5, 0.9])
        mask: (B,) or (B, T) - Optional mask (1.0 = valid, 0.0 = invalid)
    
    Returns:
        loss: Scalar tensor - Average pinball loss
    """
    # Ensure quantiles is on same device
    quantiles = quantiles.to(y_true.device)
    
    # Handle different input shapes
    if y_true.dim() == 1:
        # (B,) -> (B, 1) for broadcasting
        y_true = y_true.unsqueeze(-1)  # (B, 1)
        # y_pred: (B, Q)
        # Expand y_true to (B, Q) for element-wise operations
        y_true = y_true.expand(-1, y_pred.size(-1))  # (B, Q)
    elif y_true.dim() == 2:
        # (B, T) -> (B, T, 1)
        y_true = y_true.unsqueeze(-1)  # (B, T, 1)
        # y_pred: (B, T, Q)
        # Expand y_true to (B, T, Q)
        y_true = y_true.expand(-1, -1, y_pred.size(-1))  # (B, T, Q)
    else:
        raise ValueError(f"y_true must be 1D or 2D, got shape {y_true.shape}")
    
    # Expand quantiles for broadcasting
    if y_pred.dim() == 2:
        # (B, Q) case
        quantiles = quantiles.unsqueeze(0).expand(y_pred.size(0), -1)  # (B, Q)
    elif y_pred.dim() == 3:
        # (B, T, Q) case
        quantiles = quantiles.unsqueeze(0).unsqueeze(0).expand(
            y_pred.size(0), y_pred.size(1), -1
        )  # (B, T, Q)
    else:
        raise ValueError(f"y_pred must be 2D or 3D, got shape {y_pred.shape}")
    
    # Compute error: y_true - y_pred
    error = y_true - y_pred  # (B, Q) or (B, T, Q)
    
    # Pinball loss: max(q * error, (q - 1) * error)
    loss_q = torch.max(
        quantiles * error,
        (quantiles - 1.0) * error,
    )  # (B, Q) or (B, T, Q)
    
    # Average over quantiles
    loss = loss_q.mean(dim=-1)  # (B,) or (B, T)
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 1:
            # (B,) -> (B, 1) for broadcasting
            mask = mask.unsqueeze(-1)  # (B, 1)
        # mask: (B,) or (B, T)
        loss = loss * mask
        # Normalize by sum of mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = loss.sum() / mask_sum
        else:
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    else:
        # Average over batch (and time if applicable)
        loss = loss.mean()
    
    return loss


def quantile_crossing_penalty(
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Penalty for quantile crossing violations.
    
    Assumes quantiles are ordered [q10, q50, q90].
    Penalizes when q10 > q50 or q50 > q90.
    
    Args:
        y_pred: (..., 3) - Predicted quantiles [q10, q50, q90]
        mask: (...,) - Optional mask (1.0 = valid, 0.0 = invalid)
        margin: Margin for penalty (default: 0.0)
    
    Returns:
        penalty: Scalar tensor - Average crossing penalty
    """
    if y_pred.size(-1) != 3:
        raise ValueError(
            f"y_pred must have 3 quantiles (q10, q50, q90), got {y_pred.size(-1)}"
        )
    
    # Extract quantiles: [q10, q50, q90]
    q10 = y_pred[..., 0]  # (...,)
    q50 = y_pred[..., 1]  # (...,)
    q90 = y_pred[..., 2]  # (...,)
    
    # Penalty: relu(q10 - q50 + margin) + relu(q50 - q90 + margin)
    penalty_10_50 = torch.relu(q10 - q50 + margin)  # (...,)
    penalty_50_90 = torch.relu(q50 - q90 + margin)  # (...,)
    
    penalty = penalty_10_50 + penalty_50_90  # (...,)
    
    # Apply mask if provided
    if mask is not None:
        penalty = penalty * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            penalty = penalty.sum() / mask_sum
        else:
            penalty = torch.tensor(0.0, device=penalty.device, dtype=penalty.dtype)
    else:
        penalty = penalty.mean()
    
    return penalty


def quantile_crossing_rate(
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute fraction of timesteps where quantiles cross.
    
    Args:
        y_pred: (..., 3) - Predicted quantiles [q10, q50, q90]
        mask: (...,) - Optional mask (1.0 = valid, 0.0 = invalid)
    
    Returns:
        rate: float - Fraction of valid timesteps with crossings
    """
    if y_pred.size(-1) != 3:
        raise ValueError(
            f"y_pred must have 3 quantiles (q10, q50, q90), got {y_pred.size(-1)}"
        )
    
    q10 = y_pred[..., 0]  # (...,)
    q50 = y_pred[..., 1]  # (...,)
    q90 = y_pred[..., 2]  # (...,)
    
    # Check crossings
    crosses_10_50 = (q10 > q50).float()  # (...,)
    crosses_50_90 = (q50 > q90).float()  # (...,)
    crosses = (crosses_10_50 + crosses_50_90 > 0).float()  # (...,)
    
    # Apply mask if provided
    if mask is not None:
        crosses = crosses * mask
        mask_sum = mask.sum().item()
        if mask_sum > 0:
            rate = crosses.sum().item() / mask_sum
        else:
            rate = 0.0
    else:
        rate = crosses.mean().item()
    
    return rate


def quantile_coverage(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute fraction of y_true values within [q10, q90] interval.
    
    Args:
        y_true: (B,) or (B, T) - Ground truth RUL values
        y_pred: (B, 3) or (B, T, 3) - Predicted quantiles [q10, q50, q90]
        mask: (B,) or (B, T) - Optional mask (1.0 = valid, 0.0 = invalid)
    
    Returns:
        coverage: float - Fraction of valid samples within [q10, q90]
    """
    if y_pred.size(-1) != 3:
        raise ValueError(
            f"y_pred must have 3 quantiles (q10, q50, q90), got {y_pred.size(-1)}"
        )
    
    q10 = y_pred[..., 0]  # (...,)
    q90 = y_pred[..., 2]  # (...,)
    
    # Check if y_true is within [q10, q90]
    if y_true.dim() == 1:
        # (B,) -> (B, 1) for broadcasting
        y_true = y_true.unsqueeze(-1)  # (B, 1)
        # Expand to match q10/q90 shapes
        y_true = y_true.expand(-1, q10.size(-1) if q10.dim() > 1 else 1)
    
    in_interval = ((y_true >= q10) & (y_true <= q90)).float()  # (...,)
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 1 and in_interval.dim() == 2:
            mask = mask.unsqueeze(-1)  # (B, 1)
        in_interval = in_interval * mask
        mask_sum = mask.sum().item()
        if mask_sum > 0:
            coverage = in_interval.sum().item() / mask_sum
        else:
            coverage = 0.0
    else:
        coverage = in_interval.mean().item()
    
    return coverage

