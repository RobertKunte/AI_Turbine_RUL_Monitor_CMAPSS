from __future__ import annotations

from typing import List, Sequence

import torch


def pinball_loss(
    y_true: torch.Tensor,
    y_pred_q: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """
    Pinball (quantile) loss.

    Args:
        y_true: [B] or [B,1]
        y_pred_q: [B,Q]
        quantiles: list of Q quantile levels in (0,1)

    Returns:
        scalar loss (mean over batch and quantiles)
    """
    if y_true.dim() > 1:
        y_true = y_true.view(-1)
    if y_pred_q.dim() != 2:
        raise ValueError(f"y_pred_q must be [B,Q], got {tuple(y_pred_q.shape)}")
    q_list: List[float] = [float(x) for x in quantiles]
    if len(q_list) != int(y_pred_q.size(1)):
        raise ValueError(
            f"quantiles length {len(q_list)} must match y_pred_q.shape[1]={int(y_pred_q.size(1))}"
        )
    qs = torch.tensor(q_list, device=y_pred_q.device, dtype=y_pred_q.dtype).view(1, -1)  # [1,Q]
    y = y_true.view(-1, 1)  # [B,1]
    u = y - y_pred_q  # [B,Q]
    return torch.maximum(qs * u, (qs - 1.0) * u).mean()


def non_crossing_penalty(y_pred_q: torch.Tensor) -> torch.Tensor:
    """
    Penalize quantile crossings (enforce non-decreasing quantiles).
    """
    if y_pred_q.dim() != 2 or int(y_pred_q.size(1)) < 2:
        return torch.zeros((), device=y_pred_q.device, dtype=y_pred_q.dtype)
    diffs = y_pred_q[:, :-1] - y_pred_q[:, 1:]  # should be <= 0
    return torch.relu(diffs).mean()


