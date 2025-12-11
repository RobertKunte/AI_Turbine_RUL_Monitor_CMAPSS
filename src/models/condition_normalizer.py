from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConditionNormalizer(nn.Module):
    """
    Maps condition feature vectors to expected 'healthy' sensor values.

    Used to compute sensor_resid = sensor_raw - g(cond).

    This module is intentionally small and dataset-agnostic; it operates only
    on tensors and knows nothing about specific CMAPSS feature names.
    """

    def __init__(
        self,
        cond_dim: int,
        sensor_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.sensor_dim = int(sensor_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.sensor_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: [B, T, cond_dim] condition feature sequence.

        Returns:
            baseline: [B, T, sensor_dim] predicted healthy sensor baseline.
        """
        if cond.dim() != 3:
            raise ValueError(f"cond must be [B,T,C], got shape {tuple(cond.shape)}")

        B, T, C = cond.shape
        if C != self.cond_dim:
            raise ValueError(
                f"[ConditionNormalizer] Expected cond_dim={self.cond_dim}, got {C}"
            )

        x = cond.reshape(B * T, C)
        y = self.net(x)  # [B*T, sensor_dim]
        return y.view(B, T, self.sensor_dim)


