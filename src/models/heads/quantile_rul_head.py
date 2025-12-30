"""
Quantile RUL Prediction Head.

Predicts multiple quantiles (q10, q50, q90) of RUL distribution.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


class QuantileRULHead(nn.Module):
    """
    Quantile RUL prediction head.
    
    Predicts multiple quantiles of the RUL distribution.
    
    Args:
        d_model: Input dimension
        quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 96,
        quantiles: List[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.d_model = d_model
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Small MLP: Dropout -> Linear -> GELU -> Linear
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_quantiles),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, T, d_model) or (B, d_model) - Input features
        
        Returns:
            quantiles: (B, T, Q) or (B, Q) - Quantile predictions
                where Q = num_quantiles
        """
        return self.mlp(x)  # (B, T, Q) or (B, Q)

