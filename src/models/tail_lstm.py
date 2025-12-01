"""
Tail-LSTM Regressor for EOL prediction on tail degradation data.

This module provides a LSTM-based regressor specifically designed for
predicting Remaining Useful Life (RUL) from tail degradation sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TailLSTMConfig:
    """Configuration for TailLSTMRegressor."""
    input_dim: int            # num_features = 24
    hidden_dim: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.1


class TailLSTMRegressor(nn.Module):
    """
    LSTM-basierter Tail-EOL-Regressor.

    Erwartet Eingaben der Form:
        x: [batch_size, seq_len, input_dim]
    und gibt RUL als Skalar pro Sample zurück:
        y_hat: [batch_size]
    """

    def __init__(self, config: TailLSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,  # x: [B, T, F]
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        lstm_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_out_dim, 1),
        )

        # Optional: init etwas „sanfter“
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
        # LSTM-Output und letzter Hidden State
        # output: [B, T, H * num_directions]
        # h_n: [num_layers * num_directions, B, H]
        output, (h_n, c_n) = self.lstm(x)

        if self.config.bidirectional:
            # Letzte Layer, beide Richtungen: concat
            h_last_fwd = h_n[-2]  # [B, H]
            h_last_bwd = h_n[-1]  # [B, H]
            h_last = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        y_hat = self.head(h_last).squeeze(-1)  # [B]
        return y_hat

