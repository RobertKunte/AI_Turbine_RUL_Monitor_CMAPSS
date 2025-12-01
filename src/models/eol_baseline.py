# src/models/eol_baseline.py
"""
Simple EOL-Baseline LSTM Model.

This is a clean, standalone EOL predictor that serves as a reference
for comparing World Model EOL performance.
"""
from typing import Optional

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this module. Please install torch."
    ) from exc


class EOLLSTM(nn.Module):
    """
    Simple EOL-Baseline:
    - LSTM über die letzten `past_len` Zyklen
    - EOL-Prediction aus dem letzten Hidden-State
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, F) - Input sequences

        Returns:
            y: (B,) - EOL predictions
        """
        # x: (B, L, F)
        out, (h_n, c_n) = self.lstm(x)
        # letzter Layer, letzter Zeitschritt: (num_layers, B, H) -> (B, H)
        h_last = h_n[-1]
        y = self.head(h_last).squeeze(-1)  # (B,)
        return y

