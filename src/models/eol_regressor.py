# src/models/eol_regressor.py

from typing import Sequence
try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this notebook. Please install torch."
    ) from exc

class EOLRegressor(nn.Module):
    """
    Small MLP head that takes an encoder summary embedding and predicts
    a scalar End-of-Life RUL value.

    This is designed to sit on top of the encoder of the WorldModelEncoderDecoderMultiTask.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

        layers.append(nn.Linear(last_dim, 1))  # scalar RUL

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: encoder summary, shape [batch_size, in_dim]

        Returns:
            RUL prediction, shape [batch_size, 1]
        """
        return self.net(x)
