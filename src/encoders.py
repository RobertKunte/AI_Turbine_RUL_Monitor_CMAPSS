"""
Sequence Encoders for RUL Prediction Models.

This module provides a clean abstraction for sequence encoders (LSTM, Transformer)
used in the EOL prediction pipeline.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import math


class BaseSequenceEncoder(nn.Module):
    """
    Base class for sequence encoders.
    
    All encoders should implement forward() to return:
    - Full sequence output: [B, T, H]
    - Last time step representation: [B, H]
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize base encoder.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output hidden dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
        
        Returns:
            out: [B, T, H] - Full sequence output
            last: [B, H] - Last time step representation
        """
        raise NotImplementedError


class LSTMEncoder(BaseSequenceEncoder):
    """
    LSTM-based sequence encoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        output_dim = hidden_dim * (2 if bidirectional else 1)
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LSTM."""
        out, (h_n, c_n) = self.lstm(x)  # out: [B, T, H * num_directions]
        last = h_n[-1]  # [B, H * num_directions] - last hidden state
        return out, last


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            x with positional encoding added: [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(BaseSequenceEncoder):
    """
    Transformer-based sequence encoder with positional encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 300,
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_len: Maximum sequence length for positional encoding
        """
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
                f"Please adjust d_model or nhead."
            )
        
        super().__init__(input_dim, d_model)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Transformer."""
        # Project input to model dimension
        x_proj = self.input_proj(x)  # [B, T, d_model]
        
        # Add positional encoding
        x_proj = self.pos_encoder(x_proj)
        
        # Transformer encoder
        out = self.transformer(x_proj)  # [B, T, d_model]
        
        # Return last time step
        last = out[:, -1, :]  # [B, d_model]
        
        return out, last


def build_encoder(encoder_type: str, input_dim: int, **kwargs) -> BaseSequenceEncoder:
    """
    Factory function to build an encoder from config.
    
    Args:
        encoder_type: "lstm" or "transformer"
        input_dim: Input feature dimension
        **kwargs: Encoder-specific parameters
    
    Returns:
        Encoder instance
    """
    if encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            bidirectional=kwargs.get("bidirectional", False),
        )
    elif encoder_type == "transformer":
        return TransformerEncoder(
            input_dim=input_dim,
            d_model=kwargs.get("d_model", 48),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 3),
            dim_feedforward=kwargs.get("dim_feedforward", 256),
            dropout=kwargs.get("dropout", 0.1),
            max_len=kwargs.get("max_len", 300),
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'lstm' or 'transformer'")

