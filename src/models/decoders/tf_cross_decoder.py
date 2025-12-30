"""
Transformer Cross-Attention Decoder for Non-Autoregressive RUL Prediction.

Uses TransformerDecoder with cross-attention to encoder memory.
Non-causal: no masking required for target sequence.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class CrossAttentionTFDecoder(nn.Module):
    """
    Non-autoregressive Transformer Decoder with Cross-Attention.
    
    Uses PyTorch's TransformerDecoderLayer and TransformerDecoder.
    Non-causal: tgt_mask=None (all future tokens can attend to each other).
    
    Args:
        d_model: Model dimension (must match encoder)
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # TransformerDecoderLayer with cross-attention
        # Note: PyTorch TransformerDecoder does NOT support batch_first=True
        # We'll handle batch dimension manually
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # PyTorch requires batch_first=False
            norm_first=True,  # Pre-norm architecture
            activation="gelu",
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query_tokens: torch.Tensor,  # (B, T_future, d_model)
        memory: torch.Tensor,  # (B, T_past, d_model)
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_past) bool, True=PAD
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_future) bool, True=PAD
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_tokens: (B, T_future, d_model) - Query tokens (future time embeddings)
            memory: (B, T_past, d_model) - Encoder output sequence
            memory_key_padding_mask: (B, T_past) bool, True=PAD (PyTorch convention)
            tgt_key_padding_mask: (B, T_future) bool, True=PAD (optional, for future padding)
        
        Returns:
            dec_out: (B, T_future, d_model) - Decoder output
        """
        # PyTorch TransformerDecoder expects (S, B, E) format
        # Transpose: (B, T, d_model) -> (T, B, d_model)
        tgt = query_tokens.transpose(0, 1)  # (T_future, B, d_model)
        mem = memory.transpose(0, 1)  # (T_past, B, d_model)
        
        # Non-causal: no tgt_mask (all future tokens can attend to each other)
        tgt_mask = None
        
        # Forward through decoder
        dec_out = self.decoder(
            tgt=tgt,
            memory=mem,
            tgt_mask=tgt_mask,  # None = no causal masking
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T_future, B, d_model)
        
        # Transpose back: (T_future, B, d_model) -> (B, T_future, d_model)
        dec_out = dec_out.transpose(0, 1)  # (B, T_future, d_model)
        
        # Layer norm
        dec_out = self.layer_norm(dec_out)
        
        return dec_out  # (B, T_future, d_model)

