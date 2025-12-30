"""
Future Query Builder for Non-Autoregressive Transformer Decoder.

Builds query tokens for future time steps without any sensor leakage.
Only uses time embeddings and optional condition embeddings.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class FutureQueryBuilder(nn.Module):
    """
    Builds query tokens for future time steps.
    
    NO LEAKAGE: Only uses time embeddings and optional condition embeddings.
    No future sensor values are used.
    
    Args:
        d_model: Model dimension (must match encoder)
        max_future: Maximum future sequence length (for embedding table)
        cond_dim: Dimension of condition embedding (0 if unused)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 96,
        max_future: int = 256,
        cond_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_future = max_future
        self.cond_dim = cond_dim
        self.use_cond = cond_dim > 0
        
        # Time embedding: positions 1..max_future (NOT 0)
        # We use 1-indexed positions: position 1 = first future step
        self.time_embedding = nn.Embedding(max_future + 1, d_model)  # +1 for 0-index
        
        # Optional condition projection
        if self.use_cond:
            self.cond_proj = nn.Linear(cond_dim, d_model)
        else:
            self.cond_proj = None
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        T_future: int,
        batch_size: int,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build query tokens for future time steps.
        
        Args:
            T_future: Number of future steps to predict
            batch_size: Batch size
            device: Device for tensors
            cond: Optional condition tensor (B, cond_dim) or None
        
        Returns:
            query_tokens: (B, T_future, d_model) - Query tokens for decoder
        """
        if T_future > self.max_future:
            raise ValueError(
                f"T_future ({T_future}) exceeds max_future ({self.max_future}). "
                f"Increase max_future in FutureQueryBuilder initialization."
            )
        
        # Time positions: 1, 2, ..., T_future (1-indexed)
        time_positions = torch.arange(1, T_future + 1, device=device)  # (T_future,)
        time_positions = time_positions.unsqueeze(0).expand(batch_size, -1)  # (B, T_future)
        
        # Time embeddings: (B, T_future, d_model)
        query_tokens = self.time_embedding(time_positions)  # (B, T_future, d_model)
        
        # Optional condition embedding: broadcast to all future tokens
        if self.use_cond and cond is not None:
            if cond.dim() != 2 or cond.size(0) != batch_size or cond.size(1) != self.cond_dim:
                raise ValueError(
                    f"cond must be (B={batch_size}, cond_dim={self.cond_dim}), "
                    f"got {cond.shape}"
                )
            cond_emb = self.cond_proj(cond)  # (B, d_model)
            cond_emb = cond_emb.unsqueeze(1).expand(-1, T_future, -1)  # (B, T_future, d_model)
            query_tokens = query_tokens + cond_emb
        
        # Dropout and layer norm
        query_tokens = self.dropout(query_tokens)
        query_tokens = self.layer_norm(query_tokens)
        
        return query_tokens  # (B, T_future, d_model)

