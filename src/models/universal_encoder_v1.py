"""
Universal Encoder V1 - CNN + Transformer Hybrid Architecture.

This module implements a universal sequence encoder that combines:
- Multi-scale temporal CNN front-end
- Sensor-wise attention
- Condition Fusion (FiLM)
- Transformer encoder
- Aggregation to single sequence embedding

Designed to work across all FD001-FD004 datasets.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


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


class UniversalEncoderV1(nn.Module):
    """
    Universal Encoder V1: CNN + Transformer Hybrid with Condition Fusion.
    
    Architecture:
    1. Multi-scale CNN front-end (kernels 3, 5, 7)
    2. Sensor-wise attention (per-feature gating)
    3. Condition Fusion via FiLM (Feature-wise Linear Modulation)
    4. Transformer encoder over time
    5. Aggregation (last token + mean pooling)
    
    Args:
        input_dim: Number of input features
        d_model: Model dimension (default: 48)
        cnn_channels: Number of channels per CNN branch (default: d_model // 3)
        num_layers: Number of transformer encoder layers (default: 3)
        nhead: Number of attention heads (default: 4)
        dim_feedforward: Feedforward network dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        num_conditions: Number of unique conditions (None for single condition)
        cond_emb_dim: Dimension of condition embeddings (default: 4)
        max_seq_len: Maximum sequence length for positional encoding (default: 300)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 48,
        cnn_channels: Optional[int] = None,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        max_seq_len: int = 300,
    ):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.use_condition_fusion = num_conditions is not None and num_conditions > 1
        
        # Default cnn_channels
        if cnn_channels is None:
            cnn_channels = d_model // 3
            # Ensure it's at least 8
            cnn_channels = max(8, cnn_channels)
        
        # ===================================================================
        # 1. Multi-scale CNN front-end
        # ===================================================================
        kernel_sizes = [3, 5, 7]
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=cnn_channels,
                    kernel_size=ks,
                    padding=ks // 2,  # Same padding
                ),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_channels),
                nn.Dropout(dropout),
            )
            for ks in kernel_sizes
        ])
        
        # Project concatenated CNN outputs to d_model
        self.cnn_out_proj = nn.Conv1d(
            len(kernel_sizes) * cnn_channels,
            d_model,
            kernel_size=1,
        )
        
        # ===================================================================
        # 2. Sensor-wise attention
        # ===================================================================
        self.sensor_att_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),  # Gating weights in (0, 1)
        )
        
        # ===================================================================
        # 3. Condition Fusion (FiLM)
        # ===================================================================
        if self.use_condition_fusion:
            self.cond_emb = nn.Embedding(num_conditions, cond_emb_dim)
            # FiLM: generates gamma (scale) and beta (shift) from condition embedding
            self.cond_film = nn.Linear(cond_emb_dim, 2 * d_model)
        else:
            self.cond_emb = None
            self.cond_film = None
        
        # ===================================================================
        # 4. Transformer encoder
        # ===================================================================
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===================================================================
        # 5. Aggregation
        # ===================================================================
        self.out_proj = nn.Linear(2 * d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming/He initialization for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
            cond_ids: [B] - Optional condition IDs (int64). Required if num_conditions > 1.
        
        Returns:
            seq_emb: [B, d_model] - Sequence embedding
        """
        B, T, F = x.shape
        
        # ===================================================================
        # 1. Multi-scale CNN
        # ===================================================================
        # Transpose for Conv1d: (B, T, F) -> (B, F, T)
        x_conv = x.transpose(1, 2)  # [B, F, T]
        
        # Apply CNN branches
        cnn_outputs = []
        for branch in self.cnn_branches:
            out = branch(x_conv)  # [B, cnn_channels, T]
            cnn_outputs.append(out)
        
        # Concatenate along channel dimension
        x_cnn = torch.cat(cnn_outputs, dim=1)  # [B, 3*cnn_channels, T]
        
        # Project to d_model
        x_cnn = self.cnn_out_proj(x_cnn)  # [B, d_model, T]
        
        # Transpose back: (B, d_model, T) -> (B, T, d_model)
        x_cnn = x_cnn.transpose(1, 2)  # [B, T, d_model]
        
        # ===================================================================
        # 2. Sensor-wise attention
        # ===================================================================
        att_weights = self.sensor_att_mlp(x_cnn)  # [B, T, d_model]
        x_att = x_cnn * att_weights  # Element-wise gating
        
        # ===================================================================
        # 3. Condition Fusion (FiLM)
        # ===================================================================
        if self.use_condition_fusion:
            if cond_ids is None:
                raise ValueError("cond_ids required when num_conditions > 1")
            
            # Condition embedding
            c = self.cond_emb(cond_ids)  # [B, cond_emb_dim]
            
            # FiLM: generate gamma and beta
            gamma_beta = self.cond_film(c)  # [B, 2*d_model]
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each: [B, d_model]
            
            # Reshape for broadcasting over time
            gamma = gamma.unsqueeze(1)  # [B, 1, d_model]
            beta = beta.unsqueeze(1)    # [B, 1, d_model]
            
            # FiLM modulation: x_fused = x * (1 + gamma) + beta
            x_fused = x_att * (1 + gamma) + beta  # [B, T, d_model]
        else:
            x_fused = x_att
        
        # ===================================================================
        # 4. Transformer encoder
        # ===================================================================
        x_pos = self.pos_encoding(x_fused)  # [B, T, d_model]
        h = self.transformer(x_pos)  # [B, T, d_model]
        
        # ===================================================================
        # 5. Aggregation
        # ===================================================================
        h_last = h[:, -1, :]  # [B, d_model] - last token
        h_mean = h.mean(dim=1)  # [B, d_model] - mean pooling
        
        h_cat = torch.cat([h_last, h_mean], dim=-1)  # [B, 2*d_model]
        seq_emb = self.out_proj(h_cat)  # [B, d_model]
        
        return seq_emb


class RULHIUniversalModel(nn.Module):
    """
    RUL + Health Index model using UniversalEncoderV1.
    
    This model wraps UniversalEncoderV1 and adds RUL and HI prediction heads.
    It maintains the same API as EOLFullLSTMWithHealth for compatibility.
    
    Args:
        encoder: UniversalEncoderV1 instance
        d_model: Model dimension (should match encoder.d_model)
        dropout: Dropout rate for heads
    """
    
    def __init__(
        self,
        encoder: UniversalEncoderV1,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.d_model = d_model
        
        # For compatibility with existing training code
        self.use_condition_embedding = encoder.use_condition_fusion
        
        # Shared feature extraction
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # RUL head
        self.fc_rul = nn.Linear(d_model, 1)
        
        # Health Index head
        self.fc_health = nn.Linear(d_model, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
            cond_ids: [B] - Optional condition IDs (int64)
        
        Returns:
            rul_pred: [B] - RUL predictions in cycles
            health_last: [B] - Health Index predictions in [0, 1] at final time step
            health_seq: [B, T, 1] - Health Index predictions over full sequence
                       (for now, we replicate health_last to match API)
        """
        # Get sequence embedding
        h = self.encoder(x, cond_ids)  # [B, d_model]
        
        # Shared feature extraction
        h_shared = self.shared_head(h)  # [B, d_model]
        
        # RUL prediction
        rul_pred = self.fc_rul(h_shared).squeeze(-1)  # [B]
        
        # Health Index prediction (sigmoid to [0, 1])
        health_logit = self.fc_health(h_shared)  # [B, 1]
        health_last = torch.sigmoid(health_logit).squeeze(-1)  # [B]
        
        # For compatibility with existing training code, we need health_seq
        # We'll replicate health_last across time steps
        # (In a future version, we could compute HI at each time step)
        B, T = x.shape[0], x.shape[1]
        health_seq = health_last.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1)  # [B, T, 1]
        
        return rul_pred, health_last, health_seq


class UniversalEncoderV2(nn.Module):
    """
    Universal Encoder V2: Enhanced Multi-Scale CNN + Higher-Capacity Sequence Encoder.
    
    Phase 3.2: Designed to improve FD004 performance with:
    - Stronger multi-scale CNN front-end (kernels [3, 5, 9])
    - Higher capacity sequence encoder (d_model=64, Transformer or LSTM)
    - Clean condition fusion via concatenation + projection
    - Better degradation trajectory learning
    
    Architecture:
    1. Multi-scale CNN front-end (kernels 3, 5, 9) with layer norm
    2. Merge multi-scale outputs (concatenation + projection)
    3. Condition fusion (concatenate condition embedding, project to d_model)
    4. Higher-capacity sequence encoder (Transformer or LSTM)
    5. Aggregation (last token + mean pooling)
    
    Args:
        input_dim: Number of input features
        d_model: Model dimension (default: 64 for higher capacity)
        num_layers: Number of sequence encoder layers (default: 3)
        nhead: Number of attention heads (Transformer only, default: 4)
        dim_feedforward: Feedforward dimension (Transformer only, default: 4*d_model)
        dropout: Dropout rate (default: 0.1)
        num_conditions: Number of unique conditions (None for single condition)
        cond_emb_dim: Dimension of condition embeddings (default: 4)
        max_seq_len: Maximum sequence length for positional encoding (default: 300)
        kernel_sizes: List of CNN kernel sizes (default: [3, 5, 9])
        seq_encoder_type: "transformer" or "lstm" (default: "transformer")
        use_layer_norm: Whether to use LayerNorm after CNN (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        max_seq_len: int = 300,
        kernel_sizes: List[int] = None,
        seq_encoder_type: str = "transformer",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model  # Standard Transformer scaling
        
        if seq_encoder_type == "transformer" and d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.use_condition_fusion = num_conditions is not None and num_conditions > 1
        self.seq_encoder_type = seq_encoder_type
        
        # CNN channels per branch (smaller than d_model to keep params reasonable)
        cnn_channels = d_model // 2  # e.g., 32 for d_model=64
        cnn_channels = max(16, cnn_channels)  # Ensure minimum
        
        # ===================================================================
        # 1. Multi-scale CNN front-end
        # ===================================================================
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=cnn_channels,
                    kernel_size=ks,
                    padding=ks // 2,  # Same padding
                ),
                nn.GELU(),  # GELU for smoother gradients
                nn.BatchNorm1d(cnn_channels),
                nn.Dropout(dropout),
            )
            for ks in kernel_sizes
        ])
        
        # Merge multi-scale CNN outputs
        # Option 1: Concatenate and project
        total_cnn_channels = len(kernel_sizes) * cnn_channels
        self.cnn_merge = nn.Sequential(
            nn.Conv1d(total_cnn_channels, d_model, kernel_size=1),
            nn.GELU(),
        )
        
        if use_layer_norm:
            self.cnn_norm = nn.LayerNorm(d_model)
        else:
            self.cnn_norm = None
        
        # ===================================================================
        # 2. Condition Fusion
        # ===================================================================
        if self.use_condition_fusion:
            self.cond_emb = nn.Embedding(num_conditions, cond_emb_dim)
            # Concatenate condition embedding with CNN features, then project
            self.cond_fusion = nn.Sequential(
                nn.Linear(d_model + cond_emb_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.cond_emb = None
            self.cond_fusion = None
        
        # ===================================================================
        # 3. Higher-capacity sequence encoder
        # ===================================================================
        if seq_encoder_type == "transformer":
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif seq_encoder_type == "lstm":
            self.seq_encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unknown seq_encoder_type: {seq_encoder_type}. Must be 'transformer' or 'lstm'")
        
        # ===================================================================
        # 4. Aggregation
        # ===================================================================
        self.out_proj = nn.Linear(2 * d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming/He initialization for GELU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
            cond_ids: [B] - Optional condition IDs (int64). Required if num_conditions > 1.
            return_sequence: If True, return sequence (B, T, d_model) instead of embedding
        
        Returns:
            If return_sequence=False:
                seq_emb: [B, d_model] - Sequence embedding
            If return_sequence=True:
                memory: [B, T, d_model] - Encoder sequence output
        """
        B, T, F = x.shape
        
        # ===================================================================
        # 1. Multi-scale CNN
        # ===================================================================
        # Transpose for Conv1d: (B, T, F) -> (B, F, T)
        x_conv = x.transpose(1, 2)  # [B, F, T]
        
        # Apply CNN branches
        cnn_outputs = []
        for branch in self.cnn_branches:
            out = branch(x_conv)  # [B, cnn_channels, T]
            cnn_outputs.append(out)
        
        # Concatenate along channel dimension
        x_cnn = torch.cat(cnn_outputs, dim=1)  # [B, len(kernel_sizes)*cnn_channels, T]
        
        # Merge to d_model
        x_cnn = self.cnn_merge(x_cnn)  # [B, d_model, T]
        
        # Transpose back: (B, d_model, T) -> (B, T, d_model)
        x_cnn = x_cnn.transpose(1, 2)  # [B, T, d_model]
        
        # Optional layer norm
        if self.cnn_norm is not None:
            x_cnn = self.cnn_norm(x_cnn)
        
        # ===================================================================
        # 2. Condition Fusion
        # ===================================================================
        if self.use_condition_fusion:
            if cond_ids is None:
                raise ValueError("cond_ids required when num_conditions > 1")
            
            # Condition embedding
            c = self.cond_emb(cond_ids)  # [B, cond_emb_dim]
            
            # Broadcast condition embedding to each time step
            c_expanded = c.unsqueeze(1).expand(-1, T, -1)  # [B, T, cond_emb_dim]
            
            # Concatenate and project
            x_fused = torch.cat([x_cnn, c_expanded], dim=-1)  # [B, T, d_model + cond_emb_dim]
            x_fused = self.cond_fusion(x_fused)  # [B, T, d_model]
        else:
            x_fused = x_cnn
        
        # ===================================================================
        # 3. Sequence encoder
        # ===================================================================
        if self.seq_encoder_type == "transformer":
            x_pos = self.pos_encoding(x_fused)  # [B, T, d_model]
            h = self.seq_encoder(x_pos)  # [B, T, d_model]
        else:  # lstm
            h_out, (h_n, c_n) = self.seq_encoder(x_fused)  # h_out: [B, T, d_model]
            h = h_out
        
        # If return_sequence=True, return the sequence directly
        if return_sequence:
            return h  # [B, T, d_model]
        
        # ===================================================================
        # 4. Aggregation
        # ===================================================================
        h_last = h[:, -1, :]  # [B, d_model] - last token
        h_mean = h.mean(dim=1)  # [B, d_model] - mean pooling
        
        h_cat = torch.cat([h_last, h_mean], dim=-1)  # [B, 2*d_model]
        seq_emb = self.out_proj(h_cat)  # [B, d_model]
        
        return seq_emb


class RULHIUniversalModelV2(nn.Module):
    """
    RUL + Health Index model using UniversalEncoderV2.
    
    Compatible with existing training pipeline and EOLFullLSTMWithHealth API.
    
    Args:
        encoder: UniversalEncoderV2 instance
        d_model: Model dimension (should match encoder.d_model)
        dropout: Dropout rate for heads
    """
    
    def __init__(
        self,
        encoder: UniversalEncoderV2,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.d_model = d_model
        
        # For compatibility with existing training code
        self.use_condition_embedding = encoder.use_condition_fusion
        
        # Shared feature extraction
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # RUL head
        self.fc_rul = nn.Linear(d_model, 1)
        
        # Health Index head
        self.fc_health = nn.Linear(d_model, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - Input sequences
            cond_ids: [B] - Optional condition IDs (int64)
        
        Returns:
            rul_pred: [B] - RUL predictions in cycles
            health_last: [B] - Health Index predictions in [0, 1] at final time step
            health_seq: [B, T, 1] - Health Index predictions over full sequence
        """
        # Get sequence embedding
        h = self.encoder(x, cond_ids)  # [B, d_model]
        
        # Shared feature extraction
        h_shared = self.shared_head(h)  # [B, d_model]
        
        # RUL prediction
        rul_pred = self.fc_rul(h_shared).squeeze(-1)  # [B]
        
        # Health Index prediction (sigmoid to [0, 1])
        health_logit = self.fc_health(h_shared)  # [B, 1]
        health_last = torch.sigmoid(health_logit).squeeze(-1)  # [B]
        
        # For compatibility with existing training code, we need health_seq
        # Replicate health_last across time steps
        B, T = x.shape[0], x.shape[1]
        health_seq = health_last.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1)  # [B, T, 1]
        
        return rul_pred, health_last, health_seq


class UniversalEncoderV3Attention(nn.Module):
    """
    Universal encoder + EOL/HI heads with Transformer + temporal attention.

    This model is intended as a drop-in replacement for EOLFullLSTMWithHealth /
    RULHIUniversalModelV2 in the Phase-3/4 training pipeline:

    - Input:  [B, T, F] Phase-4 residual feature sequences (past_len=T, ~464 features)
    - Output: (rul_pred, health_last, health_seq) with the same API as other EOL+HI models

    Architecture:
    1) Optional multi-scale CNN front-end (Conv1d over time, kernels [3, 5, 9])
    2) Condition fusion via learned embeddings (for multi-condition datasets)
    3) Transformer encoder over time (nn.TransformerEncoder)
    4) Temporal attention pooling over encoder states → single sequence embedding
    5) Shared head + two heads:
       - RUL head  (scalar EOL RUL)
       - HI head   (scalar HI in [0, 1]); HI sequence is broadcast for compatibility
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        use_ms_cnn: bool = True,
        kernel_sizes: Optional[List[int]] = None,
        num_conditions: Optional[int] = None,
        condition_embedding_dim: int = 4,
        max_seq_len: int = 300,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.use_ms_cnn = use_ms_cnn
        self.use_condition_fusion = num_conditions is not None and num_conditions > 1
        # For compatibility with evaluation utilities (forward_rul_only / evaluate_eol_full_lstm)
        # which check this flag to decide whether to pass cond_ids.
        self.use_condition_embedding = self.use_condition_fusion

        # ------------------------------------------------------------------
        # 1. Optional multi-scale CNN front-end
        # ------------------------------------------------------------------
        if use_ms_cnn:
            # Similar to UniversalEncoderV2 but self-contained
            cnn_channels = max(16, d_model // 2)
            self.cnn_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=input_dim,
                            out_channels=cnn_channels,
                            kernel_size=ks,
                            padding=ks // 2,
                        ),
                        nn.GELU(),
                        nn.BatchNorm1d(cnn_channels),
                        nn.Dropout(dropout),
                    )
                    for ks in kernel_sizes
                ]
            )
            total_cnn_channels = len(kernel_sizes) * cnn_channels
            self.cnn_merge = nn.Sequential(
                nn.Conv1d(total_cnn_channels, d_model, kernel_size=1),
                nn.GELU(),
            )
            self.cnn_norm = nn.LayerNorm(d_model)
        else:
            # Simple input projection (no CNN)
            self.input_proj = nn.Linear(input_dim, d_model)
            self.cnn_branches = None
            self.cnn_merge = None
            self.cnn_norm = None

        # ------------------------------------------------------------------
        # 2. Condition fusion
        # ------------------------------------------------------------------
        if self.use_condition_fusion:
            self.cond_emb = nn.Embedding(num_conditions, condition_embedding_dim)
            self.cond_fusion = nn.Sequential(
                nn.Linear(d_model + condition_embedding_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.cond_emb = None
            self.cond_fusion = None

        # ------------------------------------------------------------------
        # 3. Transformer encoder over time
        # ------------------------------------------------------------------
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------------------------------------------------------------
        # 4. Temporal attention pooling
        # ------------------------------------------------------------------
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),  # score per time step
        )

        # ------------------------------------------------------------------
        # 5. Shared head + RUL / HI heads
        # ------------------------------------------------------------------
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_rul = nn.Linear(d_model, 1)
        self.fc_health = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming/He initialization for linear and conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, T, F] input sequences
            cond_ids: [B] optional condition IDs (required if num_conditions > 1)

        Returns:
            rul_pred:    [B]       – scalar RUL prediction
            health_last: [B]       – scalar HI in [0, 1] at EOL point
            health_seq:  [B, T, 1] – HI sequence (currently broadcast from health_last)
        """
        B, T, F = x.shape

        # ------------------------------------------------------------------
        # 1. CNN or projection
        # ------------------------------------------------------------------
        if self.use_ms_cnn:
            # (B, T, F) -> (B, F, T)
            x_conv = x.transpose(1, 2)
            cnn_outs = [branch(x_conv) for branch in self.cnn_branches]  # list of [B, C, T]
            x_cnn = torch.cat(cnn_outs, dim=1)  # [B, total_C, T]
            x_cnn = self.cnn_merge(x_cnn)       # [B, d_model, T]
            x_seq = x_cnn.transpose(1, 2)       # [B, T, d_model]
            if self.cnn_norm is not None:
                x_seq = self.cnn_norm(x_seq)
        else:
            # Simple linear projection per time step
            x_seq = self.input_proj(x)  # [B, T, d_model]

        # ------------------------------------------------------------------
        # 2. Condition fusion
        # ------------------------------------------------------------------
        if self.use_condition_fusion:
            if cond_ids is None:
                raise ValueError("cond_ids required when num_conditions > 1")
            c = self.cond_emb(cond_ids)  # [B, cond_dim]
            c_exp = c.unsqueeze(1).expand(-1, T, -1)  # [B, T, cond_dim]
            x_cat = torch.cat([x_seq, c_exp], dim=-1)  # [B, T, d_model + cond_dim]
            x_fused = self.cond_fusion(x_cat)  # [B, T, d_model]
        else:
            x_fused = x_seq

        # ------------------------------------------------------------------
        # 3. Transformer encoder
        # ------------------------------------------------------------------
        x_pos = self.pos_encoding(x_fused)  # [B, T, d_model]
        h = self.transformer(x_pos)         # [B, T, d_model]

        # ------------------------------------------------------------------
        # 4. Temporal attention pooling
        # ------------------------------------------------------------------
        # Compute scores and softmax over time dimension
        scores = self.attn_pool(h)          # [B, T, 1]
        attn_weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        ctx = (h * attn_weights).sum(dim=1)          # [B, d_model]

        # ------------------------------------------------------------------
        # 5. Shared head + RUL / HI heads
        # ------------------------------------------------------------------
        h_shared = self.shared_head(ctx)    # [B, d_model]

        rul_pred = self.fc_rul(h_shared).squeeze(-1)  # [B]

        health_logit = self.fc_health(h_shared)       # [B, 1]
        health_last = torch.sigmoid(health_logit).squeeze(-1)  # [B]

        # Broadcast HI over time to match existing training API
        health_seq = health_last.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1)  # [B, T, 1]

        return rul_pred, health_last, health_seq

