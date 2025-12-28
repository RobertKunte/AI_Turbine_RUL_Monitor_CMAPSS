"""
Transformer Autoregressive Decoder for World Model v3.

Implements a causal Transformer decoder as a drop-in replacement for LSTM decoder.
Supports both self-attention only and cross-attention variants.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import math


class TransformerARDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for trajectory prediction.
    
    Supports two modes:
    - Self-attention only (use_cross_attention=False): Causal self-attention stack
    - Cross-attention (use_cross_attention=True): Cross-attention to encoder sequence
    
    Args:
        d_model: Model dimension (must match encoder)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        horizon: Output sequence length
        use_cross_attention: If True, use cross-attention to encoder; else self-attn only
    """
    
    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        horizon: int = 30,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.horizon = horizon
        self.use_cross_attention = use_cross_attention
        
        # Token embeddings
        # Start token: learned parameter
        self.start_token = nn.Parameter(torch.randn(1, d_model))
        
        # Input projection: maps RUL values (scalar) to d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding: learnable embeddings
        max_len = horizon + 2  # start + optional context + H steps
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        # Transformer layers
        if use_cross_attention:
            # Use TransformerDecoderLayer for cross-attention
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            # Use TransformerEncoderLayer with causal masking (self-attention only)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: maps d_model -> scalar RUL
        self.output_head = nn.Linear(d_model, 1)
        
        # Layer norm (optional, for stability)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for self-attention.
        
        Args:
            seq_len: Sequence length
            device: Device for mask tensor
            
        Returns:
            mask: (seq_len, seq_len) with upper-triangular -inf
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        enc_token: torch.Tensor,  # (B, d_model) - encoder output
        y_teacher: Optional[torch.Tensor] = None,  # (B, H, 1) - ground truth (training)
        enc_seq: Optional[torch.Tensor] = None,  # (B, T_past, d_model) - encoder sequence (if cross-attn)
        cond_ctx: Optional[torch.Tensor] = None,  # (B, cond_emb_dim) - condition context (optional, unused for now)
        mode: str = "train",
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            enc_token: (B, d_model) - Encoder embedding
            y_teacher: (B, H, 1) or None - Ground truth trajectory (for teacher forcing)
            enc_seq: (B, T_past, d_model) or None - Encoder sequence (for cross-attention)
            cond_ctx: (B, cond_emb_dim) or None - Condition context (optional, reserved for future use)
            mode: "train" or "inference"
            
        Returns:
            y_hat_traj: (B, H, 1) - Predicted trajectory
        """
        B = enc_token.size(0)
        device = enc_token.device
        
        # Determine horizon
        if y_teacher is not None:
            H = y_teacher.size(1)
        else:
            H = self.horizon
        
        # Build token sequence
        if mode == "train" and y_teacher is not None:
            # Teacher forcing: shift-right strategy
            # Start token
            start_tokens = self.start_token.expand(B, -1)  # (B, d_model)
            
            # Project teacher inputs: (B, H, 1) -> (B, H, d_model)
            teacher_emb = self.input_proj(y_teacher)  # (B, H, d_model)
            
            # Shift right: prepend start token, remove last step
            # [start, y_0, y_1, ..., y_{H-2}] (H tokens total)
            teacher_shifted = teacher_emb[:, :-1, :]  # (B, H-1, d_model)
            token_seq = torch.cat([start_tokens.unsqueeze(1), teacher_shifted], dim=1)  # (B, H, d_model)
            
        else:
            # Inference: autoregressive rollout
            # Start with start token
            token_seq = self.start_token.expand(B, -1).unsqueeze(1)  # (B, 1, d_model)
            
            # Autoregressive loop: predict H steps
            predictions = []
            for t in range(H):
                # Current sequence length
                seq_len = token_seq.size(1)
                
                # Add positional encoding
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)  # (B, seq_len)
                pos_emb = self.pos_embed(pos_ids)  # (B, seq_len, d_model)
                token_seq_pos = token_seq + pos_emb
                token_seq_pos = self.dropout(token_seq_pos)
                
                # Apply transformer
                if self.use_cross_attention:
                    # Cross-attention: tgt = token_seq, memory = enc_seq
                    if enc_seq is None:
                        # Fallback: use enc_token as single-step memory
                        memory = enc_token.unsqueeze(1).transpose(0, 1)  # (1, B, d_model)
                    else:
                        # Transpose for PyTorch TransformerDecoder: (S, B, E)
                        memory = enc_seq.transpose(0, 1)  # (T_past, B, d_model)
                    
                    # Causal mask for tgt
                    tgt_mask = self._create_causal_mask(seq_len, device)
                    
                    # Transformer decoder
                    out = self.transformer_layers(
                        tgt=token_seq_pos,
                        memory=memory,
                        tgt_mask=tgt_mask,
                    )  # (B, seq_len, d_model)
                else:
                    # Self-attention only: causal mask
                    src_mask = self._create_causal_mask(seq_len, device)
                    out = self.transformer_layers(token_seq_pos, mask=src_mask)  # (B, seq_len, d_model)
                
                # Layer norm
                out = self.layer_norm(out)
                
                # Predict next token (scalar RUL)
                next_rul = self.output_head(out[:, -1:, :])  # (B, 1, 1)
                predictions.append(next_rul)
                
                # Project RUL back to embedding space for next iteration
                next_token = self.input_proj(next_rul)  # (B, 1, d_model)
                
                # Append to sequence
                token_seq = torch.cat([token_seq, next_token], dim=1)  # (B, seq_len+1, d_model)
            
            # Concatenate predictions: (B, H, 1)
            y_hat_traj = torch.cat(predictions, dim=1)
            return y_hat_traj
        
        # Teacher forcing path: apply transformer to full sequence
        seq_len = token_seq.size(1)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)  # (B, H)
        pos_emb = self.pos_embed(pos_ids)  # (B, H, d_model)
        token_seq = token_seq + pos_emb
        token_seq = self.dropout(token_seq)
        
        # Apply transformer to final sequence
        if self.use_cross_attention:
            if enc_seq is None:
                memory = enc_token.unsqueeze(1).transpose(0, 1)  # (1, B, d_model)
            else:
                memory = enc_seq.transpose(0, 1)  # (T_past, B, d_model)
            
            tgt_mask = self._create_causal_mask(seq_len, device)
            out = self.transformer_layers(token_seq, memory=memory, tgt_mask=tgt_mask)
        else:
            src_mask = self._create_causal_mask(seq_len, device)
            out = self.transformer_layers(token_seq, mask=src_mask)
        
        # Layer norm
        out = self.layer_norm(out)
        
        # Output head: (B, H, d_model) -> (B, H, 1)
        y_hat_traj = self.output_head(out)  # (B, H, 1)
        
        return y_hat_traj

