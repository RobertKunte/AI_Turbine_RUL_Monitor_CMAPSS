from __future__ import annotations

"""
RUL Trajectory Decoder v1 on top of a frozen EOLFullTransformerEncoder.

This module is intentionally minimal and dataset-agnostic: it operates only
on tensors (latent encoder sequence + HI sequences) and does not know about
NASA CMAPSS specifics. The training script is responsible for providing
properly prepared inputs.
"""

from typing import Optional

import torch
import torch.nn as nn


class RULTrajectoryDecoderV1(nn.Module):
    """
    Simple RUL trajectory decoder on top of frozen encoder + HI sequences.

    Inputs:
      - z_seq:         [B, T, D]  latent encoder sequence from EOLFullTransformerEncoder
      - hi_phys_seq:   [B, T]     physics HI_phys_v3 (continuous, monotone decreasing)
                                  If not available, pass a zero/one tensor of same shape.
      - hi_damage_seq: [B, T]     learned damage HI (v3d) from CumulativeDamageHead

    Output:
      - rul_seq_pred:  [B, T]     predicted RUL trajectory over the same T timesteps
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # project HIs to small embedding
        hi_input_dim = 2  # [hi_phys, hi_damage]
        self.hi_proj = nn.Linear(hi_input_dim, latent_dim)

        # combine latent encoder sequence + HI embedding
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_seq: torch.Tensor,
        hi_phys_seq: torch.Tensor,
        hi_damage_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_seq:        [B, T, D] latent encoder sequence
            hi_phys_seq:  [B, T]    physics HI (or zeros/ones if unavailable)
            hi_damage_seq:[B, T]    learned damage HI

        Returns:
            rul_seq_pred: [B, T] predicted RUL trajectory
        """
        if hi_phys_seq is None:
            raise ValueError("hi_phys_seq must be a tensor; pass zeros/ones if unavailable.")
        if hi_damage_seq is None:
            raise ValueError("hi_damage_seq must be provided.")

        if z_seq.dim() != 3:
            raise ValueError(f"z_seq must be [B,T,D], got shape {tuple(z_seq.shape)}")

        B, T, D = z_seq.shape

        # Ensure HI sequences are [B, T]
        if hi_phys_seq.dim() == 3:
            hi_phys_seq = hi_phys_seq.squeeze(-1)
        if hi_damage_seq.dim() == 3:
            hi_damage_seq = hi_damage_seq.squeeze(-1)

        if hi_phys_seq.shape[:2] != (B, T):
            raise ValueError(
                f"hi_phys_seq must be [B,T], got {tuple(hi_phys_seq.shape)} vs expected {(B, T)}"
            )
        if hi_damage_seq.shape[:2] != (B, T):
            raise ValueError(
                f"hi_damage_seq must be [B,T], got {tuple(hi_damage_seq.shape)} vs expected {(B, T)}"
            )

        # Stack and project HI channels
        hi = torch.stack([hi_phys_seq, hi_damage_seq], dim=-1)  # [B, T, 2]
        hi_emb = torch.relu(self.hi_proj(hi))                   # [B, T, D]

        # Concatenate latent sequence and HI embedding
        x = torch.cat([z_seq, hi_emb], dim=-1)                  # [B, T, 2D]
        x = torch.relu(self.input_proj(x))                      # [B, T, H]

        out_seq, _ = self.gru(x)                                # [B, T, H]
        rul_seq = self.out(out_seq).squeeze(-1)                 # [B, T]

        return rul_seq


