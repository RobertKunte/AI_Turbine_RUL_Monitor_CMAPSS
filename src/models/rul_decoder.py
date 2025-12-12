from __future__ import annotations

"""
RUL Trajectory Decoders on top of a frozen EOLFullTransformerEncoder.

This module is intentionally minimal and dataset-agnostic: it operates only
on tensors (latent encoder sequence + HI sequences) and does not know about
NASA CMAPSS specifics. The training scripts are responsible for providing
properly prepared inputs.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Project HIs to small embedding
        hi_input_dim = 2  # [hi_phys, hi_damage]
        self.hi_proj = nn.Linear(hi_input_dim, latent_dim)

        # Combine latent encoder sequence + HI embedding
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

        B, T, _ = z_seq.shape

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
        hi_emb = torch.relu(self.hi_proj(hi))  # [B, T, D]

        # Concatenate latent sequence and HI embedding
        x = torch.cat([z_seq, hi_emb], dim=-1)  # [B, T, 2D]
        x = torch.relu(self.input_proj(x))  # [B, T, H]

        out_seq, _ = self.gru(x)  # [B, T, H]
        rul_seq = self.out(out_seq).squeeze(-1)  # [B, T]

        return rul_seq


class RULTrajectoryDecoderV2(nn.Module):
    """
    Enhanced RUL trajectory decoder with richer HI inputs and a deeper head.

    Inputs:
      - z_seq:         [B, T, D_z] latent encoder sequence
      - hi_phys_seq:   [B, T]      raw physics HI_phys_v3
      - hi_cal_seq:    [B, T]      calibrated HI_cal_v1 (global monotone mapping of HI_phys_v3)
      - hi_damage_seq: [B, T]      learned damage HI sequence from encoder damage head

    Output:
      - rul_seq_pred:  [B, T]      predicted RUL trajectory
    """

    def __init__(
        self,
        latent_dim: int,
        hi_feature_dim: int = 3,  # HI_phys, HI_cal, HI_damage
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_zone_weights: bool = True,  # kept for possible future extensions
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hi_feature_dim = hi_feature_dim
        self.use_zone_weights = use_zone_weights

        # Simple projection of concatenated [z_seq, hi_stack] into hidden_dim
        self.input_proj = nn.Linear(latent_dim + hi_feature_dim, hidden_dim)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # MLP head for RUL prediction (per timestep)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        hi_phys_seq: torch.Tensor,
        hi_cal_seq: torch.Tensor,
        hi_damage_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_seq:         [B, T, D_z] latent encoder sequence
            hi_phys_seq:   [B, T]     physics HI
            hi_cal_seq:    [B, T]     calibrated HI
            hi_damage_seq: [B, T]     learned damage HI

        Returns:
            rul_seq_pred: [B, T] predicted RUL trajectory
        """
        if z_seq.dim() != 3:
            raise ValueError(f"z_seq must be [B,T,D_z], got shape {tuple(z_seq.shape)}")

        B, T, _ = z_seq.shape

        # Ensure HI sequences are [B, T]
        def _ensure_bt(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 3:
                x = x.squeeze(-1)
            if x.shape[:2] != (B, T):
                raise ValueError(
                    f"{name} must be [B,T], got {tuple(x.shape)} vs expected {(B, T)}"
                )
            return x

        hi_phys_seq = _ensure_bt(hi_phys_seq, "hi_phys_seq")
        hi_cal_seq = _ensure_bt(hi_cal_seq, "hi_cal_seq")
        hi_damage_seq = _ensure_bt(hi_damage_seq, "hi_damage_seq")

        # Stack HI channels and concatenate with latent sequence
        hi_stack = torch.stack([hi_phys_seq, hi_cal_seq, hi_damage_seq], dim=-1)  # [B, T, 3]
        x = torch.cat([z_seq, hi_stack], dim=-1)  # [B, T, D_z + 3]

        x_proj = torch.relu(self.input_proj(x))  # [B, T, H]
        rnn_out, _ = self.rnn(x_proj)  # [B, T, H]

        rul_seq_pred = self.head(rnn_out).squeeze(-1)  # [B, T]
        return rul_seq_pred


class DecoderV2Wrapper(nn.Module):
    """
    Thin wrapper that combines a frozen encoder with RULTrajectoryDecoderV2.

    This is primarily intended for inference/diagnostics; training scripts can
    directly call encoder.encode_with_hi + decoder_v2.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: RULTrajectoryDecoderV2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        X: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        hi_phys_seq: Optional[torch.Tensor] = None,
        hi_cal_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            X:           [B, T, F] encoder input sequence
            cond_ids:    [B] optional condition IDs for encoder
            hi_phys_seq: [B, T] optional physics HI sequence; if None, zeros are used
            hi_cal_seq:  [B, T] optional calibrated HI sequence; if None, falls back to hi_phys_seq

        Returns:
            rul_seq_pred: [B, T] decoder RUL trajectory
        """
        device = next(self.parameters()).device
        X = X.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)

        with torch.no_grad():
            z_seq, _, hi_damage_seq = self.encoder.encode_with_hi(
                X,
                cond_ids=cond_ids,
                cond_vec=None,
            )

        if hi_phys_seq is None:
            hi_phys_seq = torch.zeros_like(hi_damage_seq, device=device)
        else:
            hi_phys_seq = hi_phys_seq.to(device)

        if hi_cal_seq is None:
            hi_cal_seq = hi_phys_seq
        else:
            hi_cal_seq = hi_cal_seq.to(device)

        if hi_damage_seq.dim() == 3:
            hi_damage_seq = hi_damage_seq.squeeze(-1)

        rul_seq_pred = self.decoder(z_seq, hi_phys_seq, hi_cal_seq, hi_damage_seq)
        return rul_seq_pred


class RULTrajectoryDecoderV3(nn.Module):
    """
    RUL trajectory decoder with explicit modelling of degradation dynamics.
    """

    def __init__(
        self,
        latent_dim: int,
        hi_feature_dim: int = 4,
        slope_feature_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        # NEW: optional uncertainty head for RUL trajectory (sigma per timestep)
        use_uncertainty: bool = False,
        min_sigma: float = 1e-3,
    ) -> None:
        super().__init__()

        self.lat = latent_dim
        self.hi_feature_dim = hi_feature_dim
        self.slope_feature_dim = slope_feature_dim
        self.use_uncertainty: bool = bool(use_uncertainty)
        self.min_sigma: float = float(min_sigma)

        input_dim = latent_dim + hi_feature_dim + slope_feature_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Main RUL trajectory head
        self.rul_head = nn.Linear(hidden_dim, 1)
        # Optional: log-sigma head (per timestep)
        self.rul_log_sigma_head = nn.Linear(hidden_dim, 1) if self.use_uncertainty else None

        # Auxiliary degradation-rate head (per time step)
        self.degradation_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_seq: torch.Tensor,
        hi_phys_seq: torch.Tensor,
        hi_cal1_seq: torch.Tensor,
        hi_cal2_seq: torch.Tensor,
        hi_damage_seq: torch.Tensor,
        slope_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq:         [B, T, D_z] latent encoder sequence
            hi_phys_seq:   [B, T]     physics HI
            hi_cal1_seq:   [B, T]     HI_cal_v1 (0..1, healthy high)
            hi_cal2_seq:   [B, T]     HI_cal_v2 (1..0, healthy->EOL)
            hi_damage_seq: [B, T]     learned damage HI
            slope_feats:   [B, T, S]  slope features for health signals

        Returns:
            rul_seq_pred:  [B, T] predicted RUL trajectory
            rul_sigma_seq: [B, T] optional predicted sigma per timestep (std dev in cycles)
            degr_rate_pred:[B, T] predicted degradation rate per timestep
        """
        if z_seq.dim() != 3:
            raise ValueError(f"z_seq must be [B,T,D_z], got shape {tuple(z_seq.shape)}")

        B, T, _ = z_seq.shape

        def _ensure_bt(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 3:
                x = x.squeeze(-1)
            if x.shape[:2] != (B, T):
                raise ValueError(
                    f"{name} must be [B,T], got {tuple(x.shape)} vs expected {(B, T)}"
                )
            return x

        hi_phys_seq = _ensure_bt(hi_phys_seq, "hi_phys_seq")
        hi_cal1_seq = _ensure_bt(hi_cal1_seq, "hi_cal1_seq")
        hi_cal2_seq = _ensure_bt(hi_cal2_seq, "hi_cal2_seq")
        hi_damage_seq = _ensure_bt(hi_damage_seq, "hi_damage_seq")

        if slope_feats.dim() != 3 or slope_feats.shape[:2] != (B, T):
            raise ValueError(
                f"slope_feats must be [B,T,S], got {tuple(slope_feats.shape)}"
            )

        hi_stack = torch.stack(
            [hi_phys_seq, hi_cal1_seq, hi_cal2_seq, hi_damage_seq], dim=-1
        )  # [B, T, 4]
        x = torch.cat([z_seq, hi_stack, slope_feats], dim=-1)  # [B, T, latent+4+S]

        x_proj = torch.relu(self.input_proj(x))  # [B, T, H]
        rnn_out, _ = self.rnn(x_proj)  # [B, T, H]

        rul_seq_pred = self.rul_head(rnn_out).squeeze(-1)  # [B, T]
        if self.use_uncertainty:
            if self.rul_log_sigma_head is None:
                raise RuntimeError("use_uncertainty=True but rul_log_sigma_head is None.")
            log_sigma = self.rul_log_sigma_head(rnn_out).squeeze(-1)  # [B, T]
            rul_sigma_seq = F.softplus(log_sigma) + float(self.min_sigma)  # [B, T]
        else:
            rul_sigma_seq = None
        degr_rate_pred = self.degradation_head(rnn_out).squeeze(-1)  # [B, T]

        if self.use_uncertainty:
            return rul_seq_pred, rul_sigma_seq, degr_rate_pred
        return rul_seq_pred, degr_rate_pred
