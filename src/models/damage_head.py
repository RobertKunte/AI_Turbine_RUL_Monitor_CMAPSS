from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeDamageHead(nn.Module):
    """
    Per-timestep cumulative damage head operating on the latent encoder sequence.

    Design:
        ΔD_t = ΔD_base + ΔD_cond_t + ΔD_obs_t  (all terms >= 0)
        D_t  = cumulative sum over time of ΔD_t
        HI_phys(t) = 1 - D_t (normalised/clamped to [0, 1])
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: Optional[int] = None,
        L_ref: float = 300.0,
        alpha_base: float = 0.1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.L_ref = float(L_ref)
        self.alpha_base = float(alpha_base)

        # Global base-damage parameter
        self.base_bias = nn.Parameter(torch.zeros(1))

        # Optional condition MLP (stress contribution)
        if cond_dim is not None and cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.cond_mlp = None

        # Latent / sensor-based damage (observable degradation term)
        self.sens_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq:    [B, T, d_model] latent sequence from encoder
            cond_seq: [B, T, C] continuous condition vector or None

        Returns:
            hi_seq:          [B, T]  physical HI trajectory
            hi_last:         [B]     last-timestep HI
            damage_seq:      [B, T]  cumulative damage (normalised to [0,1])
            delta_damage_seq:[B, T]  per-step increments ΔD_t
        """
        if z_seq.dim() != 3:
            raise ValueError(f"z_seq must be [B,T,D], got shape {tuple(z_seq.shape)}")

        B, T, D = z_seq.shape

        # Base damage per step (global, non-negative)
        base_step = (self.alpha_base / self.L_ref) * F.softplus(self.base_bias)
        base_delta = base_step.expand(B, T)  # [B, T]

        # Condition-dependent damage contribution
        if self.cond_mlp is not None and cond_seq is not None:
            if cond_seq.dim() != 3 or cond_seq.shape[0] != B or cond_seq.shape[1] != T:
                raise ValueError(
                    f"[CumulativeDamageHead] cond_seq shape mismatch: expected ({B}, {T}, C), "
                    f"got {tuple(cond_seq.shape)}"
                )
            cond_in = cond_seq.reshape(B * T, -1)
            cond_out = self.cond_mlp(cond_in)  # [B*T, 1]
            cond_delta = F.softplus(cond_out).view(B, T) / self.L_ref
        else:
            cond_delta = z_seq.new_zeros(B, T)

        # Latent / sensor-based damage contribution
        z_flat = z_seq.reshape(B * T, D)
        sens_out = self.sens_mlp(z_flat)  # [B*T, 1]
        sens_delta = F.softplus(sens_out).view(B, T) / self.L_ref

        # Total per-step increment (all >= 0)
        delta_damage = base_delta + cond_delta + sens_delta  # [B, T]

        # Cumulative damage over time (monotone increasing by construction)
        damage = torch.cumsum(delta_damage, dim=1)  # [B, T]

        # Soft clamp and normalisation into [0,1]
        damage_clamped = torch.clamp(damage, 0.0, 1.5)
        max_damage = damage_clamped.amax(dim=1, keepdim=True).clamp(min=1.0)
        damage_norm = torch.clamp(damage_clamped / max_damage, 0.0, 1.0)

        hi_seq = 1.0 - damage_norm
        hi_last = hi_seq[:, -1]

        return hi_seq, hi_last, damage_norm, delta_damage


