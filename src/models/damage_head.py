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
        # NEW: Optional MLP-based damage/HI head (v3c)
        use_mlp: bool = False,
        mlp_hidden_factor: int = 2,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.1,
        # NEW: v3d delta-cumsum mode
        use_delta_cumsum: bool = False,
        delta_alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            d_model: latent dimension from encoder
            cond_dim: optional condition feature dimension
            L_ref, alpha_base, hidden_dim: legacy cumulative-damage parameters
            use_mlp, mlp_*: if True, use a direct MLP on encoder states.
                If use_delta_cumsum=False (v3c), predicts HI_seq directly via sigmoid.
                If use_delta_cumsum=True (v3d), predicts delta_damage via softplus and accumulates.
            use_delta_cumsum: enable delta-damage + cumsum mode (v3d)
            delta_alpha: scaling factor for delta damage in v3d mode
            eps: small epsilon for division
        """
        super().__init__()

        self.L_ref = float(L_ref)
        self.alpha_base = float(alpha_base)
        self.use_mlp = bool(use_mlp)
        self.use_delta_cumsum = bool(use_delta_cumsum)
        self.delta_alpha = float(delta_alpha)
        self.eps = float(eps)

        # ------------------------------------------------------------------
        # NEW: simple MLP head (damage_v3c / v3d) – operates directly on z_seq
        # ------------------------------------------------------------------
        if self.use_mlp:
            hidden_dim_mlp = int(mlp_hidden_factor * d_model)
            layers = []
            in_dim = d_model
            for _ in range(mlp_num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim_mlp))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(mlp_dropout))
                in_dim = hidden_dim_mlp
            layers.append(nn.Linear(in_dim, 1))
            self.mlp_head = nn.Sequential(*layers)

            # For the MLP-mode we still register dummy parameters to keep
            # the state_dict structure compatible, but they will not be used.
            self.base_bias = nn.Parameter(torch.zeros(1))
            self.cond_mlp = None
            self.sens_mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Global scaling parameters are not used in the simple MLP path,
            # but we keep them for backwards compatibility with checkpoints.
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            # ------------------------------------------------------------------
            # Legacy cumulative damage head (default for all existing runs)
            # ------------------------------------------------------------------
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

            # Global scaling parameters for damage → [0, 1]
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))

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

        # ------------------------------------------------------------------
        # NEW: Simple MLP-based HI head (v3c) or Delta-Cumsum (v3d)
        # ------------------------------------------------------------------
        if self.use_mlp:
            z_flat = z_seq.reshape(B * T, D)
            out = self.mlp_head(z_flat)  # [B*T, 1]
            
            if self.use_delta_cumsum:
                # v3d: Delta-Damage + Cumsum
                delta_raw = out.view(B, T)
                
                # Only positive increments
                delta_damage = F.softplus(delta_raw) * self.delta_alpha
                
                # Cumulative damage
                damage_seq = torch.cumsum(delta_damage, dim=1)
                
                # Map to Health Index
                hi_seq = 1.0 - damage_seq / (self.L_ref + self.eps)
                hi_seq = torch.clamp(hi_seq, 0.0, 1.0)
                
                hi_last = hi_seq[:, -1]
                
                # DEBUG print (rarely)
                if getattr(self, "debug", False) and torch.rand(1).item() < 0.001:
                     print(f"[DEBUG V3d DamageHead] delta_damage[0, :5]: {delta_damage[0, :5].detach().cpu().numpy()}")
                     print(f"[DEBUG V3d DamageHead] damage_seq[0, :5]:   {damage_seq[0, :5].detach().cpu().numpy()}")
                     print(f"[DEBUG V3d DamageHead] hi_seq[0, :5]:       {hi_seq[0, :5].detach().cpu().numpy()}")

                return hi_seq, hi_last, damage_seq, delta_damage

            else:
                # v3c: Direct HI prediction
                hi_seq = torch.sigmoid(out).view(B, T)
                hi_last = hi_seq[:, -1]
                # For compatibility with callers expecting damage sequences,
                # we construct a surrogate damage sequence as 1 - HI and set
                # per-step increments to zeros.
                damage_seq = 1.0 - hi_seq
                delta_damage = z_seq.new_zeros(B, T)
                return hi_seq, hi_last, damage_seq, delta_damage

        # ------------------------------------------------------------------
        # Legacy cumulative damage path (default for all existing runs)
        # ------------------------------------------------------------------
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

        # Global scaling into [0, 1] via learnable affine + sigmoid.
        damage_scaled = torch.sigmoid(self.gamma * damage + self.beta)
        damage_seq = torch.clamp(damage_scaled, 0.0, 1.0)

        hi_seq = 1.0 - damage_seq
        hi_last = hi_seq[:, -1]

        return hi_seq, hi_last, damage_seq, delta_damage


