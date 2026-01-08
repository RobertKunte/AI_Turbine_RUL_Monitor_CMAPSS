"""
ParamHeadTheta6: Maps latent z_t to 6D degradation modifiers.

This module implements the m(t) degradation modifier head for Mode 1 factorized
efficiencies: η_eff = η_nom(ops) × m(t)

The modifiers represent slow-varying degradation factors that multiply the
nominal (healthy) efficiency values.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ParamHeadTheta6(nn.Module):
    """Maps latent representation z_t to 6D degradation modifiers.
    
    Output: m(t) = [m_fan, m_lpc, m_hpc, m_hpt, m_lpt, m_dp_comb]
    
    These modifiers multiply the nominal efficiencies:
    - η_eff_comp = η_nom_comp × m_comp(t)
    - dp_eff = dp_nom × m_dp_comb(t)
    
    Bounds are enforced by construction via sigmoid scaling.
    
    Args:
        z_dim: Dimension of input latent z_t
        hidden_dim: Hidden layer dimension (default 64)
        m_bounds_eta: (min, max) bounds for efficiency modifiers (default (0.85, 1.00))
        m_bounds_dp: (min, max) bounds for dp modifier (default (0.90, 1.00))
        num_layers: Number of hidden layers (default 2)
    """
    
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        m_bounds_eta: Tuple[float, float] = (0.85, 1.00),
        m_bounds_dp: Tuple[float, float] = (0.90, 1.00),
        num_layers: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.m_bounds_eta = m_bounds_eta
        self.m_bounds_dp = m_bounds_dp
        
        # Build MLP
        layers = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 6))
        self.mlp = nn.Sequential(*layers)
        
        # Bounds: first 5 are eta modifiers, 6th is dp modifier
        eta_min, eta_max = m_bounds_eta
        dp_min, dp_max = m_bounds_dp
        
        # Store bounds as buffers
        mins = torch.tensor([eta_min] * 5 + [dp_min])
        ranges = torch.tensor([eta_max - eta_min] * 5 + [dp_max - dp_min])
        self.register_buffer("m_min", mins)
        self.register_buffer("m_range", ranges)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for mid-range output at start (degraded state).
        
        Key insight: We want θ to start ~0.92 (mid-range) so gradients flow.
        If θ starts near 1.0 (healthy), it saturates and gradients vanish.
        
        sigmoid(-0.5) ≈ 0.38 → m = 0.85 + 0.38 * 0.15 ≈ 0.91
        """
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # Small weights for stability
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Neutral bias for hidden layers
        
        # Override final layer for mid-range θ initialization
        final_layer = self.mlp[-1]
        if hasattr(final_layer, "bias") and final_layer.bias is not None:
            # sigmoid(-0.5) ≈ 0.38 → m ≈ min + 0.38 * range
            # For eta: 0.85 + 0.38 * 0.15 ≈ 0.91 (mid-range, not saturated)
            nn.init.constant_(final_layer.bias, -0.5)
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Compute degradation modifiers from latent representation.
        
        Args:
            z_t: Latent representation (B, T, z_dim) or (B, z_dim)
            
        Returns:
            m_t: Degradation modifiers (B, T, 6) or (B, 6)
                 [m_fan, m_lpc, m_hpc, m_hpt, m_lpt, m_dp_comb]
        """
        # Handle both (B, z_dim) and (B, T, z_dim)
        input_shape = z_t.shape[:-1]
        z_flat = z_t.reshape(-1, self.z_dim)  # (B*T, z_dim)
        
        # Forward through MLP
        raw = self.mlp(z_flat)  # (B*T, 6)
        
        # Apply eps-bounded sigmoid to prevent saturation
        # Standard sigmoid maps to (0, 1), but we constrain to [0.02, 0.98]
        # This ensures m never reaches exactly the bounds, preserving gradient flow
        u = torch.sigmoid(raw)
        u = u * 0.96 + 0.02  # Maps [0.02, 0.98] - never hits extremes
        
        m_t = self.m_min + self.m_range * u
        
        # Reshape back
        return m_t.reshape(*input_shape, 6)
    
    def get_saturation_stats(self, m_t: torch.Tensor) -> dict:
        """Compute saturation statistics for diagnostics.
        
        Args:
            m_t: Degradation modifiers (*, 6)
            
        Returns:
            Dict with saturation fractions and bounds info
        """
        # Flatten to (N, 6)
        m_flat = m_t.reshape(-1, 6)
        
        # Compute normalized position within bounds [0, 1]
        m_normalized = (m_flat - self.m_min) / self.m_range.clamp(min=1e-6)
        
        # Saturation: near 0 or 1
        near_low = (m_normalized < 0.02).float().mean(dim=0)
        near_high = (m_normalized > 0.98).float().mean(dim=0)
        saturation_frac = near_low + near_high
        
        param_names = ["m_fan", "m_lpc", "m_hpc", "m_hpt", "m_lpt", "m_dp_comb"]
        
        return {
            "saturation_frac_per_param": {
                name: float(saturation_frac[i])
                for i, name in enumerate(param_names)
            },
            "saturation_frac_total": float(saturation_frac.mean()),
            "m_min_actual": {
                name: float(m_flat[:, i].min())
                for i, name in enumerate(param_names)
            },
            "m_max_actual": {
                name: float(m_flat[:, i].max())
                for i, name in enumerate(param_names)
            },
            "m_mean": {
                name: float(m_flat[:, i].mean())
                for i, name in enumerate(param_names)
            },
        }
