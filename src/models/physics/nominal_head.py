"""
NominalHead: Maps operating conditions to nominal (healthy) efficiency values.

This module implements the η_nom(ops/cond) component of the Mode 1 factorized
efficiency model: η_eff = η_nom(ops) × m(t)

The nominal head captures operating-condition-dependent efficiency variations
while keeping capacity low to prevent it from explaining degradation.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class NominalHead(nn.Module):
    """Low-capacity head for operating-condition-dependent nominal efficiencies.
    
    Maps operating settings (or ConditionID) to nominal efficiency values η_nom.
    These represent the "healthy" efficiency at each operating point, before
    degradation is applied.
    
    Two modes are supported:
    - 'table': ConditionID embedding table (preferred for multi-condition datasets)
    - 'mlp': Small MLP from operating settings (fallback)
    
    Args:
        head_type: 'table' or 'mlp'
        num_conditions: Number of operating conditions (required for 'table' mode)
        ops_dim: Dimension of operating settings input (default 3)
        hidden_dim: Hidden dimension for MLP mode (default 16)
        eta_bounds: (min, max) bounds for nominal efficiencies (default (0.80, 0.99))
        output_dp_nom: If True, also output dp_nom as 6th channel (default False)
    
    Returns:
        eta_nom: (B, T, 5) or (B, T, 6) if output_dp_nom=True
        Components: [η_fan, η_lpc, η_hpc, η_hpt, η_lpt] (+ dp_nom optionally)
    """
    
    def __init__(
        self,
        head_type: Literal["table", "mlp"] = "table",
        num_conditions: Optional[int] = None,
        ops_dim: int = 3,
        hidden_dim: int = 16,
        eta_bounds: tuple[float, float] = (0.80, 0.99),
        output_dp_nom: bool = False,
    ):
        super().__init__()
        self.head_type = head_type
        self.eta_bounds = eta_bounds
        self.output_dp_nom = output_dp_nom
        self.out_dim = 6 if output_dp_nom else 5
        
        eta_min, eta_max = eta_bounds
        self.register_buffer("eta_min", torch.tensor(eta_min))
        self.register_buffer("eta_range", torch.tensor(eta_max - eta_min))
        
        if head_type == "table":
            if num_conditions is None or num_conditions < 1:
                raise ValueError(
                    "num_conditions must be >= 1 for table mode. "
                    "Use head_type='mlp' if ConditionID is not available."
                )
            self.num_conditions = num_conditions
            # Learnable embedding table: (num_conditions, out_dim)
            # Initialize near middle of range (logit ~0)
            self.eta_table = nn.Parameter(torch.zeros(num_conditions, self.out_dim))
            self.ops_net = None
        else:
            # MLP mode: ops -> hidden -> out_dim
            self.num_conditions = None
            self.eta_table = None
            self.ops_net = nn.Sequential(
                nn.Linear(ops_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.out_dim),
            )
            self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights for stable outputs."""
        if self.ops_net is not None:
            for m in self.ops_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        ops_t: Optional[torch.Tensor] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute nominal efficiencies.
        
        Args:
            ops_t: Operating settings (B, T, ops_dim) or (B, ops_dim)
                   Required for 'mlp' mode
            cond_ids: Condition IDs (B,) or (B, T) — int64
                      Required for 'table' mode
        
        Returns:
            eta_nom: (B, T, out_dim) or (B, out_dim) depending on input shape
        """
        if self.head_type == "table":
            if cond_ids is None:
                raise ValueError("cond_ids required for 'table' mode NominalHead")
            return self._forward_table(cond_ids, ops_t)
        else:
            if ops_t is None:
                raise ValueError("ops_t required for 'mlp' mode NominalHead")
            return self._forward_mlp(ops_t)
    
    def _forward_table(
        self,
        cond_ids: torch.Tensor,
        ops_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Table-based forward: look up eta_nom from ConditionID."""
        # cond_ids: (B,) or (B, T)
        # Clamp to valid range
        cond_ids_clamped = cond_ids.clamp(0, self.num_conditions - 1)
        
        # Lookup: (B,) -> (B, out_dim) or (B, T) -> (B, T, out_dim)
        raw = self.eta_table[cond_ids_clamped]  # (..., out_dim)
        
        # Apply sigmoid scaling to enforce bounds
        eta_nom = self.eta_min + self.eta_range * torch.sigmoid(raw)
        
        # If ops_t has time dimension but cond_ids doesn't, expand
        if ops_t is not None and ops_t.dim() == 3 and cond_ids.dim() == 1:
            # (B, out_dim) -> (B, T, out_dim)
            T = ops_t.shape[1]
            eta_nom = eta_nom.unsqueeze(1).expand(-1, T, -1)
        
        return eta_nom
    
    def _forward_mlp(self, ops_t: torch.Tensor) -> torch.Tensor:
        """MLP-based forward: compute eta_nom from operating settings."""
        # ops_t: (B, T, ops_dim) or (B, ops_dim)
        input_shape = ops_t.shape[:-1]
        ops_flat = ops_t.reshape(-1, ops_t.shape[-1])  # (B*T, ops_dim)
        
        raw = self.ops_net(ops_flat)  # (B*T, out_dim)
        
        # Apply sigmoid scaling to enforce bounds
        eta_nom = self.eta_min + self.eta_range * torch.sigmoid(raw)
        
        # Reshape back
        return eta_nom.reshape(*input_shape, self.out_dim)
    
    def get_num_conditions(self) -> Optional[int]:
        """Return number of conditions (for table mode) or None."""
        return self.num_conditions
