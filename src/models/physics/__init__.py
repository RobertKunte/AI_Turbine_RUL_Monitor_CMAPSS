"""
Physics Layer Modules for Cycle Branch.

This package provides differentiable thermodynamic cycle components:
- NominalHead: Maps operating conditions to nominal (healthy) efficiencies
- CycleLayerMVP: Algebraic 0D cycle model for sensor prediction
"""

from .nominal_head import NominalHead
from .cycle_layer_mvp import CycleLayerMVP

__all__ = ["NominalHead", "CycleLayerMVP"]
