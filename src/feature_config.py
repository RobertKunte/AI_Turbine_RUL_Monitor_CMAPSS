"""
Feature configuration for ablation studies.

Defines configurable feature groups and temporal window settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TemporalWindowConfig:
    """Configuration for temporal (multi-scale) window features."""
    
    short_windows: Tuple[int, ...] = (5, 10)
    long_windows: Tuple[int, ...] = (20, 30)
    include_derivatives: bool = True
    include_rolling_stats: bool = True  # mean, std, min, max
    include_deltas: bool = True  # x_t - x_{t-1}


@dataclass
class FeatureGroupsConfig:
    """High-level feature configuration for ablation studies."""
    
    use_settings: bool = True  # Setting1-3
    use_sensors: bool = True  # Sensor1-21 (whatever is standard)
    use_physics_core: bool = True  # Effizienz_HPC_Proxy, EGT_Drift, Fan_HPC_Ratio
    use_temporal_windows: bool = True  # rolling stats / deltas
    use_condition_id: bool = False  # optional integer ConditionID
    temporal_cfg: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)

