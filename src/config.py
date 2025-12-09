# src/config.py

from dataclasses import dataclass, field
from typing import List

CMAPSS_DATASETS = {
    "FD001": {"desc": "1 cond, 1 fault (HPC)",        "n_train": 100, "n_test": 100},
    "FD002": {"desc": "6 cond, 1 fault (HPC)",        "n_train": 260, "n_test": 259},
    "FD003": {"desc": "1 cond, 2 faults (HPC+Fan)",   "n_train": 100, "n_test": 100},
    "FD004": {"desc": "6 cond, 2 faults (HPC+Fan)",   "n_train": 248, "n_test": 249},
}


@dataclass
class ResidualFeatureConfig:
    """
    Configuration for residual/digital-twin features.
    Residual features compare sensor values against a "healthy" baseline.
    """
    enabled: bool = False
    mode: str = "per_engine"  # "per_engine" or "per_condition"
    baseline_len: int = 30  # Number of early cycles for baseline
    include_original: bool = True  # Keep original features as well


@dataclass
class PhysicsFeatureConfig:
    """
    Configuration for physics-informed features.
    Controls which physical features are created and how.
    """
    use_core: bool = True
    use_extended: bool = False
    use_residuals: bool = False
    use_temporal_on_physics: bool = False

    # Explicit names for core physical features
    core_features: List[str] = field(default_factory=lambda: [
        "HPC_Eff_Proxy",
        "EGT_Drift",
        "Fan_HPC_Ratio",
    ])

    # Optional granular toggles for extended features
    use_temp_ratios: bool = True
    use_press_ratios: bool = True
    use_corrected_speeds: bool = False
    
    # Residual feature configuration
    residual: ResidualFeatureConfig = field(default_factory=ResidualFeatureConfig)

MAX_RUL = 125
SEQUENCE_LENGTH = 30

# Multi-task Health Index configuration
USE_HEALTH_HEAD: bool = False  # Set to True to enable multi-task mode
HEALTH_LOSS_WEIGHT: float = 1.0  # lambda_health: trade-off between RUL and health loss (increased from 0.3)
RUL_WEIGHTING_TAU: float = 40.0  # tau: scale parameter for exponential RUL weighting
RUL_LOSS_SCALE: float = 1.0 / 125.0  # Scale factor for RUL loss to balance with HI losses

# Health Index target + regularization
HI_RUL_PLATEAU_THRESH: int = 80   # RUL threshold (cycles) above which HI_target=1
HI_EOL_THRESH: int = 25            # RUL threshold (cycles) for EOL-tail zone (20-30 cycles)
USE_HI_MONOTONICITY: bool = True   # enable monotonicity regularizer
HI_MONO_WEIGHT: float = 0.05       # weight of monotonicity penalty in loss (increased from 0.02)
HI_EOL_WEIGHT: float = 10.0        # extra weight for EOL health penalty (HI→0) - increased from 4.0
HI_MONO_RUL_BETA: float = 60.0     # RUL threshold for late region in monotonicity penalty (cycles)
                                    # Only penalize HI increases when RUL <= beta (extended from 30)
HI_GLOBAL_MONO_WEIGHT: float = 0.005  # Weight for global trend loss (penalizes HI increases over entire lifecycle)

# --- Health Index / Damage Head ----------------------------------------
USE_DAMAGE_HEALTH_HEAD: bool = True
"""
If True, the Health Index head uses a cumulative damage model:
HI_t = exp(-alpha * cumsum(rate_t)), with rate_t >= 0.
This ensures HI is monotonically decreasing by construction.
"""

DAMAGE_ALPHA_INIT: float = 0.01
"""
Initial value for the damage scaling factor alpha (>0).
HI = exp(-alpha * damage). Smaller alpha => slower decay.
"""

DAMAGE_SOFTPLUS_BETA: float = 1.0
"""
Softplus beta for the damage rates. For now we can keep 1.0, but this
allows later tuning how sharp the non-negativity constraint behaves.
"""

# --- Condition Calibration for Health Index --------------------------------
HI_CONDITION_CALIB_WEIGHT: float = 0.05  # Default weight for condition calibration loss
"""
Weight for condition calibration loss. 
Forces early-life HI to be similar across different operating conditions.
Set to 0.0 to disable condition calibration.
"""
HI_CONDITION_CALIB_PLATEAU_THRESH: float = 80.0
"""
RUL threshold above which samples are considered "early-life" for condition calibration.
Should match HI_RUL_PLATEAU_THRESH for consistency.
"""
HI_CONDITION_CALIB_VAR_ALPHA: float = 0.1
"""
Weight for variance penalty between condition means in condition calibration loss.
Higher values force condition means to be more similar.
"""

# --- Condition-wise Feature Scaling -----------------------------------------
USE_CONDITION_WISE_SCALING: bool = True
"""
If True, use separate StandardScaler per Condition-ID for feature scaling.
This helps normalize features across different operating conditions (FD002/FD004).
If False, use a single global scaler (backward compatible).
"""

# --- Phase 2: Condition Embeddings -------------------------------------------
USE_CONDITION_EMBEDDING: bool = False  # Phase 2: Enable condition embeddings
"""
If True, learn embeddings for operating conditions (ConditionID) and concatenate
them to input features. Helps stabilize Health Index across different conditions.
Default: False (backward compatible).
"""
COND_EMB_DIM: int = 4  # Phase 2: Dimension of condition embeddings
"""
Dimension of condition embedding vectors. Typical values: 4-8.
Only used if USE_CONDITION_EMBEDDING=True.
"""

# --- Phase 2: Health Index Smoothness ----------------------------------------
SMOOTH_HI_WEIGHT: float = 0.0  # Phase 2: Weight for HI smoothness loss
"""
Weight for health index smoothness loss. Penalizes large HI changes between
consecutive time steps, especially in early-life (high RUL) region.
Typical values: 0.01-0.05. Set to 0.0 to disable.
"""
SMOOTH_HI_PLATEAU_THRESH: float = 80.0  # Phase 2: RUL threshold for smoothness masking
"""
RUL threshold above which smoothness loss is applied (early-life region).
Should match HI_RUL_PLATEAU_THRESH for consistency.
"""

# --- Phase 2: Encoder Type (LSTM / Transformer) -----------------------------
ENCODER_TYPE: str = "lstm"  # Phase 2: Encoder type ("lstm" or "transformer")
"""
Encoder type for sequence processing. Options:
- "lstm": Standard LSTM encoder (default, backward compatible)
- "transformer": Transformer encoder (experimental)
"""
TRANSFORMER_D_MODEL: int = 128  # Phase 2: Transformer model dimension
"""
Transformer model dimension (d_model). Should be divisible by nhead.
Only used if ENCODER_TYPE="transformer".
"""
TRANSFORMER_NHEAD: int = 4  # Phase 2: Number of attention heads
"""
Number of attention heads in transformer. d_model must be divisible by nhead.
Only used if ENCODER_TYPE="transformer".
"""
TRANSFORMER_NUM_LAYERS: int = 2  # Phase 2: Number of transformer layers
"""
Number of transformer encoder layers.
Only used if ENCODER_TYPE="transformer".
"""
TRANSFORMER_DIM_FEEDFORWARD: int = 256  # Phase 2: Feedforward dimension
"""
Dimension of feedforward network in transformer layers.
Only used if ENCODER_TYPE="transformer".
"""

HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1

LEARNING_RATE = 1e-3
NUM_EPOCHS = 25  # oder 30, wenn du magst

# Sensors we keep (common across FD001–FD004)
GLOBAL_SENSOR_FEATURES = [
    "Sensor2", "Sensor3", "Sensor4",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14",
    "Sensor15", "Sensor16", "Sensor17", "Sensor18",
    "Sensor19", "Sensor20", "Sensor21",
]

GLOBAL_SETTING_FEATURES = ["Setting1", "Setting2", "Setting3"]

GLOBAL_PHYS_FEATURES = [
    "Effizienz_HPC_Proxy",
    "EGT_Drift",
    "Fan_HPC_Ratio",
]

# FD-ID as simple numeric feature (0..3, will be scaled)
GLOBAL_META_FEATURES = ["FD_ID"]

GLOBAL_FEATURE_COLS = (
    GLOBAL_SETTING_FEATURES
    + GLOBAL_SENSOR_FEATURES
    + GLOBAL_PHYS_FEATURES
    + GLOBAL_META_FEATURES
)

GLOBAL_DROP_COLS = [
    "Sensor1", "Sensor5", "Sensor10",  # quasi-konstant
    "MaxTime",                         # nur Hilfsspalte für RUL
]