"""
Unit Conversion Utilities for Thermodynamics

This module provides unit conversion functions and denormalization utilities
to ensure physically correct SI unit handling in the cycle branch.

CRITICAL: Always denormalize scaled values BEFORE converting to SI!
Never apply rankine_to_kelvin() or psia_to_pascal() to StandardScaler outputs.

Correct workflow:
    cycle_pred_scaled (mean≈0, std≈1)
    → denorm_by_condition()
    → cycle_pred_raw (°R, PSIA)
    → rankine_to_kelvin(), psia_to_pascal()
    → cycle_pred_SI (K, Pa)
    → validate_thermo_values_SI()
"""

from typing import Union, Dict, Tuple, Any, Literal
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Unit Conversions (Type-Preserving)
# ============================================================================

def rankine_to_kelvin(T_R: Union[float, np.ndarray, 'torch.Tensor']) -> Union[float, np.ndarray, 'torch.Tensor']:
    """Convert Rankine to Kelvin: K = R × 5/9

    Args:
        T_R: Temperature in Rankine (°R)

    Returns:
        Temperature in Kelvin (K)

    Example:
        >>> rankine_to_kelvin(491.67)  # 0°C
        273.15
        >>> rankine_to_kelvin(np.array([500, 600, 700]))
        array([277.78, 333.33, 388.89])
    """
    return T_R * (5.0 / 9.0)


def kelvin_to_rankine(T_K: Union[float, np.ndarray, 'torch.Tensor']) -> Union[float, np.ndarray, 'torch.Tensor']:
    """Convert Kelvin to Rankine: R = K × 9/5

    Args:
        T_K: Temperature in Kelvin (K)

    Returns:
        Temperature in Rankine (°R)
    """
    return T_K * (9.0 / 5.0)


def psia_to_pascal(P_psia: Union[float, np.ndarray, 'torch.Tensor']) -> Union[float, np.ndarray, 'torch.Tensor']:
    """Convert PSIA to Pascal: Pa = psia × 6894.757293168

    Args:
        P_psia: Pressure in pounds per square inch absolute (PSIA)

    Returns:
        Pressure in Pascals (Pa)

    Example:
        >>> psia_to_pascal(14.7)  # ~1 atm
        101352.93
    """
    return P_psia * 6894.757293168


def pascal_to_psia(P_pa: Union[float, np.ndarray, 'torch.Tensor']) -> Union[float, np.ndarray, 'torch.Tensor']:
    """Convert Pascal to PSIA: psia = Pa / 6894.757293168

    Args:
        P_pa: Pressure in Pascals (Pa)

    Returns:
        Pressure in PSIA
    """
    return P_pa / 6894.757293168


# ============================================================================
# Denormalization (CRITICAL for correct SI conversion)
# ============================================================================

def denorm_by_condition(
    x_scaled: Union[np.ndarray, 'torch.Tensor'],
    mean_by_cond: Union[np.ndarray, 'torch.Tensor'],
    std_by_cond: Union[np.ndarray, 'torch.Tensor'],
    cond_ids: Union[np.ndarray, 'torch.Tensor'],
    eps: float = 1e-6,
) -> Union[np.ndarray, 'torch.Tensor']:
    """Denormalize scaled values using condition-specific mean/std.

    CRITICAL: This function must be called BEFORE converting units to SI.
    Never apply rankine_to_kelvin() or psia_to_pascal() to scaled values!

    Args:
        x_scaled: Scaled values, shape (B, T, D) or (B, D)
                  Mean ≈ 0, Std ≈ 1 (StandardScaler output)
        mean_by_cond: Mean per condition, shape (num_conditions, D)
        std_by_cond: Std per condition, shape (num_conditions, D)
        cond_ids: Condition ID per sample, shape (B,)
        eps: Small constant for numerical stability

    Returns:
        x_raw: Denormalized values in original units (e.g., °R, PSIA)
               Shape matches x_scaled

    Formula:
        x_raw = x_scaled * std[cond_id] + mean[cond_id]

    Example:
        >>> x_scaled = np.array([[0.5, -0.5]])  # (1, 2) standardized
        >>> mean_by_cond = np.array([[1500, 500]])  # (1, 2) condition 0 stats
        >>> std_by_cond = np.array([[100, 50]])
        >>> cond_ids = np.array([0])
        >>> denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids)
        array([[1550., 475.]])  # Denormalized to raw units

    Raises:
        ValueError: If shapes are incompatible or cond_ids out of range
    """
    # Determine if torch or numpy
    is_torch = torch is not None and isinstance(x_scaled, torch.Tensor)

    if is_torch:
        # Convert to numpy for uniform handling
        x_np = x_scaled.detach().cpu().numpy() if x_scaled.requires_grad else x_scaled.cpu().numpy()
        mean_np = mean_by_cond.detach().cpu().numpy() if isinstance(mean_by_cond, torch.Tensor) and mean_by_cond.requires_grad else (mean_by_cond.cpu().numpy() if isinstance(mean_by_cond, torch.Tensor) else mean_by_cond)
        std_np = std_by_cond.detach().cpu().numpy() if isinstance(std_by_cond, torch.Tensor) and std_by_cond.requires_grad else (std_by_cond.cpu().numpy() if isinstance(std_by_cond, torch.Tensor) else std_by_cond)
        cond_np = cond_ids.detach().cpu().numpy() if isinstance(cond_ids, torch.Tensor) and cond_ids.requires_grad else (cond_ids.cpu().numpy() if isinstance(cond_ids, torch.Tensor) else cond_ids)
    else:
        x_np = np.asarray(x_scaled)
        mean_np = np.asarray(mean_by_cond)
        std_np = np.asarray(std_by_cond)
        cond_np = np.asarray(cond_ids)

    # Validate cond_ids range
    if cond_np.min() < 0 or cond_np.max() >= mean_np.shape[0]:
        raise ValueError(
            f"cond_ids out of range: [{cond_np.min()}, {cond_np.max()}], "
            f"but mean_by_cond has {mean_np.shape[0]} conditions"
        )

    # Index mean/std by condition: (B, D)
    sample_mean = mean_np[cond_np.astype(int)]  # (B, D)
    sample_std = std_np[cond_np.astype(int)]    # (B, D)

    # Handle sequence dimension if present
    if x_np.ndim == 3:
        # x_scaled: (B, T, D)
        # Expand to (B, 1, D) for broadcasting
        sample_mean = sample_mean[:, np.newaxis, :]  # (B, 1, D)
        sample_std = sample_std[:, np.newaxis, :]
    elif x_np.ndim == 2:
        # x_scaled: (B, D) - already correct shape
        pass
    else:
        raise ValueError(f"x_scaled must be 2D (B, D) or 3D (B, T, D), got shape {x_np.shape}")

    # Denormalize: x_raw = x_scaled * std + mean
    x_raw_np = x_np * (sample_std + eps) + sample_mean

    # Convert back to torch if input was torch
    if is_torch:
        return torch.from_numpy(x_raw_np).to(x_scaled.device).to(x_scaled.dtype)
    else:
        return x_raw_np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ThermoConfig:
    """Configuration for thermodynamics unit system and validation."""

    # Unit mappings per sensor (for documentation/validation)
    sensor_units: Dict[str, Literal["RANKINE", "PSIA", "SI_K", "SI_PA", "NONE"]] = field(default_factory=lambda: {
        "T24": "RANKINE",
        "T30": "RANKINE",
        "P30": "PSIA",
        "T50": "RANKINE",
    })

    # Physical bounds for sanity checks (in SI units)
    temp_bounds_K: Tuple[float, float] = (200.0, 2000.0)  # Reasonable for turbine cycle
    pressure_bounds_Pa: Tuple[float, float] = (1e4, 1e8)  # 10 kPa to 100 MPa
    efficiency_bounds: Tuple[float, float] = (0.70, 0.98)  # Turbomachinery range
    theta_bounds: Tuple[float, float] = (0.83, 1.00)  # Degradation multipliers

    # Fail-fast thresholds
    warn_if_viol_frac: float = 0.01  # Warn if >1% of values violate bounds
    fail_if_viol_frac: float = 0.05  # Fail if >5% violate (strict mode only)
    strict_mode: bool = False  # If True, raise on fail threshold

    # Operating settings physical ranges (for [0,1] normalization)
    # These are the ACTUAL physical ranges in CMAPSS data
    TRA_range: Tuple[float, float] = (0.0, 100.0)  # Throttle resolver angle (degrees)
    altitude_range_ft: Tuple[float, float] = (0.0, 42000.0)  # Altitude (feet)
    mach_range: Tuple[float, float] = (0.0, 0.9)  # Mach number (dimensionless)

    # Ops normalization tolerance
    ops_tolerance: float = 0.05  # Allow ops_t ∈ [-0.05, 1.05] with warning


def validate_thermo_values_SI(
    temps_K: np.ndarray,
    pressures_Pa: np.ndarray,
    config: ThermoConfig,
    context: str = "thermo_validation",
) -> Dict[str, Any]:
    """Validate thermodynamic values in SI units against physical bounds.

    IMPORTANT: temps_K and pressures_Pa must be in SI units (K, Pa).
    Call denorm_by_condition() first if working with scaled values!

    Args:
        temps_K: Temperatures in Kelvin
        pressures_Pa: Pressures in Pascal
        config: ThermoConfig with validation bounds
        context: Description for error messages

    Returns:
        Dict with:
        - violations: {type: count}
        - violation_fractions: {type: fraction}
        - warnings: List[str]
        - should_fail: bool (True if > fail_if_viol_frac)

    Example:
        >>> temps_K = np.array([300, 400, 500])
        >>> pressures_Pa = np.array([1e5, 2e5, 3e5])
        >>> config = ThermoConfig()
        >>> result = validate_thermo_values_SI(temps_K, pressures_Pa, config)
        >>> result["violation_fractions"]["temp"]
        0.0
    """
    result = {
        "violations": {},
        "violation_fractions": {},
        "warnings": [],
        "should_fail": False,
    }

    # Temperature validation
    min_K, max_K = config.temp_bounds_K
    temp_violations = ((temps_K < min_K) | (temps_K > max_K))
    temp_viol_count = int(temp_violations.sum())
    temp_viol_frac = float(temp_violations.mean())

    result["violations"]["temp"] = temp_viol_count
    result["violation_fractions"]["temp"] = temp_viol_frac

    if temp_viol_frac > config.warn_if_viol_frac:
        result["warnings"].append(
            f"[{context}] Temperature: {temp_viol_frac:.2%} outside [{min_K}-{max_K}K]"
        )

    # Pressure validation
    min_Pa, max_Pa = config.pressure_bounds_Pa
    pressure_violations = ((pressures_Pa < min_Pa) | (pressures_Pa > max_Pa))
    pressure_viol_count = int(pressure_violations.sum())
    pressure_viol_frac = float(pressure_violations.mean())

    result["violations"]["pressure"] = pressure_viol_count
    result["violation_fractions"]["pressure"] = pressure_viol_frac

    if pressure_viol_frac > config.warn_if_viol_frac:
        result["warnings"].append(
            f"[{context}] Pressure: {pressure_viol_frac:.2%} outside [{min_Pa:.0e}-{max_Pa:.0e}Pa]"
        )

    # Overall fail-fast decision
    overall_viol_frac = (temp_viol_frac + pressure_viol_frac) / 2
    if overall_viol_frac > config.fail_if_viol_frac:
        result["should_fail"] = True
        if config.strict_mode:
            raise ValueError(
                f"[{context}] {overall_viol_frac:.2%} of thermodynamic values violate physical bounds "
                f"(threshold: {config.fail_if_viol_frac:.2%}). Set strict_mode=False to allow."
            )

    return result
