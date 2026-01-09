"""
Thermodynamics Sanity Checks and Diagnostics

This module provides comprehensive validation of thermodynamic values,
ensuring unit correctness and physical plausibility.

CRITICAL: UNDERSTAND THE THREE SPACES
========================================
- Space A (ML Input): Scaled features (z-scored), ops in [0,1]
- Space B (Loss):     cycle_pred_scaled vs cycle_target_scaled (both O(1))
- Space C (Physics):  Raw imperial (°R, psia) → SI (K, Pa) for validation

DATA FLOW:
1. cycle_target_scaled comes from X_batch (already scaled by StandardScaler)
2. cycle_pred_raw comes from CycleLayerMVP (raw imperial units)
3. cycle_pred_scaled = (cycle_pred_raw - μ) / σ  [done in CycleBranchLoss]
4. For diagnostics:
   - Reconstruct target_raw = cycle_target_scaled * σ + μ
   - Convert pred_raw and target_raw to SI
   - Validate SI values against physical bounds
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.utils.unit_conversions import (
    rankine_to_kelvin,
    psia_to_pascal,
    denorm_by_condition,
    validate_thermo_values_SI,
    ThermoConfig,
)

# Import theta health utilities
try:
    from src.analysis.cycle_diagnostics import compute_theta_health_batch
except ImportError:
    def compute_theta_health_batch(m_t, **kwargs):
        return {"saturation_frac": [0.0] * 6, "delta_l1_mean": 0.0}


def compute_thermo_sanity_report(
    cycle_pred_raw: np.ndarray,
    cycle_target_scaled: np.ndarray,
    m_t: Optional[np.ndarray],
    eta_nom: Optional[np.ndarray],
    cond_ids: np.ndarray,
    scaler_stats: Tuple[np.ndarray, np.ndarray],
    sensor_names: List[str],
    epoch: int,
    step: int,
    config: ThermoConfig,
) -> Dict[str, Any]:
    """Generate comprehensive thermo sanity report.

    SPACE CONTRACTS (enforced):
    - cycle_pred_raw:     RAW imperial units from CycleLayerMVP (°R, PSIA)
    - cycle_target_scaled: SCALED targets from X_batch (z-scored, ~N(0,1))
    - scaler_stats:        (μ_by_cond, σ_by_cond) for reconstructing raw

    Args:
        cycle_pred_raw: Cycle predictions (B, T, D) or (B, D) - RAW imperial
        cycle_target_scaled: Cycle targets (B, T, D) or (B, D) - SCALED (z-scored)
        m_t: Degradation modifiers (B, T, 6) or (B, 6), None if not available
        eta_nom: Nominal efficiencies (B, T, 5) or (B, 5), None if not available
        cond_ids: Condition IDs (B,)
        scaler_stats: Tuple of (mean_by_cond, std_by_cond), each (num_conditions, D)
        sensor_names: List of sensor names (e.g., ["T24", "T30", "P30", "T50"])
        epoch: Current epoch number
        step: Current training step
        config: ThermoConfig with validation bounds

    Returns:
        Dict with comprehensive diagnostics including scaled, raw, and SI stats
    """
    mean_by_cond, std_by_cond = scaler_stats
    
    report = {
        "metadata": {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_samples": int(len(cycle_pred_raw)),
            "space_info": {
                "target_space": "SCALED (from X_batch, z-scored)",
                "pred_space": "RAW (from CycleLayerMVP, imperial units)",
            }
        },
        "theta_health": {},
        "cycle_prediction_health": {"per_sensor": {}},
        "scale_contract": {},
        "fail_fast_flags": {},
        "warnings": [],
    }

    # =========================================================================
    # 0. Space Contract Validation
    # =========================================================================
    # Check: scaled targets should be O(1)
    target_abs_mean = float(np.abs(cycle_target_scaled).mean())
    if target_abs_mean > 5:
        report["warnings"].append(
            f"cycle_target_scaled abs mean={target_abs_mean:.1f} >> 1. "
            f"Possible bug: raw values passed instead of scaled?"
        )
        report["fail_fast_flags"]["target_not_scaled"] = True
    
    # Check: scaler_stats should not be identity
    mu_abs_mean = float(np.abs(mean_by_cond).mean())
    sigma_mean = float(std_by_cond.mean())
    if mu_abs_mean < 10 or sigma_mean < 2:
        report["warnings"].append(
            f"Scaler stats look like identity (μ_abs={mu_abs_mean:.1f}, σ={sigma_mean:.1f}). "
            f"Check scaler_dict wiring."
        )
        report["fail_fast_flags"]["identity_scaler"] = True
    
    report["scale_contract"] = {
        "target_scaled_abs_mean": target_abs_mean,
        "target_scaled_std": float(cycle_target_scaled.std()),
        "scaler_mu_abs_mean": mu_abs_mean,
        "scaler_sigma_mean": sigma_mean,
        "sample_mu_cond0": mean_by_cond[0].tolist() if len(mean_by_cond) > 0 else [],
        "sample_sigma_cond0": std_by_cond[0].tolist() if len(std_by_cond) > 0 else [],
    }

    # =========================================================================
    # 1. Theta Health Metrics
    # =========================================================================
    if m_t is not None:
        theta_health = compute_theta_health_batch(m_t)
        report["theta_health"]["global"] = theta_health

        sat_frac_list = theta_health.get("saturation_frac", [])
        if sat_frac_list:
            sat_frac_total = float(np.mean(sat_frac_list))
            report["theta_health"]["saturation_frac_total"] = sat_frac_total

            if sat_frac_total > config.warn_if_viol_frac:
                report["warnings"].append(
                    f"Theta saturation detected: {sat_frac_total:.2%} of values near bounds"
                )
                report["fail_fast_flags"]["theta_saturated"] = True

    # =========================================================================
    # 2. Reconstruct target_raw from scaled using denormalization
    # =========================================================================
    # target_raw = target_scaled * σ + μ
    cycle_target_raw = denorm_by_condition(
        cycle_target_scaled, mean_by_cond, std_by_cond, cond_ids
    )
    
    # Compute pred_scaled for reporting (pred is already raw)
    cycle_pred_scaled = np.empty_like(cycle_pred_raw)
    for i in range(len(cond_ids)):
        cond = int(cond_ids[i])
        mu = mean_by_cond[min(cond, len(mean_by_cond)-1)]
        sigma = std_by_cond[min(cond, len(std_by_cond)-1)]
        if cycle_pred_raw.ndim == 3:
            cycle_pred_scaled[i] = (cycle_pred_raw[i] - mu) / (sigma + 1e-6)
        else:
            cycle_pred_scaled[i] = (cycle_pred_raw[i] - mu) / (sigma + 1e-6)
    
    pred_scaled_abs_mean = float(np.abs(cycle_pred_scaled).mean())
    report["scale_contract"]["pred_scaled_abs_mean"] = pred_scaled_abs_mean
    report["scale_contract"]["pred_scaled_std"] = float(cycle_pred_scaled.std())
    
    # Check: pred_scaled should ideally be O(1) if model is trained
    if pred_scaled_abs_mean > 10:
        report["warnings"].append(
            f"pred_scaled abs mean={pred_scaled_abs_mean:.1f} >> 1. "
            f"Model predictions are far from targets in scaled space."
        )
        report["fail_fast_flags"]["pred_not_O1"] = True

    # =========================================================================
    # 3. Per-Sensor Stats (Scaled + Raw + SI)
    # =========================================================================
    temp_sensors = ["T24", "T30", "T50", "T2"]
    pressure_sensors = ["P30", "P2", "P15"]

    for i, name in enumerate(sensor_names):
        if i >= cycle_pred_raw.shape[-1]:
            break

        # Extract sensors
        pred_raw_i = cycle_pred_raw[..., i]
        target_raw_i = cycle_target_raw[..., i]
        pred_scaled_i = cycle_pred_scaled[..., i]
        target_scaled_i = cycle_target_scaled[..., i]

        sensor_report = {
            # Scaled space (training health)
            "pred_scaled_mean": float(pred_scaled_i.mean()),
            "pred_scaled_std": float(pred_scaled_i.std()),
            "target_scaled_mean": float(target_scaled_i.mean()),
            "target_scaled_std": float(target_scaled_i.std()),
            
            # Raw imperial space
            "pred_raw_mean": float(pred_raw_i.mean()),
            "pred_raw_std": float(pred_raw_i.std()),
            "pred_raw_min": float(pred_raw_i.min()),
            "pred_raw_max": float(pred_raw_i.max()),
            "target_raw_recon_mean": float(target_raw_i.mean()),
            "target_raw_recon_std": float(target_raw_i.std()),
            "target_raw_recon_min": float(target_raw_i.min()),
            "target_raw_recon_max": float(target_raw_i.max()),
        }

        # SI conversion (only on RAW values)
        if name in temp_sensors:
            pred_SI = rankine_to_kelvin(pred_raw_i)
            target_SI = rankine_to_kelvin(target_raw_i)
            unit_SI = "K"
            bounds = config.temp_bounds_K
        elif name in pressure_sensors:
            pred_SI = psia_to_pascal(pred_raw_i)
            target_SI = psia_to_pascal(target_raw_i)
            unit_SI = "Pa"
            bounds = config.pressure_bounds_Pa
        else:
            # Unknown sensor type, skip SI conversion
            report["cycle_prediction_health"]["per_sensor"][name] = sensor_report
            continue

        sensor_report.update({
            f"pred_SI_{unit_SI}_mean": float(pred_SI.mean()),
            f"pred_SI_{unit_SI}_std": float(pred_SI.std()),
            f"pred_SI_{unit_SI}_min": float(pred_SI.min()),
            f"pred_SI_{unit_SI}_max": float(pred_SI.max()),
            f"target_SI_{unit_SI}_mean": float(target_SI.mean()),
            f"target_SI_{unit_SI}_std": float(target_SI.std()),
            f"target_SI_{unit_SI}_min": float(target_SI.min()),
            f"target_SI_{unit_SI}_max": float(target_SI.max()),
        })

        # Validate bounds (SI space only, using RAW-reconstructed values)
        min_bound, max_bound = bounds
        pred_violations = ((pred_SI < min_bound) | (pred_SI > max_bound))
        target_violations = ((target_SI < min_bound) | (target_SI > max_bound))
        pred_viol_frac = float(pred_violations.mean())
        target_viol_frac = float(target_violations.mean())

        sensor_report["pred_violations_frac"] = pred_viol_frac
        sensor_report["target_violations_frac"] = target_viol_frac

        if pred_viol_frac > config.warn_if_viol_frac:
            if unit_SI == "K":
                report["warnings"].append(
                    f"{name}: {pred_viol_frac:.2%} predictions outside [{min_bound:.0f}-{max_bound:.0f}K]"
                )
            else:
                report["warnings"].append(
                    f"{name}: {pred_viol_frac:.2%} predictions outside [{min_bound:.0e}-{max_bound:.0e}Pa]"
                )

        if target_viol_frac > config.warn_if_viol_frac:
            if unit_SI == "K":
                report["warnings"].append(
                    f"{name}: {target_viol_frac:.2%} targets outside [{min_bound:.0f}-{max_bound:.0f}K] - check data!"
                )
            else:
                report["warnings"].append(
                    f"{name}: {target_viol_frac:.2%} targets outside [{min_bound:.0e}-{max_bound:.0e}Pa] - check data!"
                )

        report["cycle_prediction_health"]["per_sensor"][name] = sensor_report

    # =========================================================================
    # 4. Fail-Fast Decision
    # =========================================================================
    all_viol_fracs = [
        report["cycle_prediction_health"]["per_sensor"][s].get("pred_violations_frac", 0.0)
        for s in sensor_names if s in report["cycle_prediction_health"]["per_sensor"]
    ]

    if all_viol_fracs:
        total_violation_frac = float(np.mean(all_viol_fracs))
        report["cycle_prediction_health"]["overall_violation_frac"] = total_violation_frac

        if total_violation_frac > config.fail_if_viol_frac:
            report["fail_fast_flags"]["physics_violation"] = True
            if config.strict_mode:
                raise ValueError(
                    f"[THERMO SANITY] {total_violation_frac:.2%} of predictions violate physical bounds "
                    f"(threshold: {config.fail_if_viol_frac:.2%}). Set strict_mode=False to allow."
                )

    return report


def save_thermo_sanity_report(
    report: Dict[str, Any],
    run_dir: Path,
    epoch: int,
) -> None:
    """Save thermo sanity report to JSON files."""
    diag_dir = run_dir / "diagnostics" / "thermo_sanity"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch-specific report
    report_path = diag_dir / f"thermo_sanity_epoch{epoch:03d}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Also save as latest
    latest_path = diag_dir / "thermo_sanity_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[THERMO SANITY] Saved report to {report_path}")


def print_thermo_sanity_summary(report: Dict[str, Any]) -> None:
    """Print compact console summary of thermo sanity report."""
    epoch = report["metadata"]["epoch"]
    print(f"\n[THERMO SANITY] Epoch {epoch} Summary:")

    # Scale contract
    scale = report.get("scale_contract", {})
    print(f"  Scale contract: target_scaled_abs_mean={scale.get('target_scaled_abs_mean', 0):.2f}, "
          f"pred_scaled_abs_mean={scale.get('pred_scaled_abs_mean', 0):.2f}")

    # Theta saturation
    theta_sat = report.get("theta_health", {}).get("saturation_frac_total", 0.0)
    print(f"  Theta saturation: {theta_sat:.2%}")

    # Overall violation fraction
    overall_viol = report.get("cycle_prediction_health", {}).get("overall_violation_frac", 0.0)
    print(f"  Overall physical violations: {overall_viol:.2%}")

    # Warnings
    warnings = report.get("warnings", [])
    print(f"  Warnings: {len(warnings)}")
    for warning in warnings[:5]:
        print(f"    - {warning}")

    # Fail-fast flags
    flags = report.get("fail_fast_flags", {})
    if any(flags.values()):
        print(f"  ⚠️ FAIL-FAST FLAGS: {flags}")
