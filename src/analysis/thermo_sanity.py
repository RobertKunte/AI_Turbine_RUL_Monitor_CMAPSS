"""
Thermodynamics Sanity Checks and Diagnostics

This module provides comprehensive validation of thermodynamic values,
ensuring unit correctness and physical plausibility.

CRITICAL WORKFLOW:
1. Denormalize scaled predictions/targets using denorm_by_condition()
2. Convert denormalized values from imperial (°R, PSIA) to SI (K, Pa)
3. Validate SI values against physical bounds
4. Generate detailed JSON report

Never convert StandardScaler outputs directly to SI!
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
    # Fallback if not available
    def compute_theta_health_batch(m_t, **kwargs):
        return {"saturation_frac": [0.0] * 6, "delta_l1_mean": 0.0}


def compute_thermo_sanity_report(
    cycle_pred: np.ndarray,
    cycle_target: np.ndarray,
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

    Args:
        cycle_pred: Cycle predictions (B, T, D) or (B, D) - RAW imperial units (°R, PSIA)
        cycle_target: Cycle targets (B, T, D) or (B, D) - RAW imperial units (°R, PSIA)
        m_t: Degradation modifiers (B, T, 6) or (B, 6), None if not available
        eta_nom: Nominal efficiencies (B, T, 5) or (B, 5), None if not available
        cond_ids: Condition IDs (B,)
        scaler_stats: Tuple of (mean_by_cond, std_by_cond), each (num_conditions, D)
        sensor_names: List of sensor names (e.g., ["T24", "T30", "P30", "T50"])
        epoch: Current epoch number
        step: Current training step
        config: ThermoConfig with validation bounds

    Returns:
        Dict with comprehensive diagnostics including:
        - metadata
        - theta_health
        - cycle_prediction_health (raw stats + SI stats + violations)
        - warnings
        - fail_fast_flags
    """
    report = {
        "metadata": {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_samples": int(len(cycle_pred)),
        },
        "theta_health": {},
        "cycle_prediction_health": {"per_sensor": {}},
        "fail_fast_flags": {},
        "warnings": [],
    }

    # =========================================================================
    # 1. Theta Health Metrics
    # =========================================================================
    if m_t is not None:
        theta_health = compute_theta_health_batch(m_t)
        report["theta_health"]["global"] = theta_health

        # Check saturation
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
    # 2. Per-Sensor Raw Stats (Imperial Units)
    # =========================================================================
    for i, name in enumerate(sensor_names):
        if i >= cycle_pred.shape[-1]:
            break

        pred_raw = cycle_pred[..., i]
        target_raw = cycle_target[..., i]

        report["cycle_prediction_health"]["per_sensor"][name] = {
            "pred_raw_mean": float(pred_raw.mean()),
            "pred_raw_std": float(pred_raw.std()),
            "pred_raw_min": float(pred_raw.min()),
            "pred_raw_max": float(pred_raw.max()),
            "target_raw_mean": float(target_raw.mean()),
            "target_raw_std": float(target_raw.std()),
            "target_raw_min": float(target_raw.min()),
            "target_raw_max": float(target_raw.max()),
        }

    # =========================================================================
    # 3. SI Conversion and Validation
    # =========================================================================
    temp_sensors = ["T24", "T30", "T50", "T2"]  # Rankine sensors
    pressure_sensors = ["P30", "P2", "P15"]      # PSIA sensors

    # Process temperature sensors
    for name in temp_sensors:
        if name not in sensor_names:
            continue

        idx = sensor_names.index(name)

        # Convert to SI
        pred_K = rankine_to_kelvin(cycle_pred[..., idx])
        target_K = rankine_to_kelvin(cycle_target[..., idx])

        report["cycle_prediction_health"]["per_sensor"][name].update({
            "pred_SI_K_mean": float(pred_K.mean()),
            "pred_SI_K_std": float(pred_K.std()),
            "pred_SI_K_min": float(pred_K.min()),
            "pred_SI_K_max": float(pred_K.max()),
            "target_SI_K_mean": float(target_K.mean()),
            "target_SI_K_std": float(target_K.std()),
        })

        # Validate bounds
        min_K, max_K = config.temp_bounds_K
        pred_violations = ((pred_K < min_K) | (pred_K > max_K))
        pred_viol_frac = float(pred_violations.mean())
        target_violations = ((target_K < min_K) | (target_K > max_K))
        target_viol_frac = float(target_violations.mean())

        report["cycle_prediction_health"]["per_sensor"][name]["pred_violations_frac"] = pred_viol_frac
        report["cycle_prediction_health"]["per_sensor"][name]["target_violations_frac"] = target_viol_frac

        if pred_viol_frac > config.warn_if_viol_frac:
            report["warnings"].append(
                f"{name}: {pred_viol_frac:.2%} predictions outside [{min_K}-{max_K}K]"
            )

        if target_viol_frac > config.warn_if_viol_frac:
            report["warnings"].append(
                f"{name}: {target_viol_frac:.2%} targets outside [{min_K}-{max_K}K] - check data!"
            )

    # Process pressure sensors
    for name in pressure_sensors:
        if name not in sensor_names:
            continue

        idx = sensor_names.index(name)

        # Convert to SI
        pred_Pa = psia_to_pascal(cycle_pred[..., idx])
        target_Pa = psia_to_pascal(cycle_target[..., idx])

        report["cycle_prediction_health"]["per_sensor"][name].update({
            "pred_SI_Pa_mean": float(pred_Pa.mean()),
            "pred_SI_Pa_std": float(pred_Pa.std()),
            "pred_SI_Pa_min": float(pred_Pa.min()),
            "pred_SI_Pa_max": float(pred_Pa.max()),
            "target_SI_Pa_mean": float(target_Pa.mean()),
            "target_SI_Pa_std": float(target_Pa.std()),
        })

        # Validate bounds
        min_Pa, max_Pa = config.pressure_bounds_Pa
        pred_violations = ((pred_Pa < min_Pa) | (pred_Pa > max_Pa))
        pred_viol_frac = float(pred_violations.mean())
        target_violations = ((target_Pa < min_Pa) | (target_Pa > max_Pa))
        target_viol_frac = float(target_violations.mean())

        report["cycle_prediction_health"]["per_sensor"][name]["pred_violations_frac"] = pred_viol_frac
        report["cycle_prediction_health"]["per_sensor"][name]["target_violations_frac"] = target_viol_frac

        if pred_viol_frac > config.warn_if_viol_frac:
            report["warnings"].append(
                f"{name}: {pred_viol_frac:.2%} predictions outside [{min_Pa:.0e}-{max_Pa:.0e}Pa]"
            )

        if target_viol_frac > config.warn_if_viol_frac:
            report["warnings"].append(
                f"{name}: {target_viol_frac:.2%} targets outside [{min_Pa:.0e}-{max_Pa:.0e}Pa] - check data!"
            )

    # =========================================================================
    # 4. Fail-Fast Decision
    # =========================================================================
    # Compute average violation fraction across all sensors
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
    """Save thermo sanity report to JSON files.

    Args:
        report: The report dict from compute_thermo_sanity_report()
        run_dir: Results directory (e.g., results/FD004/experiment_name/)
        epoch: Epoch number for filename
    """
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
    """Print compact console summary of thermo sanity report.

    Args:
        report: The report dict from compute_thermo_sanity_report()
    """
    epoch = report["metadata"]["epoch"]
    print(f"\n[THERMO SANITY] Epoch {epoch} Summary:")

    # Theta saturation
    theta_sat = report.get("theta_health", {}).get("saturation_frac_total", 0.0)
    print(f"  Theta saturation: {theta_sat:.2%}")

    # Overall violation fraction
    overall_viol = report.get("cycle_prediction_health", {}).get("overall_violation_frac", 0.0)
    print(f"  Overall physical violations: {overall_viol:.2%}")

    # Warnings
    warnings = report.get("warnings", [])
    print(f"  Warnings: {len(warnings)}")
    for warning in warnings[:5]:  # Print first 5 warnings
        print(f"    - {warning}")

    # Fail-fast flags
    flags = report.get("fail_fast_flags", {})
    if any(flags.values()):
        print(f"  ⚠️ FAIL-FAST FLAGS: {flags}")
