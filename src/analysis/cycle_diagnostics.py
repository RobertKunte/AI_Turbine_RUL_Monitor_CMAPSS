"""
Cycle Branch Diagnostics.

This module provides diagnostic plots and analysis for the differentiable
cycle branch, including:
- θ (degradation modifier) trajectory plots
- Cycle sensor fit overlays
- Residual distributions
- Nominal vs effective efficiency analysis
- Variance attribution analysis
- Bounds and saturation reports
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
except ImportError:
    plt = None

try:
    import torch
except ImportError:
    torch = None


def run_cycle_branch_diagnostics(
    run_dir: Path,
    theta_data: Dict[str, np.ndarray],
    cycle_pred_data: Dict[str, np.ndarray],
    cycle_target_data: Dict[str, np.ndarray],
    eta_nom_data: Optional[Dict[str, np.ndarray]] = None,
    eta_eff_data: Optional[Dict[str, np.ndarray]] = None,
    engine_groups: Optional[Dict[str, List[int]]] = None,
    sensor_names: List[str] = None,
) -> Dict[str, Any]:
    """Run full cycle branch diagnostics and save artifacts.
    
    Args:
        run_dir: Directory to save diagnostic outputs
        theta_data: Dict mapping engine_id -> theta trajectory (T, 6)
        cycle_pred_data: Dict mapping engine_id -> predicted sensors (T, n_targets)
        cycle_target_data: Dict mapping engine_id -> target sensors (T, n_targets)
        eta_nom_data: Optional dict mapping engine_id -> nominal etas (T, 5)
        eta_eff_data: Optional dict mapping engine_id -> effective etas (T, 5)
        engine_groups: Dict with keys 'best20', 'worst20', 'mid20' -> list of engine IDs
        sensor_names: List of sensor names matching target dimension
        
    Returns:
        Summary dict with metrics and warnings
    """
    if plt is None:
        print("matplotlib not available, skipping cycle diagnostics plots")
        return {}
    
    diag_dir = run_dir / "diagnostics" / "cycle_branch"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    if sensor_names is None:
        sensor_names = ["T24", "T30", "P30", "T50"]
    
    # Default engine groups if not provided
    if engine_groups is None:
        all_engines = list(theta_data.keys())
        n = len(all_engines)
        k = min(20, n // 3)
        engine_groups = {
            "worst20": all_engines[:k],
            "mid20": all_engines[n//2 - k//2 : n//2 + k//2],
            "best20": all_engines[-k:],
        }
    
    summary = {}
    
    # 1. Theta trajectory plots
    for group_name, engine_ids in engine_groups.items():
        fig = plot_theta_trajectories(theta_data, engine_ids, group_name)
        if fig:
            fig.savefig(diag_dir / f"theta_traj_{group_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    
    # 2. Cycle sensor fit plots
    for sensor_idx, sensor_name in enumerate(sensor_names):
        for group_name, engine_ids in engine_groups.items():
            fig = plot_cycle_sensor_fit(
                cycle_pred_data, cycle_target_data,
                engine_ids, sensor_idx, sensor_name, group_name
            )
            if fig:
                fig.savefig(
                    diag_dir / f"cycle_fit_{sensor_name}_{group_name}.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
    
    # 3. Residual distributions
    residuals = compute_residual_stats(cycle_pred_data, cycle_target_data, sensor_names)
    summary["residuals"] = residuals
    
    for sensor_idx, sensor_name in enumerate(sensor_names):
        fig = plot_residual_distribution(
            cycle_pred_data, cycle_target_data,
            sensor_idx, sensor_name
        )
        if fig:
            fig.savefig(
                diag_dir / f"residual_dist_{sensor_name}.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
    
    # 4. Nominal vs Effective analysis (D1)
    if eta_nom_data and eta_eff_data:
        for comp_idx, comp_name in enumerate(["fan", "lpc", "hpc", "hpt", "lpt"]):
            fig = plot_eta_nom_vs_eff(
                eta_nom_data, eta_eff_data, theta_data,
                comp_idx, comp_name, engine_groups.get("worst20", [])
            )
            if fig:
                fig.savefig(
                    diag_dir / f"eta_nom_vs_eff_{comp_name}.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
    
    # 5. Variance attribution (D2)
    if eta_nom_data:
        variance_attr = compute_variance_attribution_from_data(
            eta_nom_data, theta_data, eta_eff_data
        )
        summary["variance_attribution"] = variance_attr
        
        with open(diag_dir / "variance_attribution.json", "w") as f:
            json.dump(variance_attr, f, indent=2)
    
    # 6. Bounds and saturation report
    bounds_stats = compute_theta_bounds_stats(theta_data)
    summary["theta_bounds"] = bounds_stats
    
    # 7. Add warnings
    warnings = {}
    if "variance_attribution" in summary:
        # Check for low degradation usage
        low_usage = all(
            v.get("var_share_deg", 0) < 0.05
            for k, v in summary["variance_attribution"].items()
            if not k.startswith("_")
        )
        warnings["deg_usage_warning"] = low_usage
    
    if "theta_bounds" in summary:
        sat_frac = summary["theta_bounds"].get("saturation_frac_total", 0)
        warnings["saturation_warning"] = sat_frac > 0.2
    
    summary["_warnings"] = warnings
    
    # Save summary
    with open(diag_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save metrics CSV
    if residuals:
        import csv
        with open(diag_dir / "metrics_cycle.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sensor", "mae", "mse", "std"])
            for sensor_name, stats in residuals.items():
                writer.writerow([
                    sensor_name,
                    stats.get("mae", ""),
                    stats.get("mse", ""),
                    stats.get("std", ""),
                ])
    
    print(f"[cycle_diagnostics] Saved artifacts to {diag_dir}")
    return summary


def plot_theta_trajectories(
    theta_data: Dict[str, np.ndarray],
    engine_ids: List[int],
    group_name: str,
) -> Optional[Any]:
    """Plot theta (m_t) trajectories for a group of engines."""
    if not engine_ids or plt is None:
        return None
    
    param_names = ["m_fan", "m_lpc", "m_hpc", "m_hpt", "m_lpt", "m_dp"]
    n_params = 6
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for param_idx in range(n_params):
        ax = axes[param_idx]
        for eng_id in engine_ids[:10]:  # Limit to 10 for readability
            if str(eng_id) in theta_data:
                theta = theta_data[str(eng_id)]
                if theta.ndim == 2 and theta.shape[1] >= param_idx + 1:
                    ax.plot(theta[:, param_idx], alpha=0.5, linewidth=0.8)
        
        ax.set_xlabel("Cycle")
        ax.set_ylabel(param_names[param_idx])
        ax.set_title(param_names[param_idx])
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Degradation Modifiers θ(t) — {group_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_cycle_sensor_fit(
    pred_data: Dict[str, np.ndarray],
    target_data: Dict[str, np.ndarray],
    engine_ids: List[int],
    sensor_idx: int,
    sensor_name: str,
    group_name: str,
) -> Optional[Any]:
    """Plot predicted vs actual sensor values for a group of engines."""
    if not engine_ids or plt is None:
        return None
    
    n_engines = min(6, len(engine_ids))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, eng_id in enumerate(engine_ids[:n_engines]):
        ax = axes[i]
        eng_key = str(eng_id)
        
        if eng_key in pred_data and eng_key in target_data:
            pred = pred_data[eng_key]
            target = target_data[eng_key]
            
            if pred.ndim == 2 and pred.shape[1] > sensor_idx:
                ax.plot(target[:, sensor_idx], label="Actual", alpha=0.8)
                ax.plot(pred[:, sensor_idx], label="Predicted", alpha=0.8, linestyle="--")
                ax.legend(fontsize=8)
        
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor_name)
        ax.set_title(f"Engine {eng_id}")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_engines, 6):
        axes[i].set_visible(False)
    
    fig.suptitle(f"{sensor_name} Fit — {group_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_residual_distribution(
    pred_data: Dict[str, np.ndarray],
    target_data: Dict[str, np.ndarray],
    sensor_idx: int,
    sensor_name: str,
) -> Optional[Any]:
    """Plot histogram of residuals for a sensor."""
    if plt is None:
        return None
    
    all_residuals = []
    for eng_key in pred_data:
        if eng_key in target_data:
            pred = pred_data[eng_key]
            target = target_data[eng_key]
            if pred.ndim == 2 and pred.shape[1] > sensor_idx:
                residual = pred[:, sensor_idx] - target[:, sensor_idx]
                all_residuals.extend(residual.flatten().tolist())
    
    if not all_residuals:
        return None
    
    residuals = np.array(all_residuals)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(residuals.mean(), color="orange", linestyle="-", linewidth=1.5, 
               label=f"Mean: {residuals.mean():.3f}")
    
    ax.set_xlabel(f"Residual ({sensor_name})")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution — {sensor_name}\n"
                 f"MAE: {np.abs(residuals).mean():.4f}, Std: {residuals.std():.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_eta_nom_vs_eff(
    eta_nom_data: Dict[str, np.ndarray],
    eta_eff_data: Dict[str, np.ndarray],
    theta_data: Dict[str, np.ndarray],
    comp_idx: int,
    comp_name: str,
    engine_ids: List[int],
) -> Optional[Any]:
    """Plot nominal, effective, and modifier for a component (D1 analysis)."""
    if not engine_ids or plt is None:
        return None
    
    n_engines = min(4, len(engine_ids))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, eng_id in enumerate(engine_ids[:n_engines]):
        ax = axes[i]
        eng_key = str(eng_id)
        
        if eng_key in eta_nom_data and eng_key in theta_data:
            eta_nom = eta_nom_data[eng_key]
            theta = theta_data[eng_key]
            
            if eta_nom.ndim == 2 and eta_nom.shape[1] > comp_idx:
                ax.plot(eta_nom[:, comp_idx], label="η_nom (ops)", alpha=0.8)
            
            if theta.ndim == 2 and theta.shape[1] > comp_idx:
                ax.plot(theta[:, comp_idx], label=f"m_{comp_name} (deg)", alpha=0.8)
            
            if eng_key in eta_eff_data:
                eta_eff = eta_eff_data[eng_key]
                if eta_eff.ndim == 2 and eta_eff.shape[1] > comp_idx:
                    ax.plot(eta_eff[:, comp_idx], label="η_eff", alpha=0.8, linestyle="--")
            
            ax.legend(fontsize=8)
        
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Value")
        ax.set_title(f"Engine {eng_id}")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"η_nom vs η_eff vs m(t) — {comp_name.upper()}", fontsize=14)
    plt.tight_layout()
    return fig


def compute_residual_stats(
    pred_data: Dict[str, np.ndarray],
    target_data: Dict[str, np.ndarray],
    sensor_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-sensor residual statistics."""
    n_sensors = len(sensor_names)
    result = {}
    
    for sensor_idx, sensor_name in enumerate(sensor_names):
        all_residuals = []
        for eng_key in pred_data:
            if eng_key in target_data:
                pred = pred_data[eng_key]
                target = target_data[eng_key]
                if pred.ndim == 2 and pred.shape[1] > sensor_idx:
                    residual = pred[:, sensor_idx] - target[:, sensor_idx]
                    all_residuals.extend(residual.flatten().tolist())
        
        if all_residuals:
            residuals = np.array(all_residuals)
            result[sensor_name] = {
                "mae": float(np.abs(residuals).mean()),
                "mse": float((residuals ** 2).mean()),
                "std": float(residuals.std()),
                "mean": float(residuals.mean()),
            }
    
    return result


def compute_variance_attribution_from_data(
    eta_nom_data: Dict[str, np.ndarray],
    theta_data: Dict[str, np.ndarray],
    eta_eff_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute variance attribution between nominal and degradation (D2 analysis)."""
    comp_names = ["fan", "lpc", "hpc", "hpt", "lpt"]
    
    # Collect all data
    eta_nom_all = []
    m_all = []
    eta_eff_all = []
    
    for eng_key in eta_nom_data:
        if eng_key in theta_data:
            eta_nom = eta_nom_data[eng_key]
            theta = theta_data[eng_key]
            
            if eta_nom.ndim == 2:
                eta_nom_all.append(eta_nom[:, :5])
            if theta.ndim == 2:
                m_all.append(theta[:, :5])
            
            if eta_eff_data and eng_key in eta_eff_data:
                eta_eff = eta_eff_data[eng_key]
                if eta_eff.ndim == 2:
                    eta_eff_all.append(eta_eff[:, :5])
    
    if not eta_nom_all or not m_all:
        return {}
    
    eta_nom_concat = np.concatenate(eta_nom_all, axis=0)  # (N, 5)
    m_concat = np.concatenate(m_all, axis=0)  # (N, 5)
    eta_eff_concat = np.concatenate(eta_eff_all, axis=0) if eta_eff_all else None
    
    result = {}
    for i, name in enumerate(comp_names):
        var_nom = float(np.var(eta_nom_concat[:, i]))
        var_m = float(np.var(m_concat[:, i]))
        total_var = var_nom + var_m
        var_share_deg = var_m / total_var if total_var > 1e-10 else 0.0
        
        result[name] = {
            "var_eta_nom": var_nom,
            "var_m": var_m,
            "var_share_deg": var_share_deg,
        }
        
        if eta_eff_concat is not None:
            result[name]["var_eta_eff"] = float(np.var(eta_eff_concat[:, i]))
    
    return result


def compute_theta_bounds_stats(
    theta_data: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute bounds and saturation statistics for theta."""
    param_names = ["m_fan", "m_lpc", "m_hpc", "m_hpt", "m_lpt", "m_dp_comb"]
    
    # Collect all theta values
    all_theta = []
    for eng_key, theta in theta_data.items():
        if theta.ndim == 2 and theta.shape[1] == 6:
            all_theta.append(theta)
    
    if not all_theta:
        return {}
    
    theta_concat = np.concatenate(all_theta, axis=0)  # (N, 6)
    
    result = {
        "min_per_param": {},
        "max_per_param": {},
        "mean_per_param": {},
        "saturation_frac_per_param": {},
    }
    
    # Assume bounds roughly [0.85, 1.0] for eta and [0.9, 1.0] for dp
    bounds = [(0.85, 1.0)] * 5 + [(0.90, 1.0)]
    
    total_sat = 0
    for i, name in enumerate(param_names):
        vals = theta_concat[:, i]
        result["min_per_param"][name] = float(vals.min())
        result["max_per_param"][name] = float(vals.max())
        result["mean_per_param"][name] = float(vals.mean())
        
        # Saturation check (within 2% of bounds)
        lb, ub = bounds[i]
        range_val = ub - lb
        near_low = (vals < lb + 0.02 * range_val).mean()
        near_high = (vals > ub - 0.02 * range_val).mean()
        sat_frac = float(near_low + near_high)
        result["saturation_frac_per_param"][name] = sat_frac
        total_sat += sat_frac
    
    result["saturation_frac_total"] = total_sat / len(param_names)
    
    return result
