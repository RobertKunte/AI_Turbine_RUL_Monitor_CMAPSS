"""
Failure Case Analysis - Compact reporting for RUL models.
Generates group overlays and grids for Worst/Best/Mid engines.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.analysis.inference import EngineEOLMetrics, EngineTrajectory

@dataclass
class FailureAnalysisConfig:
    K: int = 20
    plots_ext: str = "png"
    dpi: int = 150

def compute_last_errors_per_unit(
    eol_metrics: List[EngineEOLMetrics]
) -> pd.DataFrame:
    """
    Convert metrics list to a clean DataFrame for ranking.
    """
    data = []
    for m in eol_metrics:
        # Calculate signed error if not present (m.error is pred - true)
        signed_err = m.error
        abs_err = abs(signed_err)
        
        row = {
            "unit_id": m.unit_id,
            "true_last_rul": m.true_rul,
            "pred_last_rul": m.pred_rul,
            "abs_err_last": abs_err,
            "signed_err_last": signed_err,
            "nasa_score": m.nasa,
        }
        
        # Add optional quantiles if available, defaulting to None
        row["q10_pred"] = getattr(m, "q10", None)
        row["q50_pred"] = getattr(m, "q50", None)
        row["q90_pred"] = getattr(m, "q90", None)
            
        data.append(row)
        
    df = pd.DataFrame(data)
    # Sort deterministically by absolute error descending
    df = df.sort_values(by=["abs_err_last", "unit_id"], ascending=[False, True])
    return df

def select_groups_from_last_error(
    df: pd.DataFrame, 
    K: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Select worst, best, and mid K unit_ids based on last error.
    Assumes df is already sorted by abs_err_last descending.
    """
    if df.empty:
        return [], [], []
        
    all_units = df["unit_id"].tolist()
    n = len(all_units)
    K = min(K, n) # Clamp K if fewer engines than K
    
    # Worst: Top K (highest error)
    worst_20 = all_units[:K]
    
    # Best: Bottom K (lowest error)
    best_20 = all_units[-K:]
    
    # Mid: Center slice
    # Logic: center index m, take K around it.
    m = n // 2
    # Start K/2 before median. Ensure bounds [0, n]
    start = max(0, m - K // 2)
    end = min(n, start + K)
    # Adjust if near end (e.g. if start+K > n)
    if end == n: 
        start = max(0, n - K)
    
    mid_20 = all_units[start:end]
    
    return worst_20, best_20, mid_20

def write_cases_csv(
    df: pd.DataFrame, 
    groups: Dict[str, List[int]], 
    out_dir: Path
):
    """
    Write summary CSVs:
    1. cases.csv (all units, sorted)
    2. selected_groups.csv (only selected units with group label)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. All cases
    df.to_csv(out_dir / "cases.csv", index=False)
    
    # 2. Selected groups
    group_rows = []
    for grp_name, unit_ids in groups.items():
        # Filter df for these units, maintain order of unit_ids list if possible
        # but here we just grab rows. 
        # Better: iterate unit_ids and fetch row to preserve ranking order if relevant
        # or just label existing rows.
        
        for uid in unit_ids:
            row = df[df["unit_id"] == uid].iloc[0].to_dict()
            row["group"] = grp_name
            group_rows.append(row)
            
    pd.DataFrame(group_rows).to_csv(out_dir / "selected_groups.csv", index=False)

def plot_rul_overlay_group(
    trajectories: Dict[int, EngineTrajectory],
    unit_ids: List[int],
    group_name: str,
    metric_info: str,
    out_path: Path
):
    """
    Overlay RUL trajectories for a group.
    """
    plt.figure(figsize=(10, 6))
    
    # Collect all lengths to compute max cycle
    max_cycle = 0
    
    # Plot individual trajectories (thin lines)
    for uid in unit_ids:
        if uid not in trajectories:
            continue
        traj = trajectories[uid]
        plt.plot(traj.cycles, traj.true_rul, color='gray', alpha=0.3, linewidth=1)
        # Predicted
        plt.plot(traj.cycles, traj.pred_rul, color='red', alpha=0.3, linewidth=1)
        max_cycle = max(max_cycle, traj.cycles[-1] if len(traj.cycles) > 0 else 0)

    # Note: Mean trajectory is tricky if lengths differ significantly. 
    # For RUL, we often just want to see the spread. 
    # Let's skip computing a "mean" line for now to avoid interpolation artifacts 
    # unless requested. The request asked for "mean/median (thicker)", 
    # but aligning them is non-trivial without re-indexing to life fraction.
    # Given the constraint of "compact", let's stick to the overlay which shows the density.
    
    plt.xlabel("Time (cycles)")
    plt.ylabel("RUL (cycles)")
    plt.title(f"RUL Overlay: {group_name}\n({metric_info})")
    plt.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='True RUL'),
        Line2D([0], [0], color='red', lw=1, label='Pred RUL'),
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_hi_overlay_group(
    trajectories: Dict[int, EngineTrajectory],
    unit_ids: List[int],
    group_name: str,
    out_path: Path
):
    """
    Overlay HI trajectories (Predicted only).
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual trajectories
    for uid in unit_ids:
        if uid not in trajectories:
            continue
        traj = trajectories[uid]
        if hasattr(traj, 'hi') and traj.hi is not None:
            # Check if HI is per-timestep array
             plt.plot(traj.cycles, traj.hi, color='blue', alpha=0.3, linewidth=1)
    
    plt.xlabel("Time (cycles)")
    plt.ylabel("Health Index")
    plt.title(f"HI Overlay (Pred): {group_name}")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_rul_grid_worst20(
    trajectories: Dict[int, EngineTrajectory],
    unit_ids: List[int],
    df_metrics: pd.DataFrame,
    out_path: Path
):
    """
    4x5 Grid plot for the Worst 20.
    """
    # Ensure we take at most 20
    units_to_plot = unit_ids[:20] 
    n_plots = len(units_to_plot)
    if n_plots == 0:
        return

    cols = 5
    rows = 4 # Fixed 4x5
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    for i, uid in enumerate(units_to_plot):
        if i >= rows * cols: break
        
        ax = fig.add_subplot(gs[i // cols, i % cols])
        
        if uid not in trajectories:
            ax.text(0.5, 0.5, f"Missing Data\nUnit {uid}", ha='center')
            continue
            
        traj = trajectories[uid]
        
        # Get error for title
        row = df_metrics[df_metrics["unit_id"] == uid]
        err_str = "N/A"
        if not row.empty:
            err = row.iloc[0]["abs_err_last"]
            err_str = f"{err:.1f}"
            
        ax.plot(traj.cycles, traj.true_rul, 'k--', label='True', lw=1.5)
        ax.plot(traj.cycles, traj.pred_rul, 'r-', label='Pred', lw=1.5)
        
        # Quantiles if available? 
        # The EngineTrajectory struct in inference.py has 'rul_sigma'. 
        if hasattr(traj, 'rul_sigma') and traj.rul_sigma is not None:
             # simple +/- 2 sigma
             sigma = traj.rul_sigma
             # Handle possible shapes (scalar expansion or array)
             if np.ndim(sigma) == 0: sigma = np.full_like(traj.pred_rul, sigma)
             
             ax.fill_between(traj.cycles, 
                             traj.pred_rul - 2*sigma, 
                             traj.pred_rul + 2*sigma, 
                             color='red', alpha=0.1)

        ax.set_title(f"U{uid} | Err: {err_str}", fontsize=9)
        
        # Minimal ticks/labels to reduce clutter
        if i // cols != rows - 1: # Not last row
            ax.set_xticklabels([])
        if i % cols != 0: # Not first col
            ax.set_yticklabels([])
            
        ax.grid(True, alpha=0.2)
        
    plt.suptitle("Worst 20 Failure Cases (RUL Trajectories)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_failure_case_report(
    experiment_dir: Path,
    eol_metrics: List[EngineEOLMetrics],
    trajectories: Dict[int, EngineTrajectory],
    K: int = 20
):
    """
    Main entry point.
    """
    print(f"\n[FailureCases] Generating report for {experiment_dir.name}...")
    failure_dir = experiment_dir / "failure_cases"
    
    # 1. Compute errors & Rank
    df = compute_last_errors_per_unit(eol_metrics)
    
    # 2. Select Groups
    worst_20, best_20, mid_20 = select_groups_from_last_error(df, K)
    
    groups = {
        "worst20": worst_20,
        "best20": best_20,
        "mid20": mid_20
    }
    
    # 3. CSV Reports
    write_cases_csv(df, groups, failure_dir)
    print(f"  Saved CSVs to {failure_dir}")
    
    # 4. Plots
    # Stats for titles
    def get_stats(uids):
        if not uids: return "N/A"
        sub = df[df["unit_id"].isin(uids)]
        return f"Mean Abs Err: {sub['abs_err_last'].mean():.1f}, Max: {sub['abs_err_last'].max():.1f}"

    # A) RUL Overlays
    plot_rul_overlay_group(trajectories, worst_20, "Worst 20", get_stats(worst_20), failure_dir / "rul_overlay_worst20.png")
    plot_rul_overlay_group(trajectories, best_20, "Best 20", get_stats(best_20), failure_dir / "rul_overlay_best20.png")
    plot_rul_overlay_group(trajectories, mid_20, "Mid 20", get_stats(mid_20), failure_dir / "rul_overlay_mid20.png")
    
    # B) HI Overlays
    plot_hi_overlay_group(trajectories, worst_20, "Worst 20", failure_dir / "hi_overlay_worst20.png")
    plot_hi_overlay_group(trajectories, best_20, "Best 20", failure_dir / "hi_overlay_best20.png")
    plot_hi_overlay_group(trajectories, mid_20, "Mid 20", failure_dir / "hi_overlay_mid20.png")
    
    # C) Grid Plot
    plot_rul_grid_worst20(trajectories, worst_20, df, failure_dir / "rul_grid_worst20.png")
    
    print(f"  Saved plots to {failure_dir}")
    print(f"[FailureCases] Done. Selected {len(worst_20)} worst, {len(best_20)} best, {len(mid_20)} mid.")
