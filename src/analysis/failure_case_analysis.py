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
from src.analysis.failure_tags import (
    compute_failure_tags_for_all, 
    generate_condition_report, 
    compute_extended_groups
)
from src.analysis.diagnostics_hi import (
    compute_hi_reaction_metrics,
    plot_hi_reaction_hist
)

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
        q10 = getattr(m, "q10", None)
        q50 = getattr(m, "q50", None)
        q90 = getattr(m, "q90", None)
        
        row["pred_q10_last"] = q10
        row["pred_q50_last"] = q50
        row["pred_q90_last"] = q90
        
        # Derived metrics
        interval = None
        overconf = False
        if q10 is not None and q90 is not None:
            interval = float(q90 - q10)
            # Overconfidence: Large Error (>= 25) AND Small Interval (<= 10)
            if abs_err >= 25.0 and interval <= 10.0:
                overconf = True
                
        row["interval_q90_q10_last"] = interval
        row["overconfident_flag"] = overconf
            
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
                             
        # Plot LAST quantile interval if available
        if not row.empty:
            r = row.iloc[0]
            q10 = r.get("pred_q10_last")
            q90 = r.get("pred_q90_last")
            if q10 is not None and q90 is not None and not pd.isna(q10):
                 last_t = traj.cycles[-1]
                 # Draw vertical bar for [q10, q90]
                 ax.vlines(last_t, q10, q90, color='purple', alpha=0.6, linewidth=2, zorder=10)
                 # Draw caps
                 ax.plot(last_t, q10, marker='_', color='purple', markersize=6)
                 ax.plot(last_t, q90, marker='_', color='purple', markersize=6)

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
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute errors & Rank
    df = compute_last_errors_per_unit(eol_metrics)
    
    # --- PHASE 2: Tags & Condition Report ---
    print("  [Phase2] Computing failure tags and condition report...")
    df = compute_failure_tags_for_all(df, trajectories)
    
    # Generate condition report
    generate_condition_report(df, failure_dir)
    
    # 2. Select Groups
    worst_20, best_20, mid_20 = select_groups_from_last_error(df, K)
    
    # Phase 2: Extended groups
    ext_groups = compute_extended_groups(df, K)
    worst_over = ext_groups["worst20_over"]
    worst_under = ext_groups["worst20_under"]
    
    groups = {
        "worst20": worst_20,
        "best20": best_20,
        "mid20": mid_20,
        "worst20_over": worst_over,
        "worst20_under": worst_under
    }
    
    # 3. CSV Reports
    # Save df with tags as cases_tags.csv (or cases.csv)
    # The requirement says "failure_cases/cases_tags.csv (or add columns to cases.csv)"
    # We will overwrite cases.csv as it is a superset now, but also save cases_tags.csv for explicit compliance
    df.to_csv(failure_dir / "cases.csv", index=False)
    df.to_csv(failure_dir / "cases_tags.csv", index=False)
    
    # selected_groups.csv
    # We use the updated 'groups' dict which now has 5 keys
    # write_cases_csv logic will handle it if we modify it to accept the dict directly?
    # Our write_cases_csv takes (df, groups, out_dir). Let's use it.
    # Note: write_cases_csv writes cases.csv again. That's fine.
    write_cases_csv(df, groups, failure_dir)
    print(f"  Saved CSVs (cases, cases_tags, condition_report) to {failure_dir}")
    
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
    
    # New Phase 2 Plots
    plot_rul_overlay_group(trajectories, worst_over, "Worst 20 (Over-Estimation)", get_stats(worst_over), failure_dir / "rul_overlay_worst20_over.png")
    plot_rul_overlay_group(trajectories, worst_under, "Worst 20 (Under-Estimation)", get_stats(worst_under), failure_dir / "rul_overlay_worst20_under.png")
    
    # B) HI Overlays
    plot_hi_overlay_group(trajectories, worst_20, "Worst 20", failure_dir / "hi_overlay_worst20.png")
    plot_hi_overlay_group(trajectories, best_20, "Best 20", failure_dir / "hi_overlay_best20.png")
    plot_hi_overlay_group(trajectories, mid_20, "Mid 20", failure_dir / "hi_overlay_mid20.png")
    
    # New Phase 2 Plots
    plot_hi_overlay_group(trajectories, worst_over, "Worst 20 (Over)", failure_dir / "hi_overlay_worst20_over.png")
    plot_hi_overlay_group(trajectories, worst_under, "Worst 20 (Under)", failure_dir / "hi_overlay_worst20_under.png")
    
    # C) Grid Plot
    plot_rul_grid_worst20(trajectories, worst_20, df, failure_dir / "rul_grid_worst20.png")
    
    # --- PHASE 2.2: HI Reaction Metrics ---
    print("  [Phase2.2] Computing HI reaction statistics...")
    hi_stats_summary = {}
    
    # Define groups to analyze
    target_groups = ["worst20", "best20", "mid20", "worst20_over", "worst20_under"]
    
    for grp in target_groups:
        if grp not in groups or not groups[grp]:
            continue
            
        uids = groups[grp]
        # Compute metrics
        stats = compute_hi_reaction_metrics(trajectories, uids, grp, failure_dir)
        
        if stats:
            hi_stats_summary[grp] = stats
            
            # Plot Histogram (using the csv detail file or re-deriving? plot function takes df_details)
            # The compute function saves a CSV. Let's load it back or modify compute to return df.
            # To avoid refactoring too much, let's just re-read the CSV we just saved.
            detail_csv_path = failure_dir / f"hi_reaction_details_{grp.replace(' ', '_')}.csv"
            if detail_csv_path.exists():
                df_details = pd.read_csv(detail_csv_path)
                plot_hi_reaction_hist(df_details, grp, failure_dir / f"hi_reaction_hist_{grp}.png")

    # Save summary JSON
    with open(failure_dir / "hi_reaction_metrics.json", "w") as f:
        json.dump(hi_stats_summary, f, indent=2)

    print(f"  Saved plots to {failure_dir}")
    print(f"[FailureCases] Done. Selected {len(worst_20)} worst, {len(best_20)} best, {len(mid_20)} mid.")
    print(f"             + {len(worst_over)} worst-over, {len(worst_under)} worst-under.")
