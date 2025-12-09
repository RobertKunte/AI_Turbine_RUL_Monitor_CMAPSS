"""
Plotting utilities for experiment analysis.

This module provides functions to visualize EOL errors, NASA scores, and
Health Index trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_eol_error_hist(
    eol_metrics: List,
    title: str,
    out_path: Path,
    bins: int = 30,
) -> None:
    """
    Plot histogram of EOL RUL errors.
    
    Args:
        eol_metrics: List of EngineEOLMetrics
        title: Plot title
        out_path: Output file path
        bins: Number of histogram bins
    """
    errors = [m.error for m in eol_metrics]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.set_xlabel('EOL RUL Error (pred - true) [cycles]')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved EOL error histogram to {out_path}")


def plot_nasa_per_engine(
    eol_metrics: List,
    title: str,
    out_path: Path,
    max_engines: Optional[int] = None,
) -> None:
    """
    Plot bar chart of NASA contributions per engine (sorted).
    
    Args:
        eol_metrics: List of EngineEOLMetrics
        title: Plot title
        out_path: Output file path
        max_engines: Maximum number of engines to show (None = all)
    """
    # Sort by NASA score (descending)
    sorted_metrics = sorted(eol_metrics, key=lambda m: m.nasa, reverse=True)
    
    if max_engines is not None:
        sorted_metrics = sorted_metrics[:max_engines]
    
    unit_ids = [m.unit_id for m in sorted_metrics]
    nasa_scores = [m.nasa for m in sorted_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(unit_ids)), nasa_scores, edgecolor='black', alpha=0.7)
    
    # Color bars: red for high NASA (bad), green for low NASA (good)
    colors = ['red' if s > np.median(nasa_scores) else 'green' for s in nasa_scores]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    ax.set_xlabel('Engine ID (sorted by NASA score)')
    ax.set_ylabel('NASA Contribution')
    ax.set_title(title)
    ax.set_xticks(range(len(unit_ids)))
    ax.set_xticklabels([f"#{uid}" for uid in unit_ids], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_nasa = np.mean(nasa_scores)
    ax.axhline(mean_nasa, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_nasa:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved NASA per engine plot to {out_path}")


def plot_hi_trajectories_for_selected_engines(
    trajectories: Dict[int, any],
    selected_unit_ids: List[int],
    title: str,
    out_path: Path,
    plot_rul: bool = True,
) -> None:
    """
    Plot HI trajectories for selected engines.
    
    Args:
        trajectories: Dict mapping unit_id -> EngineTrajectory
        selected_unit_ids: List of unit_ids to plot
        title: Plot title
        out_path: Output file path
        plot_rul: If True, also plot RUL on secondary y-axis
    """
    num_engines = len(selected_unit_ids)
    if num_engines == 0:
        print("Warning: No engines selected for trajectory plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(num_engines, 1, figsize=(10, 3 * num_engines))
    if num_engines == 1:
        axes = [axes]
    
    for idx, unit_id in enumerate(selected_unit_ids):
        if unit_id not in trajectories:
            print(f"Warning: Unit {unit_id} not found in trajectories")
            continue
        
        traj = trajectories[unit_id]
        ax = axes[idx]
        
        # Plot HI
        ax.plot(traj.cycles, traj.hi, 'b-', linewidth=2, label='Health Index')
        ax.set_ylabel('Health Index', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Plot RUL on secondary axis if requested
        if plot_rul:
            ax2 = ax.twinx()
            ax2.plot(traj.cycles, traj.true_rul, 'g--', linewidth=1.5, alpha=0.7, label='True RUL')
            ax2.plot(traj.cycles, traj.pred_rul, 'r--', linewidth=1.5, alpha=0.7, label='Pred RUL')
            ax2.set_ylabel('RUL [cycles]', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='upper right')
        
        ax.set_xlabel('Cycle Number')
        ax.set_title(f'Engine #{unit_id}')
        ax.legend(loc='upper left')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved HI trajectories plot to {out_path}")


def plot_true_vs_pred_rul(
    eol_metrics: List,
    title: str,
    out_path: Path,
) -> None:
    """
    Plot scatter plot of true RUL vs predicted RUL.
    
    Args:
        eol_metrics: List of EngineEOLMetrics
        title: Plot title
        out_path: Output file path
    """
    true_rul = [m.true_rul for m in eol_metrics]
    pred_rul = [m.pred_rul for m in eol_metrics]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_rul, pred_rul, alpha=0.6, s=50)
    
    # Diagonal line (perfect prediction)
    min_val = min(min(true_rul), min(pred_rul))
    max_val = max(max(true_rul), max(pred_rul))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True RUL [cycles]')
    ax.set_ylabel('Predicted RUL [cycles]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved true vs pred RUL scatter to {out_path}")

