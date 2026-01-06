import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.analysis.inference import EngineTrajectory

# Constants for Reaction Timing
# "Reaction" = First time HI drops below threshold OR slope drops below threshold
HI_REACT_THRESH = 0.98
SLOPE_REACT_THRESH = -0.001
SLOPE_WINDOW = 10

def compute_hi_reaction_metrics(
    trajectories: Dict[int, EngineTrajectory],
    unit_ids: List[int],
    group_name: str,
    out_dir: Path
) -> Dict[str, Any]:
    """
    Compute reaction timing statistics for a group of units.
    
    Metrics:
    - t_react: First cycle where (HI < 0.98) OR (Slope < -0.001)
    - reaction_delay: t_eol - t_react
    - reaction_delay: t_eol - t_react
    - reaction_delay_norm: (t_eol - t_react) / t_eol
    - never_reacted_rate: Fraction of units where condition never met
    - late_reaction_rate: Fraction of units where HI never drops below 0.98 (regardless of slope)
    - mean/median reaction_delay
    - mean/p90 overshoot_last: Prediction error at EOL (pred - true)
    """

    if not unit_ids:
        return {}

    rows = []
    
    for uid in unit_ids:
        if uid not in trajectories:
            continue
            
        traj = trajectories[uid]
        cycles = traj.cycles
        hi = traj.hi if hasattr(traj, "hi") and traj.hi is not None else None
        pred_rul = traj.pred_rul if hasattr(traj, "pred_rul") and traj.pred_rul is not None else None
        true_rul = traj.true_rul if hasattr(traj, "true_rul") and traj.true_rul is not None else None
        
        if hi is None:
            continue
            
        # Ensure hi is numpy
        if not isinstance(hi, np.ndarray):
            hi = np.array(hi)
            
        t_eol = cycles[-1]
        
        # 1. Threshold Crossing
        # Find first index where hi < HI_REACT_THRESH
        # Note: HI might start < 0.98 if bad initialization, but usually near 1.0.
        # We can ignore first few cycles? Let's take first index.
        cross_indices = np.where(hi < HI_REACT_THRESH)[0]
        t_react_thresh = cycles[cross_indices[0]] if len(cross_indices) > 0 else None
        
        # Late reaction: purely based on threshold
        late_reaction = (len(cross_indices) == 0)
        
        # 2. Slope Logic (optional but good for noisy HI)
        # Compute rolling slope
        if len(hi) > SLOPE_WINDOW:
            s_hi = pd.Series(hi)
            # Rolling slope? Just rolling diff for simplicity
            # diff = hi[t] - hi[t-1]. Rolling mean of diff.
            rolling_diff = s_hi.diff().rolling(window=SLOPE_WINDOW).mean()
            # Find first index where rolling diff < SLOPE_REACT_THRESH
            slope_indices = np.where(rolling_diff < SLOPE_REACT_THRESH)[0]
            t_react_slope = cycles[slope_indices[0]] if len(slope_indices) > 0 else None
        else:
            t_react_slope = None
            
        # Overall Reaction Time: min of triggered triggers
        react_times = [t for t in [t_react_thresh, t_react_slope] if t is not None]
        
        if react_times:
            t_react = min(react_times)
            reacted = True
            delay = t_eol - t_react
            delay_norm = delay / t_eol
        else:
            t_react = t_eol # Placeholder? Or None?
            reacted = False
            delay = 0
            delay = 0
            delay_norm = 0.0
            
        # Overshoot Last (pred - true at EOL)
        overshoot_last = 0.0
        if pred_rul is not None and true_rul is not None:
             # EOL is the last index
             overshoot_last = float(pred_rul[-1] - true_rul[-1])
            
        rows.append({
            "unit_id": uid,
            "t_eol": float(t_eol),
            "t_react": float(t_react) if reacted else None,
            "reacted": reacted,
            "late_reaction": late_reaction, # True if HI > 0.98 always
            "delay": float(delay),
            "delay_norm": float(delay_norm),
            "overshoot_last": float(overshoot_last)
        })
        
    if not rows:
        return {}
        
    df = pd.DataFrame(rows)
    
    # Aggregates
    n_total = len(df)
    n_reacted = df["reacted"].sum()
    never_reacted_rate = 1.0 - (n_reacted / n_total)
    
    # Late reaction rate (specific to threshold)
    late_reaction_rate = df["late_reaction"].mean()
    
    # Overshoot stats
    mean_overshoot_last = df["overshoot_last"].mean()
    p90_overshoot_last = df["overshoot_last"].quantile(0.90)
    
    mean_delay = df[df["reacted"]]["delay"].mean() if n_reacted > 0 else 0.0
    median_delay = df[df["reacted"]]["delay"].median() if n_reacted > 0 else 0.0
    
    mean_delay_norm = df[df["reacted"]]["delay_norm"].mean() if n_reacted > 0 else 0.0
    
    stats = {
        "group": group_name,
        "n_units": n_total,
        "n_reacted": int(n_reacted),
        "n_reacted": int(n_reacted),
        "never_reacted_rate": float(never_reacted_rate),
        "late_reaction_rate": float(late_reaction_rate),
        "mean_delay": float(mean_delay),
        "median_delay": float(median_delay),
        "mean_delay_norm": float(mean_delay_norm),
        "mean_overshoot_last": float(mean_overshoot_last),
        "p90_overshoot_last": float(p90_overshoot_last)
    }
    
    # Print summary to console for P1 visibility
    print(f"  [{group_name}] never_reacted={never_reacted_rate:.2%}, late_reaction={late_reaction_rate:.2%}, "
          f"overshoot_last(mean/p90)={mean_overshoot_last:.1f}/{p90_overshoot_last:.1f}")
    
    # Save CSV details
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"hi_reaction_details_{group_name.replace(' ', '_')}.csv", index=False)
    
    return stats

def plot_hi_reaction_hist(
    df_details: pd.DataFrame,
    group_name: str,
    out_path: Path
):
    """Plot histogram of reaction delays."""
    if df_details.empty or df_details["reacted"].sum() == 0:
        return
        
    reacted = df_details[df_details["reacted"]]
    
    plt.figure(figsize=(6, 4))
    plt.hist(reacted["delay"], bins=15, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel("Reaction Delay (cycles before EOL)")
    plt.ylabel("Count")
    plt.title(f"HI Reaction Delay: {group_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
