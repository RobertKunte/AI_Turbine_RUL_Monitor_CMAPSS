import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

# Constants for Tagging
DEFAULT_MAX_RUL = 125.0
PLATEAU_FACTOR = 0.95
LATE_PORTION = 0.40
EARLY_PORTION = 0.30
STEP_JUMP_THRESH = 15.0
ERROR_LARGE_THRESH = 25.0
INTERVAL_SMALL_THRESH = 10.0

def compute_failure_tags_for_all(
    cases_df: pd.DataFrame,
    trajectories: Dict[int, Any],  # Dict[int, EngineTrajectory]
    max_rul: float = DEFAULT_MAX_RUL
) -> pd.DataFrame:
    """
    Apply failure mode tags to each unit in cases_df.
    Returns a copy of cases_df with a new 'tags' column.
    
    Tags:
    - LATE_DETECTION: Pred stays > 95% max_rul for > 40% of time, and final err is positive large.
    - EARLY_COLLAPSE: Pred drops < 70% max_rul in first 30% of time, and final err is negative.
    - STEP_ARTIFACT: Max single-step jump > 15 cycles.
    - OVERCONFIDENT: (abs_err >= 25) & (interval <= 10). (From Phase 1)
    - REGIME_SENSITIVE: (Assigned locally or later? We can assign here based on condition stats if we pass them, 
                        or user can merge later. Let's do basic tags here.)
    """
    df = cases_df.copy()
    all_tags = []
    
    # Pre-calculate condition means for REGIME_SENSITIVE check?
    # Simplest: do regime sensitive in report step, or passed in. 
    # Let's focus on shape tags here.
    
    for idx, row in df.iterrows():
        uid = int(row["unit_id"])
        tags = []
        
        # 1. Existing Overconfidence (if passed from Phase 1 logic in row)
        # Using Phase 1 definition:
        is_overconf = False
        if "overconfident_flag" in row and row["overconfident_flag"]:
             tags.append("OVERCONFIDENT")
             is_overconf = True
        
        # 2. Trajectory Shape Tags
        if uid in trajectories:
            traj = trajectories[uid]
            # normalized time [0, 1]
            if len(traj.cycles) > 5: # Need minimal length
                y_pred = traj.pred_rul
                max_cycle = traj.cycles[-1]
                t_ratio = traj.cycles / max_cycle
                
                # --- LATE_DETECTION ---
                # Pred stays high (near saturation) for a long time
                # Check fraction of points where pred > 0.95 * MAX
                n_saturated = np.sum(y_pred >= (max_rul * PLATEAU_FACTOR))
                frac_saturated = n_saturated / len(y_pred)
                
                # Condition: High saturation duration AND ends with positive error (overestimation)
                # AND final error is large
                signed_err = row.get("signed_err_last", 0.0)
                abs_err = row.get("abs_err_last", 0.0)
                
                if (frac_saturated >= LATE_PORTION and 
                    signed_err > 0 and 
                    abs_err >= ERROR_LARGE_THRESH):
                    tags.append("LATE_DETECTION")
                    
                # --- EARLY_COLLAPSE ---
                # Pred drops significantly very early
                # Look at first EARLY_PORTION of life
                early_mask = t_ratio <= EARLY_PORTION
                if early_mask.any():
                    early_preds = y_pred[early_mask]
                    # If any early prediction is surprisingly low (e.g. < 70% max)
                    # AND final error is negative (underestimation)
                    # (Usually early collapse implies we underestimated RUL throughout)
                    if (np.min(early_preds) < (0.7 * max_rul) and 
                        signed_err < 0 and
                        abs_err >= ERROR_LARGE_THRESH):
                        tags.append("EARLY_COLLAPSE")
                        
                # --- STEP_ARTIFACT ---
                # Large jumps
                diffs = np.abs(np.diff(y_pred))
                if np.max(diffs) > STEP_JUMP_THRESH:
                    tags.append("STEP_ARTIFACT")
                    
        else:
            # No trajectory, can only tag overconfident based on columns
            # But we handled OVERCONFIDENT above
            pass
            
        all_tags.append(";".join(tags))
        
    df["tags"] = all_tags
    return df

def generate_condition_report(
    cases_df: pd.DataFrame,
    out_dir: Path
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Generate aggregate metrics per Condition ID.
    """
    if "condition_id" not in cases_df.columns:
        # If missing, try to infer or return dummy
        print("[Tags] Warning: condition_id not in cases_df. Skipping condition report.")
        return {}, pd.DataFrame()
        
    # Group by condition
    grouped = cases_df.groupby("condition_id")
    
    rows = []
    global_mae = cases_df["abs_err_last"].mean()
    
    for cond_id, gdf in grouped:
        n = len(gdf)
        mae = gdf["abs_err_last"].mean()
        mse = (gdf["abs_err_last"]**2).mean()
        rmse = np.sqrt(mse)
        mean_signed = gdf["signed_err_last"].mean()
        
        # Frac in worst 20? 
        # We need to know who is in worst 20 globally.
        # But worst 20 is a property of the whole set, let's assume cases_df is WHOLE set.
        # Find global ranking again?
        # Or just use rank?
        # Let's count how many have 'abs_err_last' in the top 20 of the whole/global df.
        top20_thresh = cases_df["abs_err_last"].nlargest(20).min()
        n_worst20 = (gdf["abs_err_last"] >= top20_thresh).sum()
        frac_worst20 = n_worst20 / n
        
        frac_overconf = 0.0
        if "overconfident_flag" in gdf.columns:
            frac_overconf = gdf["overconfident_flag"].mean()
            
        rows.append({
            "condition_id": int(cond_id),
            "n_units": int(n),
            "mean_abs_err_last": float(mae),
            "mean_signed_err_last": float(mean_signed),
            "rmse_last": float(rmse),
            "frac_in_worst20": float(frac_worst20),
            "frac_overconfident": float(frac_overconf)
        })
        
    report_df = pd.DataFrame(rows)
    # Sort by MAE descent
    report_df = report_df.sort_values("mean_abs_err_last", ascending=False)
    
    # Flag REGIME_SENSITIVE for conditions with high error
    # Heuristic: Condition MAE > 1.25 * Global MAE
    regime_sensitive_conds = set(report_df[report_df["mean_abs_err_last"] > 1.25 * global_mae]["condition_id"])
    
    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_dir / "condition_report.csv", index=False)
    
    summary_dict = {
        "global_mae": float(global_mae),
        "regime_sensitive_conditions": [int(c) for c in regime_sensitive_conds],
        "details": report_df.to_dict(orient="records")
    }
    with open(out_dir / "condition_report.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
        
    return summary_dict, report_df

def compute_extended_groups(
    cases_df: pd.DataFrame,
    K: int = 20
) -> Dict[str, List[int]]:
    """
    Select sub-groups: Worst20 Over, Worst20 Under.
    """
    # Worst20 Over: Largest POSITIVE signed error
    # Filter > 0, sort desc
    over_df = cases_df[cases_df["signed_err_last"] > 0].sort_values("signed_err_last", ascending=False)
    worst_over = over_df["unit_id"].head(K).tolist()
    
    # Worst20 Under: Largest NEGATIVE signed error (most negative)
    # Filter < 0, sort asc (e.g. -50 < -10)
    under_df = cases_df[cases_df["signed_err_last"] < 0].sort_values("signed_err_last", ascending=True)
    worst_under = under_df["unit_id"].head(K).tolist()
    
    return {
        "worst20_over": worst_over,
        "worst20_under": worst_under
    }
