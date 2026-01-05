import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from src.data.windowing import WindowConfig, TargetConfig, build_test_windows_last

@dataclass
class QuantileCalibrationResult:
    quantiles: List[float]
    global_metrics: Dict[str, float]
    per_condition: Dict[int, Dict[str, float]]
    df_table: pd.DataFrame
    
    # Raw arrays for further analysis/plotting
    unit_ids: np.ndarray
    y_true: np.ndarray
    y_pred_q10: np.ndarray
    y_pred_q50: np.ndarray
    y_pred_q90: np.ndarray
    
    # Derived per-unit metrics
    interval_width: np.ndarray
    abs_err_q50: np.ndarray
    is_overconfident: np.ndarray

def compute_quantile_calibration_last(
    model: torch.nn.Module,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    scaler_dict: Dict[int, Any],
    quantiles: List[float] = [0.1, 0.5, 0.9],
    past_len: int = 30,
    max_rul: int = 125,
    device: torch.device = None,
    batch_size: int = 256,
) -> Optional[QuantileCalibrationResult]:
    """
    Compute quantile calibration metrics for the LAST prediction of each unit.
    
    Args:
        model: Trained WorldModelUniversalV3 (must support quantiles)
        df_test: Test DataFrame
        y_test_true: True RUL at EOL
        feature_cols: Feature column names
        scaler_dict: Dictionary of condition-wise scalers
        quantiles: Expected [q_low, q_med, q_high], default [0.1, 0.5, 0.9]
        past_len: Past window length
        max_rul: Maximum RUL value
        device: PyTorch device
        batch_size: Batch size for inference
        
    Returns:
        QuantileCalibrationResult object or None if model doesn't output quantiles.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    
    # 1. Prepare Data (LAST window per unit)
    # Using defaults matching evaluate_world_model_v3_eol
    window_cfg = WindowConfig(past_len=int(past_len), horizon=1, pad_mode="clamp")
    target_cfg = TargetConfig(max_rul=int(max_rul), cap_targets=True)
    
    built = build_test_windows_last(
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=window_cfg,
        target_cfg=target_cfg,
    )
    
    X = built["X"]  # (N, P, F)
    y_true = built["y_true"]  # (N,)
    unit_ids = built["unit_ids"]
    cond_ids = built["cond_ids"]
    
    # Condition-wise scaling
    X_scaled = np.empty_like(X, dtype=np.float32)
    unique_conds = np.unique(cond_ids)
    
    for cond in unique_conds:
        cond = int(cond)
        idxs = np.where(cond_ids == cond)[0]
        scaler = scaler_dict.get(cond, scaler_dict.get(0)) # Fallback to 0
        if scaler is None:
             # Fallback if 0 also not found (rare)
             scaler = list(scaler_dict.values())[0]

        flat = X[idxs].reshape(-1, len(feature_cols))
        # Ensure scaler is fitted
        X_scaled[idxs] = scaler.transform(flat).reshape(-1, int(past_len), len(feature_cols)).astype(np.float32)

    # 2. Inference
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    cond_t = torch.tensor(cond_ids, dtype=torch.long).to(device)
    
    y_pred_q_all = []
    
    # Helper for batching
    dataset = torch.utils.data.TensorDataset(X_t, cond_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for bx, bcond in loader:
            outputs = model(
                encoder_inputs=bx,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
                cond_ids=bcond,
            )
            
            # Check for quantiles
            if isinstance(outputs, dict) and "rul_q_last" in outputs:
                # (B, 3) -> [q10, q50, q90]
                q_out = outputs["rul_q_last"].cpu().numpy()
                y_pred_q_all.append(q_out)
            else:
                # Model does not support quantiles
                return None
                
    if not y_pred_q_all:
        return None
        
    y_pred_q = np.concatenate(y_pred_q_all, axis=0) # (N, 3)
    
    # Clip predictions
    y_pred_q = np.clip(y_pred_q, 0.0, float(max_rul))
    
    # Map to specific quantiles
    # Assuming model outputs exactly [q10, q50, q90] corresponding to input config
    # We should verify quantiles match, or just assume first/middle/last if 3.
    # The assignment text says "quantiles": [0.1, 0.5, 0.9]
    
    q10 = y_pred_q[:, 0]
    q50 = y_pred_q[:, 1]
    q90 = y_pred_q[:, 2]
    
    # 3. Compute Metrics
    metrics_res = _compute_metrics_from_arrays(
        y_true, q10, q50, q90, unit_ids, cond_ids, unique_conds
    )
    
    return QuantileCalibrationResult(
        quantiles=quantiles,
        global_metrics=metrics_res["global"],
        per_condition=metrics_res["per_condition"],
        df_table=metrics_res["df_table"],
        unit_ids=unit_ids,
        y_true=y_true,
        y_pred_q10=q10,
        y_pred_q50=q50,
        y_pred_q90=q90,
        interval_width=metrics_res["interval_width"],
        abs_err_q50=metrics_res["abs_err_q50"],
        is_overconfident=metrics_res["is_overconfident"]
    )

def _compute_metrics_from_arrays(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    unit_ids: np.ndarray,
    cond_ids: np.ndarray,
    unique_conds: np.ndarray
) -> Dict[str, Any]:
    
    abs_err = np.abs(q50 - y_true)
    interval = q90 - q10
    
    # Coverage: y_true <= q
    cov_10 = (y_true <= q10).mean()
    cov_50 = (y_true <= q50).mean()
    cov_90 = (y_true <= q90).mean()
    
    mean_int = interval.mean()
    mean_abs_err = abs_err.mean()
    
    # Overconfidence: Large Error (>= 25) AND Small Interval (<= 10)
    # Thresholds from requirements
    overconf_mask = (abs_err >= 25.0) & (interval <= 10.0)
    frac_overconf = overconf_mask.mean()
    
    global_metrics = {
        "n_units": int(len(y_true)),
        "coverage_q10": float(cov_10),
        "coverage_q50": float(cov_50),
        "coverage_q90": float(cov_90),
        "mean_interval_q90_q10": float(mean_int),
        "mean_abs_err_last_q50": float(mean_abs_err),
        "frac_overconfident": float(frac_overconf)
    }
    
    # Per-condition
    per_cond = {}
    rows = []
    
    # Add global row first
    g_row = {"condition_id": "global"}
    g_row.update(global_metrics)
    rows.append(g_row)
    
    for c in unique_conds:
        c = int(c)
        mask = (cond_ids == c)
        if not mask.any():
            continue
            
        n_c = mask.sum()
        cov_10_c = (y_true[mask] <= q10[mask]).mean()
        cov_50_c = (y_true[mask] <= q50[mask]).mean()
        cov_90_c = (y_true[mask] <= q90[mask]).mean()
        mean_int_c = interval[mask].mean()
        mean_abs_err_c = abs_err[mask].mean()
        frac_over_c = overconf_mask[mask].mean()
        
        m = {
            "n_units": int(n_c),
            "coverage_q10": float(cov_10_c),
            "coverage_q50": float(cov_50_c),
            "coverage_q90": float(cov_90_c),
            "mean_interval_q90_q10": float(mean_int_c),
            "mean_abs_err_last_q50": float(mean_abs_err_c),
            "frac_overconfident": float(frac_over_c)
        }
        per_cond[c] = m
        
        row = {"condition_id": c}
        row.update(m)
        rows.append(row)
        
    df_table = pd.DataFrame(rows)
    # Reorder columns
    cols = ["condition_id", "n_units", "coverage_q10", "coverage_q50", "coverage_q90", 
            "mean_interval_q90_q10", "mean_abs_err_last_q50", "frac_overconfident"]
    df_table = df_table[cols]
    
    return {
        "global": global_metrics,
        "per_condition": per_cond,
        "df_table": df_table,
        "interval_width": interval,
        "abs_err_q50": abs_err,
        "is_overconfident": overconf_mask
    }

def write_quantile_calibration_artifacts(
    out_dir: Path,
    result: QuantileCalibrationResult
) -> None:
    """Writes report JSON and CSV artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON
    report = {
        "quantiles": result.quantiles,
        "global": result.global_metrics,
        "per_condition": result.per_condition
    }
    with open(out_dir / "quantile_calibration_last.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # 2. CSV
    result.df_table.to_csv(out_dir / "quantile_calibration_last.csv", index=False)
    
    print(f"[Quantile] Saved calibration reports to {out_dir}")

def plot_quantile_reliability(
    out_path: Path,
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    title: str = "Quantile Reliability (Last)"
) -> None:
    """
    Generates a simple reliability plot: Nominal vs Empirical Coverage.
    We only have 3 points: 0.1, 0.5, 0.9.
    """
    nominal = [0.1, 0.5, 0.9]
    empirical = [
        (y_true <= q10).mean(),
        (y_true <= q50).mean(),
        (y_true <= q90).mean()
    ]
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.plot(nominal, empirical, "o-", markersize=8, linewidth=2, label="Model")
    
    # Add text labels
    for nom, emp in zip(nominal, empirical):
        plt.text(nom, emp + 0.02, f"{emp:.2f}", ha="center", va="bottom", fontsize=9)
        
    plt.xlabel("Nominal Quantile Level")
    plt.ylabel("Empirical Coverage")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Quantile] Saved reliability plot to {out_path}")
