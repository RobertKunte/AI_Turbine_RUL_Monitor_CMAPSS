"""
Evaluation utilities for EOL prediction models.
Includes literature benchmarks, comparison functions, and physics sanity check plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # Removed to allow interactive backends in notebooks

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ===================================================================
# Health Index Post-Processing
# ===================================================================

def enforce_monotone_hi(
    hi: np.ndarray,
    direction: str = "decreasing",
) -> np.ndarray:
    """
    Enforce monotonic HI over time using a running min/max.
    
    This function applies a monotone projection to ensure HI follows
    the expected physical behavior (decreasing towards EOL).
    
    Args:
        hi: (T,) health index over time (earlier -> healthy, later -> degraded).
            Can also be (batch, T) or (batch, T, 1) - will be processed per sample.
        direction: "decreasing" (HI decreases towards EOL) or "increasing" (HI increases)
    
    Returns:
        hi_mono: Same shape as input, with monotonicity enforced.
                 For "decreasing": hi_mono[t] = min_{k>=t} hi[k] (no rises towards EOL)
                 For "increasing": hi_mono[t] = max_{k>=t} hi[k]
    """
    hi = np.asarray(hi, dtype=float)
    original_shape = hi.shape
    
    # Handle multi-dimensional inputs
    if hi.ndim == 3:
        # (batch, T, 1) -> (batch, T)
        hi = hi.squeeze(-1)
        squeeze_last = True
    else:
        squeeze_last = False
    
    if hi.ndim == 2:
        # (batch, T) - process each sample independently
        result = np.array([enforce_monotone_hi(hi[i], direction) for i in range(hi.shape[0])])
        if squeeze_last:
            result = result[..., np.newaxis]
        return result.reshape(original_shape)
    elif hi.ndim != 1:
        raise ValueError(f"hi must be 1D, 2D, or 3D, got shape {original_shape}")
    
    # 1D case: process single sequence
    if direction == "decreasing":
        # Reverse, apply running minimum, reverse back
        # This ensures: hi_mono[t] = min_{k>=t} hi[k]
        rev = hi[::-1]
        rev_mono = np.minimum.accumulate(rev)
        result = rev_mono[::-1]
    elif direction == "increasing":
        # Reverse, apply running maximum, reverse back
        rev = hi[::-1]
        rev_mono = np.maximum.accumulate(rev)
        result = rev_mono[::-1]
    else:
        raise ValueError(f"Unknown direction: {direction}, must be 'decreasing' or 'increasing'")
    
    # Reshape to match original if needed
    if squeeze_last:
        result = result[..., np.newaxis]
    
    return result.reshape(original_shape)


def forward_rul_only(model: nn.Module, X: torch.Tensor, cond_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Helper function to extract RUL predictions from models that may return
    either a single tensor (single-task) or a tuple (multi-task: RUL, Health, or RUL, Health, HealthSeq).
    
    This ensures backward compatibility with evaluation functions that expect
    model(X) to return only RUL predictions.
    
    Args:
        model: EOLFullLSTM or EOLFullLSTMWithHealth model
        X: Input tensor [B, T, F]
        cond_ids: Optional condition IDs [B] (required if model uses condition embeddings)
    
    Returns:
        rul_pred: RUL predictions [B]
    """
    # Check if model uses condition embeddings
    use_cond_emb = getattr(model, 'use_condition_embedding', False)
    
    # Also check for WorldModelUniversalV3 which uses num_conditions attribute
    num_conditions = getattr(model, 'num_conditions', None)
    if num_conditions is not None and num_conditions > 1:
        use_cond_emb = True
    
    # For UniversalEncoderV2, cond_ids are always required if num_conditions > 1
    # But checking specific model type is brittle; rely on the flag + argument presence
    
    if use_cond_emb:
        if cond_ids is None:
            # Try to use X without cond_ids if the model allows it (e.g. default value in forward)
            # But most models raise ValueError. We'll try passing it if provided.
             raise ValueError("cond_ids required when model uses condition embeddings")
        out = model(X, cond_ids=cond_ids)
    else:
        # Try passing cond_ids anyway if provided, in case the flag was missing but model needs it
        if cond_ids is not None:
            try:
                out = model(X, cond_ids=cond_ids)
            except TypeError:
                # Fallback if model doesn't accept cond_ids
                out = model(X)
        else:
            out = model(X)
    
    if isinstance(out, (tuple, list)):
        # Default: first element is point prediction (mu).
        # For quantile runs, prefer q50 as point prediction to match uncertainty-aware reporting.
        rul_pred = out[0]
        try:
            # Non-aux contract: (rul_pred, hi_last, hi_seq, rul_sigma, rul_quantiles, ...)
            if len(out) >= 5 and out[4] is not None and torch.is_tensor(out[4]) and out[4].dim() == 2:
                q = out[4]  # [B,Q]
                qs = getattr(model, "rul_quantiles", (0.1, 0.5, 0.9))
                qs_t = torch.tensor(list(qs), device=q.device, dtype=q.dtype)
                idx50 = int(torch.argmin(torch.abs(qs_t - 0.5)).item())
                rul_pred = q[:, idx50]
            # Aux contract: (..., rul_quantiles at index 6)
            elif len(out) >= 7 and out[6] is not None and torch.is_tensor(out[6]) and out[6].dim() == 2:
                q = out[6]
                qs = getattr(model, "rul_quantiles", (0.1, 0.5, 0.9))
                qs_t = torch.tensor(list(qs), device=q.device, dtype=q.dtype)
                idx50 = int(torch.argmin(torch.abs(qs_t - 0.5)).item())
                rul_pred = q[:, idx50]
        except Exception:
            # Never break inference due to quantile selection logic.
            pass
    elif isinstance(out, dict):
        # Handle dict outputs (e.g. WorldModelUniversalV3)
        if "eol" in out:
            rul_pred = out["eol"]
        elif "rul" in out:
            rul_pred = out["rul"]
        else:
            # Fallback: maybe the first value? 
            # Better to return the dict so caller fails with clearer error, 
            # or raise ValueError here.
            raise ValueError(f"Model returned dict without 'eol' or 'rul' keys: {list(out.keys())}")
    else:
        rul_pred = out
    return rul_pred

# Literature benchmarks (approximate, from recent experiments / literature)
LITERATURE_BENCHMARKS = {
    "FD001": {
        "RMSE": 12.0,   # typical LSTM ~11–13
        "RMSE_eol": 2.5,
        "NASA_mean": 0.3,
    },
    "FD002": {
        "RMSE": 20.0,   # more difficult, multi-condition
        "RMSE_eol": 4.5,
        "NASA_mean": 0.5,
    },
    "FD003": {
        "RMSE": 12.0,
        "RMSE_eol": 3.0,
        "NASA_mean": 0.35,
    },
    "FD004": {
        "RMSE": 23.0,   # typical range; we mainly watch NASA tails
        "RMSE_eol": 9.0,
        "NASA_mean": 1.5,
    },
}

# Project baseline EOL-LSTM results (from user's baseline)
PROJECT_BASELINE_EOLLSTM = {
    "FD001": {
        "RMSE": 14.66,
        "RMSE_eol": 2.74,
        "NASA_mean": 0.30,
    },
    "FD002": {
        "RMSE": 20.20,
        "RMSE_eol": 4.59,
        "NASA_mean": 0.54,
    },
    "FD003": {
        "RMSE": 12.18,
        "RMSE_eol": 3.13,
        "NASA_mean": 0.35,
    },
    "FD004": {
        "RMSE": 26.14,
        "RMSE_eol": 9.30,
        "NASA_mean": 1.54,
    },
}


def compare_with_benchmarks(
    metrics: Dict[str, Any],
    fd_name: str,
    print_output: bool = True,
) -> Dict[str, Any]:
    """
    Compare current metrics with literature benchmarks and project baseline.
    
    Args:
        metrics: Dictionary with current metrics (RMSE, RMSE_eol, NASA_mean, etc.)
        fd_name: Dataset name (e.g., "FD001")
        print_output: Whether to print comparison to console
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "fd_name": fd_name,
        "current": {},
        "baseline": {},
        "literature": {},
        "improvement_vs_baseline": {},
        "improvement_vs_literature": {},
    }
    
    # Get baselines
    baseline = PROJECT_BASELINE_EOLLSTM.get(fd_name, {})
    literature = LITERATURE_BENCHMARKS.get(fd_name, {})
    
    # Compare key metrics
    for metric_name in ["RMSE", "RMSE_eol", "MAE", "MAE_eol", "NASA_mean"]:
        current_val = metrics.get(metric_name)
        baseline_val = baseline.get(metric_name)
        literature_val = literature.get(metric_name)
        
        if current_val is not None:
            comparison["current"][metric_name] = current_val
            
            if baseline_val is not None:
                comparison["baseline"][metric_name] = baseline_val
                improvement = baseline_val - current_val
                comparison["improvement_vs_baseline"][metric_name] = improvement
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                comparison["improvement_vs_baseline"][f"{metric_name}_pct"] = improvement_pct
            
            if literature_val is not None:
                comparison["literature"][metric_name] = literature_val
                improvement = literature_val - current_val
                comparison["improvement_vs_literature"][metric_name] = improvement
                improvement_pct = (improvement / literature_val * 100) if literature_val > 0 else 0
                comparison["improvement_vs_literature"][f"{metric_name}_pct"] = improvement_pct
    
    if print_output:
        print("=" * 60)
        print(f"[Benchmark Comparison] {fd_name}")
        print("=" * 60)
        
        # Pointwise RMSE
        if "RMSE" in comparison["current"]:
            current_rmse = comparison["current"]["RMSE"]
            baseline_rmse = comparison["baseline"].get("RMSE", "N/A")
            literature_rmse = comparison["literature"].get("RMSE", "N/A")
            
            print(f"Pointwise RMSE: {current_rmse:.2f}")
            if baseline_rmse != "N/A":
                improvement = comparison["improvement_vs_baseline"].get("RMSE", 0)
                print(f"  Baseline: {baseline_rmse:.2f} (improvement: {improvement:+.2f})")
            if literature_rmse != "N/A":
                improvement = comparison["improvement_vs_literature"].get("RMSE", 0)
                print(f"  Literature target: ~{literature_rmse:.2f} (improvement: {improvement:+.2f})")
        
        # EOL RMSE
        if "RMSE_eol" in comparison["current"]:
            current_rmse_eol = comparison["current"]["RMSE_eol"]
            baseline_rmse_eol = comparison["baseline"].get("RMSE_eol", "N/A")
            literature_rmse_eol = comparison["literature"].get("RMSE_eol", "N/A")
            
            print(f"EOL RMSE: {current_rmse_eol:.2f}")
            if baseline_rmse_eol != "N/A":
                improvement = comparison["improvement_vs_baseline"].get("RMSE_eol", 0)
                print(f"  Baseline: {baseline_rmse_eol:.2f} (improvement: {improvement:+.2f})")
            if literature_rmse_eol != "N/A":
                improvement = comparison["improvement_vs_literature"].get("RMSE_eol", 0)
                print(f"  Literature target: ~{literature_rmse_eol:.2f} (improvement: {improvement:+.2f})")
        
        # NASA mean
        if "NASA_mean" in comparison["current"]:
            current_nasa = comparison["current"]["NASA_mean"]
            baseline_nasa = comparison["baseline"].get("NASA_mean", "N/A")
            literature_nasa = comparison["literature"].get("NASA_mean", "N/A")
            
            print(f"NASA Score (mean): {current_nasa:.4f}")
            if baseline_nasa != "N/A":
                improvement = comparison["improvement_vs_baseline"].get("NASA_mean", 0)
                print(f"  Baseline: {baseline_nasa:.4f} (improvement: {improvement:+.4f})")
            if literature_nasa != "N/A":
                improvement = comparison["improvement_vs_literature"].get("NASA_mean", 0)
                print(f"  Literature target: ~{literature_nasa:.4f} (improvement: {improvement:+.4f})")
        
        print("=" * 60)
    
    return comparison


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays and other non-JSON-serializable objects to JSON-serializable types.
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


def save_evaluation_results(
    metrics: Dict[str, Any],
    physics_config: Optional[Any],
    feature_cols: List[str],
    output_dir: Path | str,
    fd_name: str,
) -> None:
    """
    Save evaluation results, config, and feature list to files.
    
    Args:
        metrics: Dictionary with evaluation metrics
        physics_config: PhysicsFeatureConfig (if available)
        feature_cols: List of feature column names
        output_dir: Directory to save results
        fd_name: Dataset name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON (convert numpy arrays to lists)
    metrics_file = output_dir / "metrics.json"
    metrics_serializable = _make_json_serializable(metrics)
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save physics config (if available)
    if physics_config is not None:
        config_file = output_dir / "physics_config.json"
        config_dict = {
            "use_core": physics_config.use_core,
            "use_extended": physics_config.use_extended,
            "use_residuals": physics_config.use_residuals,
            "use_temporal_on_physics": physics_config.use_temporal_on_physics,
            "core_features": physics_config.core_features,
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Save feature list
    features_file = output_dir / "feature_cols.txt"
    with open(features_file, 'w') as f:
        f.write(f"# Feature columns for {fd_name}\n")
        f.write(f"# Total: {len(feature_cols)}\n\n")
        for col in sorted(feature_cols):
            f.write(f"{col}\n")
    
    print(f"[save_evaluation_results] Saved results to {output_dir}")


def plot_physics_feature_time_series(
    df: pd.DataFrame,
    unit_ids: List[int],
    physics_cols: List[str],
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    output_dir: str | Path = "results",
    fd_name: str = "FD001",
) -> None:
    """
    For each requested unit_id and each physics feature, plot feature vs cycle.
    Save as PNG per feature+unit.
    
    Args:
        df: DataFrame with physics features and cycles
        unit_ids: List of unit IDs to plot
        physics_cols: List of physics feature column names
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        output_dir: Directory to save plots
        fd_name: Dataset name for file naming
    """
    output_dir = Path(output_dir) / "physics_sanity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    available_units = df[unit_col].unique()
    units_to_plot = [uid for uid in unit_ids if uid in available_units]
    
    if len(units_to_plot) == 0:
        print(f"[plot_physics_feature_time_series] No valid unit IDs found. Skipping plots.")
        return
    
    for unit_id in units_to_plot:
        df_unit = df[df[unit_col] == unit_id].sort_values(cycle_col)
        
        for col in physics_cols:
            if col not in df_unit.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_unit[cycle_col], df_unit[col], 'b-', linewidth=1.5, label=col)
            ax.set_xlabel(cycle_col)
            ax.set_ylabel(col)
            ax.set_title(f"{fd_name} - Unit {unit_id} - {col} vs {cycle_col}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            filename = f"{fd_name}_unit_{unit_id}_{col}_timeseries.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"[plot_physics_feature_time_series] Saved {len(units_to_plot) * len(physics_cols)} plots to {output_dir}")


def plot_physics_feature_vs_rul_eol(
    df_eol: pd.DataFrame,
    physics_cols: List[str],
    rul_col: str = "RUL",
    output_dir: str | Path = "results",
    fd_name: str = "FD001",
) -> None:
    """
    For EOL samples, create scatter plots:
    physical feature (x) vs true RUL (y),
    plus a simple correlation annotation.
    
    Args:
        df_eol: DataFrame with one row per engine at EOL (physics features + true RUL)
        physics_cols: List of physics feature column names
        rul_col: Name of RUL column
        output_dir: Directory to save plots
        fd_name: Dataset name for file naming
    """
    output_dir = Path(output_dir) / "physics_sanity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rul_col not in df_eol.columns:
        print(f"[plot_physics_feature_vs_rul_eol] RUL column '{rul_col}' not found. Skipping plots.")
        return
    
    for col in physics_cols:
        if col not in df_eol.columns:
            continue
        
        # Compute correlation
        correlation = df_eol[col].corr(df_eol[rul_col])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_eol[col], df_eol[rul_col], alpha=0.6, s=30)
        ax.set_xlabel(col)
        ax.set_ylabel(f"True {rul_col}")
        ax.set_title(f"{fd_name} - {col} vs {rul_col} (EOL)\nPearson correlation: {correlation:.3f}")
        ax.grid(True, alpha=0.3)
        
        filename = f"{fd_name}_{col}_vs_{rul_col}_EOL.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[plot_physics_feature_vs_rul_eol] Saved {len(physics_cols)} plots to {output_dir}")
