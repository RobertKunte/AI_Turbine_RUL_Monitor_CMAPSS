"""
Comprehensive diagnostics for Phase 3.2 experiments.

Generates all required plots and summaries:
- Training curves (MSE train+val, RMSE val, LR)
- RUL trajectories for selected engines
- Health Index trajectories
- EOL error histograms
- Per-engine NASA contribution bar chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # Removed to allow interactive backends in notebooks

from src.analysis.inference import (
    run_inference_for_experiment,
    EngineEOLMetrics,
    EngineTrajectory,
)
from src.analysis.plots import (
    plot_eol_error_hist,
    plot_nasa_per_engine,
    plot_hi_trajectories_for_selected_engines,
)
from src.metrics import compute_eol_errors_and_nasa
from src.world_model_training_v3 import evaluate_world_model_v3_eol

# Additional imports for FD004 diagnostics
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import subprocess
from datetime import datetime, timezone
from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
    build_condition_features,
    create_twin_features,
    group_feature_columns,
)
from src.feature_safety import remove_rul_leakage, check_feature_dimensions
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
    evaluate_on_test_data,
)
from src.models.universal_encoder_v1 import UniversalEncoderV2, RULHIUniversalModelV2


def select_engines_for_trajectories(
    eol_metrics: List[EngineEOLMetrics],
    trajectories: Dict[int, EngineTrajectory],
    num_early: int = 3,
    num_late: int = 3,
    num_random: int = 4,
) -> List[int]:
    """
    Select engines for trajectory plotting:
    - num_early: engines with shortest cycles (early failure)
    - num_late: engines with longest cycles (late failure)
    - num_random: randomly sampled mid-range engines
    
    Args:
        eol_metrics: List of EOL metrics per engine
        trajectories: Dict mapping unit_id to EngineTrajectory
        num_early: Number of early-failure engines
        num_late: Number of late-failure engines
        num_random: Number of random mid-range engines
    
    Returns:
        List of selected unit_ids
    """
    # Sort by true RUL (ascending = early failure, descending = late failure)
    sorted_by_rul = sorted(eol_metrics, key=lambda m: m.true_rul)
    
    # Early failure: lowest true RUL
    early_ids = [m.unit_id for m in sorted_by_rul[:num_early]]
    
    # Late failure: highest true RUL
    late_ids = [m.unit_id for m in sorted_by_rul[-num_late:]]
    
    # Mid-range: engines in the middle, randomly sample
    mid_start = len(sorted_by_rul) // 4
    mid_end = 3 * len(sorted_by_rul) // 4
    mid_range = sorted_by_rul[mid_start:mid_end]
    if len(mid_range) > num_random:
        mid_ids = [m.unit_id for m in np.random.choice(mid_range, num_random, replace=False)]
    else:
        mid_ids = [m.unit_id for m in mid_range]
    
    selected = early_ids + late_ids + mid_ids
    return selected


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_rmses: List[float],
    learning_rates: Optional[List[float]] = None,
    title: str = "Training Curves",
    out_path: Path = None,
) -> None:
    """
    Plot training curves: MSE train+val, RMSE val, LR over time.
    
    Args:
        train_losses: Training MSE losses per epoch
        val_losses: Validation MSE losses per epoch
        val_rmses: Validation RMSE per epoch
        learning_rates: Learning rates per epoch (optional)
        title: Plot title
        out_path: Output file path
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label='Train MSE', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val MSE', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: RMSE and LR
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # RMSE on left axis
    ax2.plot(epochs, val_rmses, label='Val RMSE', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE [cycles]', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # LR on right axis (if provided)
    if learning_rates is not None:
        ax2_twin.plot(epochs, learning_rates, label='Learning Rate', color='orange', linewidth=1, linestyle='--')
        ax2_twin.set_ylabel('Learning Rate', color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2_twin.set_yscale('log')
    
    ax2.set_title(f'{title} - RMSE & Learning Rate')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {out_path}")


def plot_rul_trajectories(
    trajectories: Dict[int, EngineTrajectory],
    selected_unit_ids: List[int],
    title: str = "RUL Trajectories",
    out_path: Path = None,
    highlight_last_n: int = 50,
) -> None:
    """
    Plot RUL trajectories for selected engines.
    
    Note: The red line (Pred RUL) is the EOL prediction, shown as a constant
    horizontal line over the engine's lifetime. This represents the model's
    single RUL prediction at the last test window (EOL point).
    
    Args:
        trajectories: Dict mapping unit_id to EngineTrajectory
        selected_unit_ids: List of unit_ids to plot
        title: Plot title
        out_path: Output file path
        highlight_last_n: Number of last cycles to highlight
    """
    num_engines = len(selected_unit_ids)
    n_cols = 3
    n_rows = (num_engines + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if num_engines == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, unit_id in enumerate(selected_unit_ids):
        if unit_id not in trajectories:
            continue
        
        traj = trajectories[unit_id]
        ax = axes[idx]
        
        cycles = traj.cycles
        true_rul = traj.true_rul
        pred_rul = traj.pred_rul
        
        # Plot trajectories
        # True RUL: linear decrease from max_rul to 0 (reconstructed for visualization)
        ax.plot(cycles, true_rul, label='True RUL', color='blue', linewidth=2, alpha=0.7)
        # Pred RUL: constant EOL prediction (single value per engine, shown as horizontal line)
        ax.plot(cycles, pred_rul, label='Pred RUL (EOL)', color='red', linewidth=2, alpha=0.7, linestyle='--')
        
        # Highlight last N cycles
        if len(cycles) > highlight_last_n:
            highlight_start = len(cycles) - highlight_last_n
            ax.axvspan(cycles[highlight_start], cycles[-1], alpha=0.2, color='yellow', label=f'Last {highlight_last_n} cycles')
        
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL [cycles]')
        ax.set_title(f'Engine {unit_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_engines, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RUL trajectories to {out_path}")


def generate_all_diagnostics(
    experiment_dir: Path,
    split: str = "test",
    device: str = "cpu",
) -> Dict:
    """
    Generate all diagnostic plots and summaries for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory (contains model.pt, summary.json)
        split: Data split to analyze ("test" or "val")
        device: Device for inference
    
    Returns:
        Dictionary with all metrics and paths to generated plots
    """
    experiment_dir = Path(experiment_dir)
    
    print(f"\n{'='*80}")
    print(f"Generating diagnostics for: {experiment_dir.name}")
    print(f"{'='*80}\n")
    
    # Run inference
    print(f"[1] Running inference on {split} set...")
    eol_metrics, trajectories = run_inference_for_experiment(
        experiment_dir=experiment_dir,
        split=split,
        device=device,
    )
    
    # Load summary.json for additional metadata and to update with diagnostics
    summary_path = experiment_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Extract test_metrics from summary if available (for comparison/validation)
    # test_metrics is stored in summary.json under "test_metrics" key
    test_metrics = summary.get("test_metrics", {})
    
    # Select engines for trajectories
    print(f"[2] Selecting engines for trajectory plots...")
    selected_engines = select_engines_for_trajectories(
        eol_metrics=eol_metrics,
        trajectories=trajectories,
        num_early=3,
        num_late=3,
        num_random=4,
    )
    print(f"Selected engines: {selected_engines}")
    
    # Generate plots
    print(f"[3] Generating diagnostic plots...")
    
    # 1. EOL error histogram
    plot_eol_error_hist(
        eol_metrics=eol_metrics,
        title=f"EOL RUL Error Distribution - {experiment_dir.name}",
        out_path=experiment_dir / "eol_error_hist.png",
    )
    
    # 2. NASA per engine bar chart
    plot_nasa_per_engine(
        eol_metrics=eol_metrics,
        title=f"NASA Contribution per Engine - {experiment_dir.name}",
        out_path=experiment_dir / "nasa_per_engine_bar.png",
        max_engines=50,
    )
    
    # 3. RUL trajectories
    plot_rul_trajectories(
        trajectories=trajectories,
        selected_unit_ids=selected_engines,
        title=f"RUL Trajectories - {experiment_dir.name}",
        out_path=experiment_dir / "rul_trajectories_10_engines.png",
        highlight_last_n=50,
    )
    
    # 4. HI trajectories
    plot_hi_trajectories_for_selected_engines(
        trajectories=trajectories,
        selected_unit_ids=selected_engines,
        title=f"Health Index Trajectories - {experiment_dir.name}",
        out_path=experiment_dir / "hi_trajectories_10_engines.png",
    )
    
    # Extract EOL values (one per engine) for consistent metric computation
    # This ensures we use the same EOL-based calculation as evaluate_on_test_data
    y_true_eol = np.array([m.true_rul for m in eol_metrics])
    y_pred_eol = np.array([m.pred_rul for m in eol_metrics])
    
    # Get max_rul from config (same as used in evaluate_on_test_data)
    max_rul = summary.get("max_rul", 125)
    if max_rul is None:
        max_rul = 125  # Default
    
    # Compute EOL-based metrics using centralized function
    # This ensures consistency with evaluate_on_test_data (same RUL capping, same formulas)
    eol_metrics_dict = compute_eol_errors_and_nasa(y_true_eol, y_pred_eol, max_rul=max_rul)
    
    # Build diagnostics summary with EOL-based metrics
    # Include all metrics for consistency with test_metrics structure
    diagnostics_summary = {
        "num_engines": eol_metrics_dict["num_engines"],
        "mse": eol_metrics_dict["mse"],
        "rmse": eol_metrics_dict["rmse"],
        "mae": eol_metrics_dict["mae"],
        "bias": eol_metrics_dict["bias"],
        "mean_error": eol_metrics_dict["mean_error"],  # alias for bias
        "std_error": eol_metrics_dict["std_error"],
        "mean_abs_error": eol_metrics_dict["mean_abs_error"],
        "median_error": eol_metrics_dict["median_error"],
        "r2": eol_metrics_dict["r2"],
        "nasa_mean": eol_metrics_dict["nasa_mean"],
        "nasa_sum": eol_metrics_dict["nasa_sum"],
        "nasa_median": eol_metrics_dict["nasa_median"],
        "selected_engines": selected_engines,
        "plots_generated": [
            "eol_error_hist.png",
            "nasa_per_engine_bar.png",
            "rul_trajectories_10_engines.png",
            "hi_trajectories_10_engines.png",
        ],
    }
    
    # Add plot paths to diagnostics summary
    diagnostics_summary["plots"] = {
        "eol_error_hist": str(experiment_dir / "eol_error_hist.png"),
        "nasa_per_engine_bar": str(experiment_dir / "nasa_per_engine_bar.png"),
        "rul_trajectories": str(experiment_dir / "rul_trajectories_10_engines.png"),
        "hi_trajectories": str(experiment_dir / "hi_trajectories_10_engines.png"),
    }
    
    # Update summary.json with diagnostics
    # Use "eol_diagnostics" key to clearly separate from "test_metrics"
    summary["eol_diagnostics"] = diagnostics_summary
    
    # Also keep "diagnostics" for backward compatibility
    summary["diagnostics"] = diagnostics_summary
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Log EOL diagnostics with consistent formatting (matching evaluate_on_test_data)
    print(f"\n[4] Diagnostics complete!")
    print(f"  EOL RMSE:              {eol_metrics_dict['rmse']:.2f} cycles")
    print(f"  EOL MAE:               {eol_metrics_dict['mae']:.2f} cycles")
    print(f"  EOL Bias (mean error): {eol_metrics_dict['bias']:.2f} cycles")
    print(f"  EOL R²:                {eol_metrics_dict['r2']:.4f}")
    print(f"  EOL NASA Score (sum):  {eol_metrics_dict['nasa_sum']:.2f}")
    print(f"  EOL NASA Score (mean): {eol_metrics_dict['nasa_mean']:.4f}")
    
    # Robust sanity check: compare with test_metrics if available
    if test_metrics:
        print(f"\n  [Diagnostics] Sanity check vs test_metrics:")
        
        # Compare bias
        test_bias = test_metrics.get("bias")
        diagnostics_bias = eol_metrics_dict['bias']
        if test_bias is not None:
            bias_diff = abs(diagnostics_bias - test_bias)
            print(f"    Bias: Test={test_bias:.4f}, Diagnostics={diagnostics_bias:.4f} (diff: {bias_diff:.4f})")
            if bias_diff > 0.5:  # Tolerance: 0.5 cycles
                print(f"    ⚠️  WARNING: Bias difference > 0.5 cycles - please verify consistency")
        
        # Compare RMSE
        test_rmse = test_metrics.get("rmse")
        diagnostics_rmse = eol_metrics_dict['rmse']
        if test_rmse is not None:
            rmse_diff = abs(diagnostics_rmse - test_rmse)
            print(f"    RMSE: Test={test_rmse:.4f}, Diagnostics={diagnostics_rmse:.4f} (diff: {rmse_diff:.4f})")
            if rmse_diff > 0.5:  # Tolerance: 0.5 cycles
                print(f"    ⚠️  WARNING: RMSE difference > 0.5 cycles - please verify consistency")
        
        # Compare NASA mean
        test_nasa_mean = test_metrics.get("nasa_mean")
        diagnostics_nasa_mean = eol_metrics_dict['nasa_mean']
        if test_nasa_mean is not None:
            nasa_diff = abs(diagnostics_nasa_mean - test_nasa_mean)
            print(f"    NASA Mean: Test={test_nasa_mean:.4f}, Diagnostics={diagnostics_nasa_mean:.4f} (diff: {nasa_diff:.4f})")
            if nasa_diff > 1.0:  # Tolerance: 1.0
                print(f"    ⚠️  ERROR: NASA mean difference > 1.0 - please check RUL capping / engine aggregation logic")
            elif nasa_diff > 0.1:
                print(f"    ⚠️  WARNING: NASA mean difference > 0.1 - minor inconsistency detected")
        
        # Overall status
        all_ok = (
            (test_bias is None or abs(diagnostics_bias - test_bias) <= 0.5) and
            (test_rmse is None or abs(diagnostics_rmse - test_rmse) <= 0.5) and
            (test_nasa_mean is None or abs(diagnostics_nasa_mean - test_nasa_mean) <= 1.0)
        )
        if all_ok:
            print(f"    ✅ OK: diagnostics metrics are consistent with test_metrics")
        else:
            print(f"    ⚠️  INCONSISTENCY DETECTED: Please review RUL capping, engine order, and metric computation")
    
    print(f"Diagnostics complete and saved to summary.json")
    
    return diagnostics_summary


# ============================================================================
# FD004 EOL + Sliding-HI Diagnostics (Notebook-based implementation)
# ============================================================================

def build_eval_data(
    dataset_name: str,
    max_rul: int,
    past_len: int,
    feature_config: FeatureConfig,
    physics_config: PhysicsFeatureConfig,
    phys_features: Optional[Dict] = None,
    ) -> Tuple[
    np.ndarray,        # X_test_scaled
    np.ndarray,        # y_true_eol (capped)
    np.ndarray,        # y_test_true (per engine)
    np.ndarray,        # unit_ids_test
    np.ndarray,        # cond_ids_test
    Dict[int, StandardScaler],  # scaler_dict
    List[str],         # feature_cols
    pd.DataFrame,      # df_test_fe (for trajectory building)
]:
    """
    Reproduziert genau den Datenpfad aus dem Notebook / Training:

    - load_cmapps_subset(dataset_name, ...)
    - create_physical_features(...)
    - Optional: build_condition_features/create_twin_features (für phys_v2 Runs)
    - create_all_features(...)
    - remove_rul_leakage(feature_cols)
    - build_full_eol_sequences_from_df(...)  -> zum Fitten der Scaler
    - build_test_sequences_from_df(...)      -> Testfenster
    - Condition-wise StandardScaler (nur auf Train-Sequenzen fitten)

    Args:
        dataset_name: Dataset name (e.g., "FD001", "FD002", "FD003", "FD004")
        max_rul: Maximum RUL value
        past_len: Past window length
        feature_config: Feature configuration
        physics_config: Physics feature configuration
        phys_features: Optional phys_features-Config (z.B. für phys_v2: use_condition_vector,
                       use_twin_features, twin_baseline_len). Muss mit run_experiments übereinstimmen.

    Returns:
        X_test_scaled: Scaled test sequences [num_engines, past_len, num_features]
        y_true_eol: True RUL at EOL (capped) [num_engines]
        y_test_true: True RUL at EOL (as returned by load_cmapps_subset) [num_engines]
        unit_ids_test: Unit IDs for test engines [num_engines]
        cond_ids_test: Condition IDs for test engines [num_engines]
        scaler_dict: Dictionary mapping cond_id -> StandardScaler
        feature_cols: List of feature column names
        df_test_fe: Test DataFrame with features (for trajectory building)
    """
    # Load dataset data
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    # Optional physikalisch informierte Features (continuous condition vector + digital twin)
    phys_features = phys_features or {}
    use_phys_condition_vec = phys_features.get("use_condition_vector", False)
    # Backwards-compatible: accept both "use_twin_features" and
    # "use_digital_twin_residuals" as enabling the same digital-twin block.
    use_twin_features = phys_features.get(
        "use_twin_features",
        phys_features.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_features.get("twin_baseline_len", 30)
    condition_vector_version = phys_features.get("condition_vector_version", 2)

    # Feature engineering – exakt wie in run_experiments:
    # 1) Physik-Features
    df_train_fe = create_physical_features(df_train.copy(), physics_config, "UnitNumber", "TimeInCycles")
    df_test_fe = create_physical_features(df_test.copy(), physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        print("  Using continuous condition vector features (Cond_*) [diagnostics]")
        df_train_fe = build_condition_features(
            df_train_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )
        df_test_fe = build_condition_features(
            df_test_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin + Residuen
    if use_twin_features:
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len}) [diagnostics]")
        df_train_fe, twin_model = create_twin_features(
            df_train_fe,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test_fe = twin_model.add_twin_and_residuals(df_test_fe)

    # 4) Temporale/multi-scale Features
    df_train_fe = create_all_features(df_train_fe, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)
    df_test_fe = create_all_features(df_test_fe, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)
    
    # Build feature columns
    feature_cols = [
        c for c in df_train_fe.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    
    # Build full EOL sequences from train data (for scaler fitting)
    result = build_full_eol_sequences_from_df(
        df=df_train_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    X_full, y_full, unit_ids_full, cond_ids_full = result[:4]
    health_phys_seq_full = result[4] if len(result) > 4 else None
    
    # Simple engine-based split (80/20) for scaler fitting
    unique_units = np.unique(unit_ids_full.numpy())
    n_total = len(unique_units)
    n_train = int(0.8 * n_total)
    train_units = unique_units[:n_train]
    
    train_mask = np.isin(unit_ids_full.numpy(), train_units)
    X_train = X_full[train_mask]
    cond_train = cond_ids_full[train_mask]
    
    # Build test sequences
    X_test, unit_ids_test, cond_ids_test = build_test_sequences_from_df(
        df_test_fe,
        feature_cols=feature_cols,
        past_len=past_len,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    
    # Condition-wise feature scaling (same as notebook)
    scaler_dict = {}
    all_cond_ids = np.unique(cond_train.numpy())
    
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    
    X_test_scaled = np.empty_like(X_test_np)
    
    for cond in all_cond_ids:
        cond = int(cond)
        train_mask_cond = (cond_train.numpy() == cond)
        test_mask_cond = (cond_ids_test.numpy() == cond)
        
        scaler = StandardScaler()
        # Fit on train data for this condition
        X_train_cond_flat = X_train_np[train_mask_cond].reshape(-1, X_train_np.shape[-1])
        scaler.fit(X_train_cond_flat)
        scaler_dict[cond] = scaler
        
        # Transform test
        if test_mask_cond.any():
            X_test_scaled[test_mask_cond] = scaler.transform(
                X_test_np[test_mask_cond].reshape(-1, X_test_np.shape[-1])
            ).reshape(-1, past_len, len(feature_cols))
    
    # Build y_true_eol (capped) - EXACTLY as in notebook Cell 7
    # In the notebook: 
    #   unit_ids_test_np = unit_ids_test.numpy()
    #   y_true_eol = np.minimum(y_test_true, max_rul)  # Direct use, same order
    # 
    # IMPORTANT: 
    # - build_test_sequences_from_df returns unit_ids in SORTED order (we fixed this)
    # - y_test_true from load_cmapps_subset is sorted by UnitNumber (1, 2, 3, ...)
    # - So they should match directly, BUT we need to verify the mapping
    unit_ids_test_np = unit_ids_test.numpy()
    
    # Verify that unit_ids_test matches the order of y_test_true
    # y_test_true[i] corresponds to UnitNumber = i+1
    if len(unit_ids_test_np) == len(y_test_true):
        # Check if they match directly (both sorted)
        expected_unit_ids = np.arange(1, len(y_test_true) + 1)
        if np.array_equal(unit_ids_test_np, expected_unit_ids):
            # Direct match - use as in notebook
            y_true_eol = np.minimum(y_test_true, max_rul)
        else:
            # Need to map: create unit_id -> y_test_true index mapping
            unit_id_to_y_idx = {i + 1: i for i in range(len(y_test_true))}
            y_true_eol = []
            for unit_id in unit_ids_test_np:
                idx = unit_id_to_y_idx.get(int(unit_id), int(unit_id) - 1)
                if 0 <= idx < len(y_test_true):
                    y_true_eol.append(y_test_true[idx])
                else:
                    print(f"  ⚠️  Warning: unit_id {unit_id} not found in y_test_true, using fallback")
                    y_true_eol.append(y_test_true[-1] if len(y_test_true) > 0 else max_rul)
            y_true_eol = np.array(y_true_eol)
            y_true_eol = np.minimum(y_true_eol, max_rul)
    else:
        # Length mismatch - create mapping
        print(f"  ⚠️  WARNING: Length mismatch! y_test_true={len(y_test_true)}, unit_ids_test={len(unit_ids_test_np)}")
        unit_id_to_y_idx = {i + 1: i for i in range(len(y_test_true))}
        y_true_eol = []
        for unit_id in unit_ids_test_np:
            idx = unit_id_to_y_idx.get(int(unit_id), int(unit_id) - 1)
            if 0 <= idx < len(y_test_true):
                y_true_eol.append(y_test_true[idx])
            else:
                y_true_eol.append(y_test_true[-1] if len(y_test_true) > 0 else max_rul)
        y_true_eol = np.array(y_true_eol)
        y_true_eol = np.minimum(y_true_eol, max_rul)
    
    return (
        X_test_scaled,
        y_true_eol,
        y_test_true,
        unit_ids_test_np,
        cond_ids_test.numpy(),
        scaler_dict,
        feature_cols,
        df_test_fe,
    )


def compute_hi_trajectory_sliding(
    df_engine: pd.DataFrame,
    feature_cols: List[str],
    scaler_dict: Dict[int, StandardScaler],
    past_len: int,
    model: nn.Module,
    device: torch.device,
    is_world_model: bool = False,
    is_world_model_v3: bool = False,
    max_rul: float = 125.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Berechnet eine HI-Trajektorie für eine Engine mit Sliding Window:
    
    - Für jeden Zyklus t >= past_len wird ein Fenster [t-past_len+1, ..., t] gebildet.
    - Das Modell sagt HI für dieses Fenster voraus.
    - HI_last (letzter Zeitschritt im Fenster) wird als HI(t) verwendet.
    
    Args:
        df_engine: DataFrame für eine Engine, sortiert nach TimeInCycles
        feature_cols: Liste der Feature-Spaltennamen
        scaler_dict: Dictionary mapping cond_id -> StandardScaler
        past_len: Länge des Sliding Windows
        model: Trainiertes Modell (RULHIUniversalModelV2 or WorldModelEncoderDecoderUniversalV2)
        device: PyTorch device
        is_world_model: Whether this is a world model (returns trajectory + eol_pred instead of rul + hi)
    
    Returns:
        cycles_hi: Zyklen, für die HI definiert ist (ab past_len)
        hi_vals:   HI-Werte pro Zyklus
        rul_vals:  RUL-Vorhersagen pro Zyklus (in Zyklen, gecappt auf [0, max_rul])
        hi_damage_vals: Optional damage-based HI values (None if model doesn't have damage_head)
    """
    df_engine = df_engine.sort_values("TimeInCycles").copy()
    
    cond_id = int(df_engine["ConditionID"].iloc[0])
    scaler = scaler_dict.get(cond_id, None)
    
    # Extract features and scale
    feats = df_engine[feature_cols].values.astype(np.float32)
    if scaler is not None:
        feats = scaler.transform(feats)
    
    cycles = df_engine["TimeInCycles"].values
    
    hi_vals: List[float] = []
    rul_vals: List[float] = []
    hi_damage_vals: List[float] = []
    hi_cycles: List[float] = []
    
    # Check if model has damage_head
    has_damage_head = hasattr(model, 'damage_head') and model.damage_head is not None
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(df_engine)):
            if idx + 1 < past_len:
                continue  # noch kein volles Fenster
            
            # Sliding window: [idx+1-past_len, ..., idx]
            window = feats[idx + 1 - past_len : idx + 1]  # (past_len, F)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, past_len, F)
            
            # Forward pass
            if is_world_model:
                # Check if model needs cond_ids
                needs_cond_ids = hasattr(model, 'num_conditions') and model.num_conditions and model.num_conditions > 1
                cond_t = torch.tensor([cond_id], dtype=torch.long).to(device) if needs_cond_ids else None
                
                if is_world_model_v3:
                    # World model v3 returns dict with "traj", "eol", "hi"
                    outputs = model(
                        encoder_inputs=x,
                        decoder_targets=None,
                        teacher_forcing_ratio=0.0,
                        horizon=1,
                        cond_ids=cond_t,
                    )
                    # Use explicit HI and EOL head output from v3
                    if isinstance(outputs, dict):
                        hi_val = float(outputs["hi"][0, 0].item())
                        eol_val = float(outputs["eol"][0, 0].item())
                    else:
                        # Fallback: treat as tuple (shouldn't happen)
                        traj_pred, eol_pred = outputs
                        eol_val = float(eol_pred[0, 0].item())
                        hi_val = max(0.0, min(1.0, 1.0 - (eol_val / max_rul)))
                    # Clamp RUL to valid range
                    eol_val = max(0.0, min(float(max_rul), eol_val))
                    hi_vals.append(hi_val)
                    rul_vals.append(eol_val)
                else:
                    # World model v2 returns (traj_outputs, eol_pred)
                    try:
                        model_output = model(
                            encoder_inputs=x,
                            decoder_targets=None,
                            teacher_forcing_ratio=0.0,
                            horizon=1,
                            cond_ids=cond_t,
                        )
                        # Handle both tuple and dict returns
                        if isinstance(model_output, dict):
                            # Unexpected: v2 returned dict, try to extract
                            eol_pred = model_output.get("eol", model_output.get("eol_pred"))
                            if eol_pred is None:
                                # Fallback: use trajectory if available
                                traj_pred = model_output.get("traj")
                                if traj_pred is not None and traj_pred.numel() > 0:
                                    eol_val = traj_pred[0, 0, 0].item() if len(traj_pred.shape) == 3 else traj_pred[0, 0].item()
                                else:
                                    eol_val = max_rul * 0.5  # Default fallback
                            else:
                                eol_val = eol_pred[0, 0].item() if len(eol_pred.shape) == 2 else eol_pred[0].item()
                        else:
                            # Expected: tuple (traj_outputs, eol_pred)
                            if len(model_output) == 2:
                                traj_pred, eol_pred = model_output
                                # eol_pred is (B, 1), extract scalar
                                eol_val = eol_pred[0, 0].item() if len(eol_pred.shape) == 2 else eol_pred[0].item()
                            else:
                                # Unexpected number of outputs
                                print(f"  Warning: World Model v2 returned {len(model_output)} values, expected 2")
                                eol_val = model_output[1][0, 0].item() if len(model_output[1].shape) == 2 else model_output[1][0].item()
                        
                        # Use EOL prediction as proxy for HI (normalized)
                        # HI = 1 - (eol_pred / max_rul), clamped to [0, 1]
                        eol_val = float(eol_val)
                        hi_proxy = max(0.0, min(1.0, 1.0 - (eol_val / max_rul)))
                        # Clamp RUL for plotting
                        eol_val = max(0.0, min(float(max_rul), eol_val))
                        hi_vals.append(hi_proxy)
                        rul_vals.append(eol_val)
                    except Exception as e:
                        print(f"  Warning: Error computing HI/RUL for sliding window: {e}")
                        # Fallback: use default HI/RUL values
                        hi_vals.append(0.5)
                        rul_vals.append(float(max_rul) / 2.0)
            else:
                # EOL model returns (rul_pred, hi_last, hi_seq)
                cond_t = torch.tensor([cond_id], dtype=torch.long).to(device)
                out = model(x, cond_ids=cond_t)
                # Newer Transformer variants may append rul_sigma (v5u):
                #   (rul_pred, hi_last, hi_seq, rul_sigma)
                if isinstance(out, (tuple, list)) and len(out) >= 3:
                    rul_pred, hi_last, hi_seq = out[0], out[1], out[2]
                else:
                    raise RuntimeError(
                        f"Unexpected encoder output type/len in sliding HI: {type(out)}"
                    )
                # HI_last ist der letzte Zeitschritt im Fenster
                hi_last_val = float(hi_seq[0, -1].item())
                # RUL prediction for this window
                if rul_pred.ndim == 2:
                    eol_val = float(rul_pred[0, 0].item())
                else:
                    eol_val = float(rul_pred[0].item())
                eol_val = max(0.0, min(float(max_rul), eol_val))
                hi_vals.append(hi_last_val)
                rul_vals.append(eol_val)
                
                # Extract damage HI if damage_head is available
                if has_damage_head:
                    try:
                        # Get encoder output to pass to damage_head
                        # For transformer_eol models, use the encode method
                        enc_out = None
                        cond_seq_for_damage = None
                        
                        if hasattr(model, 'encode'):
                            # Use encode method to get encoder output sequence
                            enc_out, _ = model.encode(x, cond_ids=cond_t, return_seq=True)
                            # Prepare cond_seq for damage_head if needed
                            if hasattr(model, 'use_cond_encoder') and model.use_cond_encoder:
                                if hasattr(model, 'cond_feature_indices') and model.cond_feature_indices is not None:
                                    cond_seq_for_damage = x[:, :, model.cond_feature_indices]
                        else:
                            # Fallback: try manual encoding for other model types
                            if hasattr(model, 'transformer') and hasattr(model, 'input_proj'):
                                x_proj = model.input_proj(x)
                                # Apply condition embedding if available
                                if hasattr(model, 'use_condition_embedding') and model.use_condition_embedding:
                                    cond_emb = model.condition_embedding(cond_t)
                                    cond_up = model.cond_proj(cond_emb)
                                    cond_up = cond_up.unsqueeze(1).expand(-1, x_proj.shape[1], -1)
                                    x_seq = x_proj + cond_up
                                else:
                                    x_seq = x_proj
                                # Add continuous condition if available
                                if hasattr(model, 'use_cond_encoder') and model.use_cond_encoder and hasattr(model, 'cond_encoder'):
                                    if hasattr(model, 'cond_feature_indices') and model.cond_feature_indices is not None:
                                        cond_seq_for_damage = x[:, :, model.cond_feature_indices]
                                        cond_emb_seq = model.cond_encoder(cond_seq_for_damage)
                                        x_seq = x_seq + cond_emb_seq
                                x_pos = model.pos_encoding(x_seq)
                                enc_out = model.transformer(x_pos)
                        
                        if enc_out is not None:
                            # Call damage_head
                            hi_seq_damage, _, _, _ = model.damage_head(enc_out, cond_seq=cond_seq_for_damage)
                            hi_damage_val = float(hi_seq_damage[0, -1].item())
                            hi_damage_vals.append(hi_damage_val)
                        else:
                            hi_damage_vals.append(None)
                    except Exception as e:
                        # If damage head extraction fails, append None
                        # Uncomment for debugging: print(f"  Warning: Could not extract damage HI: {e}")
                        hi_damage_vals.append(None)
                else:
                    hi_damage_vals.append(None)
            
            hi_cycles.append(cycles[idx])
    
    # Convert hi_damage_vals to numpy array, filtering out None values if any
    hi_damage_array = None
    if has_damage_head and len(hi_damage_vals) > 0:
        valid_count = sum(1 for v in hi_damage_vals if v is not None)
        if valid_count > 0:
            # Filter out None values and create array
            valid_damage_vals = [v if v is not None else 0.0 for v in hi_damage_vals]
            hi_damage_array = np.array(valid_damage_vals)
        # Note: Logging is done at the build_trajectories level to avoid spam
    
    return np.array(hi_cycles), np.array(hi_vals), np.array(rul_vals), hi_damage_array


def build_trajectories(
    df_test_fe: pd.DataFrame,
    feature_cols: List[str],
    scaler_dict: Dict[int, StandardScaler],
    past_len: int,
    model: nn.Module,
    device: torch.device,
    y_true_eol: np.ndarray,
    rul_pred_full_np: np.ndarray,
    unit_ids_test: np.ndarray,
    max_rul: int,
    is_world_model: bool = False,
    is_world_model_v3: bool = False,
) -> List[EngineTrajectory]:
    """
    Baut pro Engine Trajektorien für HI + RUL.
    
    Args:
        df_test_fe: Test DataFrame with features
        feature_cols: List of feature column names
        scaler_dict: Dictionary mapping cond_id -> StandardScaler
        past_len: Past window length
        model: Trained model
        device: PyTorch device
        y_true_eol: True RUL at EOL (capped) [num_engines]
        rul_pred_full_np: Predicted RUL at EOL [num_engines] (wird hier nur noch
            informativ genutzt – die Trajektorie selbst wird per Sliding Window
            berechnet)
        unit_ids_test: Unit IDs for test engines [num_engines]
        max_rul: Maximum RUL value
    
    Returns:
        List of EngineTrajectory objects
    """
    trajectories = []
    
    for i, unit_id in enumerate(unit_ids_test):
        unit_id = int(unit_id)
        
        # Get full time series for this engine
        df_engine = df_test_fe[df_test_fe["UnitNumber"] == unit_id].sort_values("TimeInCycles")
        
        if len(df_engine) == 0:
            continue
        
        # Compute HI *and* RUL trajectory using sliding window
        cycles_hi, hi_vals, pred_rul_traj, hi_damage_vals = compute_hi_trajectory_sliding(
            df_engine=df_engine,
            feature_cols=feature_cols,
            scaler_dict=scaler_dict,
            past_len=past_len,
            model=model,
            device=device,
            is_world_model=is_world_model,
            is_world_model_v3=is_world_model_v3,
            max_rul=max_rul,
        )
        
        if len(cycles_hi) == 0:
            continue  # Skip if no valid HI values
        
        # Build full cycles array for RUL trajectories
        cycles_full = df_engine["TimeInCycles"].values
        
        # True RUL trajectory: linear decline from max_rul to true_rul_eol
        true_rul_eol = y_true_eol[i]
        true_rul_full = np.linspace(max_rul, true_rul_eol, len(cycles_full))
        
        # Map cycles_hi to indices in cycles_full
        idx_map = {c: idx for idx, c in enumerate(cycles_full)}
        true_rul_traj = np.array([true_rul_full[idx_map[c]] for c in cycles_hi])
        
        trajectories.append(EngineTrajectory(
            unit_id=unit_id,
            cycles=cycles_hi,  # Only cycles where HI is defined
            hi=hi_vals,  # HI values from sliding window
            true_rul=true_rul_traj,
            pred_rul=pred_rul_traj,
            hi_damage=hi_damage_vals,  # Optional damage-based HI trajectory
        ))
    
    return trajectories


def select_degraded_engines(
    y_true_eol: np.ndarray,
    unit_ids_test: np.ndarray,
    trajectories: List[EngineTrajectory],
    rul_threshold: float = 40.0,
    max_engines: int = 10,
) -> List[EngineTrajectory]:
    """
    Wählt bis zu max_engines Engines mit kleiner True-RUL (RUL_eol < rul_threshold) aus.
    
    Args:
        y_true_eol: True RUL at EOL (capped) [num_engines]
        unit_ids_test: Unit IDs for test engines [num_engines]
        trajectories: List of EngineTrajectory objects
        rul_threshold: RUL threshold for "degraded" engines
        max_engines: Maximum number of engines to select
    
    Returns:
        List of selected EngineTrajectory objects
    """
    degraded_indices = np.where(y_true_eol < rul_threshold)[0]
    
    # Wenn weniger als max_engines gefunden werden, nimm alle; sonst die max_engines mit kleinstem RUL
    if len(degraded_indices) > max_engines:
        # sortiere nach RUL, wähle max_engines Engines mit kleinster RUL
        degraded_sorted = degraded_indices[np.argsort(y_true_eol[degraded_indices])]
        selected_degraded_indices = degraded_sorted[:max_engines]
    else:
        selected_degraded_indices = degraded_indices
    
    selected_degraded_unit_ids = [int(unit_ids_test[i]) for i in selected_degraded_indices]
    
    # Mapping von unit_id zu Trajectory
    unit_id_to_traj = {traj.unit_id: traj for traj in trajectories}
    
    selected_trajectories = []
    for uid in selected_degraded_unit_ids:
        traj = unit_id_to_traj.get(uid)
        if traj is not None:
            selected_trajectories.append(traj)
    
    return selected_trajectories


def plot_error_histogram(
    errors: np.ndarray,
    out_path: Path,
    title: str = "Error Histogram",
    xlabel: str = "Error (pred - true) [cycles]",
) -> None:
    """
    Plot error histogram (publication-ready).
    
    Args:
        errors: Error array (pred - true)
        out_path: Output file path
        title: Plot title
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} (mean={errors.mean():.2f}, std={errors.std():.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error histogram to {out_path}")


def plot_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    max_rul: int,
    title: str = "True vs Predicted RUL",
) -> None:
    """
    Plot true vs predicted RUL scatter (publication-ready).
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        out_path: Output file path
        max_rul: Maximum RUL value (for diagonal line)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([0, max_rul], [0, max_rul], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True RUL [cycles]')
    ax.set_ylabel('Predicted RUL [cycles]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved true vs pred scatter to {out_path}")


def plot_hi_rul_trajectories(
    trajectories: List[EngineTrajectory],
    out_path: Path,
    title: str = "Health Index + RUL Trajectories",
    max_engines: int = 10,
) -> None:
    """
    Plot HI + RUL trajectories (dual axis, publication-ready).
    
    Args:
        trajectories: List of EngineTrajectory objects
        out_path: Output file path
        title: Plot title
        max_engines: Maximum number of engines to plot
    """
    num_engines = min(len(trajectories), max_engines)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, traj in enumerate(trajectories[:max_engines]):
        ax1 = axes[idx]
        
        # Primary axis: HI (with dynamic scaling based on actual HI range)
        if len(traj.hi) > 0:
            hi_min, hi_max = traj.hi.min(), traj.hi.max()
            hi_range = hi_max - hi_min
            # Add 10% padding, but ensure reasonable bounds
            hi_padding = max(0.05, hi_range * 0.1) if hi_range > 0 else 0.1
            ax1.set_ylim([max(0, hi_min - hi_padding), min(1.1, hi_max + hi_padding)])
        else:
            ax1.set_ylim([0, 1.1])
        
        ax1.plot(traj.cycles, traj.hi, 'g-', linewidth=2, label='Health Index', alpha=0.7)
        
        ax1.set_xlabel('Time in Cycles')
        ax1.set_ylabel('Health Index', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis: RUL
        ax2 = ax1.twinx()
        ax2.plot(traj.cycles, traj.true_rul, 'b-', linewidth=2, label='True RUL', alpha=0.7)
        ax2.plot(traj.cycles, traj.pred_rul, 'r--', linewidth=2, label='Pred RUL', alpha=0.7)
        ax2.set_ylabel('RUL [cycles]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax1.set_title(f'Engine #{traj.unit_id} – HI + RUL (degraded)')
    
    # Hide unused subplots
    for idx in range(num_engines, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved HI + RUL trajectories to {out_path}")


def plot_hi_damage_trajectories(
    trajectories: List[EngineTrajectory],
    out_path: Path,
    title: str = "Health Index Damage Trajectories",
    max_engines: int = 10,
) -> None:
    """
    Plot HI Damage trajectories (similar to HI trajectories, but for damage-based HI).
    
    This function creates a separate plot specifically for the damage-based HI trajectory,
    using the same layout and style as the standard HI trajectory plots.
    
    Args:
        trajectories: List of EngineTrajectory objects (must have hi_damage field populated)
        out_path: Output file path
        title: Plot title
        max_engines: Maximum number of engines to plot
    """
    # Filter trajectories that have damage HI
    traj_with_damage = [t for t in trajectories if t.hi_damage is not None and len(t.hi_damage) > 0]
    
    if len(traj_with_damage) == 0:
        print(f"  Warning: No damage HI trajectories available, skipping damage plot")
        return
    
    num_engines = min(len(traj_with_damage), max_engines)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, traj in enumerate(traj_with_damage[:max_engines]):
        ax1 = axes[idx]
        
        # Primary axis: HI Damage (with dynamic scaling based on actual HI damage range)
        if len(traj.hi_damage) > 0:
            hi_damage_min, hi_damage_max = traj.hi_damage.min(), traj.hi_damage.max()
            hi_damage_range = hi_damage_max - hi_damage_min
            # Add 10% padding, but ensure reasonable bounds
            hi_damage_padding = max(0.05, hi_damage_range * 0.1) if hi_damage_range > 0 else 0.1
            ax1.set_ylim([max(0, hi_damage_min - hi_damage_padding), min(1.1, hi_damage_max + hi_damage_padding)])
        else:
            ax1.set_ylim([0, 1.1])
        
        # Plot HI Damage trajectory
        ax1.plot(traj.cycles, traj.hi_damage, 'm-', linewidth=2, label='HI Damage', alpha=0.7)
        
        # Also plot regular HI for comparison (lighter color)
        if len(traj.hi) > 0:
            ax1.plot(traj.cycles, traj.hi, 'g--', linewidth=1.5, label='HI (standard)', alpha=0.5)
        
        ax1.set_xlabel('Time in Cycles')
        ax1.set_ylabel('Health Index', color='m')
        ax1.tick_params(axis='y', labelcolor='m')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis: RUL (for reference)
        ax2 = ax1.twinx()
        ax2.plot(traj.cycles, traj.true_rul, 'b-', linewidth=2, label='True RUL', alpha=0.7)
        ax2.plot(traj.cycles, traj.pred_rul, 'r--', linewidth=2, label='Pred RUL', alpha=0.7)
        ax2.set_ylabel('RUL [cycles]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax1.set_title(f'Engine #{traj.unit_id} – HI Damage (degraded)')
    
    # Hide unused subplots
    for idx in range(num_engines, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved HI Damage trajectories to {out_path}")


def plot_hi_phys_v3_true_trajectories(
    df_test_fe: pd.DataFrame,
    trajectories: List[EngineTrajectory],
    out_path: Path,
    title: str = "HI_phys_v3 true trajectories",
    max_engines: int = 10,
) -> None:
    """
    Plot true HI_phys_v3 trajectories (no predictions) for a subset of engines.

    Args:
        df_test_fe: Test DataFrame with columns UnitNumber, TimeInCycles, HI_phys_v3.
        trajectories: List of EngineTrajectory objects (used to select unit_ids).
        out_path: Output file path.
        title: Plot title.
        max_engines: Maximum number of engines to plot.
    """
    if "HI_phys_v3" not in df_test_fe.columns:
        print("  Warning: HI_phys_v3 not found in df_test_fe, skipping true HI_phys_v3 plot.")
        return

    unit_ids = [t.unit_id for t in trajectories][:max_engines]
    if len(unit_ids) == 0:
        print("  Warning: No trajectories available for HI_phys_v3 plot.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, uid in enumerate(unit_ids):
        if idx >= len(axes):
            break
        ax = axes[idx]
        g = (
            df_test_fe[df_test_fe["UnitNumber"] == uid]
            .sort_values("TimeInCycles")
            .copy()
        )
        if g.empty or "HI_phys_v3" not in g.columns:
            continue

        ax.plot(
            g["TimeInCycles"].to_numpy(),
            g["HI_phys_v3"].to_numpy(),
            "b-",
            linewidth=2,
            label="HI_phys_v3 (true)",
        )
        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("HI_phys_v3")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Engine #{uid} – HI_phys_v3 (true)")

    # Hide unused subplots
    for j in range(len(unit_ids), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved true HI_phys_v3 trajectories to {out_path}")


def plot_hi_phys_v3_true_vs_pred(
    df_test_fe: pd.DataFrame,
    trajectories: List[EngineTrajectory],
    out_path: Path,
    metrics_path: Optional[Path] = None,
    title: str = "HI_phys_v3 true vs predicted",
    max_engines: int = 10,
) -> None:
    """
    Plot true vs predicted HI_phys_v3 trajectories for a subset of engines.

    Uses:
        - true HI_phys_v3 from df_test_fe["HI_phys_v3"]
        - predicted damage-based HI from EngineTrajectory.hi_damage
    """
    if "HI_phys_v3" not in df_test_fe.columns:
        print("  Warning: HI_phys_v3 not found in df_test_fe, skipping true-vs-pred plot.")
        return

    traj_with_damage = [
        t for t in trajectories if t.hi_damage is not None and len(t.hi_damage) > 0
    ]
    if len(traj_with_damage) == 0:
        print(
            "  Warning: No trajectories with damage HI available, skipping true-vs-pred HI_phys_v3 plot."
        )
        return

    num_engines = min(len(traj_with_damage), max_engines)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Store HI sequences for RMSE computation
    hi_rmse_per_engine: Dict[int, float] = {}

    for idx, traj in enumerate(traj_with_damage[:max_engines]):
        ax = axes[idx]
        uid = traj.unit_id
        g = (
            df_test_fe[df_test_fe["UnitNumber"] == uid]
            .sort_values("TimeInCycles")
            .copy()
        )
        if g.empty:
            continue

        # Map cycle -> true HI_phys_v3
        hi_map = dict(zip(g["TimeInCycles"].to_numpy(), g["HI_phys_v3"].to_numpy()))
        hi_true_aligned = np.array([hi_map.get(c, np.nan) for c in traj.cycles])

        ax.plot(
            traj.cycles,
            hi_true_aligned,
            "b-",
            linewidth=2,
            label="HI_phys_v3 (true)",
            alpha=0.8,
        )
        ax.plot(
            traj.cycles,
            traj.hi_damage,
            "m--",
            linewidth=2,
            label="HI_damage (pred)",
            alpha=0.8,
        )

        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("Health Index")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Engine #{uid} – HI_phys_v3 true vs pred")

        # Compute per-engine RMSE between HI_phys_v3 and HI_damage
        mask_valid = ~np.isnan(hi_true_aligned)
        if np.any(mask_valid):
            hi_true = hi_true_aligned[mask_valid].astype(float)
            hi_pred = np.asarray(traj.hi_damage, dtype=float)[mask_valid]
            T = min(len(hi_true), len(hi_pred))
            if T > 0:
                rmse = float(np.sqrt(np.mean((hi_true[:T] - hi_pred[:T]) ** 2)))
                hi_rmse_per_engine[int(uid)] = rmse

    # Hide unused subplots
    for j in range(num_engines, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved HI_phys_v3 true vs pred trajectories to {out_path}")


def plot_hi_cal_v2_trajectories(
    trajectories: List[EngineTrajectory],
    out_path: Path,
    title: str = "HI_cal_v2 (v4) trajectories",
    max_engines: int = 10,
) -> None:
    """
    Plot predicted HI_cal_v2 trajectories (encoder v4) for a subset of engines.

    Uses EngineTrajectory.hi_cal (if present), aligned to engine cycles.
    """
    traj_with_hi_cal = [
        t for t in trajectories if getattr(t, "hi_cal", None) is not None and len(t.hi_cal) > 0
    ]
    if len(traj_with_hi_cal) == 0:
        print("  Warning: No HI_cal_v2 trajectories available, skipping HI_cal_v2 plot.")
        return

    num_engines = min(len(traj_with_hi_cal), max_engines)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, traj in enumerate(traj_with_hi_cal[:max_engines]):
        ax = axes[idx]
        ax.plot(traj.cycles, traj.hi_cal, "c-", linewidth=2, label="HI_cal_v2 (pred)")
        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("HI_cal_v2")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Engine #{traj.unit_id} – HI_cal_v2 (v4)")

    for j in range(num_engines, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved HI_cal_v2 trajectories to {out_path}")


def plot_hi_cal_v2_vs_hi_phys_v3(
    df_test_fe: pd.DataFrame,
    trajectories: List[EngineTrajectory],
    out_path: Path,
    title: str = "HI_cal_v2 (v4) vs HI_phys_v3 at EOL",
) -> None:
    """
    Scatter plot: predicted HI_cal_v2 (v4) vs true HI_phys_v3 at EOL for engines.

    - HI_cal_v2 comes from EngineTrajectory.hi_cal (last timestep).
    - HI_phys_v3 comes from df_test_fe["HI_phys_v3"] at the engine's last cycle.
    """
    if "HI_phys_v3" not in df_test_fe.columns:
        print("  Warning: HI_phys_v3 not found in df_test_fe, skipping HI_cal_v2 vs HI_phys_v3 plot.")
        return

    xs: List[float] = []
    ys: List[float] = []

    for traj in trajectories:
        hi_cal_traj = getattr(traj, "hi_cal", None)
        if hi_cal_traj is None or len(hi_cal_traj) == 0:
            continue

        uid = traj.unit_id
        g = (
            df_test_fe[df_test_fe["UnitNumber"] == uid]
            .sort_values("TimeInCycles")
            .copy()
        )
        if g.empty:
            continue

        # EOL: last cycle in df_test_fe for this engine
        last_row = g.iloc[-1]
        hi_phys_eol = float(last_row.get("HI_phys_v3", np.nan))
        hi_cal_eol = float(hi_cal_traj[-1])

        if np.isnan(hi_phys_eol):
            continue

        xs.append(hi_phys_eol)
        ys.append(hi_cal_eol)

    if not xs:
        print("  Warning: No valid HI_cal_v2/HI_phys_v3 pairs for scatter plot.")
        return

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    plt.figure(figsize=(6, 5))
    plt.scatter(xs_arr, ys_arr, alpha=0.7, c="c", edgecolors="k")
    plt.xlabel("HI_phys_v3 at EOL")
    plt.ylabel("HI_cal_v2 (v4) at EOL")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved HI_cal_v2 vs HI_phys_v3 scatter to {out_path}")


def run_diagnostics_for_run(
    exp_dir: Union[str, Path],
    dataset_name: str,
    run_name: str,
    device: Optional[torch.device] = None,
) -> None:
    """
    Lädt Konfiguration + Modell aus dem Experiment-Ordner und führt vollständige
    Diagnostik durch (Notebook-basiert, für alle Datasets).
    
    Args:
        exp_dir: Base directory for experiments (e.g., "results")
        dataset_name: Dataset name (e.g., "FD001", "FD002", "FD003", "FD004")
        run_name: Experiment name (e.g., "fd004_phase3_universal_v2_ms_cnn_d96")
        device: PyTorch device (if None, auto-detect)
    """
    from src.analysis.inference import load_model_from_experiment
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path(exp_dir)
    # Experiment structure: results/<dataset>/<experiment_name>/
    # dataset is lowercase (e.g., "fd004")
    experiment_dir = exp_dir / dataset_name.lower() / run_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    print(f"\n{'='*80}")
    print(f"Running diagnostics for: {run_name} ({dataset_name})")
    print(f"Experiment directory: {experiment_dir}")
    print(f"{'='*80}\n")
    
    # Load model and config
    print("[1] Loading model and config...")
    try:
        model, config = load_model_from_experiment(experiment_dir, device=device)
        model.eval()
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠️  Skipping diagnostics for {run_name} due to model loading error")
        return
    
    # Check if this is a world model (use same logic as load_model_from_experiment)
    experiment_name = config.get("experiment_name", run_name)
    model_type = config.get("model_type")
    encoder_type = config.get("encoder_type")
    
    is_world_model = (
        model_type in ["world_model_universal_v2", "world_model_universal_v3"]
        or encoder_type in ["world_model_universal_v2", "world_model_universal_v3"]
        or "world_phase" in experiment_name.lower()
        or "world_model" in experiment_name.lower()
        or config.get("training_mode") == "world_model"
    )
    
    # Check if it is World Model V3 based on class name (most reliable)
    # load_model_from_experiment handles the fallback logic
    is_world_model_v3 = (
        model.__class__.__name__ == "WorldModelUniversalV3"
        or (
            # Fallback to config/name if class name check fails (e.g. wrapper)
            model.__class__.__name__ != "WorldModelEncoderDecoderUniversalV2" and
            (
                model_type == "world_model_universal_v3"
                or encoder_type == "world_model_universal_v3"
                or ("world" in experiment_name.lower() and "v3" in experiment_name.lower())
                or ("world" in experiment_name.lower() and "phase5" in experiment_name.lower())
            )
        )
    )
    
    if is_world_model:
        if is_world_model_v3:
            print(f"  ✓ Detected World Model v3 experiment: {experiment_name}")
            print(f"    model_type={model_type}, encoder_type={encoder_type}")
            print("  Will use HI head output for diagnostics (not EOL proxy)")
        else:
            print(f"  ✓ Detected World Model v2 experiment: {experiment_name}")
            print(f"    model_type={model_type}, encoder_type={encoder_type}")
            print("  Will use EOL prediction as HI proxy for diagnostics")
    else:
        print(f"  ✓ Detected RUL/HI experiment: {experiment_name}")
    
    # Extract config parameters
    max_rul = config.get("max_rul", 125)
    past_len = config.get("past_len", 30)
    
    # Check if this is a phase 4/5 residual experiment or a \"digital-twin\" residual run.
    # World model experiments use residual features if "phase4" or "phase5" and "residual" in name
    # or if world_model_params.use_residual_features is set.
    # EOL-Modelle mit \"resid\" im Namen (z.B. fd004_transformer_encoder_resid_v1) werden hier
    # ebenfalls als Residual-Runs behandelt.
    
    # Check both run_name and experiment_name for phase4/phase5 and residual/resid
    name_to_check = (experiment_name or run_name).lower()
    
    is_phase4_residual = (
        (("phase4" in name_to_check) or ("phase5" in name_to_check))
        and "residual" in name_to_check
    ) or (
        ("world_phase4" in name_to_check or "world_phase5" in name_to_check)
        and "residual" in name_to_check
    ) or (
        is_world_model and 
        config.get("world_model_params", {}).get("use_residual_features", False)
    ) or (
        # Digital-twin light runs: name contains "resid"
        ("resid" in name_to_check) or ("residual" in name_to_check)
    )
    
    # For world model experiments, residual features are typically enabled if "residual" in name
    if is_world_model and not is_phase4_residual:
        # Check if experiment name suggests residual features
        if "residual" in name_to_check:
            is_phase4_residual = True
            print(f"  Detected residual features from experiment name: {experiment_name}")
    
    # Final check: if it's a world model with "phase4" or "phase5" in name, assume residuals
    if is_world_model and not is_phase4_residual:
        if "phase4" in name_to_check or "phase5" in name_to_check or "world_phase" in name_to_check:
            # World model phase 4/5 experiments typically use residual features
            is_phase4_residual = True
            print(f"  Assuming residual features for world model phase 4/5 experiment: {experiment_name}")
    
    # Feature configs - must match training pipeline exactly
    from src.config import ResidualFeatureConfig
    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=is_phase4_residual,  # Enable residuals for phase 4 experiments
        use_temporal_on_physics=False,
        residual=ResidualFeatureConfig(
            enabled=is_phase4_residual,
            mode="per_engine",
            baseline_len=30,
            include_original=True,
        ) if is_phase4_residual else ResidualFeatureConfig(enabled=False),
    )

    # Mirror the feature-block configuration used during training (run_experiments.py).
    # If no explicit "features" section exists, we fall back to the legacy defaults
    # (temporal multi-scale features enabled with windows 5/10/30).
    features_cfg = config.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    use_temporal_features = features_cfg.get("use_multiscale_features", True)

    windows_short = tuple(ms_cfg.get("windows_short", (5, 10)))
    windows_medium = tuple(ms_cfg.get("windows_medium", ()))
    windows_long = tuple(ms_cfg.get("windows_long", (30,)))
    combined_long = windows_medium + windows_long

    temporal_cfg = TemporalFeatureConfig(
        base_cols=None,
        short_windows=windows_short,
        long_windows=combined_long if combined_long else (30,),
        add_rolling_mean=True,
        add_rolling_std=False,
        add_trend=True,
        add_delta=True,
        delta_lags=(5, 10),
    )
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=use_temporal_features,
        temporal=temporal_cfg,
    )
    
    # ------------------------------------------------------------------
    # Build evaluation data
    # ------------------------------------------------------------------
    # Physically-informed Transformer variants (transformer_encoder_phys_v2/v3/v4)
    # und ms_dt_v1/v2-Experimente (ms+DT) nutzen während des Trainings einen
    # kontinuierlichen Condition-Vektor + Digital-Twin-Residuals.
    # Wenn phys_features in summary.json persistiert wurden, verwenden wir sie
    # direkt; andernfalls rekonstruieren wir eine konsistente Default-Config
    # basierend auf dem Experimentnamen (muss exakt zu run_experiments passen).
    phys_features_cfg = config.get("phys_features", None)
    if phys_features_cfg is None:
        name_lower = (experiment_name or run_name).lower()
        if "transformer_encoder_phys_v4" in name_lower:
            phys_features_cfg = {
                "use_condition_vector": True,
                "use_twin_features": True,
                "twin_baseline_len": 30,
                "condition_vector_version": 3,
            }
        elif "transformer_encoder_phys_v3" in name_lower:
            phys_features_cfg = {
                "use_condition_vector": True,
                "use_twin_features": True,
                "twin_baseline_len": 30,
                "condition_vector_version": 3,
            }
        elif "transformer_encoder_phys_v2" in name_lower:
            phys_features_cfg = {
                "use_condition_vector": True,
                "use_twin_features": True,
                "twin_baseline_len": 30,
                # phys_v2 uses the original condition vector (version 2)
                "condition_vector_version": 2,
            }
        elif "transformer_encoder_ms_dt_v" in name_lower:
            # ms+DT-Experimente (ms_dt_v1/v2, inkl. damage_v2/v3/v3b/v3c) –
            # Standard-Einstellungen wie in experiment_configs.py
            phys_features_cfg = {
                "use_condition_vector": True,
                "use_twin_features": True,
                "twin_baseline_len": 30,
                "condition_vector_version": 3,
            }
    
    # Twin-/Residual-Flags explizit loggen, damit klar ist, was in Diagnostics
    # wirklich aktiv ist (Phase-4-Residuals vs. Twin-Residuals).
    use_twin_features_flag = False
    if phys_features_cfg is not None:
        use_twin_features_flag = bool(
            phys_features_cfg.get(
                "use_twin_features",
                phys_features_cfg.get("use_digital_twin_residuals", False),
            )
        )

    print(f"[2] Building evaluation data ({dataset_name} pipeline)...")
    # is_phase4_residual steuert nur die expliziten Physics-Residual-Features
    # (PhysicsFeatureConfig.use_residuals), NICHT die HealthyTwin-Residuals.
    print(f"  Phase-4 residual features enabled (physics_config): {is_phase4_residual}")
    print(f"  Digital-twin residuals enabled (phys_features):    {use_twin_features_flag}")
    if is_phase4_residual:
        print(f"  Residual config: mode={physics_config.residual.mode}, baseline_len={physics_config.residual.baseline_len}, include_original={physics_config.residual.include_original}")
    
    # Try to load scaler from experiment directory first to ensure consistency
    scaler_path = experiment_dir / "scaler.pkl"
    scaler_loaded = None
    if scaler_path.exists():
        try:
            import pickle
            with open(scaler_path, "rb") as f:
                scaler_loaded = pickle.load(f)
            print(f"  ✓ Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"  ⚠️  Could not load scaler from {scaler_path}: {e}")
    
    (
        X_test_scaled,
        y_true_eol,
        y_test_true,
        unit_ids_test,
        cond_ids_test,
        scaler_dict,
        feature_cols,
        df_test_fe,
    ) = build_eval_data(
        dataset_name=dataset_name,
        max_rul=max_rul,
        past_len=past_len,
        feature_config=feature_config,
        physics_config=physics_config,
        phys_features=phys_features_cfg,
    )

    # For damage_v3-style experiments, compute HI_phys_v3 on the test features
    # for diagnostics (true physics-based HI trajectory). We do NOT add this
    # column to feature_cols, so the encoder input dimensionality remains
    # identical to training.
    if "damage_v3" in experiment_name.lower():
        try:
            from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals

            print("  Computing HI_phys_v3 on test data for diagnostics...")
            hi_v3_test = compute_hi_phys_v3_from_residuals(
                df_test_fe,
                unit_col="UnitNumber",
                cycle_col="TimeInCycles",
            )
            df_test_fe["HI_phys_v3"] = hi_v3_test
            print(
                f"  HI_phys_v3 (test) stats: "
                f"min={float(np.nanmin(hi_v3_test)):.4f}, "
                f"max={float(np.nanmax(hi_v3_test)):.4f}, "
                f"mean={float(np.nanmean(hi_v3_test)):.4f}"
            )
        except Exception as e:
            print(f"  ⚠️  Could not compute HI_phys_v3 for diagnostics: {e}")

    # If we loaded a scaler from the experiment directory, prefer it over the
    # one fitted on-the-fly in build_eval_data to exactly mirror training –
    # BUT only if the feature dimensions match. Otherwise we keep the freshly
    # fitted diagnostics scaler to avoid 295/349-style mismatches.
    if scaler_loaded is not None:
        try:
            check_feature_dimensions(
                feature_cols=feature_cols,
                scaler=scaler_loaded,
                model=None,
                context="diagnostics (loaded scaler)",
            )
            print("  Using loaded scaler for consistency with training")
            scaler_dict = scaler_loaded
        except AssertionError as e:
            print(f"  ⚠️  Loaded scaler feature_dim mismatch, keeping diagnostics scaler: {e}")

    # Log high-level feature grouping for encoder/decoder design
    groups = group_feature_columns(feature_cols)
    print(
        "[Diagnostics] Feature groups: "
        f"total={len(feature_cols)}, "
        f"raw={len(groups['raw'])}, "
        f"ms={len(groups['ms'])}, "
        f"residual={len(groups['residual'])}, "
        f"cond={len(groups['cond'])}, "
        f"twin={len(groups['twin'])}"
    )
    
    # Verify feature count matches model expectations
    if is_world_model:
        expected_features = config.get("num_features")
        if expected_features is not None:
            if len(feature_cols) != expected_features:
                print(f"  ⚠️  WARNING: Feature count mismatch!")
                print(f"     Expected: {expected_features} (from config)")
                print(f"     Got: {len(feature_cols)} (from feature engineering)")
                print(f"     This may cause model loading errors.")
                print(f"     Attempting to continue anyway...")
            else:
                print(f"  ✓ Feature count matches model: {len(feature_cols)}")
    
    # Evaluate using the same helpers as in the training loops.
    # - World models (v2/v3): evaluate_world_model_v3_eol
    # - EOL RUL/HI Modelle (LSTM, Transformer, UniversalEncoderV3Attention, ...):
    #   evaluate_on_test_data
    print("[3] Evaluating EOL metrics via training helpers...")
    if is_world_model:
        test_metrics_diag = evaluate_world_model_v3_eol(
            model=model,
            df_test=df_test_fe,
            y_test_true=y_test_true,
            feature_cols=feature_cols,
            scaler_dict=scaler_dict,
            past_len=past_len,
            max_rul=max_rul,
            num_conditions=config.get("num_conditions", 1),
            device=device,
        )

        # Extract predictions and targets in the exact engine order used by evaluation
        rul_pred_full_np = np.array(test_metrics_diag["y_pred_eol"], dtype=float)
        y_true_eol = np.array(test_metrics_diag["y_true_eol"], dtype=float)
        unit_ids_test = np.arange(1, len(y_true_eol) + 1, dtype=int)

        # Extract errors for plots
        errors = rul_pred_full_np - y_true_eol
        
        # Get nasa_scores - evaluate_world_model_v3_eol uses compute_eol_errors_and_nasa internally
        # but doesn't return nasa_scores, so we compute them here for consistency
        from src.metrics import compute_eol_errors_and_nasa
        nasa_stats_computed = compute_eol_errors_and_nasa(y_true_eol, rul_pred_full_np, max_rul=max_rul)
        
        # Create eol_metrics_dict with same structure as EOL model path
        # Use metrics directly from evaluate_world_model_v3_eol (100% consistent with training)
        eol_metrics_dict = {
            "errors": errors,
            "mean_error": test_metrics_diag["Bias"],
            "std_error": float(np.std(errors)),
            "mean_abs_error": test_metrics_diag["MAE"],
            "median_error": float(np.median(errors)),
            "mse": test_metrics_diag.get("MSE", float(np.mean(errors ** 2))),
            "rmse": test_metrics_diag["RMSE"],
            "mae": test_metrics_diag["MAE"],
            "bias": test_metrics_diag["Bias"],
            "r2": test_metrics_diag.get("R2", 0.0),
            "nasa_scores": nasa_stats_computed["nasa_scores"],
            "nasa_mean": test_metrics_diag["nasa_score_mean"],  # Use from evaluate_world_model_v3_eol
            "nasa_sum": test_metrics_diag.get("nasa_score_sum", test_metrics_diag["nasa_score_mean"] * len(y_true_eol)),
            "nasa_median": float(np.median(nasa_stats_computed["nasa_scores"])),
            "num_engines": len(y_true_eol),
        }

        # Sanity log (same style as training)
        print(f"  Test RMSE (from evaluate_world_model_v3_eol): {test_metrics_diag['RMSE']:.2f} cycles")
        print(f"  Test MAE  (from evaluate_world_model_v3_eol): {test_metrics_diag['MAE']:.2f} cycles")
        print(f"  Test Bias (from evaluate_world_model_v3_eol): {test_metrics_diag['Bias']:.2f} cycles")
        print(f"  Test R²   (from evaluate_world_model_v3_eol): {test_metrics_diag.get('R2', 0.0):.4f}")
        print(f"  NASA Score (mean, from evaluate_world_model_v3_eol): {test_metrics_diag['nasa_score_mean']:.4f}")
        
        # Validation: Check shapes and values (for debugging)
        print(f"  EOL predictions shape: {rul_pred_full_np.shape}")
        print(f"  True EOL shape: {y_true_eol.shape}")
        print(f"  EOL predictions range: [{rul_pred_full_np.min():.2f}, {rul_pred_full_np.max():.2f}]")
        print(f"  True EOL range: [{y_true_eol.min():.2f}, {y_true_eol.max():.2f}]")
    else:
        # EOL model path (EOLFullLSTMWithHealth, UniversalEncoderV3Attention, EOLFullTransformerEncoder, ...)
        # Safety check: ensure feature dimensions match scaler and model expectations
        check_feature_dimensions(
            feature_cols=feature_cols,
            scaler=scaler_dict,
            model=model,
            context="diagnostics",
        )

        # ------------------------------------------------------------------
        # CRITICAL: restore runtime feature-index attributes for Transformer v2/v5
        # ------------------------------------------------------------------
        # Some transformer variants rely on `cond_feature_indices` (Cond_* positions)
        # and `sensor_feature_indices_for_norm` (residual sensor positions) which are
        # NOT part of the state_dict. Training code sets them at init time, but
        # loading from checkpoint does not. If missing, diagnostics can produce
        # different (often much worse) test metrics than the training script.
        try:
            from src.models.transformer_eol import EOLFullTransformerEncoder

            if isinstance(model, EOLFullTransformerEncoder):
                # Cond_* indices for continuous condition encoder
                cond_idx = [i for i, c in enumerate(feature_cols) if c.startswith("Cond_")]
                if getattr(model, "use_cond_encoder", False) and getattr(model, "cond_in_dim", 0) > 0:
                    if len(cond_idx) == int(getattr(model, "cond_in_dim", 0)):
                        model.cond_feature_indices = torch.as_tensor(cond_idx, dtype=torch.long, device=device)

                # Residual sensor indices for condition normalizer (v5)
                if getattr(model, "use_condition_normalizer", False):
                    groups = group_feature_columns(feature_cols)
                    residual_cols = set(groups.get("residual", []))
                    sens_idx = [i for i, c in enumerate(feature_cols) if c in residual_cols]
                    if len(sens_idx) > 0:
                        model.sensor_feature_indices_for_norm = torch.as_tensor(sens_idx, dtype=torch.long, device=device)
                        # Initialise condition normalizer dims if possible
                        if hasattr(model, "set_condition_normalizer_dims") and len(cond_idx) > 0:
                            model.set_condition_normalizer_dims(cond_dim=len(cond_idx), sensor_dim=len(sens_idx))
        except Exception as e:
            print(f"[diagnostics] WARNING: Could not restore transformer feature indices: {e}")

        test_metrics_diag = evaluate_on_test_data(
            model=model,
            df_test=df_test_fe,
            y_test_true=y_test_true,
            feature_cols=feature_cols,
            scaler=scaler_dict,
            past_len=past_len,
            max_rul=max_rul,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            device=device,
        )

        # evaluate_on_test_data arbeitet bereits auf EOL-Samples (ein Fenster pro Engine)
        # WICHTIG: Verwende die Metriken direkt von evaluate_on_test_data - keine Neuberechnung!
        rul_pred_full_np = np.array(test_metrics_diag["y_pred"], dtype=float)
        y_true_eol = np.array(test_metrics_diag["y_true"], dtype=float)
        unit_ids_test = np.array(test_metrics_diag.get("unit_ids"), dtype=int)

        # Verwende die Metriken direkt von evaluate_on_test_data (100% konsistent mit Training)
        pt = test_metrics_diag["pointwise"]
        nasa_pt = test_metrics_diag["nasa_pointwise"]
        
        # Extrahiere errors für Plots
        errors = rul_pred_full_np - y_true_eol
        
        # WICHTIG: Verwende Metriken DIREKT von evaluate_on_test_data - keine Neuberechnung!
        # evaluate_on_test_data verwendet intern compute_eol_errors_and_nasa und gibt bereits
        # alle korrekten Metriken zurück. Jede Neuberechnung könnte zu Inkonsistenzen führen!
        # 
        # evaluate_on_test_data gibt zurück:
        # - "pointwise": {mse, rmse, mae, bias, r2}
        # - "nasa_pointwise": {score_sum, score_mean}
        # - "y_true": capped true RUL
        # - "y_pred": capped predicted RUL
        # 
        # Diese Werte sind 100% identisch mit den Werten im Training!
        eol_metrics_dict = {
            "errors": errors,  # errors = rul_pred_full_np - y_true_eol (bereits aus capped Werten)
            "mean_error": pt["bias"],  # Direkt von evaluate_on_test_data
            "std_error": float(np.std(errors)),
            "mean_abs_error": pt["mae"],  # Direkt von evaluate_on_test_data
            "median_error": float(np.median(errors)),
            "mse": pt["mse"],  # Direkt von evaluate_on_test_data
            "rmse": pt["rmse"],  # Direkt von evaluate_on_test_data - WICHTIG!
            "mae": pt["mae"],  # Direkt von evaluate_on_test_data
            "bias": pt["bias"],  # Direkt von evaluate_on_test_data
            "r2": pt["r2"],  # Direkt von evaluate_on_test_data
            # NASA scores: evaluate_on_test_data berechnet diese bereits korrekt via compute_eol_errors_and_nasa
            # Wir müssen sie NICHT neu berechnen, sondern direkt verwenden!
            "nasa_scores": None,  # Nicht direkt verfügbar, aber nasa_mean/nasa_sum sind korrekt
            "nasa_mean": nasa_pt["score_mean"],  # Direkt von evaluate_on_test_data - WICHTIG!
            "nasa_sum": nasa_pt.get("score_sum", nasa_pt["score_mean"] * len(y_true_eol)),  # Direkt von evaluate_on_test_data
            "nasa_median": None,  # Nicht direkt verfügbar, aber nicht kritisch
            "num_engines": len(y_true_eol),
        }
        
        # Für Plots: Berechne nasa_scores nur wenn nötig (für Histogramme etc.)
        # Aber verwende für die Hauptmetriken die Werte von evaluate_on_test_data!
        from src.metrics import nasa_phm_score_single
        nasa_scores_for_plots = np.array([
            nasa_phm_score_single(true_rul, pred_rul)
            for true_rul, pred_rul in zip(y_true_eol, rul_pred_full_np)
        ])
        eol_metrics_dict["nasa_scores"] = nasa_scores_for_plots
        eol_metrics_dict["nasa_median"] = float(np.median(nasa_scores_for_plots))
        
        # SANITY CHECK: Vergleiche nasa_mean aus evaluate_on_test_data mit unserer Berechnung
        nasa_mean_computed = float(np.mean(nasa_scores_for_plots))
        nasa_mean_from_eval = nasa_pt["score_mean"]
        if abs(nasa_mean_computed - nasa_mean_from_eval) > 1e-6:
            print(f"  ⚠️  WARNING: NASA mean mismatch!")
            print(f"     From evaluate_on_test_data: {nasa_mean_from_eval:.6f}")
            print(f"     Computed from nasa_scores: {nasa_mean_computed:.6f}")
            print(f"     Difference: {abs(nasa_mean_computed - nasa_mean_from_eval):.6f}")
            print(f"     Using value from evaluate_on_test_data (training-consistent)")
        else:
            print(f"  ✓ NASA mean matches: {nasa_mean_from_eval:.6f}")

        print(f"  Test RMSE (from evaluate_on_test_data): {pt['rmse']:.2f} cycles")
        print(f"  Test MAE  (from evaluate_on_test_data): {pt['mae']:.2f} cycles")
        print(f"  Test Bias (from evaluate_on_test_data): {pt['bias']:.2f} cycles")
        print(f"  Test R²   (from evaluate_on_test_data): {pt['r2']:.4f}")
        print(f"  NASA Score (mean, from evaluate_on_test_data): {nasa_pt['score_mean']:.4f}")
        
        # Validation: Check shapes and values (for debugging)
        print(f"  EOL predictions shape: {rul_pred_full_np.shape}")
        print(f"  True EOL shape: {y_true_eol.shape}")
        print(f"  EOL predictions range: [{rul_pred_full_np.min():.2f}, {rul_pred_full_np.max():.2f}]")
        print(f"  True EOL range: [{y_true_eol.min():.2f}, {y_true_eol.max():.2f}]")

    # ------------------------------------------------------------------
    # Optional: \"Floor Correction\"–Analyse (Soft Landing bei ~45 Zyklen)
    # ------------------------------------------------------------------
    # Idee: Modell bleibt bei ~45 Zyklen \"stehen\". Wir simulieren, wie gut
    # die Metrik wäre, wenn wir diesen Bias per Postprocessing korrigieren.
    # Use rmse from eol_metrics_dict (works for both world model and EOL model paths)
    rmse = eol_metrics_dict["rmse"]
    
    y_pred_calib = rul_pred_full_np.copy()
    mask_floor = y_pred_calib < 65.0
    y_pred_calib[mask_floor] = y_pred_calib[mask_floor] - 45.0
    y_pred_calib = np.maximum(y_pred_calib, 0.0)

    errors_calib = y_pred_calib - y_true_eol
    mse_calib = float(np.mean(errors_calib ** 2))
    rmse_calib = float(np.sqrt(mse_calib))

    print(f"  [Analysis] Standard RMSE: {rmse:.2f} vs. Potential (floor-corrected) RMSE: {rmse_calib:.2f} cycles")
    
    # Build trajectories (use df_test_fe from build_eval_data)
    print("[5] Building HI trajectories (sliding window)...")
    trajectories = build_trajectories(
        df_test_fe=df_test_fe,
        feature_cols=feature_cols,
        scaler_dict=scaler_dict,
        past_len=past_len,
        model=model,
        device=device,
        y_true_eol=y_true_eol,
        rul_pred_full_np=rul_pred_full_np,
        unit_ids_test=unit_ids_test,
        max_rul=max_rul,
        is_world_model=is_world_model,
        is_world_model_v3=is_world_model_v3,
    )
    
    print(f"  Built trajectories for {len(trajectories)} engines")
    
    # Check if any trajectories have damage HI (check all trajectories first)
    trajectories_with_damage = [t for t in trajectories if t.hi_damage is not None and len(t.hi_damage) > 0]
    has_damage_hi = len(trajectories_with_damage) > 0
    if has_damage_hi:
        print(f"  ✓ Damage HI trajectories detected: {len(trajectories_with_damage)}/{len(trajectories)} engines have damage HI")
        print(f"    Will create separate damage plot")
    else:
        print(f"  ⚠️  No damage HI trajectories found (model may not have damage_head)")
    
    # Select degraded engines
    print("[6] Selecting degraded engines...")
    selected_trajectories = select_degraded_engines(
        y_true_eol=y_true_eol,
        unit_ids_test=unit_ids_test,
        trajectories=trajectories,
        rul_threshold=40.0,
        max_engines=10,
    )
    
    print(f"  Selected {len(selected_trajectories)} degraded engines")
    
    # Check if selected trajectories have damage HI
    selected_with_damage = [t for t in selected_trajectories if t.hi_damage is not None and len(t.hi_damage) > 0]
    if has_damage_hi and len(selected_with_damage) > 0:
        print(f"  ✓ {len(selected_with_damage)}/{len(selected_trajectories)} selected engines have damage HI - will plot")
    elif has_damage_hi and len(selected_with_damage) == 0:
        print(f"  ⚠️  Warning: Damage HI available but none in selected engines - damage plot may be empty")
    
    # Generate plots (save directly in experiment directory, not in diagnostics subfolder)
    print("[7] Generating publication plots...")
    
    # 1. Error histogram
    errors = rul_pred_full_np - y_true_eol
    plot_error_histogram(
        errors=errors,
        out_path=experiment_dir / "error_hist.png",
        title=f"{dataset_name} Error Histogram",
    )
    
    # 2. True vs Pred scatter
    plot_true_vs_pred_scatter(
        y_true=y_true_eol,
        y_pred=rul_pred_full_np,
        out_path=experiment_dir / "true_vs_pred.png",
        max_rul=max_rul,
        title=f"{dataset_name} True vs Predicted RUL",
    )
    
    # 3. HI + RUL trajectories (always create if we have trajectories)
    # Note: We don't need to check for trajectory heads here since we build trajectories
    # using sliding window approach which works for all model types
    
    # For EOL RUL/HI experiments we always have HI trajectories via build_trajectories,
    # even if there is no dedicated "trajectory head" in the model.
    # Therefore, rely on the presence of trajectories instead of has_traj_head.
    if len(selected_trajectories) > 0:
        plot_hi_rul_trajectories(
            trajectories=selected_trajectories,
            out_path=experiment_dir / "hi_rul_10_degraded.png",
            title=f"{dataset_name} Health Index + RUL Trajectories – 10 degraded engines",
            max_engines=10,
        )
        print(f"  Saved HI+RUL trajectory plots for {len(selected_trajectories)} engines")
        
        # 4. Separate HI Damage trajectory plot (if damage HI is available)
        # Check again on selected trajectories to ensure we have damage HI in the selected set
        selected_with_damage = [t for t in selected_trajectories if t.hi_damage is not None and len(t.hi_damage) > 0]
        if len(selected_with_damage) > 0:
            print(f"  Creating HI Damage plot for {len(selected_with_damage)} engines with damage HI...")
            plot_hi_damage_trajectories(
                trajectories=selected_trajectories,
                out_path=experiment_dir / "hi_damage_10_degraded.png",
                title=f"{dataset_name} Health Index Damage Trajectories – 10 degraded engines",
                max_engines=10,
            )
            print(f"  ✓ Saved HI Damage trajectory plots to {experiment_dir / 'hi_damage_10_degraded.png'}")

            # Additional diagnostics for HI_phys_v3-based experiments:
            # 1) True HI_phys_v3 trajectories for several engines.
            # 2) True vs predicted HI_phys_v3 (using damage-head HI as prediction),
            #    plus per-engine RMSE summary between HI_phys_v3 and HI_damage.
            #    Trigger this block for v3/v4/v5 damage encoders.
            exp_lower = experiment_name.lower()
            if "HI_phys_v3" in df_test_fe.columns and (
                "damage_v3" in exp_lower
                or "damage_v4" in exp_lower
                or "damage_v5" in exp_lower
            ):
                print("  Creating HI_phys_v3 diagnostics plots (true + true vs pred)...")
                plot_hi_phys_v3_true_trajectories(
                    df_test_fe=df_test_fe,
                    trajectories=selected_trajectories,
                    out_path=experiment_dir / "hi_phys_v3_true_10_degraded.png",
                    title=f"{dataset_name} HI_phys_v3 true trajectories – 10 degraded engines",
                    max_engines=10,
                )
                plot_hi_phys_v3_true_vs_pred(
                    df_test_fe=df_test_fe,
                    trajectories=selected_trajectories,
                    out_path=experiment_dir / "hi_phys_v3_true_vs_pred_10_degraded.png",
                    metrics_path=experiment_dir / "hi_damage_metrics.json",
                    title=f"{dataset_name} HI_phys_v3 true vs pred – 10 degraded engines",
                    max_engines=10,
                )

                # HI_cal_v2 (encoder v4) diagnostics, if trajectories contain HI_cal
                traj_with_hi_cal = [
                    t for t in selected_trajectories
                    if getattr(t, "hi_cal", None) is not None and len(t.hi_cal) > 0
                ]
                if len(traj_with_hi_cal) > 0:
                    print("  Creating HI_cal_v2 (v4) diagnostics plots...")
                    plot_hi_cal_v2_trajectories(
                        trajectories=traj_with_hi_cal,
                        out_path=experiment_dir / "hi_cal_v2_10_degraded.png",
                        title=f"{dataset_name} HI_cal_v2 (v4) trajectories – 10 degraded engines",
                        max_engines=10,
                    )
                    plot_hi_cal_v2_vs_hi_phys_v3(
                        df_test_fe=df_test_fe,
                        trajectories=traj_with_hi_cal,
                        out_path=experiment_dir / "hi_cal_v2_vs_hi_phys_v3_eol.png",
                        title=f"{dataset_name} HI_cal_v2 (v4) vs HI_phys_v3 at EOL",
                    )
        elif has_damage_hi:
            print(f"  ⚠️  Skipping HI Damage plot: damage HI exists but not in selected degraded engines")
        else:
            print(f"  ℹ️  Skipping HI Damage plot: model does not have damage_head")
    else:
        print(f"  Skipping HI+RUL trajectory plot (no trajectories available)")
    
    # Save metrics as JSON (convert numpy arrays to lists for JSON serialization)
    print("[8] Saving metrics...")
    metrics_path = experiment_dir / "eol_metrics.json"

    # Stamp metrics with code provenance so it's obvious when a run folder contains
    # stale diagnostics created by older code (e.g., before engine/target alignment fixes).
    def _get_git_sha() -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip() or None
        except Exception:
            return None
    
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    generated_git_sha = _get_git_sha()
    
    # Convert numpy arrays to lists for JSON serialization
    eol_metrics_json = {}
    for key, value in eol_metrics_dict.items():
        if isinstance(value, np.ndarray):
            eol_metrics_json[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            eol_metrics_json[key] = float(value)
        else:
            eol_metrics_json[key] = value

    # Add provenance metadata (does not affect downstream numeric consumers).
    eol_metrics_json["_meta"] = {
        "generated_at_utc": generated_at_utc,
        "generated_git_sha": generated_git_sha,
        "dataset": dataset_name,
        "run_name": run_name,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(eol_metrics_json, f, indent=2)

    # ------------------------------------------------------------------
    # Keep summary.json metrics consistent with the diagnostics we just wrote
    # ------------------------------------------------------------------
    # Rationale: users often compare `summary.json:test_metrics` with `eol_metrics.json`
    # and plots. If diagnostics is re-run later (or older runs are opened), summary.json
    # can become stale. We persist a canonical "diagnostics_test_metrics" block and also
    # update the top-level test_* fields to match.
    try:
        summary_path = experiment_dir / "summary.json"
        if summary_path.exists():
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary_obj = {}

        diag_test_metrics = {
            "rmse": float(eol_metrics_dict["rmse"]),
            "mae": float(eol_metrics_dict["mae"]),
            "bias": float(eol_metrics_dict["bias"]),
            "r2": float(eol_metrics_dict["r2"]),
            "nasa_mean": float(eol_metrics_dict["nasa_mean"]),
            "nasa_sum": float(eol_metrics_dict["nasa_sum"]),
            "num_engines": int(eol_metrics_dict["num_engines"]),
        }

        summary_obj["diagnostics_test_metrics"] = diag_test_metrics
        summary_obj["diagnostics_meta"] = {
            "generated_at_utc": generated_at_utc,
            "generated_git_sha": generated_git_sha,
        }

        # Update "official" test_metrics + flat keys to match diagnostics (single source of truth).
        summary_obj["test_metrics"] = dict(diag_test_metrics)
        summary_obj["test_rmse"] = float(diag_test_metrics["rmse"])
        summary_obj["test_mae"] = float(diag_test_metrics["mae"])
        summary_obj["test_bias"] = float(diag_test_metrics["bias"])
        summary_obj["test_r2"] = float(diag_test_metrics["r2"])
        summary_obj["test_nasa_mean"] = float(diag_test_metrics["nasa_mean"])

        summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
        print(f"  ✓ Updated summary.json test metrics for consistency: {summary_path}")
    except Exception as e:
        print(f"  ⚠️  WARNING: failed to update summary.json with diagnostics metrics: {e}")
    
    print(f"\n✅ Diagnostics complete for {dataset_name}!")
    print(f"  Plots saved to: {experiment_dir}")
    print(f"  Metrics saved to: {metrics_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run diagnostics for a trained experiment"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results",
        help="Base directory for experiments (default: results)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset name (e.g., FD004)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Experiment name (e.g., fd004_phase3_universal_v2_ms_cnn_d96)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Parse device
    if args.device is None:
        device = None  # Auto-detect
    elif args.device.lower() == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Run diagnostics
    run_diagnostics_for_run(
        exp_dir=args.exp_dir,
        dataset_name=args.dataset,
        run_name=args.run_name,
        device=device,
    )

