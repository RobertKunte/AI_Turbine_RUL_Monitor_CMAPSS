"""
Detection Horizon Analysis for Health Index.

Computes when HI crosses below thresholds (0.8, 0.5, 0.2, 0.1) for all engines
and provides statistics (mean, std, min, max) over all engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data_loading import load_cmapps_subset
from src.additional_features import create_physical_features, create_all_features
from src.feature_config import FEATURE_CONFIG
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTMWithHealth,
)
from src.eval_utils import forward_rul_only


def collect_hi_rul_per_engine(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Collect HI and RUL sequences per engine from a dataloader.
    
    Args:
        model: Trained model (should return (rul_pred, health_last, health_seq))
        dataloader: DataLoader with sequences
        device: torch.device
    
    Returns:
        Dictionary mapping unit_id -> {
            "rul": np.array([...]),      # sorted by cycle, decreasing (high RUL -> 0)
            "hi_pred": np.array([...]),   # predicted HI aligned with rul
        }
    """
    model.eval()
    model.to(device)
    
    per_engine_data: Dict[int, Dict[str, List[float]]] = {}
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) >= 3:
                X_batch, y_batch, unit_ids_batch = batch[:3]
            else:
                X_batch, y_batch = batch
                unit_ids_batch = None
            
            X_batch = X_batch.to(device)
            
            # Get model predictions
            model_output = model(X_batch)
            if isinstance(model_output, (tuple, list)) and len(model_output) >= 3:
                rul_pred, health_last, health_seq = model_output
            elif isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
                rul_pred, health_last = model_output
                # Create dummy health_seq from health_last
                seq_len = X_batch.size(1)
                health_seq = health_last.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Single-task model - skip
                continue
            
            # Prepare sequences
            if health_seq.dim() == 3:
                health_seq = health_seq.squeeze(-1)  # [batch, seq_len]
            
            # Build RUL sequence for each sample
            rul_last = y_batch.view(-1, 1).to(device)  # [batch, 1]
            seq_len = X_batch.size(1)
            offsets = torch.arange(seq_len - 1, -1, -1, device=device).float()  # [seq_len]
            rul_seq_batch = rul_last + offsets  # [batch, seq_len]
            
            # Convert to numpy
            health_seq_np = health_seq.cpu().numpy()  # [batch, seq_len]
            rul_seq_np = rul_seq_batch.cpu().numpy()  # [batch, seq_len]
            
            if unit_ids_batch is not None:
                unit_ids_np = unit_ids_batch.numpy()
            else:
                # Create dummy unit IDs
                unit_ids_np = np.arange(len(y_batch))
            
            # Collect per engine
            for i in range(len(y_batch)):
                unit_id = int(unit_ids_np[i])
                
                if unit_id not in per_engine_data:
                    per_engine_data[unit_id] = {"rul": [], "hi_pred": []}
                
                # Extract sequence for this sample
                hi_seq_i = health_seq_np[i]  # [seq_len]
                rul_seq_i = rul_seq_np[i]    # [seq_len]
                
                # Sort by RUL descending (early cycles first, high RUL -> low RUL)
                order = np.argsort(-rul_seq_i)
                hi_sorted = hi_seq_i[order]
                rul_sorted = rul_seq_i[order]
                
                # Append to engine data
                per_engine_data[unit_id]["rul"].extend(rul_sorted.tolist())
                per_engine_data[unit_id]["hi_pred"].extend(hi_sorted.tolist())
    
    # Convert lists to numpy arrays and sort each engine's data by RUL
    result = {}
    for unit_id, data in per_engine_data.items():
        rul_all = np.array(data["rul"])
        hi_all = np.array(data["hi_pred"])
        
        # Sort by RUL descending (high RUL -> low RUL)
        order = np.argsort(-rul_all)
        result[unit_id] = {
            "rul": rul_all[order],
            "hi_pred": hi_all[order],
        }
    
    return result


def compute_detection_horizon_stats(
    per_engine_data: Dict[int, Dict[str, np.ndarray]],
    thresholds: Tuple[float, ...] = (0.8, 0.5, 0.2, 0.1),
) -> Dict[float, Dict[str, float | int | np.ndarray]]:
    """
    Compute detection horizon statistics for HI threshold crossings.
    
    Args:
        per_engine_data: Output from collect_hi_rul_per_engine
        thresholds: Iterable of HI thresholds
    
    Returns:
        Dictionary mapping threshold -> {
            "num_engines": int,
            "mean_rul": float,
            "std_rul": float,
            "min_rul": float,
            "max_rul": float,
            "values": np.ndarray,  # All detection horizons for this threshold
        }
    """
    results = {}
    
    for thr in thresholds:
        horizons = []  # RUL at time of first threshold crossing
        
        for unit_id, data in per_engine_data.items():
            rul = np.asarray(data["rul"])         # [T]
            hi = np.asarray(data["hi_pred"])       # [T]
            
            # Apply running-min for monotonicity
            run_min = hi.copy()
            for t in range(len(run_min) - 2, -1, -1):
                run_min[t] = min(run_min[t], run_min[t + 1])
            
            # Find first index where HI <= threshold
            below_indices = np.where(run_min <= thr)[0]
            if below_indices.size == 0:
                continue  # Never crossed, skip this engine
            
            first_idx = below_indices[0]
            detection_rul = float(rul[first_idx])  # RUL in cycles
            horizons.append(detection_rul)
        
        if len(horizons) == 0:
            continue
        
        horizons = np.array(horizons)
        results[thr] = {
            "num_engines": len(horizons),
            "mean_rul": float(horizons.mean()),
            "std_rul": float(horizons.std()),
            "min_rul": float(horizons.min()),
            "max_rul": float(horizons.max()),
            "values": horizons,
        }
    
    return results


def plot_detection_horizon_histograms(
    results: Dict[float, Dict[str, float | int | np.ndarray]],
    dataset_name: str,
    output_dir: Path,
) -> None:
    """
    Plot histograms of detection horizons for each threshold.
    
    Args:
        results: Output from compute_detection_horizon_stats
        dataset_name: Dataset identifier (e.g., "FD001")
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for thr, stats in results.items():
        if "values" not in stats:
            continue
        
        values = stats["values"]
        
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("RUL at threshold crossing (cycles)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(
            f"{dataset_name}: Detection Horizon for HI <= {thr}\n"
            f"Mean: {stats['mean_rul']:.1f} cycles, Std: {stats['std_rul']:.1f} cycles",
            fontsize=14
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = output_dir / f"{dataset_name.lower()}_detection_horizon_hi_{thr:.1f}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved histogram to: {plot_path}")


def print_detection_horizon_summary(
    results: Dict[float, Dict[str, float | int | np.ndarray]],
    dataset_name: str,
) -> None:
    """
    Print detection horizon summary statistics.
    
    Args:
        results: Output from compute_detection_horizon_stats
        dataset_name: Dataset identifier (e.g., "FD001")
    """
    print("\n" + "=" * 80)
    print(f"[{dataset_name}] Detection Horizon Summary (RUL at HI crossing)")
    print("=" * 80)
    print("Interpretation: RUL remaining when HI first crosses below threshold")
    print("(using running-min HI for monotonicity)")
    print("-" * 80)
    
    # Create summary DataFrame
    records = []
    for thr in sorted(results.keys(), reverse=True):
        stats = results[thr]
        records.append({
            "threshold": thr,
            "num_engines": stats["num_engines"],
            "mean_rul": stats["mean_rul"],
            "std_rul": stats["std_rul"],
            "min_rul": stats["min_rul"],
            "max_rul": stats["max_rul"],
        })
    
    if records:
        df = pd.DataFrame(records)
        print(df.to_string(index=False))
    else:
        print("No detection horizons found (HI never crosses thresholds)")
    
    print("=" * 80)


def main(
    dataset: str = "FD001",
    model_type: str = "multi_task",
    checkpoint_path: Optional[str] = None,
    data_dir: str = "../data",
    results_dir: str = "../results/health_index",
    device: str = "cpu",
) -> None:
    """
    Main entry point for detection horizon analysis.
    
    Args:
        dataset: Dataset identifier (FD001, FD002, FD003, FD004)
        model_type: Model type ("multi_task" or "single_task")
        checkpoint_path: Path to model checkpoint (if None, searches in results_dir)
        data_dir: Directory containing CMAPSS data files
        results_dir: Directory for results and plots
        device: torch.device string
    """
    device = torch.device(device)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Detection Horizon Analysis for {dataset}")
    print("=" * 80)
    
    # Load data
    print(f"\n[{dataset}] Loading data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        data_dir=data_dir,
        max_rul=None,
    )
    
    # Create features
    print(f"[{dataset}] Creating features...")
    df_train = create_physical_features(
        df_train,
        physics_config=None,  # Use default
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    df_train = create_all_features(
        df_train,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        config=FEATURE_CONFIG,
        inplace=False,
    )
    
    # Get feature columns
    exclude = ["UnitNumber", "TimeInCycles", "RUL", "MaxTime", "Setting1", "Setting2", "Setting3"]
    if not FEATURE_CONFIG.use_condition_id:
        exclude.append("ConditionID")
    feature_cols = [c for c in df_train.columns if c not in exclude]
    
    # Build sequences
    print(f"[{dataset}] Building sequences...")
    X_full, y_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=30,
        max_rul=125,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
    )
    
    # Create dataloader (use full dataset for analysis)
    print(f"[{dataset}] Creating dataloader...")
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    N, T, F = X_full.shape
    X_flat = X_full.numpy().reshape(-1, F)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = torch.from_numpy(X_scaled.reshape(N, T, F)).float()
    
    # Create dataset and dataloader
    dataset_obj = TensorDataset(X_scaled, y_full, unit_ids_full, cond_ids_full)
    dataloader = DataLoader(dataset_obj, batch_size=256, shuffle=False)
    
    # Load model
    print(f"[{dataset}] Loading model...")
    if checkpoint_path is None:
        # Search for checkpoint in results_dir
        checkpoint_path = results_dir.parent / "eol_full_lstm" / f"eol_full_lstm_best_{dataset.lower()}.pt"
        if not checkpoint_path.exists():
            checkpoint_path = results_dir.parent / "eol_full_lstm" / f"eol_full_lstm_best_{dataset.lower()}_multitask.pt"
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = EOLFullLSTMWithHealth(
        input_dim=len(feature_cols),
        hidden_size=50,
        num_layers=2,
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"  Loaded model from: {checkpoint_path}")
    
    # Collect HI and RUL per engine
    print(f"\n[{dataset}] Collecting HI and RUL sequences per engine...")
    per_engine_data = collect_hi_rul_per_engine(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    
    print(f"  Collected data for {len(per_engine_data)} engines")
    
    # Compute detection horizon statistics
    print(f"\n[{dataset}] Computing detection horizon statistics...")
    thresholds = (0.8, 0.5, 0.2, 0.1)
    results = compute_detection_horizon_stats(
        per_engine_data=per_engine_data,
        thresholds=thresholds,
    )
    
    # Print summary
    print_detection_horizon_summary(results, dataset)
    
    # Plot histograms
    print(f"\n[{dataset}] Plotting histograms...")
    plot_detection_horizon_histograms(
        results=results,
        dataset_name=dataset,
        output_dir=results_dir / dataset.lower(),
    )
    
    # Save summary to CSV
    records = []
    for thr in sorted(results.keys(), reverse=True):
        stats = results[thr]
        records.append({
            "threshold": thr,
            "num_engines": stats["num_engines"],
            "mean_rul": stats["mean_rul"],
            "std_rul": stats["std_rul"],
            "min_rul": stats["min_rul"],
            "max_rul": stats["max_rul"],
        })
    
    if records:
        df = pd.DataFrame(records)
        csv_path = results_dir / f"{dataset.lower()}_detection_horizon_hi.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[{dataset}] Saved summary to: {csv_path}")
    
    print("\n" + "=" * 80)
    print(f"Detection Horizon Analysis Complete for {dataset}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection Horizon Analysis for Health Index")
    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset identifier",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="multi_task",
        choices=["multi_task", "single_task"],
        help="Model type",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if None, searches in results_dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing CMAPSS data files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/health_index",
        help="Directory for results and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch.device (cpu or cuda)",
    )
    
    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device,
    )

