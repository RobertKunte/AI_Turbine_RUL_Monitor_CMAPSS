"""
Aggregate experiment results from all summary.json files into a single CSV.

This script scans the results/ directory for all summary.json files and
creates a unified CSV table for analysis.

Usage:
    python scripts/aggregate_experiment_results.py
    python scripts/aggregate_experiment_results.py --output results/all_experiments_summary.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_all_summaries(results_dir: Path = Path("results")) -> list[dict]:
    """
    Recursively find all summary.json files and load them.
    
    Args:
        results_dir: Root directory to search
    
    Returns:
        List of summary dictionaries
    """
    summaries = []
    
    for summary_path in results_dir.rglob("summary.json"):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
                # Add path info for reference
                summary["_summary_path"] = str(summary_path.relative_to(results_dir))
                summaries.append(summary)
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")
            continue
    
    return summaries


def create_summary_table(summaries: list[dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from summaries with standardized columns.
    
    Args:
        summaries: List of summary dictionaries
    
    Returns:
        DataFrame with all experiment results
    """
    if not summaries:
        return pd.DataFrame()
    
    # Define key columns to extract (in order)
    key_columns = [
        "experiment_name",
        "dataset",
        "phase",
        "encoder_type",
        # Encoder parameters
        "hidden_dim",  # LSTM
        "d_model",  # Transformer
        "num_layers",
        "dropout",
        "nhead",  # Transformer
        "dim_feedforward",  # Transformer
        "bidirectional",  # LSTM
        # Phase 2 parameters
        "use_condition_embedding",
        "cond_emb_dim",
        "smooth_hi_weight",
        # Loss parameters
        "rul_beta",
        "health_loss_weight",
        "mono_late_weight",
        "mono_global_weight",
        "hi_condition_calib_weight",
        # Validation metrics
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_r2",
        "val_nasa_mean",
        # Test metrics
        "test_rmse",
        "test_mae",
        "test_bias",
        "test_r2",
        "test_nasa_mean",
    ]
    
    # Extract data
    rows = []
    for summary in summaries:
        row = {}
        for col in key_columns:
            row[col] = summary.get(col, None)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by dataset, then encoder_type, then experiment_name
    df = df.sort_values(["dataset", "encoder_type", "experiment_name"]).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results into CSV")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory to search for summary.json files (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/all_experiments_summary.csv"),
        help="Output CSV file path (default: results/all_experiments_summary.csv)",
    )
    
    args = parser.parse_args()
    
    print(f"Scanning {args.results_dir} for summary.json files...")
    summaries = find_all_summaries(args.results_dir)
    
    if not summaries:
        print(f"No summary.json files found in {args.results_dir}")
        return
    
    print(f"Found {len(summaries)} experiment summaries")
    
    # Create DataFrame
    df = create_summary_table(summaries)
    
    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved summary table to {args.output}")
    print(f"\nTable shape: {df.shape}")
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # Print summary statistics
    if "test_rmse" in df.columns and "test_nasa_mean" in df.columns:
        print("\n" + "=" * 80)
        print("Summary Statistics by Dataset and Encoder Type")
        print("=" * 80)
        for dataset in df["dataset"].unique():
            print(f"\n{dataset}:")
            dataset_df = df[df["dataset"] == dataset]
            for encoder_type in dataset_df["encoder_type"].unique():
                enc_df = dataset_df[dataset_df["encoder_type"] == encoder_type]
                print(f"  {encoder_type.upper()}:")
                print(f"    Count: {len(enc_df)}")
                if len(enc_df) > 0:
                    print(f"    Test RMSE: {enc_df['test_rmse'].mean():.2f} ± {enc_df['test_rmse'].std():.2f}")
                    print(f"    Test NASA Mean: {enc_df['test_nasa_mean'].mean():.2f} ± {enc_df['test_nasa_mean'].std():.2f}")
                    best_rmse = enc_df.loc[enc_df['test_rmse'].idxmin()]
                    best_nasa = enc_df.loc[enc_df['test_nasa_mean'].idxmin()]
                    print(f"    Best RMSE: {best_rmse['test_rmse']:.2f} ({best_rmse['experiment_name']})")
                    print(f"    Best NASA: {best_nasa['test_nasa_mean']:.2f} ({best_nasa['experiment_name']})")


if __name__ == "__main__":
    main()

