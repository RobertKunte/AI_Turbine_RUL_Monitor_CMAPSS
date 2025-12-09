"""
Aggregate Ablation Study Results.

This module collects results from all ablation experiments and generates:
- Master CSV with all metrics
- Markdown table for README
- Per-axis analysis (effect of horizons, residuals, etc.)
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import pandas as pd
import numpy as np


def load_ablation_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all ablation experiment results into a DataFrame.
    
    Args:
        results_dir: Base directory containing ablation experiments
                    (e.g., results/fd002/ablation_phase6/)
    
    Returns:
        DataFrame with one row per experiment
    """
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            continue
        
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            # Extract experiment name components
            exp_name = exp_dir.name
            
            # Parse experiment name: A1_full_H40_L5_residual_late
            parts = exp_name.split("_")
            variant = parts[0]  # A1
            horizon = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("H") else None
            eol_weight = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("L") else None
            use_residuals = "residual" in exp_name
            fusion_mode = parts[-1] if len(parts) > 1 else "late"
            
            # Extract metrics
            test_metrics = summary.get("test_metrics", {})
            
            row = {
                "experiment_name": exp_name,
                "variant": variant,
                "horizon": horizon,
                "eol_loss_weight": eol_weight,
                "use_residuals": use_residuals,
                "fusion_mode": fusion_mode,
                "rmse": test_metrics.get("rmse", np.nan),
                "mae": test_metrics.get("mae", np.nan),
                "bias": test_metrics.get("bias", np.nan),
                "r2": test_metrics.get("r2", np.nan),
                "nasa_sum": test_metrics.get("nasa_sum", np.nan),
                "nasa_mean": test_metrics.get("nasa_mean", np.nan),
                "num_engines": test_metrics.get("num_engines", np.nan),
                "best_val_loss": summary.get("best_val_loss", np.nan),
                "best_epoch": summary.get("best_epoch", np.nan),
            }
            
            experiments.append(row)
            
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
            continue
    
    if not experiments:
        return pd.DataFrame()
    
    df = pd.DataFrame(experiments)
    return df


def generate_master_table(df: pd.DataFrame, out_path: Path):
    """Generate master CSV table."""
    df.to_csv(out_path, index=False)
    print(f"Saved master table to {out_path}")


def generate_markdown_table(df: pd.DataFrame, out_path: Path):
    """Generate Markdown table for README."""
    # Sort by RMSE (best first)
    df_sorted = df.sort_values("rmse").copy()
    
    # Select key columns
    display_cols = [
        "variant", "horizon", "eol_loss_weight", "use_residuals", "fusion_mode",
        "rmse", "mae", "bias", "r2", "nasa_mean"
    ]
    
    df_display = df_sorted[display_cols].copy()
    
    # Format numeric columns
    df_display["rmse"] = df_display["rmse"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    df_display["mae"] = df_display["mae"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    df_display["bias"] = df_display["bias"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    df_display["r2"] = df_display["r2"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    df_display["nasa_mean"] = df_display["nasa_mean"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    df_display["use_residuals"] = df_display["use_residuals"].apply(lambda x: "Yes" if x else "No")
    
    # Generate markdown
    lines = ["## Ablation Study Results\n"]
    lines.append("| Variant | Horizon | EOL Weight | Residuals | Fusion | RMSE | MAE | Bias | R² | NASA Mean |")
    lines.append("|---------|---------|------------|-----------|--------|------|-----|------|----|-----------|")
    
    for _, row in df_display.iterrows():
        lines.append(
            f"| {row['variant']} | {row['horizon']} | {row['eol_loss_weight']} | "
            f"{row['use_residuals']} | {row['fusion_mode']} | {row['rmse']} | "
            f"{row['mae']} | {row['bias']} | {row['r2']} | {row['nasa_mean']} |"
        )
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved Markdown table to {out_path}")


def analyze_per_axis(df: pd.DataFrame, out_dir: Path):
    """Generate per-axis analysis."""
    analyses = {}
    
    # Effect of horizon
    if "horizon" in df.columns:
        horizon_analysis = df.groupby("horizon").agg({
            "rmse": ["mean", "std", "min"],
            "nasa_mean": ["mean", "std", "min"],
        }).round(2)
        analyses["horizon"] = horizon_analysis
        print("\nEffect of Horizon:")
        print(horizon_analysis)
    
    # Effect of EOL weight
    if "eol_loss_weight" in df.columns:
        eol_analysis = df.groupby("eol_loss_weight").agg({
            "rmse": ["mean", "std", "min"],
            "nasa_mean": ["mean", "std", "min"],
        }).round(2)
        analyses["eol_weight"] = eol_analysis
        print("\nEffect of EOL Loss Weight:")
        print(eol_analysis)
    
    # Effect of residuals
    if "use_residuals" in df.columns:
        residual_analysis = df.groupby("use_residuals").agg({
            "rmse": ["mean", "std", "min"],
            "nasa_mean": ["mean", "std", "min"],
        }).round(2)
        analyses["residuals"] = residual_analysis
        print("\nEffect of Residual Features:")
        print(residual_analysis)
    
    # Effect of fusion mode
    if "fusion_mode" in df.columns:
        fusion_analysis = df.groupby("fusion_mode").agg({
            "rmse": ["mean", "std", "min"],
            "nasa_mean": ["mean", "std", "min"],
        }).round(2)
        analyses["fusion"] = fusion_analysis
        print("\nEffect of Fusion Mode:")
        print(fusion_analysis)
    
    # Effect of variant
    if "variant" in df.columns:
        variant_analysis = df.groupby("variant").agg({
            "rmse": ["mean", "std", "min"],
            "nasa_mean": ["mean", "std", "min"],
        }).round(2)
        analyses["variant"] = variant_analysis
        print("\nEffect of Architecture Variant:")
        print(variant_analysis)
    
    # Save analyses
    for name, analysis in analyses.items():
        analysis_path = out_dir / f"analysis_{name}.csv"
        analysis.to_csv(analysis_path)
        print(f"Saved {name} analysis to {analysis_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate ablation study results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Base directory for ablation experiments (e.g., results/fd002/ablation_phase6/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: same as results-dir)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_dir}...")
    df = load_ablation_results(results_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Generate outputs
    generate_master_table(df, output_dir / "ablation_results_master.csv")
    generate_markdown_table(df, output_dir / "ablation_results_table.md")
    analyze_per_axis(df, output_dir)
    
    print(f"\n✅ Aggregation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

