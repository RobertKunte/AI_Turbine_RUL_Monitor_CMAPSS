"""
Generate Markdown Summary Table from Phase-1 Cross-Dataset Baseline CSV

This script reads the phase1_cross_dataset_baseline.csv and generates
a Markdown table for documentation.

Usage:
    python scripts/generate_phase1_summary_table.py
"""

import sys
from pathlib import Path
import pandas as pd

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_markdown_table(csv_path: Path, output_path: Path = None):
    """Generate Markdown table from CSV results."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Generate Markdown table
    print("=" * 80)
    print("Phase-1 Cross-Dataset Baseline - Summary Table")
    print("=" * 80)
    print()
    print("| Dataset | Test RMSE | Test MAE | Test Bias | Test R² | Test NASA Mean |")
    print("|---------|-----------|----------|-----------|---------|----------------|")
    
    for _, row in df.iterrows():
        print(
            f"| {row['dataset']:7} | "
            f"{row['test_rmse']:9.2f} | "
            f"{row['test_mae']:8.2f} | "
            f"{row['test_bias']:9.2f} | "
            f"{row['test_r2']:7.4f} | "
            f"{row['test_nasa_mean']:14.2f} |"
        )
    
    print()
    print("=" * 80)
    
    # Write to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Phase-1 Cross-Dataset Baseline Results\n\n")
            f.write("This table shows the results of applying the best Phase-1 Health-Head ")
            f.write("hyperparameters from FD004 to all datasets.\n\n")
            f.write("| Dataset | Test RMSE | Test MAE | Test Bias | Test R² | Test NASA Mean |\n")
            f.write("|---------|-----------|----------|-----------|---------|----------------|\n")
            
            for _, row in df.iterrows():
                f.write(
                    f"| {row['dataset']:7} | "
                    f"{row['test_rmse']:9.2f} | "
                    f"{row['test_mae']:8.2f} | "
                    f"{row['test_bias']:9.2f} | "
                    f"{row['test_r2']:7.4f} | "
                    f"{row['test_nasa_mean']:14.2f} |\n"
                )
        
        print(f"\n✓ Markdown table written to: {output_path}")


if __name__ == "__main__":
    csv_path = Path("results/health_index/phase1_cross_dataset_baseline.csv")
    output_path = Path("docs/phase1_cross_dataset_summary.md")
    
    generate_markdown_table(csv_path, output_path)

