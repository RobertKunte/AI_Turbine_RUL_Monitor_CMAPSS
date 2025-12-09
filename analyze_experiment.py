"""
CLI script for analyzing experiment results.

This script runs inference on trained models and generates analysis plots.

Usage:
    # Analyze a single experiment
    python analyze_experiment.py --experiment results/FD004/fd004_phase2_transformer_baseline --split test
    
    # Analyze multiple experiments with glob pattern
    python analyze_experiment.py --glob "results/FD004/fd004_phase2_transformer*" --split test
    
    # Analyze with custom output directory
    python analyze_experiment.py --experiment results/FD002/fd002_phase2_lstm_baseline --split test --output-dir custom_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.analysis import (
    run_inference_for_experiment,
    select_representative_engines,
    plot_eol_error_hist,
    plot_nasa_per_engine,
    plot_hi_trajectories_for_selected_engines,
    plot_true_vs_pred_rul,
)


def analyze_single_experiment(
    experiment_dir: Path,
    split: str = "test",
    device: str = "auto",
    output_dir: Optional[Path] = None,
) -> None:
    """
    Analyze a single experiment and generate plots.
    
    Args:
        experiment_dir: Path to experiment directory
        split: "test" or "val"
        device: Device to use ("auto", "cpu", "cuda")
        output_dir: Optional custom output directory (default: experiment_dir)
    """
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Analyzing experiment: {experiment_dir}")
    print(f"Split: {split}")
    
    # Set output directory
    if output_dir is None:
        output_dir = experiment_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    try:
        eol_metrics, trajectories = run_inference_for_experiment(
            experiment_dir=experiment_dir,
            split=split,
            device=device,
            return_hi_trajectories=True,
        )
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not eol_metrics:
        print("Warning: No metrics collected")
        return
    
    # Print summary statistics
    errors = [m.error for m in eol_metrics]
    nasa_scores = [m.nasa for m in eol_metrics]
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Number of engines: {len(eol_metrics)}")
    print(f"Mean error: {np.mean(errors):.2f} cycles")
    print(f"Mean absolute error: {np.mean(np.abs(errors)):.2f} cycles")
    print(f"RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.2f} cycles")
    print(f"Mean NASA contribution: {np.mean(nasa_scores):.2f}")
    print(f"Total NASA score: {np.sum(nasa_scores):.2f}")
    print("=" * 80)
    
    # Generate plots
    experiment_name = experiment_dir.name
    
    # 1. EOL error histogram
    plot_eol_error_hist(
        eol_metrics=eol_metrics,
        title=f"EOL RUL Error Distribution - {experiment_name}",
        out_path=output_dir / "eol_error_hist.png",
    )
    
    # 2. True vs Predicted RUL scatter
    plot_true_vs_pred_rul(
        eol_metrics=eol_metrics,
        title=f"True vs Predicted RUL - {experiment_name}",
        out_path=output_dir / "true_vs_pred_rul.png",
    )
    
    # 3. NASA per engine barplot
    plot_nasa_per_engine(
        eol_metrics=eol_metrics,
        title=f"NASA Contribution per Engine - {experiment_name}",
        out_path=output_dir / "nasa_per_engine_bar.png",
        max_engines=50,  # Show top 50 engines
    )
    
    # 4. HI trajectories for selected engines
    if trajectories:
        selected_engines = select_representative_engines(eol_metrics, num=5)
        plot_hi_trajectories_for_selected_engines(
            trajectories=trajectories,
            selected_unit_ids=selected_engines,
            title=f"Health Index Trajectories - {experiment_name}",
            out_path=output_dir / "hi_trajectories_5_engines.png",
            plot_rul=True,
        )
        print(f"\nSelected engines for trajectory plot: {selected_engines}")
    
    print(f"\n✓ Analysis complete. Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Path to experiment directory (contains summary.json and model.pt)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Glob pattern to match multiple experiments (e.g., 'results/FD004/fd004_phase2_transformer*')",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Data split to analyze (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for plots (default: experiment directory)",
    )
    
    args = parser.parse_args()
    
    if not args.experiment and not args.glob:
        parser.print_help()
        print("\nError: Must specify either --experiment or --glob")
        sys.exit(1)
    
    # Collect experiment directories
    experiment_dirs: List[Path] = []
    
    if args.experiment:
        experiment_dirs.append(args.experiment)
    
    if args.glob:
        # Find all matching directories
        results_root = Path("results")
        for path in results_root.rglob(args.glob):
            if path.is_dir() and (path / "summary.json").exists():
                experiment_dirs.append(path)
        
        if not experiment_dirs:
            print(f"Warning: No experiments found matching pattern: {args.glob}")
            sys.exit(1)
    
    # Remove duplicates
    experiment_dirs = list(set(experiment_dirs))
    
    print(f"Found {len(experiment_dirs)} experiment(s) to analyze")
    
    # Analyze each experiment
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n{'#' * 80}")
        print(f"Experiment {i}/{len(experiment_dirs)}: {exp_dir}")
        print(f"{'#' * 80}")
        
        analyze_single_experiment(
            experiment_dir=exp_dir,
            split=args.split,
            device=args.device,
            output_dir=args.output_dir,
        )
    
    print(f"\n{'=' * 80}")
    print(f"Completed analysis of {len(experiment_dirs)} experiment(s)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

