"""
Diagnostics Runner for Trained Experiments.

This script runs diagnostics for already-trained experiments without re-training.
It loads models from results/<dataset>/<experiment_name>/ and generates diagnostic plots.

Usage:
    # Run diagnostics for a specific experiment group
    python run_diagnostics.py --group world_phase4
    python run_diagnostics.py --group world_phase5
    python run_diagnostics.py --group P4
    
    # Run diagnostics for specific experiments
    python run_diagnostics.py --experiments fd002_world_phase4_universal_v2_residual fd004_world_phase4_universal_v2_residual
    
    # Run diagnostics for all experiments in a group
    python run_diagnostics.py --group all
    
    # Run diagnostics for a specific dataset
    python run_diagnostics.py --dataset FD004
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch

from src.experiment_configs import (
    get_experiment_group,
    get_experiment_by_name,
)
from src.analysis.diagnostics import run_diagnostics_for_run


def run_diagnostics_for_experiment(
    experiment_name: str,
    dataset_name: str,
    device: torch.device,
    enable_failure_cases: bool = False,
    failure_cases_k: int = 10,
) -> bool:
    """
    Run diagnostics for a single experiment.

    Args:
        experiment_name: Name of the experiment
        dataset_name: Dataset name (e.g., "FD004")
        device: PyTorch device
        enable_failure_cases: Build failure case library
        failure_cases_k: Number of top-K worst cases to select

    Returns:
        True if successful, False otherwise
    """
    results_dir = Path("results") / dataset_name.lower() / experiment_name

    if not results_dir.exists():
        print(f"  ⚠️  Experiment directory not found: {results_dir}")
        print(f"     Skipping diagnostics for {experiment_name}")
        return False

    # Check if model file exists
    model_files = list(results_dir.glob("*.pt"))
    if not model_files:
        print(f"  ⚠️  No model file found in {results_dir}")
        print(f"     Skipping diagnostics for {experiment_name}")
        return False

    try:
        print(f"\n{'='*80}")
        print(f"Running diagnostics for: {experiment_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Experiment directory: {results_dir}")
        print(f"{'='*80}\n")

        run_diagnostics_for_run(
            exp_dir=results_dir.parent.parent,  # results/ (go up from results/<dataset>/<name> to results/)
            dataset_name=dataset_name,
            run_name=experiment_name,
            device=device,
            enable_failure_cases=enable_failure_cases,
            failure_cases_k=failure_cases_k,
        )

        print(f"\n✓ Diagnostics completed for {experiment_name}")
        return True

    except Exception as e:
        print(f"\n❌ Error running diagnostics for {experiment_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics for trained experiments")
    parser.add_argument(
        "--group",
        type=str,
        choices=["A", "B", "C", "P3", "P3_2", "P3_2_FD004", "P4", "all", "lstm_baselines", "transformer_baselines", "fd004_sweep", "phase3", "universal_v1", "fd004_tuning", "phase4", "fd_all_phase4_residual", "world_phase4", "world_p4", "world_eol_heavy", "world_h10_eol_heavy", "world_phase5", "world_p5", "world_v3"],
        help="Experiment group to run diagnostics for (same groups as run_experiments.py)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Specific experiment names to run diagnostics for",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Run diagnostics for all experiments of a specific dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--failure_cases",
        action="store_true",
        help="Build failure case library for the experiment",
    )
    parser.add_argument(
        "--failure_cases_k",
        type=int,
        default=10,
        help="Number of top-K worst cases to select (default: 10)",
    )

    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Collect experiments to run diagnostics for
    experiments: List[dict] = []
    
    if args.group:
        experiments.extend(get_experiment_group(args.group))
    
    if args.experiments:
        for exp_name in args.experiments:
            try:
                experiments.append(get_experiment_by_name(exp_name))
            except ValueError as e:
                print(f"⚠️  Could not find experiment config for: {exp_name}")
                print(f"   Error: {e}")
                continue
    
    if args.dataset:
        # Add all experiments for this dataset
        from src.experiment_configs import get_lstm_baseline_config, get_transformer_baseline_config
        try:
            experiments.append(get_lstm_baseline_config(args.dataset))
            experiments.append(get_transformer_baseline_config(args.dataset))
        except Exception as e:
            print(f"⚠️  Could not add baseline configs for {args.dataset}: {e}")
        
        if args.dataset == "FD004":
            try:
                experiments.extend(get_experiment_group("C"))
            except Exception as e:
                print(f"⚠️  Could not add FD004 sweep experiments: {e}")
    
    if not experiments:
        parser.print_help()
        print("\nError: No experiments specified. Use --group, --experiments, or --dataset")
        sys.exit(1)
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_experiments = []
    for exp in experiments:
        if exp['experiment_name'] not in seen:
            seen.add(exp['experiment_name'])
            unique_experiments.append(exp)
    experiments = unique_experiments
    
    print(f"\n{'=' * 80}")
    print(f"Running diagnostics for {len(experiments)} experiment(s)")
    print(f"{'=' * 80}")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['experiment_name']} ({exp['dataset']}, {exp['encoder_type']})")
    
    # Run diagnostics
    successful = 0
    failed = 0
    
    for i, config in enumerate(experiments, 1):
        print(f"\n\n{'#' * 80}")
        print(f"Diagnostics {i}/{len(experiments)}")
        print(f"{'#' * 80}")
        
        experiment_name = config['experiment_name']
        dataset_name = config['dataset']
        
        success = run_diagnostics_for_experiment(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            device=device,
            enable_failure_cases=args.failure_cases,
            failure_cases_k=args.failure_cases_k,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n\n{'=' * 80}")
    print(f"Diagnostics Complete")
    print(f"{'=' * 80}")
    print(f"Successful: {successful}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

