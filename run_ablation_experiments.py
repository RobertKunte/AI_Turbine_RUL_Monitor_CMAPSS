"""
Ablation Study Experiment Runner.

This script runs systematic ablation experiments for World Model v3:
- Architecture variants (A1-A6)
- Horizons (20, 40, 60)
- EOL loss weights (1, 5, 10)
- Residual modes (True/False)
- Fusion modes (early/late)

Results are saved under results/<dataset>/ablation_phase6/<experiment_name>/
"""

import argparse
import sys
from pathlib import Path
from itertools import product

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
)
from src.config import ResidualFeatureConfig
from src.feature_safety import remove_rul_leakage
from src.world_model_training_ablation import AblationConfig, train_ablation_experiment
from src.analysis.diagnostics import run_diagnostics_for_run


def generate_experiment_name(
    variant: str,
    horizon: int,
    eol_weight: float,
    use_residuals: bool,
    fusion_mode: str,
) -> str:
    """Generate experiment folder name."""
    variant_names = {
        "A1": "A1_full",
        "A2": "A2_no_traj",
        "A3": "A3_no_hi",
        "A4": "A4_no_mono",
        "A5": "A5_eol_only",
        "A6": "A6_traj_only",
    }
    
    variant_name = variant_names.get(variant, variant)
    horizon_name = f"H{horizon}"
    eol_name = f"L{int(eol_weight)}"
    residual_name = "residual" if use_residuals else "nores"
    fusion_name = fusion_mode
    
    return f"{variant_name}_{horizon_name}_{eol_name}_{residual_name}_{fusion_name}"


def run_ablation_sweep(
    dataset_name: str,
    variants: list = None,
    horizons: list = None,
    eol_weights: list = None,
    residual_modes: list = None,
    fusion_modes: list = None,
    device: torch.device = None,
    max_experiments: int = None,
):
    """
    Run ablation study sweep.
    
    Args:
        dataset_name: Dataset name ("FD002" or "FD004")
        variants: List of variants to test (default: all A1-A6)
        horizons: List of horizons to test (default: [20, 40, 60])
        eol_weights: List of EOL weights to test (default: [1, 5, 10])
        residual_modes: List of residual modes (default: [True, False])
        fusion_modes: List of fusion modes (default: ["late", "early"])
        device: PyTorch device
        max_experiments: Maximum number of experiments to run (for testing)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Defaults
    if variants is None:
        variants = ["A1", "A2", "A3", "A4", "A5", "A6"]
    if horizons is None:
        horizons = [20, 40, 60]
    if eol_weights is None:
        eol_weights = [1, 5, 10]
    if residual_modes is None:
        residual_modes = [True, False]
    if fusion_modes is None:
        fusion_modes = ["late", "early"]
    
    # Generate all combinations
    all_combinations = list(product(variants, horizons, eol_weights, residual_modes, fusion_modes))
    
    if max_experiments is not None:
        all_combinations = all_combinations[:max_experiments]
    
    print(f"\n{'='*80}")
    print(f"Running Ablation Study for {dataset_name}")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_combinations)}")
    print(f"Variants: {variants}")
    print(f"Horizons: {horizons}")
    print(f"EOL Weights: {eol_weights}")
    print(f"Residual Modes: {residual_modes}")
    print(f"Fusion Modes: {fusion_modes}")
    print(f"{'='*80}\n")
    
    # Load data once
    print(f"[0] Loading {dataset_name} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )
    
    # Run experiments
    results_dir_base = Path("results") / dataset_name.lower() / "ablation_phase6"
    results_dir_base.mkdir(parents=True, exist_ok=True)
    
    completed = 0
    failed = 0
    
    for i, (variant, horizon, eol_weight, use_residuals, fusion_mode) in enumerate(all_combinations, 1):
        experiment_name = generate_experiment_name(variant, horizon, eol_weight, use_residuals, fusion_mode)
        experiment_dir = results_dir_base / experiment_name
        
        # Skip if already completed
        if (experiment_dir / "summary.json").exists():
            print(f"\n[{i}/{len(all_combinations)}] Skipping {experiment_name} (already exists)")
            completed += 1
            continue
        
        print(f"\n{'#'*80}")
        print(f"Experiment {i}/{len(all_combinations)}: {experiment_name}")
        print(f"{'#'*80}")
        
        try:
            # Feature engineering (with or without residuals)
            physics_config = PhysicsFeatureConfig(
                use_core=True,
                use_extended=False,
                use_residuals=use_residuals,
                use_temporal_on_physics=False,
                residual=ResidualFeatureConfig(
                    enabled=use_residuals,
                    mode="per_engine",
                    baseline_len=30,
                    include_original=True,
                ) if use_residuals else ResidualFeatureConfig(enabled=False),
            )
            
            feature_config = FeatureConfig(
                add_physical_core=True,
                add_temporal_features=True,
                temporal=TemporalFeatureConfig(
                    base_cols=None,
                    short_windows=(5, 10),
                    long_windows=(30,),
                    add_rolling_mean=True,
                    add_rolling_std=False,
                    add_trend=True,
                    add_delta=True,
                    delta_lags=(5, 10),
                ),
            )
            
            df_train_fe = create_physical_features(df_train.copy(), physics_config, "UnitNumber", "TimeInCycles")
            df_train_fe = create_all_features(df_train_fe, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)
            
            df_test_fe = create_physical_features(df_test.copy(), physics_config, "UnitNumber", "TimeInCycles")
            df_test_fe = create_all_features(df_test_fe, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)
            
            feature_cols = [
                c for c in df_train_fe.columns
                if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
            ]
            feature_cols, _ = remove_rul_leakage(feature_cols)
            
            print(f"  Using {len(feature_cols)} features")
            
            # Create ablation config
            ablation_config = AblationConfig(
                variant=variant,
                horizon=horizon,
                eol_loss_weight=eol_weight,
                use_residuals=use_residuals,
                fusion_mode=fusion_mode,
            )
            
            # Train
            summary = train_ablation_experiment(
                df_train=df_train_fe,
                df_test=df_test_fe,
                y_test_true=y_test_true,
                feature_cols=feature_cols,
                dataset_name=dataset_name,
                ablation_config=ablation_config,
                results_dir=experiment_dir,
                device=device,
            )

            print(f"\n✓ Completed {experiment_name}")
            print(f"  Test RMSE: {summary['test_metrics']['rmse']:.2f}")
            print(f"  Test NASA Mean: {summary['test_metrics']['nasa_mean']:.2f}")

            # ------------------------------------------------------------------
            # Run diagnostics for this ablation run:
            #   - uses the SAME pipeline as main experiments
            #   - recomputes EOL metrics directly from the checkpoint
            #   - builds sliding-window HI trajectories
            #   - saves publication-ready plots (error hist, true-vs-pred, HI+RUL)
            # Ablation results are stored under:
            #   results/<dataset>/ablation_phase6/<experiment_name>
            # We call diagnostics with run_name="ablation_phase6/<experiment_name>"
            # so that experiment_dir = results/<dataset>/ablation_phase6/<experiment_name>
            # matches the layout.
            # ------------------------------------------------------------------
            try:
                run_diagnostics_for_run(
                    exp_dir=results_dir_base.parent.parent,  # "results"
                    dataset_name=dataset_name,
                    run_name=f"ablation_phase6/{experiment_name}",
                    device=device,
                )
            except Exception as e:
                print(f"\n⚠️  Diagnostics failed for {experiment_name}: {e}")
                import traceback
                traceback.print_exc()

            completed += 1
            
        except Exception as e:
            print(f"\n❌ Error in {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"Ablation Study Complete")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{len(all_combinations)}")
    print(f"Failed: {failed}/{len(all_combinations)}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run World Model v3 ablation study")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["FD002", "FD004"],
        default="FD002",
        help="Dataset to run ablation on",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Variants to test (A1, A2, ..., A6). Default: all",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizons to test (20, 40, 60). Default: all",
    )
    parser.add_argument(
        "--eol-weights",
        type=float,
        nargs="+",
        default=None,
        help="EOL loss weights to test (1, 5, 10). Default: all",
    )
    parser.add_argument(
        "--residual-modes",
        type=str,
        nargs="+",
        choices=["True", "False", "true", "false"],
        default=None,
        help="Residual modes to test. Default: both",
    )
    parser.add_argument(
        "--fusion-modes",
        type=str,
        nargs="+",
        choices=["early", "late"],
        default=None,
        help="Fusion modes to test. Default: both",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Parse residual modes
    residual_modes = None
    if args.residual_modes:
        residual_modes = [r.lower() == "true" for r in args.residual_modes]
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    run_ablation_sweep(
        dataset_name=args.dataset,
        variants=args.variants,
        horizons=args.horizons,
        eol_weights=args.eol_weights,
        residual_modes=residual_modes,
        fusion_modes=args.fusion_modes,
        device=device,
        max_experiments=args.max_experiments,
    )


if __name__ == "__main__":
    main()

