"""
Experiment Runner for Systematic RUL Model Experiments.

This script runs experiments defined in src/experiment_configs.py and saves
results to results/<dataset>/<experiment_name>/summary.json.

Usage:
    # Run a specific experiment group
    python run_experiments.py --group A
    python run_experiments.py --group B
    python run_experiments.py --group C
    
    # Run Phase 4 residual feature experiments for all datasets
    python run_experiments.py --group P4
    # or
    python run_experiments.py --group fd_all_phase4_residual
    
    # Run all experiments
    python run_experiments.py --group all
    
    # Run specific experiments by name
    python run_experiments.py --experiments fd001_phase2_lstm_baseline fd004_phase2_transformer_baseline
    python run_experiments.py --experiments fd004_phase4_universal_v2_ms_cnn_d96_residual
    
    # Run experiments for a specific dataset
    python run_experiments.py --dataset FD004

Phase 4 Residual Features:
    Phase 4 experiments enable residual/digital-twin features that compare sensor
    values against a "healthy" baseline (mean of first 30 cycles per engine).
    This helps the model focus on degradation signals rather than absolute values.
    
    Residual features are automatically enabled for experiments with names containing
    "phase4" and "residual". The feature engineering pipeline computes:
    - Per-engine baselines from early cycles (no RUL leakage)
    - Residual features: current_value - baseline
    - Original features are kept by default (include_original=True)
    
    Example Phase 4 experiments:
    - fd001_phase4_universal_v1_residual
    - fd002_phase4_universal_v2_ms_cnn_d96_residual
    - fd003_phase4_universal_v1_residual
    - fd004_phase4_universal_v2_ms_cnn_d96_residual
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
import shutil
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

from src.experiment_configs import (
    ExperimentConfig,
    get_experiment_group,
    get_experiment_by_name,
)
from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    add_temporal_features,
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
    build_condition_features,
    create_twin_features,
    group_feature_columns,
)
from src.data.physics_hi import add_physics_hi, add_physics_hi_v2
from src.config import HI_RUL_PLATEAU_THRESH
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTMWithHealth,
    train_eol_full_lstm,
    evaluate_eol_full_lstm,
    evaluate_on_test_data,
)
from src.models.universal_encoder_v1 import (
    UniversalEncoderV1,
    RULHIUniversalModel,
    UniversalEncoderV2,
    RULHIUniversalModelV2,
    UniversalEncoderV3Attention,
)
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.feature_safety import remove_rul_leakage
from src.state_encoder_training_v3 import train_state_encoder_v3
from src.state_encoder_training_v3_physics import train_state_encoder_v3_physics


def run_single_experiment(config: ExperimentConfig, device: torch.device) -> dict:
    """
    Run a single experiment and return results summary.
    
    Args:
        config: Experiment configuration
        device: torch.device to use
    
    Returns:
        Dictionary with experiment results (for summary.json)
    """
    print("\n" + "=" * 80)
    print(f"Running Experiment: {config['experiment_name']}")
    print(f"Dataset: {config['dataset']}, Encoder: {config['encoder_type']}")
    print("=" * 80)
    
    dataset_name = config['dataset']
    experiment_name = config['experiment_name']
    # Canonical results directory for this run (used across all branches)
    results_dir = Path("results") / dataset_name.lower() / experiment_name

    # ===================================================================
    # Phase 1 Run Registry (SQLite) – best-effort (must not break runs)
    # ===================================================================
    registry = None
    run_id: Optional[str] = None
    artifact_root_current: Optional[Path] = None
    registry_results_dir_default = results_dir

    def _get_git_sha() -> Optional[str]:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return sha or None
        except Exception:
            return None

    def _metrics_from_summary(summary: dict) -> dict:
        # Keep this small and stable. Store the entire summary as metrics_json,
        # but prefer the standard metric dicts when present.
        if not isinstance(summary, dict):
            return {"summary": str(summary)}
        metrics = {}
        # Common patterns across encoders/decoders/world models
        if "test_metrics" in summary and isinstance(summary["test_metrics"], dict):
            metrics["test_metrics"] = summary["test_metrics"]
        if "val_metrics" in summary and isinstance(summary["val_metrics"], dict):
            metrics["val_metrics"] = summary["val_metrics"]
        if "metrics_test" in summary and isinstance(summary["metrics_test"], dict):
            metrics["metrics_test"] = summary["metrics_test"]
        if "metrics_val" in summary and isinstance(summary["metrics_val"], dict):
            metrics["metrics_val"] = summary["metrics_val"]
        # Always include a thin top-level snapshot for quick inspection
        for k in ["experiment_name", "dataset", "encoder_type", "model_type", "best_val_rmse", "val_rmse", "test_rmse", "test_nasa_mean"]:
            if k in summary:
                metrics[k] = summary[k]
        return metrics or summary

    def _registry_start() -> None:
        nonlocal registry, run_id, artifact_root_current
        if os.environ.get("RUN_REGISTRY_DISABLE", "").strip() in {"1", "true", "True", "yes"}:
            return
        try:
            from src.tools.run_registry import RunRegistry

            db_path = Path(os.environ.get("RUN_REGISTRY_DB", str(Path("artifacts") / "run_registry.sqlite")))
            registry = RunRegistry(db_path)
            run_id = registry.start_run(
                experiment_name=experiment_name,
                dataset=dataset_name,
                config=dict(config),
                results_dir=registry_results_dir_default,
                artifact_root=None,
                git_sha=_get_git_sha(),
            )
            # Standard per-run artifact root (used by Colab sync)
            try:
                artifact_root_current = Path("artifacts") / "runs" / str(run_id)
                artifact_root_current.mkdir(parents=True, exist_ok=True)
                registry.set_artifact_root(str(run_id), artifact_root_current)
            except Exception as e:
                print(f"[run_registry] WARNING: could not set artifact_root: {e}")
            print(f"[run_registry] run_id={run_id} db={db_path}")
        except Exception as e:
            registry = None
            run_id = None
            artifact_root_current = None
            print(f"[run_registry] WARNING: disabled due to error: {e}")

    def _copy_only_newer(src: Path, dst: Path) -> bool:
        if not dst.exists():
            return True
        try:
            sm = src.stat().st_mtime
            dm = dst.stat().st_mtime
            if sm > dm + 1e-6:
                return True
            if abs(sm - dm) <= 1e-6:
                return src.stat().st_size != dst.stat().st_size
            return False
        except Exception:
            return True

    def _snapshot_results_to_artifacts(results_dir: Path) -> None:
        """
        Copy a small, useful snapshot of the run outputs into artifacts/runs/<run_id>/.
        This avoids syncing the entire repo and keeps Colab sync predictable.
        """
        if artifact_root_current is None:
            return
        if not results_dir.exists():
            return

        dst_root = artifact_root_current / "results_snapshot"
        dst_root.mkdir(parents=True, exist_ok=True)

        patterns = [
            "summary.json",
            "summary_decoder_*.json",
            "eol_metrics.json",
            "*.png",
            "*.pkl",
            "*best*.pt",
            "*.pt",
        ]

        copied = 0
        for pat in patterns:
            for p in results_dir.glob(pat):
                if not p.is_file():
                    continue
                q = dst_root / p.name
                if _copy_only_newer(p, q):
                    shutil.copy2(p, q)
                    copied += 1

        if copied > 0:
            print(f"[run_registry] Snapshotted {copied} files to {dst_root}")

    def _registry_finish(summary: dict, *, results_dir: Optional[Path] = None, summary_path: Optional[Path] = None) -> None:
        if registry is None or run_id is None:
            return
        try:
            # Default summary.json path for most experiments
            if results_dir is None:
                results_dir = registry_results_dir_default
            if summary_path is None:
                candidate = results_dir / "summary.json"
                summary_path = candidate if candidate.exists() else None
            # Snapshot key run outputs into per-run artifact root
            _snapshot_results_to_artifacts(results_dir)
            registry.finish_run(
                run_id,
                metrics=_metrics_from_summary(summary),
                summary_path=summary_path,
                results_dir=results_dir,
                artifact_root=artifact_root_current,
                status="success",
            )
        except Exception as e:
            print(f"[run_registry] WARNING: could not finish run: {e}")
        finally:
            try:
                registry.close()
            except Exception:
                pass

    def _registry_fail(exc: BaseException, *, results_dir: Optional[Path] = None, summary_path: Optional[Path] = None) -> None:
        if registry is None or run_id is None:
            return
        try:
            if results_dir is None:
                results_dir = registry_results_dir_default
            if summary_path is None:
                candidate = results_dir / "summary.json"
                summary_path = candidate if candidate.exists() else None
            # Snapshot any files that may exist even on failure
            _snapshot_results_to_artifacts(results_dir)
            registry.fail_run(
                run_id,
                error_message=f"{type(exc).__name__}: {exc}",
                traceback_str=traceback.format_exc(),
                results_dir=results_dir,
                summary_path=summary_path,
                artifact_root=artifact_root_current,
            )
        except Exception as e:
            print(f"[run_registry] WARNING: could not fail-run log: {e}")
        finally:
            try:
                registry.close()
            except Exception:
                pass

    _registry_start()

    # ===================================================================
    # Special case: RUL Trajectory Decoders on top of frozen encoder v3d/v3e
    # ===================================================================
    encoder_type_cfg = config.get("encoder_type")

    if encoder_type_cfg in ["decoder_v1_from_encoder_v3d", "decoder_v1_from_encoder_v3e"]:
        from src.rul_decoder_training_v1 import train_rul_decoder_v1

        train_cfg = config.get("training_params", {})
        epochs = int(train_cfg.get("num_epochs", 50))
        batch_size = int(train_cfg.get("batch_size", 256))

        device_str = "cuda" if str(device).startswith("cuda") else "cpu"

        if encoder_type_cfg == "decoder_v1_from_encoder_v3d":
            encoder_experiment = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
            decoder_subdir = "decoder_v1_from_encoder_v3d"
        else:  # decoder_v1_from_encoder_v3e
            encoder_experiment = "fd004_transformer_encoder_ms_dt_v2_damage_v3e_smooth"
            decoder_subdir = "decoder_v1_from_encoder_v3e"

        print(
            "\n[decoder_v1] Launching RUL Trajectory Decoder v1 training "
            f"for experiment '{experiment_name}' on {device_str}\n"
            f"  -> encoder_experiment = {encoder_experiment}\n"
            f"  -> decoder_results_subdir = {decoder_subdir}"
        )
        train_rul_decoder_v1(
            device=device_str,
            epochs=epochs,
            batch_size=batch_size,
            encoder_experiment=encoder_experiment,
            decoder_results_subdir=decoder_subdir,
        )

        decoder_results_dir = Path("results") / dataset_name.lower() / decoder_subdir
        summary_path = decoder_results_dir / "summary_decoder_v1.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
        else:
            summary = {
                "experiment_name": experiment_name,
                "dataset": dataset_name,
                "note": "Decoder v1 training finished, but summary_decoder_v1.json was not found.",
            }
        _registry_finish(summary, results_dir=decoder_results_dir, summary_path=summary_path if summary_path.exists() else None)
        return summary

    if encoder_type_cfg == "decoder_v2":
        from src.rul_decoder_training_v2 import train_rul_decoder_v2

        print(
            "\n[decoder_v2] Launching RUL Trajectory Decoder v2 training "
            f"for experiment '{experiment_name}' on {device}\n"
            f"  -> encoder_experiment = {config.get('encoder_experiment', 'fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase')}\n"
        )
        summary = train_rul_decoder_v2(config, device)
        _registry_finish(summary, results_dir=registry_results_dir_default)
        return summary

    if encoder_type_cfg == "decoder_v3":
        from src.rul_decoder_training_v3 import train_rul_decoder_v3

        print(
            "\n[decoder_v3] Launching RUL Trajectory Decoder v3 training "
            f"for experiment '{experiment_name}' on {device}\n"
            f"  -> encoder_experiment = {config.get('encoder_experiment', 'fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase')}\n"
        )
        summary = train_rul_decoder_v3(config, device)
        _registry_finish(summary, results_dir=registry_results_dir_default)
        return summary
    
    # ===================================================================
    # Load Data
    # ===================================================================
    print(f"\n[1] Loading {dataset_name} data...")
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )
    
    # Feature Engineering
    # Check if this is a phase 4/5 residual experiment (including world model)
    # or a \"digital-twin\" residual experiment (name contains \"resid\").
    name_lower = experiment_name.lower()
    is_phase4_residual = (
        (("phase4" in name_lower) or ("phase5" in name_lower)) and "residual" in name_lower
    ) or ("residual" in name_lower) or ("resid" in name_lower)
    is_world_model = config['encoder_type'] in ["world_model_universal_v2", "world_model_universal_v3"]
    
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

    # Optional high-level feature block configuration (encoder-side)
    # This is deliberately kept simple and fully backwards compatible:
    # - If no "features" section exists, we keep the previous defaults
    # - If features['use_multiscale_features'] is provided, it toggles
    #   the standard temporal multi-scale block (add_temporal_features)
    features_cfg = config.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    # Default behaviour: temporal multi-scale features are enabled
    use_temporal_features = features_cfg.get("use_multiscale_features", True)

    # Map the advisor-style windows (short/medium/long) onto the existing
    # TemporalFeatureConfig (short_windows + long_windows).  We simply
    # concatenate medium+long into the "long" tuple.
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
    
    # Optional physics-informed condition/twin features (enabled via config)
    phys_opts = config.get("phys_features", {})
    use_phys_condition_vec = phys_opts.get("use_condition_vector", False)
    # Backwards-compatible: accept both "use_twin_features" (existing) and
    # "use_digital_twin_residuals" (advisor spec) as enabling the same block.
    use_twin_features = phys_opts.get(
        "use_twin_features",
        phys_opts.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_opts.get("twin_baseline_len", 30)
    condition_vector_version = phys_opts.get("condition_vector_version", 2)

    # 1) Physics features
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector (uses physics + settings/sensors)
    if use_phys_condition_vec:
        print("  Using continuous condition vector features (Cond_*)")
        df_train = build_condition_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )
        df_test = build_condition_features(
            df_test,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            version=condition_vector_version,
        )

    # 3) Digital twin (healthy predictor) + residuals
    if use_twin_features:
        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len})")
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        # Apply the same twin model to test data
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Temporal / multi-scale features (as before)
    # 4) Temporal / multi-scale features (as before)
    df_train = create_all_features(
        df_train,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )
    df_test = create_all_features(
        df_test,
        "UnitNumber",
        "TimeInCycles",
        feature_config,
        inplace=False,
        physics_config=physics_config,
    )

    # Optional: add *additional* temporal features for extra base columns.
    # This is OFF by default and only activated when explicitly configured.
    extra_prefixes = ms_cfg.get("extra_temporal_base_prefixes", None)
    extra_max_cols = ms_cfg.get("extra_temporal_base_max_cols", None)
    if use_temporal_features and extra_prefixes:
        if not isinstance(extra_prefixes, (list, tuple)):
            raise ValueError(
                "features.multiscale.extra_temporal_base_prefixes must be a list of prefixes "
                f"(e.g. ['Twin_','Resid_']). Got: {type(extra_prefixes)}"
            )
        prefixes = [str(p) for p in extra_prefixes]
        candidates = [
            c for c in df_train.columns
            if any(c.startswith(p) for p in prefixes)
        ]
        candidates = sorted(set(candidates))
        if extra_max_cols is not None:
            candidates = candidates[: int(extra_max_cols)]
        if candidates:
            print(
                f"[Train] Adding extra temporal base cols for multiscale: "
                f"prefixes={prefixes} count={len(candidates)}"
            )
            extra_temporal_cfg = TemporalFeatureConfig(
                base_cols=candidates,
                short_windows=windows_short,
                long_windows=combined_long if combined_long else (30,),
                add_rolling_mean=True,
                add_rolling_std=False,
                add_trend=True,
                add_delta=True,
                delta_lags=(5, 10),
            )
            df_train = add_temporal_features(
                df_train,
                unit_col="UnitNumber",
                cycle_col="TimeInCycles",
                config=extra_temporal_cfg,
                inplace=False,
            )
            df_test = add_temporal_features(
                df_test,
                unit_col="UnitNumber",
                cycle_col="TimeInCycles",
                config=extra_temporal_cfg,
                inplace=False,
            )

    # Optional physics-based Health Index (only needed explicitly for the
    # physics state encoder; kept here to avoid touching core model code).
    hi_scalers = None
    if config["encoder_type"] == "transformer_state_encoder_v3_physics":
        print("  Computing physics-based HI_phys_v2 + Hybrid HI_target for state encoder training")
        # Use the v2 HI pipeline which builds HI_phys_v2 and, during training,
        # a Hybrid-HI target (HI_target_hybrid) blending physics HI with a
        # RUL-based proxy. We do NOT remove any of the intermediate damage
        # channels so that the encoder still sees the full 361-D ms+DT space.
        max_rul_cfg = config.get("data", {}).get("max_rul", 125.0)
        df_train, hi_scalers = add_physics_hi_v2(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            cond_col="ConditionID",
            is_training=True,
            max_rul=float(max_rul_cfg),
            rul_col="RUL",
            alpha_hybrid=0.7,
        )
    
    # For damage_v2/v3 experiments: add HI_phys_v2 or HI_phys_v3 using the appropriate
    # per-row computation. This creates a time-resolved HI_phys_seq target for
    # training the damage head.
    experiment_name = config.get("experiment_name", "")
    if "damage_v2" in experiment_name.lower():
        print("  Computing HI_phys_v2 for damage_v2 experiment (time-resolved target)")
        from src.hi_phys_targets import add_hi_phys_v2
        df_train = add_hi_phys_v2(
            df_train,
            unit_col="UnitNumber",
            time_col="TimeInCycles",
            hpc_eff_col="Effizienz_HPC_Proxy",
            egt_drift_col="EGT_Drift",
            residual_prefix="Resid_",
            baseline_len=30,
        )
    elif "damage_v3" in experiment_name.lower():
        print("  Computing HI_phys_v3 for damage_v3 experiment (time-resolved target)")
        from src.features.hi_phys_v3 import compute_hi_phys_v3_from_residuals
        hi_v3_series = compute_hi_phys_v3_from_residuals(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
        )
        df_train["HI_phys_v3"] = hi_v3_series
        print(
            f"  HI_phys_v3 stats: min={hi_v3_series.min():.4f}, "
            f"max={hi_v3_series.max():.4f}, mean={hi_v3_series.mean():.4f}"
        )

    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    # Never feed HI_* targets as input features to the encoder; they are used
    # as supervised targets for the physics state encoder or damage head.
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid", "HI_phys_v2", "HI_phys_v3"]
    ]

    # -------------------------------------------------------------------
    # Optional feature-group filtering (backwards compatible):
    # - If features.include_groups is missing/None -> keep ALL features (old behavior)
    # - If provided -> select only the requested semantic groups
    # -------------------------------------------------------------------
    groups = group_feature_columns(feature_cols)
    include_groups = features_cfg.get("include_groups", None)
    print(
        "[Train] Feature groups: "
        f"total={len(feature_cols)}, raw={len(groups.get('raw', []))}, ms={len(groups.get('ms', []))}, "
        f"residual={len(groups.get('residual', []))}, cond={len(groups.get('cond', []))}, twin={len(groups.get('twin', []))}"
    )
    if include_groups is None:
        print("[Train] include_groups=None -> using ALL features")
        # Optional info log for the canonical FD004 ms+DT + cond + twin feature space.
        if (
            dataset_name.upper() == "FD004"
            and bool(use_temporal_features)
            and bool(use_phys_condition_vec)
            and bool(use_twin_features)
        ):
            print("[Train] Info: FD004 ms+DT + Cond_* + Twin_/Resid_* typically yields ~659 features.")
    else:
        if not isinstance(include_groups, (list, tuple)):
            raise ValueError(
                "features.include_groups must be a list of group names (or omitted/None). "
                f"Got: {type(include_groups)}"
            )
        allowed = sorted(groups.keys())
        unknown = [g for g in include_groups if g not in groups]
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}. Allowed: {allowed}")
        selected_cols: List[str] = []
        for g in include_groups:
            selected_cols.extend(groups[g])
        # De-duplicate while preserving order
        seen = set()
        selected_cols = [c for c in selected_cols if not (c in seen or seen.add(c))]
        feature_cols = selected_cols
        print(f"[Train] include_groups={list(include_groups)} -> selected={len(feature_cols)}")

    print(f"Using {len(feature_cols)} features for model input.")

    # -------------------------------------------------------------------
    # Dedicated paths: Transformer State Encoders (no EOL-specific loss)
    # -------------------------------------------------------------------
    if config["encoder_type"] in ["transformer_state_encoder_v3", "transformer_state_encoder_v3_physics"]:
        from types import SimpleNamespace

        results_dir = Path("results") / dataset_name.lower() / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Resolve training and loss configs for the dedicated state-encoder loop.
        train_cfg = config.get("training", config.get("training_params", {}))
        loss_cfg = config.get("loss", {})

        # Model configuration: start from sensible defaults and override with
        # any explicit settings provided in config["model"]. This allows
        # experiments such as the damage-head variant to customise the state
        # encoder architecture while keeping backwards compatibility.
        model_cfg_raw = config.get("model", {})
        model_ns = SimpleNamespace(
            input_dim=len(feature_cols),
            d_model=model_cfg_raw.get("d_model", 96),
            num_layers=model_cfg_raw.get("num_layers", 3),
            num_heads=model_cfg_raw.get("num_heads", 4),
            dim_feedforward=model_cfg_raw.get("dim_feedforward", 256),
            dropout=model_cfg_raw.get("dropout", 0.1),
            # Continuous condition encoder is used for all current state encoders;
            # cond_in_dim is inferred later from Cond_* indices if available.
            use_cond_encoder=model_cfg_raw.get("use_cond_encoder", True),
            cond_in_dim=model_cfg_raw.get("cond_in_dim", None),
            # Optional cumulative damage head parameters (used by damage-head runs)
            use_damage_head=model_cfg_raw.get("use_damage_head", False),
            L_ref=model_cfg_raw.get("L_ref", 300.0),
            alpha_base=model_cfg_raw.get("alpha_base", 0.1),
        )

        # Common base config namespace
        cfg_ns = SimpleNamespace(
            experiment_name=experiment_name,
            dataset=dataset_name,
            data=SimpleNamespace(
                fd_name=dataset_name,
                max_rul=125.0,
                past_len=30,
                feature_cols=feature_cols,
                train_frac=config.get("data", {}).get("train_frac", 0.8),
                df_train_fe=df_train,  # feature-engineered train DF (may include HI_phys_final)
            ),
            model=model_ns,
            hi=SimpleNamespace(
                plateau_threshold=80.0,
                eol_threshold=25.0,
            ),
            training=SimpleNamespace(
                batch_size=train_cfg.get("batch_size", 256),
                lr=train_cfg.get("lr", 1e-4),
                weight_decay=train_cfg.get("weight_decay", 1e-4),
                num_epochs=train_cfg.get("num_epochs", 80),
                patience=train_cfg.get("patience", 10),
            ),
            loss=SimpleNamespace(
                hi_weight=loss_cfg.get("hi_weight", 1.0),
                rul_weight=loss_cfg.get("rul_weight", 0.1),
                # Optional RUL–HI alignment loss (disabled by default unless set)
                align_weight=loss_cfg.get("align_weight", 0.0),
                use_rul_hi_alignment_loss=loss_cfg.get("use_rul_hi_alignment_loss", False),
                # Optional damage-head specific weights (only used by damage runs)
                hi_phys_weight=loss_cfg.get("hi_phys_weight", 1.0),
                hi_aux_weight=loss_cfg.get("hi_aux_weight", 0.3),
                mono_weight=loss_cfg.get("mono_weight", 0.01),
                smooth_weight=loss_cfg.get("smooth_weight", 0.01),
            ),
            paths=SimpleNamespace(
                result_dir=str(results_dir),
                # Optional HI scalers for diagnostics / inference when using
                # the physics-based Health Index. None for other encoders.
                hi_scalers=hi_scalers,
            ),
            seed=train_cfg.get("random_seed", config.get("training_params", {}).get("random_seed", 42)),
        )

        if config["encoder_type"] == "transformer_state_encoder_v3":
            summary = train_state_encoder_v3(cfg_ns)
        else:
            summary = train_state_encoder_v3_physics(cfg_ns)
        _registry_finish(summary, results_dir=results_dir)
        return summary
    
    # ===================================================================
    # World Model Training (Phase 4/5 Residual)
    # ===================================================================
    if is_world_model:
        import importlib
        from src import world_model_training
        importlib.reload(world_model_training)  # Force reload in Colab
        
        # Check if it's V3 or V2
        if config['encoder_type'] == "world_model_universal_v3":
            from src.world_model_training_v3 import (
                train_world_model_universal_v3,
                train_transformer_world_model_v1,
            )
            from src.world_model_training import WorldModelTrainingConfig
            
            world_model_params = config.get('world_model_params', {})
            
            # Create WorldModelTrainingConfig (shared between UniversalV3 and Transformer-WorldModel V1)
            # For WorldModelV1 variants we prefer an explicit "future_horizon"
            # if provided; otherwise fall back to the generic "horizon".
            wm_horizon = world_model_params.get('future_horizon', world_model_params.get('horizon', 40))

            world_model_config = WorldModelTrainingConfig(
                forecast_horizon=wm_horizon,
                traj_loss_weight=world_model_params.get('traj_loss_weight', 1.0),
                eol_loss_weight=world_model_params.get('eol_loss_weight', 5.0),
                hi_loss_weight=world_model_params.get('hi_loss_weight', 2.0),
                mono_late_weight=world_model_params.get('mono_late_weight', 0.1),
                mono_global_weight=world_model_params.get('mono_global_weight', 0.1),
                traj_step_weighting=world_model_params.get('traj_step_weighting', None),
                past_len=world_model_params.get('past_len', 30),
                max_rul=world_model_params.get('max_rul', 125),
                use_condition_wise_scaling=world_model_params.get('use_condition_wise_scaling', True),
                # v3 extensions: HI fusion + tail-weighted EOL loss
                use_hi_in_eol=world_model_params.get('use_hi_in_eol', False),
                use_hi_slope_in_eol=world_model_params.get('use_hi_slope_in_eol', False),
                eol_tail_rul_threshold=world_model_params.get('eol_tail_rul_threshold', None),
                eol_tail_weight=world_model_params.get('eol_tail_weight', 1.0),
                # Stage-1: 3-phase schedule + extra HI shape losses (optional; defaults keep old behavior)
                three_phase_schedule=world_model_params.get('three_phase_schedule', False),
                phase_a_frac=world_model_params.get('phase_a_frac', 0.2),
                phase_b_end_frac=world_model_params.get('phase_b_end_frac', 0.8),
                phase_b_frac=world_model_params.get('phase_b_frac', None),
                schedule_type=world_model_params.get('schedule_type', world_model_params.get('eol_ramp', 'linear')),
                eol_w_max=world_model_params.get('eol_w_max', 1.0),
                # EOL ramp stabilization (optional; default off)
                normalize_eol=bool(world_model_params.get('normalize_eol', False)),
                eol_scale=world_model_params.get('eol_scale', 'rul_cap'),
                eol_loss_type=world_model_params.get('eol_loss_type', 'mse'),
                eol_huber_beta=float(world_model_params.get('eol_huber_beta', 0.1)),
                clip_grad_norm=world_model_params.get('clip_grad_norm', None),
                freeze_encoder_epochs_after_eol_on=int(world_model_params.get('freeze_encoder_epochs_after_eol_on', 0)),
                # Selection / alignment knobs (optional; default off)
                select_best_after_eol_active=bool(world_model_params.get("select_best_after_eol_active", False)),
                eol_active_min_mult=float(world_model_params.get("eol_active_min_mult", 0.01)),
                best_metric=str(world_model_params.get("best_metric", "val_total")),
                eol_target_mode=str(world_model_params.get("eol_target_mode", "future0")),
                cap_rul_targets_to_max_rul=bool(world_model_params.get("cap_rul_targets_to_max_rul", False)),
                eval_clip_y_true_to_max_rul=bool(world_model_params.get("eval_clip_y_true_to_max_rul", False)),
                init_eol_bias_to_target_mean=bool(world_model_params.get("init_eol_bias_to_target_mean", False)),
                # Padded/clamped horizon target building + optional masking
                use_padded_horizon_targets=bool(world_model_params.get("use_padded_horizon_targets", False)),
                target_clamp_min=float(world_model_params.get("target_clamp_min", 0.0)),
                use_horizon_mask=bool(world_model_params.get("use_horizon_mask", False)),
                hi_early_slope_weight=world_model_params.get('hi_early_slope_weight', 0.0),
                hi_early_slope_epsilon=world_model_params.get('hi_early_slope_epsilon', 1e-3),
                hi_early_slope_rul_threshold=world_model_params.get('hi_early_slope_rul_threshold', None),
                hi_curvature_weight=world_model_params.get('hi_curvature_weight', 0.0),
                hi_curvature_abs=world_model_params.get('hi_curvature_abs', True),
                # Stage-2 (optional): HI→EOL consistency coupling for WorldModel (default off)
                w_eol_hi=world_model_params.get('w_eol_hi', 0.0),
                eol_hi_threshold=world_model_params.get('eol_hi_threshold', 0.2),
                eol_hi_temperature=world_model_params.get('eol_hi_temperature', 0.05),
                eol_hi_p_min=world_model_params.get('eol_hi_p_min', 0.2),
            )
            # Additional world-model V1 specific loss weights (sensor / future HI / future RUL)
            # and architectural flags are stored as dynamic attributes on the
            # config for ease of access inside train_transformer_world_model_v1.
            world_model_config.sensor_loss_weight = world_model_params.get('sensor_loss_weight', 1.0)
            world_model_config.hi_future_loss_weight = world_model_params.get('hi_future_loss_weight', 0.0)
            world_model_config.rul_future_loss_weight = world_model_params.get('rul_future_loss_weight', 0.0)
            world_model_config.target_mode = world_model_params.get('target_mode', 'sensors')
            world_model_config.init_from_rul_hi = world_model_params.get('init_from_rul_hi', False)
            world_model_config.future_horizon = world_model_params.get('future_horizon', world_model_params.get('horizon', 20))
            world_model_config.decoder_hidden_dim = world_model_params.get('decoder_hidden_dim', 256)
            world_model_config.num_layers_decoder = world_model_params.get('num_layers_decoder', 1)
            # Optional: encoder checkpoint + freezing for Transformer World Model V1
            world_model_config.freeze_encoder = world_model_params.get('freeze_encoder', False)
            world_model_config.encoder_checkpoint = world_model_params.get('encoder_checkpoint', None)
            # Dynamic latent world-model flags (Branch A+)
            world_model_config.use_latent_history = world_model_params.get('use_latent_history', False)
            world_model_config.use_hi_anchor = world_model_params.get('use_hi_anchor', False)
            world_model_config.use_future_conds = world_model_params.get('use_future_conds', False)
            # A+ latent decoder conditioning + EOL fusion
            world_model_config.use_eol_fusion = world_model_params.get("use_eol_fusion", False)
            world_model_config.eol_fusion_mode = world_model_params.get("eol_fusion_mode", "token")
            world_model_config.predict_latent = world_model_params.get("predict_latent", False)
            world_model_config.latent_decoder_type = world_model_params.get("latent_decoder_type", "gru")
            world_model_config.latent_decoder_num_layers = world_model_params.get("latent_decoder_num_layers", 2)
            world_model_config.latent_decoder_nhead = world_model_params.get("latent_decoder_nhead", 4)
            # Training staging (freeze -> partial unfreeze)
            world_model_config.freeze_encoder_epochs = world_model_params.get("freeze_encoder_epochs", 0)
            world_model_config.unfreeze_encoder_layers = world_model_params.get("unfreeze_encoder_layers", 0)
            world_model_config.encoder_lr_mult = world_model_params.get("encoder_lr_mult", 0.1)
            world_model_config.eol_scalar_loss_weight = world_model_params.get("eol_scalar_loss_weight", 0.0)
            world_model_config.grad_clip_norm = world_model_params.get("grad_clip_norm", None)

            # If this is one of the Transformer World Model V1 experiments, route to
            # the dedicated training function; otherwise use the existing
            # UniversalEncoderV2-based World Model v3.
            if experiment_name in [
                "fd004_transformer_worldmodel_v1",
                "fd004_transformer_worldmodel_v1_h10_eolfocus",
                "fd004_transformer_worldmodel_v1_h20_residuals_eolfocus",
                "fd004_transformer_latent_worldmodel_v1",
                "fd004_transformer_latent_worldmodel_msfreeze_v1",
                "fd004_transformer_latent_worldmodel_dynamic_v1",
                "fd004_transformer_latent_worldmodel_dynamic_freeze_v1",
                "fd004_transformer_latent_worldmodel_dynamic_delta_v2",
                "fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659",
                "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1",
                "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_rulonly_v1",
            ]:
                print("\n[2] Training Transformer World Model V1 (Transformer encoder + GRU decoder)...")
                summary = train_transformer_world_model_v1(
                    df_train=df_train,
                    df_test=df_test,
                    feature_cols=feature_cols,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                    world_model_config=world_model_config,
                    encoder_d_model=config['encoder_kwargs']['d_model'],
                    encoder_num_layers=config['encoder_kwargs']['num_layers'],
                    encoder_nhead=config['encoder_kwargs']['nhead'],
                    encoder_dim_feedforward=config['encoder_kwargs']['dim_feedforward'],
                    encoder_dropout=config['encoder_kwargs']['dropout'],
                    num_sensors_out=21,
                    cond_dim=9,
                    batch_size=config['training_params']['batch_size'],
                    num_epochs=config['training_params']['num_epochs'],
                    lr=world_model_params.get('learning_rate', config['optimizer_params']['lr']),
                    weight_decay=world_model_params.get('weight_decay', config['optimizer_params']['weight_decay']),
                    patience=config['training_params']['patience'],
                    results_dir=results_dir,
                    device=device,
                )
            else:
                print("\n[2] Training World Model v3 (UniversalEncoderV2 + HI Head)...")
                summary = train_world_model_universal_v3(
                    df_train=df_train,
                    df_test=df_test,
                    y_test_true=y_test_true,
                    feature_cols=feature_cols,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                    d_model=config['encoder_kwargs']['d_model'],
                    num_layers=config['encoder_kwargs']['num_layers'],
                    nhead=config['encoder_kwargs']['nhead'],
                    dim_feedforward=config['encoder_kwargs']['dim_feedforward'],
                    dropout=config['encoder_kwargs']['dropout'],
                    kernel_sizes=config['encoder_kwargs']['kernel_sizes'],
                    seq_encoder_type=config['encoder_kwargs']['seq_encoder_type'],
                    decoder_num_layers=config['encoder_kwargs']['decoder_num_layers'],
                    batch_size=config['training_params']['batch_size'],
                    num_epochs=config['training_params']['num_epochs'],
                    lr=config['optimizer_params']['lr'],
                    weight_decay=config['optimizer_params']['weight_decay'],
                    patience=config['training_params']['patience'],
                    engine_train_ratio=config['training_params']['engine_train_ratio'],
                    random_seed=config['training_params']['random_seed'],
                    world_model_config=world_model_config,
                    results_dir=results_dir,
                    device=device,
                )
            
        else:
            # V2 (Phase 4 default)
            from src.world_model_training import train_world_model_universal_v2_residual, WorldModelTrainingConfig
            
            print("\n[2] Training World Model v2 with UniversalEncoderV2...")
            world_model_params = config.get('world_model_params', {})
            
            # Create WorldModelTrainingConfig from experiment config
            world_model_config = WorldModelTrainingConfig(
                forecast_horizon=world_model_params.get('horizon', 20),
                traj_loss_weight=world_model_params.get('traj_loss_weight', 1.0),
                eol_loss_weight=world_model_params.get('eol_loss_weight', 1.0),
                traj_step_weighting=world_model_params.get('traj_step_weighting', None),
                past_len=world_model_params.get('past_len', 30),
                max_rul=world_model_params.get('max_rul', 125),
                use_condition_wise_scaling=world_model_params.get('use_condition_wise_scaling', True),
                two_phase_training=world_model_params.get('two_phase_training', False),
                phase2_eol_weight=world_model_params.get('phase2_eol_weight', 10.0),
            )
            
            summary = train_world_model_universal_v2_residual(
                df_train=df_train,
                df_test=df_test,
                y_test_true=y_test_true,
                feature_cols=feature_cols,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
                d_model=config['encoder_kwargs']['d_model'],
                num_layers=config['encoder_kwargs']['num_layers'],
                nhead=config['encoder_kwargs']['nhead'],
                dim_feedforward=config['encoder_kwargs']['dim_feedforward'],
                dropout=config['encoder_kwargs']['dropout'],
                kernel_sizes=config['encoder_kwargs']['kernel_sizes'],
                seq_encoder_type=config['encoder_kwargs']['seq_encoder_type'],
                decoder_num_layers=config['encoder_kwargs']['decoder_num_layers'],
                batch_size=config['training_params']['batch_size'],
                num_epochs=config['training_params']['num_epochs'],
                lr=config['optimizer_params']['lr'],
                weight_decay=config['optimizer_params']['weight_decay'],
                patience=config['training_params']['patience'],
                engine_train_ratio=config['training_params']['engine_train_ratio'],
                random_seed=config['training_params']['random_seed'],
                world_model_config=world_model_config,
                results_dir=results_dir,
                device=device,
            )
        
        print("\n" + "=" * 80)
        print(f"World Model Experiment Complete: {experiment_name}")
        print("=" * 80)

        # For classic Universal V3 world models we have full EOL test metrics.
        # For the new Transformer World Model V1 we currently only track val_loss
        # and do not compute EOL RUL metrics yet, so 'test_metrics' may be absent.
        if "test_metrics" in summary:
            print(f"Test Metrics:")
            print(f"  RMSE: {summary['test_metrics']['rmse']:.2f} cycles")
            print(f"  MAE:  {summary['test_metrics']['mae']:.2f} cycles")
            print(f"  Bias: {summary['test_metrics']['bias']:.2f} cycles")
            print(f"  R²:   {summary['test_metrics']['r2']:.4f}")
            print(f"  NASA Mean: {summary['test_metrics']['nasa_mean']:.2f}")
        else:
            print("Test Metrics: (not computed for Transformer World Model V1 yet)")
            print(f"  Best val_loss (train_transformer_world_model_v1): {summary.get('val_loss', float('nan')):.4f}")
        print("=" * 80)

        _registry_finish(summary, results_dir=results_dir, summary_path=results_dir / "summary.json")
        return summary
    
    # ===================================================================
    # Build Sequences and Dataloaders (EOL Models)
    # ===================================================================
    print("\n[2] Building full-trajectory sequences...")
    window_cfg_run = config.get("window_cfg", {}) if isinstance(config, dict) else {}
    target_cfg_run = config.get("target_cfg", {}) if isinstance(config, dict) else {}

    past_len = int(window_cfg_run.get("past_len", 30))
    horizon = int(window_cfg_run.get("horizon", 40))
    max_rul = int(target_cfg_run.get("max_rul", 125))
    
    X_full, y_full, unit_ids_full, cond_ids_full, health_phys_seq_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        horizon=horizon,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )
    
    # Determine number of unique conditions
    unique_conditions = torch.unique(cond_ids_full).cpu().numpy()
    num_conditions = len(unique_conditions)
    print(f"Found {num_conditions} unique conditions: {unique_conditions}")
    
    # Override num_conditions from config if condition embedding is enabled
    phase_2_params = config['phase_2_params']
    if phase_2_params['use_condition_embedding']:
        phase_2_params['num_conditions'] = num_conditions
    
    print("\n[3] Creating dataloaders...")
    train_loader, val_loader, scaler, _, _ = create_full_dataloaders(
        X=X_full,
        y=y_full,
        unit_ids=unit_ids_full,
        cond_ids=cond_ids_full,
        health_phys_seq=health_phys_seq_full,
        batch_size=config['training_params']['batch_size'],
        engine_train_ratio=config['training_params']['engine_train_ratio'],
        shuffle_engines=config['training_params']['shuffle_engines'],
        random_seed=config['training_params']['random_seed'],
        use_condition_wise_scaling=True,
        censoring_aware_training=bool(config["training_params"].get("censoring_aware_training", False)),
        num_truncations_per_engine=int(config["training_params"].get("num_truncations_per_engine", 5)),
        trunc_p_full=float(config["training_params"].get("trunc_p_full", 0.25)),
        trunc_r_min=float(config["training_params"].get("trunc_r_min", 0.4)),
        trunc_r_max=float(config["training_params"].get("trunc_r_max", 1.0)),
        use_multiview_censoring=bool(config["training_params"].get("use_multiview_censoring", False)),
        num_windows_per_truncation=int(config["training_params"].get("num_windows_per_truncation", 8)),
        trunc_ratio_min=float(config["training_params"].get("trunc_ratio_min", 0.4)),
        trunc_ratio_max=float(config["training_params"].get("trunc_ratio_max", 1.0)),
        aux_sample_ratio=float(config["training_params"].get("aux_sample_ratio", 0.3)),
        run_seed=int(config["training_params"].get("run_seed", config["training_params"].get("random_seed", 42))),
    )
    
    # ===================================================================
    # Initialize Model
    # ===================================================================
    print("\n[4] Initializing model...")
    encoder_kwargs = config['encoder_kwargs']
    encoder_type = config['encoder_type']
    
    # Map encoder_kwargs to model parameters and create model
    if encoder_type == "universal_v2":
        # Phase 3.2: Universal Encoder V2 (Enhanced)
        d_model = encoder_kwargs.get('d_model', 64)
        num_layers = encoder_kwargs.get('num_layers', 3)
        dropout = encoder_kwargs.get('dropout', 0.1)
        nhead = encoder_kwargs.get('nhead', 4)
        dim_feedforward = encoder_kwargs.get('dim_feedforward', None)  # Will default to 4*d_model
        kernel_sizes = encoder_kwargs.get('kernel_sizes', [3, 5, 9])
        seq_encoder_type = encoder_kwargs.get('seq_encoder_type', 'transformer')
        use_layer_norm = encoder_kwargs.get('use_layer_norm', True)
        
        # Determine num_conditions
        if phase_2_params['use_condition_embedding']:
            num_conditions = phase_2_params['num_conditions']
        else:
            num_conditions = None  # Single condition or no condition fusion
        
        # Create Universal Encoder V2
        encoder = UniversalEncoderV2(
            input_dim=X_full.shape[-1],
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=phase_2_params.get('cond_emb_dim', 4),
            max_seq_len=300,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
        )
        
        # Create RUL+HI model
        model = RULHIUniversalModelV2(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    elif encoder_type == "universal_v1":
        # Phase 3: Universal Encoder V1
        d_model = encoder_kwargs.get('d_model', 48)
        cnn_channels = encoder_kwargs.get('cnn_channels', None)  # Will default to d_model // 3
        num_layers = encoder_kwargs.get('num_layers', 3)
        dropout = encoder_kwargs.get('dropout', 0.1)
        nhead = encoder_kwargs.get('nhead', 4)
        dim_feedforward = encoder_kwargs.get('dim_feedforward', 256)
        
        # Determine num_conditions
        if phase_2_params['use_condition_embedding']:
            num_conditions = phase_2_params['num_conditions']
        else:
            num_conditions = None  # Single condition or no condition fusion
        
        # Create Universal Encoder
        encoder = UniversalEncoderV1(
            input_dim=X_full.shape[-1],
            d_model=d_model,
            cnn_channels=cnn_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=phase_2_params.get('cond_emb_dim', 4),
            max_seq_len=300,
        )
        
        # Create RUL+HI model
        model = RULHIUniversalModel(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    elif encoder_type == "universal_v3_attention":
        # Transformer + Attention encoder (UniversalEncoderV3Attention)
        d_model = encoder_kwargs.get("d_model", 64)
        num_layers = encoder_kwargs.get("num_layers", 3)
        dropout = encoder_kwargs.get("dropout", 0.1)
        nhead = encoder_kwargs.get("nhead", 4)
        dim_feedforward = encoder_kwargs.get("dim_feedforward", None)
        kernel_sizes = encoder_kwargs.get("kernel_sizes", [3, 5, 9])
        use_ms_cnn = encoder_kwargs.get("use_ms_cnn", True)

        # Determine number of conditions
        if phase_2_params["use_condition_embedding"]:
            num_conditions = phase_2_params["num_conditions"]
        else:
            num_conditions = None

        model = UniversalEncoderV3Attention(
            input_dim=X_full.shape[-1],
            d_model=d_model,
            num_layers=num_layers,
            n_heads=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_ms_cnn=use_ms_cnn,
            kernel_sizes=kernel_sizes,
            num_conditions=num_conditions,
            condition_embedding_dim=phase_2_params.get("cond_emb_dim", 4),
            max_seq_len=300,
        )
    elif encoder_type == "transformer_encoder_v1":
        # New Transformer-Encoder EOL model (EOLFullTransformerEncoder)
        d_model = encoder_kwargs.get("d_model", 64)
        num_layers = encoder_kwargs.get("num_layers", 3)
        dropout = encoder_kwargs.get("dropout", 0.1)
        n_heads = encoder_kwargs.get("n_heads", encoder_kwargs.get("nhead", 4))
        dim_feedforward = encoder_kwargs.get("dim_feedforward", None)

        # NEW: continuous condition encoder (Cond_* per timestep) support.
        # If enabled via encoder_kwargs['use_cond_encoder'], infer cond_in_dim from Cond_* features.
        use_cond_encoder = encoder_kwargs.get("use_cond_encoder", False)
        cond_in_dim = encoder_kwargs.get("cond_in_dim", 0)
        if use_cond_encoder and (cond_in_dim is None or cond_in_dim == 0):
            cond_feature_indices = [i for i, c in enumerate(feature_cols) if c.startswith("Cond_")]
            cond_in_dim = len(cond_feature_indices)
            encoder_kwargs["cond_in_dim"] = cond_in_dim
        else:
            cond_feature_indices = None

        # Optional: sensor feature indices for condition normalisation (v5)
        sensor_feature_indices_for_norm = None
        try:
            groups = group_feature_columns(feature_cols)
            residual_cols = set(groups.get("residual", []))
            if residual_cols:
                sensor_feature_indices_for_norm = [
                    i for i, c in enumerate(feature_cols) if c in residual_cols
                ]
        except Exception as e:
            print(f"[Init] Warning: could not derive sensor_feature_indices_for_norm: {e}")

        # Damage head parameters (from encoder_kwargs or config['model'])
        model_cfg_raw = config.get("model", {})
        use_damage_head = encoder_kwargs.get("use_damage_head", model_cfg_raw.get("use_damage_head", False))
        damage_L_ref = encoder_kwargs.get("L_ref", model_cfg_raw.get("L_ref", 300.0))
        damage_alpha_base = encoder_kwargs.get("alpha_base", model_cfg_raw.get("alpha_base", 0.1))
        damage_hidden_dim = encoder_kwargs.get("damage_hidden_dim", model_cfg_raw.get("damage_hidden_dim", 64))
        # NEW (v3c): optional MLP-based damage head configuration
        damage_use_mlp = encoder_kwargs.get(
            "damage_use_mlp", model_cfg_raw.get("damage_use_mlp", False)
        )
        damage_mlp_hidden_factor = encoder_kwargs.get(
            "damage_mlp_hidden_factor", model_cfg_raw.get("damage_mlp_hidden_factor", 2)
        )
        damage_mlp_num_layers = encoder_kwargs.get(
            "damage_mlp_num_layers", model_cfg_raw.get("damage_mlp_num_layers", 2)
        )
        damage_mlp_dropout = encoder_kwargs.get(
            "damage_mlp_dropout", model_cfg_raw.get("damage_mlp_dropout", 0.1)
        )

        # Store in encoder_kwargs so they get saved to summary.json
        encoder_kwargs["use_damage_head"] = use_damage_head
        encoder_kwargs["L_ref"] = damage_L_ref
        encoder_kwargs["alpha_base"] = damage_alpha_base
        encoder_kwargs["damage_hidden_dim"] = damage_hidden_dim
        encoder_kwargs["damage_use_mlp"] = damage_use_mlp
        encoder_kwargs["damage_mlp_hidden_factor"] = damage_mlp_hidden_factor
        encoder_kwargs["damage_mlp_num_layers"] = damage_mlp_num_layers
        encoder_kwargs["damage_mlp_dropout"] = damage_mlp_dropout
        
        # NEW (v3d): delta cumsum parameters
        damage_use_delta_cumsum = encoder_kwargs.get(
            "damage_use_delta_cumsum", model_cfg_raw.get("damage_use_delta_cumsum", False)
        )
        damage_delta_alpha = encoder_kwargs.get(
            "damage_delta_alpha", model_cfg_raw.get("damage_delta_alpha", 1.0)
        )
        # NEW (v3e): temporal smoothing
        damage_use_temporal_conv = encoder_kwargs.get(
            "damage_use_temporal_conv", model_cfg_raw.get("damage_use_temporal_conv", False)
        )
        damage_temporal_conv_kernel_size = encoder_kwargs.get(
            "damage_temporal_conv_kernel_size", model_cfg_raw.get("damage_temporal_conv_kernel_size", 3)
        )
        damage_temporal_conv_num_layers = encoder_kwargs.get(
            "damage_temporal_conv_num_layers", model_cfg_raw.get("damage_temporal_conv_num_layers", 1)
        )

        encoder_kwargs["damage_use_delta_cumsum"] = damage_use_delta_cumsum
        encoder_kwargs["damage_delta_alpha"] = damage_delta_alpha
        encoder_kwargs["damage_use_temporal_conv"] = damage_use_temporal_conv
        encoder_kwargs["damage_temporal_conv_kernel_size"] = damage_temporal_conv_kernel_size
        encoder_kwargs["damage_temporal_conv_num_layers"] = damage_temporal_conv_num_layers

        model = EOLFullTransformerEncoder(
            input_dim=X_full.shape[-1],
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_condition_embedding=phase_2_params["use_condition_embedding"],
            num_conditions=phase_2_params["num_conditions"],
            cond_emb_dim=phase_2_params["cond_emb_dim"],
            max_seq_len=300,
            use_cond_encoder=use_cond_encoder,
            cond_in_dim=cond_in_dim,
            cond_encoder_dim=encoder_kwargs.get("cond_encoder_dim", None),
            use_cond_recon_head=encoder_kwargs.get("use_cond_recon_head", False),
            # Damage head parameters
            use_damage_head=use_damage_head,
            damage_L_ref=damage_L_ref,
            damage_alpha_base=damage_alpha_base,
            damage_hidden_dim=damage_hidden_dim,
            damage_use_mlp=damage_use_mlp,
            damage_mlp_hidden_factor=damage_mlp_hidden_factor,
            damage_mlp_num_layers=damage_mlp_num_layers,
            damage_mlp_dropout=damage_mlp_dropout,
            damage_use_delta_cumsum=damage_use_delta_cumsum,
            damage_delta_alpha=damage_delta_alpha,
            damage_use_temporal_conv=damage_use_temporal_conv,
            damage_temporal_conv_kernel_size=damage_temporal_conv_kernel_size,
            damage_temporal_conv_num_layers=damage_temporal_conv_num_layers,
            # v4/v5: calibrated HI head and optional HI_cal fusion / condition normaliser
            use_hi_cal_head=encoder_kwargs.get("use_hi_cal_head", False),
            use_condition_normalizer=encoder_kwargs.get("use_condition_normalizer", False),
            condition_normalizer_hidden_dim=encoder_kwargs.get("condition_normalizer_hidden_dim", 64),
            use_hi_cal_fusion_for_rul=encoder_kwargs.get("use_hi_cal_fusion_for_rul", False),
            # v5u: optional uncertainty head for RUL at last observed cycle
            use_rul_uncertainty_head=encoder_kwargs.get("use_rul_uncertainty_head", False),
            rul_uncertainty_min_sigma=encoder_kwargs.get("rul_uncertainty_min_sigma", 1e-3),
            # v5q: optional quantile head for RUL at last observed cycle
            use_rul_quantiles_head=encoder_kwargs.get("use_rul_quantiles_head", False),
            rul_quantiles=tuple(encoder_kwargs.get("rul_quantiles", (0.1, 0.5, 0.9))),
            # risk: residual quantile risk head (overshoot)
            use_residual_risk_head=encoder_kwargs.get("use_residual_risk_head", False),
            residual_risk_hidden_dim=int(encoder_kwargs.get("residual_risk_hidden_dim", 128)),
            # Censoring-aware: optional bucket head for RUL
            use_bucket_head=encoder_kwargs.get("use_bucket_head", False),
            rul_bucket_edges=tuple(encoder_kwargs.get("rul_bucket_edges", (25.0, 50.0, 75.0, 100.0, 125.0))),
        )

        # If we inferred Cond_* feature indices, attach them to the model so it can
        # derive cond_seq internally from the full feature sequence.
        if cond_feature_indices is not None and hasattr(model, "cond_feature_indices"):
            model.cond_feature_indices = torch.as_tensor(
                cond_feature_indices, dtype=torch.long, device=device
            )

        # Attach sensor indices for condition normalisation (v5) and
        # initialise ConditionNormalizer once dims are known.
        if (
            sensor_feature_indices_for_norm is not None
            and getattr(model, "use_condition_normalizer", False)
            and len(sensor_feature_indices_for_norm) > 0
        ):
            model.sensor_feature_indices_for_norm = torch.as_tensor(
                sensor_feature_indices_for_norm, dtype=torch.long, device=device
            )
            if cond_feature_indices is not None:
                model.set_condition_normalizer_dims(
                    cond_dim=len(cond_feature_indices),
                    sensor_dim=len(sensor_feature_indices_for_norm),
                )
    elif encoder_type == "transformer_state_encoder_v3":
        # State encoder V3 is trained via a dedicated routine (train_state_encoder_v3)
        # and does not use the generic EOL training loop below. We handle it
        # separately further down in run_single_experiment.
        print("[Init] TransformerStateEncoderV3 will be initialised inside train_state_encoder_v3.")
        model = None

        # Optional: configure advanced RUL head (phys_v3/phys_v4) in a backwards-compatible way.
        rul_head_params = config.get("rul_head_params", {})
        rul_head_type = rul_head_params.get("rul_head_type", "linear")
        if rul_head_type == "improved":
            from src.models.transformer_eol import ImprovedRULHead

            model.rul_head_type = "improved"
            model.max_rul = max_rul
            model.tau = config["loss_params"]["rul_beta"]

            hidden_dim = rul_head_params.get("hidden_dim", 128)
            num_hidden_layers = rul_head_params.get("num_hidden_layers", 3)
            head_dropout = rul_head_params.get("dropout", dropout)
            use_skip = rul_head_params.get("use_skip", True)
            use_hi_fusion = rul_head_params.get("use_hi_fusion", True)
            use_piecewise = rul_head_params.get("use_piecewise_mapping", True)

            hi_dim = 1 if use_hi_fusion else None

            model.rul_head = ImprovedRULHead(
                input_dim=d_model,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=head_dropout,
                use_skip=use_skip,
                use_hi_fusion=use_hi_fusion,
                hi_dim=hi_dim,
                use_piecewise_mapping=use_piecewise,
                max_rul=max_rul,
                tau=config["loss_params"]["rul_beta"],
            )
        elif rul_head_type == "v4":
            from src.models.transformer_eol import RULHeadV4

            model.rul_head_type = "v4"
            model.max_rul = max_rul
            model.tau = config["loss_params"]["rul_beta"]

            model.rul_head = RULHeadV4(
                d_model=d_model,
                max_rul=max_rul,
            )
    elif encoder_type == "lstm":
        hidden_dim = encoder_kwargs['hidden_dim']
        num_layers = encoder_kwargs['num_layers']
        dropout = encoder_kwargs['dropout']
        bidirectional = encoder_kwargs.get('bidirectional', False)
        transformer_nhead = 4  # Not used for LSTM
        transformer_dim_feedforward = 256  # Not used for LSTM
        
        model = EOLFullLSTMWithHealth(
            input_dim=X_full.shape[-1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            lambda_health=config['loss_params']['health_loss_weight'],
            # Phase 2: Condition embeddings
            use_condition_embedding=phase_2_params['use_condition_embedding'],
            num_conditions=phase_2_params['num_conditions'],
            cond_emb_dim=phase_2_params['cond_emb_dim'],
            # Phase 2: Encoder type
            encoder_type=encoder_type,
            transformer_nhead=transformer_nhead,
            transformer_dim_feedforward=transformer_dim_feedforward,
        )
    else:  # transformer
        hidden_dim = encoder_kwargs['d_model']
        num_layers = encoder_kwargs['num_layers']
        dropout = encoder_kwargs['dropout']
        bidirectional = False  # Not applicable for Transformer
        transformer_nhead = encoder_kwargs['nhead']
        transformer_dim_feedforward = encoder_kwargs['dim_feedforward']
        
        model = EOLFullLSTMWithHealth(
            input_dim=X_full.shape[-1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            lambda_health=config['loss_params']['health_loss_weight'],
            # Phase 2: Condition embeddings
            use_condition_embedding=phase_2_params['use_condition_embedding'],
            num_conditions=phase_2_params['num_conditions'],
            cond_emb_dim=phase_2_params['cond_emb_dim'],
            # Phase 2: Encoder type
            encoder_type=encoder_type,
            transformer_nhead=transformer_nhead,
            transformer_dim_feedforward=transformer_dim_feedforward,
        )
    model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    if phase_2_params['use_condition_embedding']:
        print(f"  - Condition embeddings: {phase_2_params['num_conditions']} conditions, "
              f"{phase_2_params['cond_emb_dim']}D embeddings")
    print(f"  - Encoder type: {encoder_type}")
    if encoder_type == "universal_v2":
        dim_ff = dim_feedforward if dim_feedforward is not None else 4 * d_model
        print(f"  - Universal Encoder V2: d_model={d_model}, nhead={nhead}, "
              f"dim_feedforward={dim_ff}, num_layers={num_layers}, seq_encoder={seq_encoder_type}")
        print(f"  - CNN kernel sizes: {kernel_sizes}")
    elif encoder_type == "universal_v1":
        print(f"  - Universal Encoder V1: d_model={d_model}, nhead={nhead}, "
              f"dim_feedforward={dim_feedforward}, num_layers={num_layers}")
        if cnn_channels is not None:
            print(f"  - CNN channels: {cnn_channels}")
    elif encoder_type == "transformer":
        print(f"  - Transformer: d_model={hidden_dim}, nhead={transformer_nhead}, "
              f"dim_feedforward={transformer_dim_feedforward}, num_layers={num_layers}")
    
    # ===================================================================
    # Training
    # ===================================================================
    print("\n[5] Training model...")
    results_dir = Path("results") / dataset_name.lower() / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Optional two-phase schedule for damage HI (used in damage_v3c / v3d / v4)
    training_cfg = config.get("training_params", {})
    damage_two_phase = training_cfg.get("damage_two_phase", False)
    damage_warmup_epochs = training_cfg.get("damage_warmup_epochs", 0)
    damage_phase1_damage_weight = training_cfg.get(
        "damage_phase1_damage_weight", config["loss_params"].get("damage_hi_weight", 0.0)
    )
    damage_phase2_damage_weight = training_cfg.get(
        "damage_phase2_damage_weight", config["loss_params"].get("damage_hi_weight", 0.0)
    )

    model, history = train_eol_full_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training_params']['num_epochs'],
        lr=config['optimizer_params']['lr'],
        weight_decay=config['optimizer_params']['weight_decay'],
        patience=config['training_params']['patience'],
        device=device,
        results_dir=results_dir,
        run_name=experiment_name,
        use_mixed_precision=config['training_params']['use_mixed_precision'],
        use_health_head=True,
        max_rul=max_rul,
        tau=config['loss_params']['rul_beta'],
        lambda_health=config['loss_params']['health_loss_weight'],
        hi_condition_calib_weight=config['loss_params']['hi_condition_calib_weight'],
        hi_plateau_threshold=HI_RUL_PLATEAU_THRESH,
        hi_mono_late_weight=config['loss_params']['mono_late_weight'],
        hi_mono_global_weight=config['loss_params']['mono_global_weight'],
        # Phase 2: Smoothness loss
        smooth_hi_weight=phase_2_params['smooth_hi_weight'],
        smooth_hi_plateau_threshold=phase_2_params['smooth_hi_plateau_threshold'],
        # Phase 2: Condition embedding flag
        use_condition_embedding=phase_2_params['use_condition_embedding'],
        # NEW: auxiliary condition reconstruction loss (Transformer encoder V2)
        cond_recon_weight=config['loss_params'].get('cond_recon_weight', 0.0),
        # NEW: cumulative damage-based HI loss (enabled only for damage-head configs)
        damage_hi_weight=config['loss_params'].get('damage_hi_weight', 0.0),
        # NEW (v3c): two-phase damage HI training schedule
        damage_two_phase=damage_two_phase,
        damage_warmup_epochs=damage_warmup_epochs,
        damage_phase1_damage_weight=damage_phase1_damage_weight,
        damage_phase2_damage_weight=damage_phase2_damage_weight,
        # NEW (v4): calibrated HI (HI_cal_v2) supervision and slope regularisation
        hi_cal_weight=config['loss_params'].get('w_hi_cal', 0.0),
        hi_cal_mono_weight=config['loss_params'].get('w_mono_hi_cal', 0.0),
        hi_cal_slope_weight=config['loss_params'].get('w_slope_hi_cal', 0.0),
        hi_calibrator_path=config.get('hi_calibrator_path', None),
        # v5u: Gaussian NLL for RUL using predicted sigma
        rul_nll_weight=config['loss_params'].get('rul_nll_weight', 0.0),
        rul_nll_min_sigma=encoder_kwargs.get("rul_uncertainty_min_sigma", 1e-3),
        rul_nll_detach_mu=bool(config["loss_params"].get("rul_nll_detach_mu", False)),
        # v5q: quantile loss for RUL at last observed cycle
        rul_quantile_weight=config["loss_params"].get("rul_quantile_weight", 0.0),
        rul_quantiles=config["loss_params"].get("rul_quantiles", None),
        rul_quantile_cross_weight=config["loss_params"].get("rul_quantile_cross_weight", 0.0),
        rul_quantile_p50_mse_weight=config["loss_params"].get("rul_quantile_p50_mse_weight", 0.0),
        # Quantile usability: upper-tail risk penalty + optional bias calibration
        rul_risk_weight=float(config["loss_params"].get("lambda_risk", config["loss_params"].get("rul_risk_weight", 0.0))),
        rul_risk_margin=float(config["loss_params"].get("risk_margin", config["loss_params"].get("rul_risk_margin", 0.0))),
        rul_q50_bias_weight=float(config["loss_params"].get("lambda_q50_bias", config["loss_params"].get("rul_q50_bias_weight", 0.0))),
        rul_bias_calibration_mode=str(config["loss_params"].get("bias_calibration_mode", config["loss_params"].get("rul_bias_calibration_mode", "off"))),
        rul_bias_ema_beta=float(config["loss_params"].get("bias_ema_beta", config["loss_params"].get("rul_bias_ema_beta", 0.98))),
        # Residual risk head (safe_RUL): predict overshoot quantile and subtract in inference
        use_residual_risk_head=bool(encoder_kwargs.get("use_residual_risk_head", False)),
        residual_risk_tau=float(config["loss_params"].get("risk_tau", 0.90)),
        residual_risk_weight=float(config["loss_params"].get("lambda_residual_risk", config["loss_params"].get("lambda_risk", 0.0))),
        # Residual risk v2b: emphasize low-RUL region (safety-critical tail)
        residual_risk_low_rul_threshold=float(config["loss_params"].get("low_rul_threshold", 20.0)),
        residual_risk_low_weight=float(config["loss_params"].get("risk_low_weight", 1.0)),
        residual_risk_low_only=bool(config["loss_params"].get("risk_low_only", False)),
        # Training monitor metric (scheduler + early stopping)
        monitor_metric=str(training_cfg.get("monitor_metric", "val_loss")),
        # Censoring-aware: ranking loss on mu
        use_ranking_loss=bool(config.get("loss_params", {}).get("use_ranking_loss", True)),
        lambda_rank=float(config.get("loss_params", {}).get("lambda_rank", 0.1)),
        rank_margin=float(config.get("loss_params", {}).get("rank_margin", 1.0)),
        # Censoring-aware: bucket head loss (must match model construction; default OFF for backward compatibility)
        use_bucket_head=bool(encoder_kwargs.get("use_bucket_head", False)),
        lambda_bucket=float(config.get("loss_params", {}).get("lambda_bucket", 0.1)),
        rul_bucket_edges=encoder_kwargs.get("rul_bucket_edges", None),
    )
    
    # ===================================================================
    # Evaluation
    # ===================================================================
    print("\n[6] Evaluating on validation set...")
    val_metrics = evaluate_eol_full_lstm(
        model=model,
        val_loader=val_loader,
        device=device,
    )
    
    print("\n[7] Evaluating on test set...")
    test_metrics = evaluate_on_test_data(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler=scaler,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        device=device,
    )
    
    # ===================================================================
    # Build Summary
    # ===================================================================
    # Determine phase from experiment name
    if "phase3" in experiment_name:
        phase = "phase3"
    elif "phase2" in experiment_name:
        phase = "phase2"
    else:
        phase = "phase1"
    
    # Build summary with clean structure
    rul_head_params = config.get("rul_head_params", {})
    phys_features_cfg = config.get("phys_features", {})
    features_cfg = config.get("features", {})

    # Persist window/target policy (single source of truth for windowing)
    # EOL models train on scalar targets (current RUL at window end), but we also
    # build padded horizon targets internally for consistent EOL-near-0 coverage/logs.
    window_cfg_summary = {
        "past_len": int(past_len),
        "horizon": int(config.get("window_cfg", {}).get("horizon", 40)),
        "stride": int(config.get("window_cfg", {}).get("stride", 1)),
        "pad_mode": str(config.get("window_cfg", {}).get("pad_mode", "clamp")),
        "require_full_horizon": bool(config.get("window_cfg", {}).get("require_full_horizon", False)),
    }
    target_cfg_summary = {
        "max_rul": int(max_rul),
        "cap_targets": True,
        "eol_target_mode": str(config.get("target_cfg", {}).get("eol_target_mode", "current_from_df")),
        "clip_eval_y_true": True,
    }
    summary = {
        "_meta": {
            "generated_git_sha": _get_git_sha(),
        },
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "phase": phase,
        "encoder_type": encoder_type,
        **encoder_kwargs,  # Include all encoder parameters
        "use_condition_embedding": phase_2_params['use_condition_embedding'],
        "cond_emb_dim": phase_2_params['cond_emb_dim'] if phase_2_params['use_condition_embedding'] else None,
        "smooth_hi_weight": phase_2_params['smooth_hi_weight'],
        **config['loss_params'],  # Include all loss parameters
        # RUL head configuration (for Transformer-Encoder variants)
        "rul_head_type": rul_head_params.get("rul_head_type", "linear"),
        "use_piecewise_rul_mapping": rul_head_params.get("use_piecewise_mapping", False),
        "rul_head_hidden_dim": rul_head_params.get("hidden_dim"),
        "rul_head_num_layers": rul_head_params.get("num_hidden_layers"),
        "rul_head_dropout": rul_head_params.get("dropout"),
        "rul_head_use_skip": rul_head_params.get("use_skip"),
        "rul_head_use_hi_fusion": rul_head_params.get("use_hi_fusion"),
        # Feature configuration (multi-scale etc.) – persisted so diagnostics/inference
        # can exactly mirror the training-time feature pipeline.
        "features": features_cfg,
        # Window/target policy (shared across pipelines)
        "window_cfg": window_cfg_summary,
        "target_cfg": target_cfg_summary,
        # Condition-vector configuration (phys_v2/v3/v4)
        "condition_vector_version": phys_features_cfg.get("condition_vector_version", 2),
        # Persist phys_features so diagnostics can exactly mirror training pipeline
        "phys_features": phys_features_cfg,
        # Validation metrics
        "val_metrics": {
            "rmse": val_metrics["pointwise"]["rmse"],
            "mae": val_metrics["pointwise"]["mae"],
            "bias": val_metrics["pointwise"]["bias"],
            "r2": val_metrics["pointwise"]["r2"],
            "nasa_mean": val_metrics["nasa_pointwise"]["score_mean"],
            "nasa_sum": val_metrics["nasa_pointwise"]["score_sum"],
        },
        # Test metrics (official evaluation, EOL-based)
        "test_metrics": {
            "rmse": test_metrics["pointwise"]["rmse"],
            "mae": test_metrics["pointwise"]["mae"],
            "bias": test_metrics["pointwise"]["bias"],
            "r2": test_metrics["pointwise"]["r2"],
            "nasa_mean": test_metrics["nasa_pointwise"]["score_mean"],
            "nasa_sum": test_metrics["nasa_pointwise"]["score_sum"],
            "num_engines": len(test_metrics.get("y_true", [])),
        },
        # Backward compatibility: keep flat keys
        "val_rmse": val_metrics["pointwise"]["rmse"],
        "val_mae": val_metrics["pointwise"]["mae"],
        "val_bias": val_metrics["pointwise"]["bias"],
        "val_r2": val_metrics["pointwise"]["r2"],
        "val_nasa_mean": val_metrics["nasa_pointwise"]["score_mean"],
        "test_rmse": test_metrics["pointwise"]["rmse"],
        "test_mae": test_metrics["pointwise"]["mae"],
        "test_bias": test_metrics["pointwise"]["bias"],
        "test_r2": test_metrics["pointwise"]["r2"],
        "test_nasa_mean": test_metrics["nasa_pointwise"]["score_mean"],
    }
    
    # Save scaler for inference
    import pickle
    scaler_path = results_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[8] Saved summary to {summary_path}")
    
    # ===================================================================
    # Phase 3.2: Generate comprehensive diagnostics
    # ===================================================================
    print("\n[9] Generating comprehensive diagnostics...")
    try:
        # Use the new notebook-based diagnostics with sliding-window HI for all datasets
        from src.analysis.diagnostics import run_diagnostics_for_run
        
        print(f"Using diagnostics (sliding-window HI, degraded engines) for {dataset_name}...")
        # run_diagnostics_for_run expects: exp_dir="results", dataset_name="FD004", run_name="fd004_phase3_..."
        # results_dir is already "results/<dataset>/<experiment_name>"
        run_diagnostics_for_run(
            exp_dir=results_dir.parent.parent,  # results/ (go up from results/<dataset>/<name> to results/)
            dataset_name=dataset_name,
            run_name=experiment_name,
            device=device,
        )
        
        # Reload summary to get updated info
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Warning: Could not generate diagnostics: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Experiment Complete: {experiment_name}")
    print("=" * 80)
    # Print training metrics (authoritative) + diagnostics metrics (secondary) if present.
    print("Test Metrics (training eval_on_test_data):")
    print(f"  RMSE: {test_metrics['pointwise']['rmse']:.2f} cycles")
    print(f"  MAE:  {test_metrics['pointwise']['mae']:.2f} cycles")
    print(f"  Bias: {test_metrics['pointwise']['bias']:.2f} cycles")
    print(f"  R²:   {test_metrics['pointwise']['r2']:.4f}")
    print(f"  NASA Mean: {test_metrics['nasa_pointwise']['score_mean']:.2f}")

    diag_tm = None
    if isinstance(summary, dict):
        diag_tm = summary.get("diagnostics_test_metrics") or summary.get("test_metrics_diagnostics")
    if isinstance(diag_tm, dict) and all(k in diag_tm for k in ["rmse", "mae", "bias", "r2", "nasa_mean"]):
        print("Test Metrics (diagnostics):")
        print(f"  RMSE: {diag_tm['rmse']:.2f} cycles")
        print(f"  MAE:  {diag_tm['mae']:.2f} cycles")
        print(f"  Bias: {diag_tm['bias']:.2f} cycles")
        print(f"  R²:   {diag_tm['r2']:.4f}")
        print(f"  NASA Mean: {diag_tm['nasa_mean']:.2f}")
    print("=" * 80)
    
    _registry_finish(summary, results_dir=results_dir, summary_path=results_dir / "summary.json")
    return summary


def main():
    # Example (Dynamic Latent World Model A+ on FD004):
    #   python run_experiments.py --experiments fd004_transformer_latent_worldmodel_dynamic_v1 --device cuda
    #   python run_experiments.py --experiments fd004_transformer_latent_worldmodel_dynamic_delta_v2 --device cuda
    parser = argparse.ArgumentParser(description="Run RUL prediction experiments")
    parser.add_argument(
        "--group",
        type=str,
        choices=[
            "A", "B", "C",
            "P3", "P3_2", "P3_2_FD004",
            "P4", "phase4", "fd_all_phase4_residual",
            "all",
            "lstm_baselines", "transformer_baselines",
            "fd004_sweep", "phase3", "universal_v1", "fd004_tuning",
            "world_phase4", "world_p4", "world_eol_heavy", "world_h10_eol_heavy",
            "world_phase5", "world_p5", "world_v3",
            "world_v3_fd001_fd002", "world_phase5_fd001_fd002",
            "world_v3_all", "world_phase5_all",
            "TE", "transformer_encoder_all",
        ],
        help=(
            "Experiment group to run (A: LSTM baselines, B: Transformer baselines, "
            "C: FD004 sweep, P3: UniversalEncoderV1, P4: Phase 4 residual features for all datasets, "
            "world_phase5/world_p5/world_v3: World Model v3 experiments, "
            "TE/transformer_encoder_all: Transformer-Encoder EOL+HI on FD001–FD004, "
            "all: all groups)"
        ),
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Specific experiment names to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Run all experiments for a specific dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help=(
            "Override num_epochs for a quick smoke run (e.g. 3–5). "
            "Does not change the experiment configs on disk."
        ),
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Collect experiments to run
    experiments: List[ExperimentConfig] = []
    
    if args.group:
        experiments.extend(get_experiment_group(args.group))
    
    if args.experiments:
        for exp_name in args.experiments:
            experiments.append(get_experiment_by_name(exp_name))
    
    if args.dataset:
        # Add all experiments for this dataset
        from src.experiment_configs import get_lstm_baseline_config, get_transformer_baseline_config
        experiments.append(get_lstm_baseline_config(args.dataset))
        experiments.append(get_transformer_baseline_config(args.dataset))
        if args.dataset == "FD004":
            experiments.extend(get_experiment_group("C"))
    
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
    print(f"Running {len(experiments)} experiment(s)")
    print(f"{'=' * 80}")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['experiment_name']} ({exp['dataset']}, {exp['encoder_type']})")
    
    # Run experiments
    all_summaries = []
    for i, config in enumerate(experiments, 1):
        print(f"\n\n{'#' * 80}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'#' * 80}")
        try:
            if args.smoke_epochs is not None:
                try:
                    config.setdefault("training_params", {})
                    config["training_params"]["num_epochs"] = int(args.smoke_epochs)
                    # Keep patience bounded so smoke runs don't early-stop instantly.
                    if "patience" in config["training_params"]:
                        config["training_params"]["patience"] = min(
                            int(config["training_params"]["patience"]),
                            int(args.smoke_epochs),
                        )
                except Exception as e:
                    print(f"  ⚠️  Could not apply --smoke-epochs override: {e}")
            summary = run_single_experiment(config, device)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n❌ Error running {config['experiment_name']}: {e}")
            traceback.print_exc()
            # Best-effort: mark the latest running registry entry as failed.
            try:
                if os.environ.get("RUN_REGISTRY_DISABLE", "").strip() not in {"1", "true", "True", "yes"}:
                    from src.tools.run_registry import RunRegistry

                    db_path = Path(os.environ.get("RUN_REGISTRY_DB", str(Path("artifacts") / "run_registry.sqlite")))
                    reg = RunRegistry(db_path)
                    try:
                        rid = reg.find_latest_run_id(
                            experiment_name=config.get("experiment_name", ""),
                            dataset=config.get("dataset", ""),
                            status="running",
                        )
                        if rid is not None:
                            reg.fail_run(
                                rid,
                                error_message=f"{type(e).__name__}: {e}",
                                traceback_str=traceback.format_exc(),
                            )
                            print(f"[run_registry] Marked run_id={rid} as failed")
                    finally:
                        reg.close()
            except Exception as reg_e:
                print(f"[run_registry] WARNING: could not mark failed run: {reg_e}")
            continue
    
    print(f"\n\n{'=' * 80}")
    print(f"Completed {len(all_summaries)}/{len(experiments)} experiments")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

# Example usage for the Multi-Scale + Digital-Twin Transformer-Encoder experiments:
#   python run_experiments.py --experiments fd004_transformer_encoder_ms_dt_v1
# or to run all four datasets in one go:
#   python run_experiments.py --group transformer_encoder_ms_dt_all
# Diagnostics (after training) can then be invoked via:
#   python run_diagnostics.py --dataset FD004 --run fd004_transformer_encoder_ms_dt_v1

