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
import sys
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
    
    # For damage_v2 experiments: add HI_phys_v2 using the simpler per-row computation
    # This creates a time-resolved HI_phys_seq target for training the damage head
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

    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    # Never feed HI_* targets as input features to the encoder; they are used
    # as supervised targets for the physics state encoder.
    feature_cols = [
        c
        for c in feature_cols
        if c not in ["HI_phys_final", "HI_target_hybrid"]
    ]

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
                    lr=config['optimizer_params']['lr'],
                    weight_decay=config['optimizer_params']['weight_decay'],
                    patience=config['training_params']['patience'],
                    results_dir=Path("results") / dataset_name.lower() / experiment_name,
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
                    results_dir=Path("results") / dataset_name.lower() / experiment_name,
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
                results_dir=Path("results") / dataset_name.lower() / experiment_name,
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

        return summary
    
    # ===================================================================
    # Build Sequences and Dataloaders (EOL Models)
    # ===================================================================
    print("\n[2] Building full-trajectory sequences...")
    past_len = 30
    max_rul = 125
    
    X_full, y_full, unit_ids_full, cond_ids_full, health_phys_seq_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
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

        # Damage head parameters (from encoder_kwargs or config['model'])
        model_cfg_raw = config.get("model", {})
        use_damage_head = encoder_kwargs.get("use_damage_head", model_cfg_raw.get("use_damage_head", False))
        damage_L_ref = encoder_kwargs.get("L_ref", model_cfg_raw.get("L_ref", 300.0))
        damage_alpha_base = encoder_kwargs.get("alpha_base", model_cfg_raw.get("alpha_base", 0.1))
        damage_hidden_dim = encoder_kwargs.get("damage_hidden_dim", model_cfg_raw.get("damage_hidden_dim", 64))
        
        # Store in encoder_kwargs so they get saved to summary.json
        encoder_kwargs["use_damage_head"] = use_damage_head
        encoder_kwargs["L_ref"] = damage_L_ref
        encoder_kwargs["alpha_base"] = damage_alpha_base
        encoder_kwargs["damage_hidden_dim"] = damage_hidden_dim
        
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
        )

        # If we inferred Cond_* feature indices, attach them to the model so it can
        # derive cond_seq internally from the full feature sequence.
        if cond_feature_indices is not None and hasattr(model, "cond_feature_indices"):
            model.cond_feature_indices = cond_feature_indices
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
    summary = {
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
    print(f"Test Metrics:")
    print(f"  RMSE: {test_metrics['pointwise']['rmse']:.2f} cycles")
    print(f"  MAE: {test_metrics['pointwise']['mae']:.2f} cycles")
    print(f"  Bias: {test_metrics['pointwise']['bias']:.2f} cycles")
    print(f"  R²: {test_metrics['pointwise']['r2']:.4f}")
    print(f"  NASA Mean: {test_metrics['nasa_pointwise']['score_mean']:.2f}")
    print("=" * 80)
    
    return summary


def main():
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
            summary = run_single_experiment(config, device)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n❌ Error running {config['experiment_name']}: {e}")
            import traceback
            traceback.print_exc()
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

