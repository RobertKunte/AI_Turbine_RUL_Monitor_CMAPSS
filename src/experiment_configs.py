"""
Experiment Configuration Definitions.

This module centralizes all experiment configurations for systematic runs
across FD001-FD004 with LSTM and Transformer encoders.
"""

from typing import TypedDict, Dict, Any, List
import re
import copy
from pathlib import Path


class ExperimentConfig(TypedDict):
    """Experiment configuration structure."""
    experiment_name: str
    dataset: str  # "FD001", "FD002", "FD003", "FD004"
    encoder_type: str  # "lstm" or "transformer"
    encoder_kwargs: Dict[str, Any]
    loss_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    training_params: Dict[str, Any]
    phase_2_params: Dict[str, Any]  # Condition embeddings, smoothness, etc.


# ===================================================================
# Group A: LSTM Baselines (Phase-2 style) - All Datasets
# ===================================================================

def get_lstm_baseline_config(dataset: str) -> ExperimentConfig:
    """Get LSTM baseline config for a dataset."""
    # Determine if condition embedding should be enabled
    use_cond_emb = dataset in ["FD002", "FD004"]  # Multi-condition datasets
    num_conditions = 7 if use_cond_emb else 1
    
    return ExperimentConfig(
        experiment_name=f"{dataset.lower()}_phase2_lstm_baseline",
        dataset=dataset,
        encoder_type="lstm",
        encoder_kwargs={
            "hidden_dim": 50,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.35,
            "mono_late_weight": 0.03,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_emb,
            "cond_emb_dim": 4 if use_cond_emb else 0,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


# ===================================================================
# Group B: Transformer Cross-Dataset Baseline - All Datasets
# ===================================================================

def get_transformer_baseline_config(dataset: str) -> ExperimentConfig:
    """Get Transformer baseline config for a dataset (based on successful FD004 run)."""
    # Determine if condition embedding should be enabled
    use_cond_emb = dataset in ["FD002", "FD004"]  # Multi-condition datasets
    num_conditions = 7 if use_cond_emb else 1
    
    return ExperimentConfig(
        experiment_name=f"{dataset.lower()}_phase2_transformer_baseline",
        dataset=dataset,
        encoder_type="transformer",
        encoder_kwargs={
            "d_model": 48,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.35,
            "mono_late_weight": 0.03,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_emb,
            "cond_emb_dim": 4 if use_cond_emb else 0,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


# ===================================================================
# Group C: FD004 Transformer Hyperparameter Sweep
# ===================================================================

def get_fd004_transformer_hi_strong() -> ExperimentConfig:
    """P2T-A: Stronger HI/monotonicity for FD004."""
    return ExperimentConfig(
        experiment_name="fd004_phase2_transformer_hi_strong",
        dataset="FD004",
        encoder_type="transformer",
        encoder_kwargs={
            "d_model": 48,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.50,  # Increased from 0.35
            "mono_late_weight": 0.08,  # Increased from 0.03
            "mono_global_weight": 0.005,  # Increased from 0.003
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_transformer_hi_condcalib() -> ExperimentConfig:
    """P2T-B: Condition calibration enabled for FD004."""
    return ExperimentConfig(
        experiment_name="fd004_phase2_transformer_hi_condcalib",
        dataset="FD004",
        encoder_type="transformer",
        encoder_kwargs={
            "d_model": 48,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.50,
            "mono_late_weight": 0.08,
            "mono_global_weight": 0.005,
            "hi_condition_calib_weight": 0.03,  # Condition calibration enabled
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_transformer_small_regularized() -> ExperimentConfig:
    """P2T-C: Smaller encoder with stronger regularization for FD004."""
    return ExperimentConfig(
        experiment_name="fd004_phase2_transformer_small_regularized",
        dataset="FD004",
        encoder_type="transformer",
        encoder_kwargs={
            "d_model": 32,  # Reduced from 48
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.2,  # Increased from 0.1
            "max_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.35,  # Back to baseline
            "mono_late_weight": 0.03,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def _base_transformer_encoder_config(
    dataset: str,
    experiment_name: str,
    use_cond_emb: bool,
    num_conditions: int,
) -> ExperimentConfig:
    """
    Helper to build a Transformer-Encoder EOL+HI config for a given dataset.
    Shared hyperparameters across FD001–FD004; only condition embedding differs.
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset=dataset,
        encoder_type="transformer_encoder_v1",
        encoder_kwargs={
            "d_model": 64,
            "n_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,
            "mono_late_weight": 0.02,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_emb,
            "cond_emb_dim": 4 if use_cond_emb else 0,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd001_transformer_encoder_v1_config() -> ExperimentConfig:
    """Transformer-Encoder EOL+HI experiment on FD001 (single-condition)."""
    dataset = "FD001"
    return _base_transformer_encoder_config(
        dataset=dataset,
        experiment_name="fd001_transformer_encoder_v1",
        use_cond_emb=False,
        num_conditions=1,
    )


def get_fd002_transformer_encoder_v1_config() -> ExperimentConfig:
    """Transformer-Encoder EOL+HI experiment on FD002 (multi-condition)."""
    dataset = "FD002"
    # FD002 has 6 operating conditions; we reuse 7 as in existing FD004 configs.
    return _base_transformer_encoder_config(
        dataset=dataset,
        experiment_name="fd002_transformer_encoder_v1",
        use_cond_emb=True,
        num_conditions=7,
    )


def get_fd003_transformer_encoder_v1_config() -> ExperimentConfig:
    """Transformer-Encoder EOL+HI experiment on FD003 (single-condition, 2 faults)."""
    dataset = "FD003"
    return _base_transformer_encoder_config(
        dataset=dataset,
        experiment_name="fd003_transformer_encoder_v1",
        use_cond_emb=False,
        num_conditions=1,
    )


def get_fd004_transformer_attention_v1_config() -> ExperimentConfig:
    """
    First Transformer+Attention encoder experiment on FD004.

    Uses UniversalEncoderV3Attention (seq encoder with MS-CNN + Transformer + attention)
    in the standard EOL+HI Phase-2/3 training pipeline.
    """
    dataset = "FD004"
    # Condition embeddings enabled for FD004
    use_cond_emb = True
    num_conditions = 7

    return ExperimentConfig(
        experiment_name="fd004_transformer_attention_v1",
        dataset=dataset,
        encoder_type="universal_v3_attention",
        encoder_kwargs={
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "use_layer_norm": True,
            "seq_encoder_type": "transformer",
            "use_ms_cnn": True,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,
            "mono_late_weight": 0.02,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_emb,
            "cond_emb_dim": 4 if use_cond_emb else 0,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_transformer_encoder_v1_config() -> ExperimentConfig:
    """
    Transformer-Encoder-basierter EOL+HI Versuch auf FD004.

    Verwendet EOLFullTransformerEncoder (reiner Transformer-Encoder mit
    Condition-Embedding + Attention-Pooling) im gleichen Phase-2/3 EOL-Setup
    wie fd004_transformer_attention_v1, jedoch ohne MS-CNN-Front-End.
    """
    dataset = "FD004"
    return _base_transformer_encoder_config(
        dataset=dataset,
        experiment_name="fd004_transformer_encoder_v1",
        use_cond_emb=True,
        num_conditions=7,
    )


def get_fd004_transformer_encoder_resid_v1_config() -> ExperimentConfig:
    """
    Transformer-Encoder EOL+HI Experiment auf FD004 mit Residual-Features
    (Digital-Twin-Light: Residuals über Physik-/Sensor-Feature-Space).
    
    Identisch zu fd004_transformer_encoder_v1 bzgl. Modell- und Loss-Params,
    aber mit aktivierten Residual-Features in der Feature-Pipeline.
    """
    base_cfg = get_fd004_transformer_encoder_v1_config()
    # Kopiere und passe nur den Experimentnamen an; Residual-Features werden
    # über den Namen (\"resid\") in run_experiments/diagnostics aktiviert.
    base_cfg["experiment_name"] = "fd004_transformer_encoder_resid_v1"
    return base_cfg


def get_fd004_transformer_encoder_phys_v2_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD004.

    - Uses continuous condition vector (Cond_* features)
    - Uses a global HealthyTwinRegressor for twin predictions + residuals
    - Disables discrete ConditionID embeddings (no categorical conditions)
    """
    cfg = get_fd004_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_phys_v2"

    # Disable discrete condition embeddings – rely on continuous condition vector instead
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    # Physics / twin feature options used in run_experiments.py
    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd004_transformer_encoder_phys_v3_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD004 (v3).

    Differences to phys_v2:
        - Extended continuous condition vector (condition_vector_version=3)
        - Improved RUL head with deeper MLP, optional HI fusion and
          piecewise RUL mapping.
    """
    cfg = get_fd004_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_phys_v3"

    # Disable discrete condition embeddings – rely on continuous condition vector instead
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    # Physics / twin feature options used in run_experiments.py
    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    # Improved RUL head configuration (consumed in run_experiments.py)
    cfg["rul_head_params"] = {
        "rul_head_type": "improved",
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "dropout": 0.1,
        "use_skip": True,
        "use_hi_fusion": True,
        "use_piecewise_mapping": True,
    }

    # Optional explicit max_rul (kept in sync with training/evaluation)
    cfg["max_rul"] = 125.0

    return cfg


def get_fd004_transformer_encoder_phys_v4_config() -> ExperimentConfig:
    """
    Transformer V4: physics-informed Transformer-Encoder EOL+HI on FD004.

    - Extended continuous condition vector (condition_vector_version=3)
    - HealthyTwinRegressor residuals
    - RULHeadV4 (gated residual head with clamping to [0, max_rul])
    """
    cfg = get_fd004_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_phys_v4"

    # Disable discrete condition embeddings – rely on continuous condition vector instead
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    # Physics / twin feature options used in run_experiments.py
    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    # V4 RUL head configuration (consumed in run_experiments.py)
    cfg["rul_head_params"] = {
        "rul_head_type": "v4",
    }

    # Optional explicit max_rul (kept in sync with training/evaluation)
    cfg["max_rul"] = 125.0

    return cfg


def get_fd001_transformer_encoder_ms_dt_v1_config() -> ExperimentConfig:
    """
    Multi-Scale + Digital-Twin Transformer-Encoder experiment on FD001.

    - Uses the base Transformer-Encoder EOL+HI config for FD001
    - Enables explicit multi-scale temporal windows via `features`
    - Enables continuous condition vector + digital-twin residuals via `phys_features`
    """
    cfg = get_fd001_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd001_transformer_encoder_ms_dt_v1"

    # Feature block: explicit multi-scale configuration
    cfg["features"] = {
        "use_multiscale_features": True,
        "multiscale": {
            "windows_short": [10],
            "windows_medium": [30],
            "windows_long": [60, 120],
        },
    }

    # Physical / Digital-Twin / Condition-Vector block
    cfg["phys_features"] = {
        "use_condition_vector": True,
        "condition_vector_version": 3,
        "use_digital_twin_residuals": True,  # alias for use_twin_features
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd002_transformer_encoder_ms_dt_v1_config() -> ExperimentConfig:
    """
    Multi-Scale + Digital-Twin Transformer-Encoder experiment on FD002.

    - Uses the base Transformer-Encoder EOL+HI config for FD002
    - Enables explicit multi-scale temporal windows via `features`
    - Enables continuous condition vector + digital-twin residuals via `phys_features`
    """
    cfg = get_fd002_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd002_transformer_encoder_ms_dt_v1"

    cfg["features"] = {
        "use_multiscale_features": True,
        "multiscale": {
            "windows_short": [10],
            "windows_medium": [30],
            "windows_long": [60, 120],
        },
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "condition_vector_version": 3,
        "use_digital_twin_residuals": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd004_transformer_worldmodel_v1_h10_eolfocus_config() -> ExperimentConfig:
    """
    FD004 – Transformer World Model V1
    - Horizon = 10 (kürzerer Forecast)
    - Stärkerer Fokus auf HI/RUL (EOL)
    - Gleiche ms+DT-Featurepipeline wie fd004_transformer_encoder_ms_dt_v1
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()
    config = copy.deepcopy(base)

    config["experiment_name"] = "fd004_transformer_worldmodel_v1_h10_eolfocus"
    # World-model style routing in run_experiments.py
    config["encoder_type"] = "world_model_universal_v3"
    config["dataset"] = "FD004"

    # Mirror encoder_kwargs mapping used by the base WorldModel V1 config
    config["encoder_kwargs"] = {
        "d_model": base["encoder_kwargs"]["d_model"],
        "num_layers": base["encoder_kwargs"]["num_layers"],
        "nhead": base["encoder_kwargs"]["n_heads"],
        "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
        "dropout": base["encoder_kwargs"]["dropout"],
        "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    world_params = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 10,
        "max_rul": base.get("max_rul", 125),
        "d_model": base["encoder_kwargs"].get("d_model", 128),
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": base["encoder_kwargs"].get("dropout", 0.1),
        # Loss-Gewichte: stärkerer Fokus auf HI/RUL
        "sensor_loss_weight": 0.5,
        "hi_future_loss_weight": 1.0,
        "rul_future_loss_weight": 1.0,
        # Trainingshyperparameter (werden in run_experiments.py z.T. separat gesetzt)
        "learning_rate": base.get("optimizer_params", {}).get("lr", 3e-4),
        "weight_decay": base.get("optimizer_params", {}).get("weight_decay", 1e-4),
        "num_epochs": base.get("training_params", {}).get("num_epochs", 80),
        "batch_size": base.get("training_params", {}).get("batch_size", 256),
        # World-Model-Modi
        "target_mode": "sensors",   # Targets: zukünftige Sensorsignale (normiert)
        "init_from_rul_hi": True,   # Decoder-Init aus aktuellem RUL/HI
    }

    config["world_model_params"] = world_params
    return config


def get_fd004_transformer_worldmodel_v1_h20_residuals_eolfocus_config() -> ExperimentConfig:
    """
    FD004 – Transformer World Model V1
    - Horizon = 20 (wie V1)
    - Residual-Targets (Sensor - Twin)
    - Stärkerer Fokus auf HI/RUL (EOL)
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()
    config = copy.deepcopy(base)

    config["experiment_name"] = "fd004_transformer_worldmodel_v1_h20_residuals_eolfocus"
    config["encoder_type"] = "world_model_universal_v3"
    config["dataset"] = "FD004"

    # Mirror encoder_kwargs mapping used by the base WorldModel V1 config
    config["encoder_kwargs"] = {
        "d_model": base["encoder_kwargs"]["d_model"],
        "num_layers": base["encoder_kwargs"]["num_layers"],
        "nhead": base["encoder_kwargs"]["n_heads"],
        "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
        "dropout": base["encoder_kwargs"]["dropout"],
        "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    world_params = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 20,
        "max_rul": base.get("max_rul", 125),
        "d_model": base["encoder_kwargs"].get("d_model", 128),
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": base["encoder_kwargs"].get("dropout", 0.1),
        "sensor_loss_weight": 0.5,
        "hi_future_loss_weight": 1.0,
        "rul_future_loss_weight": 1.0,
        "learning_rate": base.get("optimizer_params", {}).get("lr", 3e-4),
        "weight_decay": base.get("optimizer_params", {}).get("weight_decay", 1e-4),
        "num_epochs": base.get("training_params", {}).get("num_epochs", 80),
        "batch_size": base.get("training_params", {}).get("batch_size", 256),
        # World-Model-Modi
        "target_mode": "residuals",  # Targets: Resid_Sensor* (normiert)
        "init_from_rul_hi": True,
    }

    config["world_model_params"] = world_params
    return config


def get_fd004_transformer_latent_worldmodel_v1_config() -> ExperimentConfig:
    """
    Branch A: Latent World Model V1 for FD004

    - Uses the best ms+DT encoder config (fd004_transformer_encoder_ms_dt_v1)
    - Encoder will be loaded from a pretrained checkpoint and frozen
    - Decoder optimizes only HI_future + RUL_future (sensor_loss_weight = 0)
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()
    config = copy.deepcopy(base)

    config["experiment_name"] = "fd004_transformer_latent_worldmodel_v1"
    config["encoder_type"] = "world_model_universal_v3"
    config["dataset"] = "FD004"

    # Mirror encoder_kwargs mapping used by the other WorldModel V1 configs
    config["encoder_kwargs"] = {
        "d_model": base["encoder_kwargs"]["d_model"],
        "num_layers": base["encoder_kwargs"]["num_layers"],
        "nhead": base["encoder_kwargs"]["n_heads"],
        "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
        "dropout": base["encoder_kwargs"]["dropout"],
        "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    world_params = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 20,
        "max_rul": base.get("max_rul", 125),
        "d_model": base["encoder_kwargs"].get("d_model", 128),
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": base["encoder_kwargs"].get("dropout", 0.1),
        # Latent-only: no sensor trajectory loss
        "sensor_loss_weight": 0.0,
        # Focus on HI + RUL future
        "hi_future_loss_weight": 0.5,
        "rul_future_loss_weight": 0.5,
        # Optimizer / training hyperparameters (used by training loop)
        "learning_rate": base.get("optimizer_params", {}).get("lr", 1e-4),
        "weight_decay": base.get("optimizer_params", {}).get("weight_decay", 1e-4),
        "num_epochs": base.get("training_params", {}).get("num_epochs", 80),
        "batch_size": base.get("training_params", {}).get("batch_size", 256),
        # World-Model modes
        "target_mode": "sensors",  # still use future sensors as auxiliary targets (loss weight 0)
        "init_from_rul_hi": True,
        # Branch A: freeze encoder and load from pretrained ms+DT checkpoint
        "freeze_encoder": True,
        "encoder_checkpoint": str(
            Path("results")
            / "fd004"
            / "fd004_transformer_encoder_ms_dt_v1"
            / "eol_full_lstm_best_fd004_transformer_encoder_ms_dt_v1.pt"
        ),
    }

    config["world_model_params"] = world_params
    return config


def get_fd004_transformer_latent_worldmodel_msfreeze_v1_config() -> ExperimentConfig:
    """
    Branch B: Latent World Model V1 for FD004 with explicit ms+DT encoder freeze.

    This shares the same configuration as `fd004_transformer_latent_worldmodel_v1`,
    but uses a distinct experiment_name so that results are stored separately.
    """
    base_cfg = get_fd004_transformer_latent_worldmodel_v1_config()
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_msfreeze_v1"
    return cfg


def get_fd004_transformer_latent_worldmodel_dynamic_v1_config() -> ExperimentConfig:
    """
    Branch A+: Dynamic latent Transformer World Model V1 for FD004.

    - Uses the best ms+DT encoder config (fd004_transformer_encoder_ms_dt_v1)
    - Dynamic latent decoder with short latent history, HI anchor and future Cond_* vectors
    - Latent-only training: HI_future + RUL_future (sensor_loss_weight = 0)
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()
    config = copy.deepcopy(base)

    config["experiment_name"] = "fd004_transformer_latent_worldmodel_dynamic_v1"
    config["encoder_type"] = "world_model_universal_v3"
    config["dataset"] = "FD004"

    # Mirror encoder_kwargs mapping used by the other WorldModel V1 configs
    config["encoder_kwargs"] = {
        "d_model": base["encoder_kwargs"]["d_model"],
        "num_layers": base["encoder_kwargs"]["num_layers"],
        "nhead": base["encoder_kwargs"]["n_heads"],
        "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
        "dropout": base["encoder_kwargs"]["dropout"],
        "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    world_params = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 30,
        "max_rul": base.get("max_rul", 125),
        "d_model": base["encoder_kwargs"].get("d_model", 128),
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": base["encoder_kwargs"].get("dropout", 0.1),
        # Latent-only: no sensor trajectory loss
        "sensor_loss_weight": 0.0,
        # Strong focus on future HI + RUL
        "hi_future_loss_weight": 10.0,
        "rul_future_loss_weight": 1.0,
        # Optimizer / training hyperparameters (used by training loop)
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": base.get("training_params", {}).get("num_epochs", 80),
        "batch_size": base.get("training_params", {}).get("batch_size", 256),
        # Dynamic latent world-model flags
        "target_mode": "latent_hi_rul",
        "init_from_rul_hi": False,
        "use_latent_history": True,
        "use_hi_anchor": True,
        "use_future_conds": True,
        # A+ latent transformer decoder + EOL fusion
        "use_eol_fusion": True,
        "eol_fusion_mode": "token",
        "predict_latent": True,
        "latent_decoder_type": "transformer",
        "latent_decoder_num_layers": 2,
        "latent_decoder_nhead": 4,
        # Training staging: freeze encoder then partially unfreeze
        "freeze_encoder_epochs": 10,
        "unfreeze_encoder_layers": 1,
        "encoder_lr_mult": 0.1,
        "eol_scalar_loss_weight": 0.1,
        # Start from best ms+DT encoder – no hard freeze in the main dynamic experiment
        "freeze_encoder": False,
        "encoder_checkpoint": str(
            Path("results")
            / "fd004"
            / "fd004_transformer_encoder_ms_dt_v1"
            / "eol_full_lstm_best_fd004_transformer_encoder_ms_dt_v1.pt"
        ),
    }

    config["world_model_params"] = world_params
    return config


def get_fd004_transformer_latent_worldmodel_dynamic_freeze_v1_config() -> ExperimentConfig:
    """
    Ablation: Dynamic latent World Model V1 with frozen ms+DT encoder.
    """
    base_cfg = get_fd004_transformer_latent_worldmodel_dynamic_v1_config()
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_dynamic_freeze_v1"
    # Enable encoder freezing for this ablation
    cfg["world_model_params"]["freeze_encoder"] = True
    return cfg


def get_fd004_transformer_latent_worldmodel_dynamic_delta_v2_config() -> ExperimentConfig:
    """
    Dynamic Delta World Model V2 on FD004.

    - Uses the best ms+DT encoder config (fd004_transformer_encoder_ms_dt_v1)
    - Delta-style latent world model:
        * latent context: [z_t, v1=z_t-z_{t-1}, v2=z_t-z_{t-2}] (3*d_model)
        * predicts HI/RUL deltas relative to encoder anchors
        * accumulates and clamps to valid ranges (HI∈[0,1], normalized RUL∈[0,1])
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()
    config = copy.deepcopy(base)

    config["experiment_name"] = "fd004_transformer_latent_worldmodel_dynamic_delta_v2"
    config["encoder_type"] = "world_model_universal_v3"
    config["dataset"] = "FD004"

    # Mirror encoder_kwargs mapping used by the other WorldModel V1 configs
    config["encoder_kwargs"] = {
        "d_model": base["encoder_kwargs"]["d_model"],
        "num_layers": base["encoder_kwargs"]["num_layers"],
        "nhead": base["encoder_kwargs"]["n_heads"],
        "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
        "dropout": base["encoder_kwargs"]["dropout"],
        "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    world_params = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 30,
        "max_rul": base.get("max_rul", 125),
        "d_model": base["encoder_kwargs"].get("d_model", 128),
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": base["encoder_kwargs"].get("dropout", 0.1),
        # Latent-only: no sensor trajectory loss
        "sensor_loss_weight": 0.0,
        # Strong focus on future HI + RUL
        "hi_future_loss_weight": 3.0,
        "rul_future_loss_weight": 1.0,
        # Optimizer / training hyperparameters (used by training loop)
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs": base.get("training_params", {}).get("num_epochs", 60),
        "batch_size": base.get("training_params", {}).get("batch_size", 128),
        # Dynamic delta latent world-model flags
        "target_mode": "latent_hi_rul_dynamic_delta_v2",
        "init_from_rul_hi": False,
        "use_latent_history": True,
        "use_hi_anchor": True,      # also enables RUL anchor inside the model
        "use_future_conds": True,
        # A+ latent transformer decoder + EOL fusion
        "use_eol_fusion": True,
        "eol_fusion_mode": "token",
        "predict_latent": True,
        "latent_decoder_type": "transformer",
        "latent_decoder_num_layers": 2,
        "latent_decoder_nhead": 4,
        # Training staging: freeze encoder then partially unfreeze
        "freeze_encoder_epochs": 10,
        "unfreeze_encoder_layers": 1,
        "encoder_lr_mult": 0.1,
        "eol_scalar_loss_weight": 0.1,
        # Start from best ms+DT encoder – typically not frozen for main delta run
        "freeze_encoder": False,
        "encoder_checkpoint": str(
            Path("results")
            / "fd004"
            / "fd004_transformer_encoder_ms_dt_v1"
            / "eol_full_lstm_best_fd004_transformer_encoder_ms_dt_v1.pt"
        ),
    }

    config["world_model_params"] = world_params
    return config


def get_fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659_config() -> ExperimentConfig:
    """
    Dynamic latent Transformer World Model V1 (Branch A+) that starts from the
    ms_dt_v2 damage_v5 cond_norm multiview residual-risk encoder checkpoint
    and uses the FULL feature space (~659 on FD004).

    Notes:
    - Feature filtering is intentionally OFF (features.include_groups omitted)
    - Encoder checkpoint load is guarded by an input_dim mismatch check
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20_config()
    cfg = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659"
    cfg["encoder_type"] = "world_model_universal_v3"
    cfg["dataset"] = "FD004"

    # Map encoder kwargs to the WM-V1 training loop expectations (nhead key)
    enc_base = base.get("encoder_kwargs", {})
    cfg["encoder_kwargs"] = {
        "d_model": enc_base.get("d_model", 96),
        "num_layers": enc_base.get("num_layers", 3),
        "nhead": enc_base.get("n_heads", enc_base.get("nhead", 4)),
        "dim_feedforward": enc_base.get("dim_feedforward", 256),
        "dropout": enc_base.get("dropout", 0.1),
        # keep these for symmetry with other WM configs (unused in WM-V1)
        "kernel_sizes": base.get("encoder_kwargs", {}).get("kernel_sizes", [3, 5, 9]),
        "seq_encoder_type": "transformer",
        "decoder_num_layers": 1,
    }

    # Ensure feature filtering stays OFF: do not set include_groups
    feats = cfg.get("features", {})
    feats.pop("include_groups", None)
    # Match the v5 encoder feature-set dimensionality (659) by augmenting the
    # multiscale block with extra temporal base columns for Twin_/Resid_.
    # This is experiment-local and does NOT affect other runs.
    ms = feats.get("multiscale", {})
    ms.setdefault("extra_temporal_base_prefixes", ["Twin_", "Resid_"])
    ms.setdefault("extra_temporal_base_max_cols", 31)
    feats["multiscale"] = ms
    cfg["features"] = feats

    # World model params (dynamic latent A+)
    ckpt_path = (
        "results/fd004/"
        "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20/"
        "eol_full_lstm_best_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20.pt"
    )

    cfg["world_model_params"] = {
        "past_len": base.get("past_len", 30),
        "future_horizon": 30,
        "max_rul": base.get("max_rul", 125),
        "d_model": cfg["encoder_kwargs"]["d_model"],
        "decoder_hidden_dim": 256,
        "num_layers_decoder": 1,
        "dropout": cfg["encoder_kwargs"]["dropout"],
        # Latent-only: no sensor trajectory loss
        "sensor_loss_weight": 0.0,
        # Strong focus on future HI + RUL
        "hi_future_loss_weight": 10.0,
        "rul_future_loss_weight": 1.0,
        # Optimizer / training hyperparameters (used by training loop)
        # Training stability knobs (WM-V1)
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "num_epochs": cfg.get("training_params", {}).get("num_epochs", 80),
        "batch_size": cfg.get("training_params", {}).get("batch_size", 256),
        # Dynamic latent world-model flags
        "target_mode": "latent_hi_rul",
        "init_from_rul_hi": False,
        "use_latent_history": True,
        "use_hi_anchor": True,
        "use_future_conds": True,
        # A+ latent transformer decoder + EOL fusion
        "use_eol_fusion": True,
        "eol_fusion_mode": "token",
        "predict_latent": True,
        "latent_decoder_type": "transformer",
        "latent_decoder_num_layers": 2,
        "latent_decoder_nhead": 4,
        "eol_scalar_loss_weight": 0.1,
        # Encoder checkpoint + staged training
        "encoder_checkpoint": ckpt_path,
        "freeze_encoder": True,
        "freeze_encoder_epochs": 10,
        "unfreeze_encoder_layers": 1,
        "encoder_lr_mult": 0.1,
    }

    return cfg


def get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_config() -> ExperimentConfig:
    """
    Ablation to address mean-collapse: rebalance HI vs RUL losses + lower LR + earlier unfreeze.

    Base: fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659
    Changes:
      - hi_future_loss_weight: 2.0 (was 10.0)
      - rul_future_loss_weight: 3.0 (was 1.0)
      - learning_rate: 2e-4 (was 5e-4)
      - freeze_encoder_epochs: 5 (was 10)
    """
    cfg = copy.deepcopy(get_fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659_config())
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["hi_future_loss_weight"] = 2.0
    wmp["rul_future_loss_weight"] = 3.0
    wmp["learning_rate"] = 2e-4
    wmp["freeze_encoder_epochs"] = 5
    return cfg


def get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_latew10_config() -> ExperimentConfig:
    """
    Quick A/B experiment: add late-window weighting (future horizon contains RUL=0) to fight "always healthy" collapse.

    Base: fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1
    Changes:
      - late_weight_enable: True
      - late_weight_factor: 10.0
    """
    cfg = copy.deepcopy(get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_config())
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_latew10"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["late_weight_enable"] = True
    wmp["late_weight_factor"] = 10.0
    # Keep defaults explicit for clarity (unit-safe detection in training loop)
    wmp["late_weight_mode"] = "future_has_zero"
    wmp["late_weight_eps_norm"] = 1e-6
    wmp["late_weight_apply_hi"] = False
    return cfg


def get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_infwin_config() -> ExperimentConfig:
    """
    Quick A/B experiment: informative window sampling to avoid RUL-cap saturation collapse.

    Base: fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1
    Changes:
      - informative_sampling_enable: True
      - informative_sampling_mode: "future_min_lt_cap"
      - keep_prob_noninformative: 0.1
    """
    cfg = copy.deepcopy(get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_config())
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_infwin"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["informative_sampling_enable"] = True
    wmp["informative_sampling_mode"] = "future_min_lt_cap"
    wmp["informative_eps_norm"] = 1e-6
    wmp["keep_prob_noninformative"] = 0.1
    wmp["log_informative_stats"] = True
    return cfg


def get_fd004_wm_v1_infwin_wiringcheck_k0_config() -> ExperimentConfig:
    """
    Minimal wiring proof run:
      - hard focus on informative windows (keep_prob_noninformative=0)
      - short epochs (3)
      - wiring debug enabled (prints only first batch of epoch 1 and writes wiring_debug.json)
    """
    cfg = copy.deepcopy(get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_infwin_config())
    cfg["experiment_name"] = "fd004_wm_v1_infwin_wiringcheck_k0"
    cfg.setdefault("training_params", {})
    cfg["training_params"]["num_epochs"] = 20
    wmp = cfg.setdefault("world_model_params", {})
    wmp["keep_prob_noninformative"] = 0.0
    wmp["debug_wiring_enable"] = True
    wmp["debug_wiring_batches"] = 1
    wmp["debug_wiring_epochs"] = 1
    wmp["debug_wiring_save_json"] = True
    # Make sure late-weighting is on for the check (as per prompt context)
    wmp["late_weight_enable"] = True
    wmp["late_weight_factor"] = 10.0
    wmp["late_weight_mode"] = "future_has_zero"
    return cfg


def get_fd004_wm_v1_infwin_capweight_k1_config() -> ExperimentConfig:
    """
    WM-V1 plateau/cap reweighting check:
      - same as wiringcheck_k0 but with cap/plateau down-weighting enabled
      - keeps wiring debug enabled for proof
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_infwin_wiringcheck_k0_config())
    cfg["experiment_name"] = "fd004_wm_v1_infwin_capweight_k1"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["cap_reweight_enable"] = True
    wmp["cap_reweight_eps"] = 1e-6
    wmp["cap_reweight_weight"] = 0.05
    wmp["cap_reweight_apply_to"] = "rul"
    return cfg


def get_fd004_wm_v1_infwin_capmask_k2_config() -> ExperimentConfig:
    """
    WM-V1 cap-mask + split-metrics check:
      - based on capweight_k1 (keeps informative sampling + late weighting + cap reweight)
      - enables cap-aware masking for RUL future loss via cap_mask_* knobs
      - keeps wiring debug enabled for proof
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_infwin_capweight_k1_config())
    cfg["experiment_name"] = "fd004_wm_v1_infwin_capmask_k2"
    cfg.setdefault("training_params", {})
    cfg["training_params"]["num_epochs"] = 20
    wmp = cfg.setdefault("world_model_params", {})
    wmp["cap_mask_enable"] = True
    wmp["cap_mask_apply_to"] = ["rul"]
    wmp["cap_mask_eps"] = 1e-6
    return cfg


def get_fd004_wm_v1_p0_softcap_k3_config() -> ExperimentConfig:
    """
    P0 cap-collapse fix (ADR-0010):
      - P0.1: soft_cap_enable=True (distance-based soft weighting, replaces binary masking)
      - P0.2: informative_sampling_mode="uncapped_frac" with threshold=0.3
      - keeps late weighting from previous experiments
    
    Based on capmask_k2.
    
    Go/No-Go criteria:
      - Bias_LAST: +70 -> <+30 cycles
      - R²_LAST: ~0 -> >0.3
      - pred_rul_seq_std: ~0.02 -> >=0.10
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_infwin_capmask_k2_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3"
    cfg.setdefault("training_params", {})
    cfg["training_params"]["num_epochs"] = 10

    wmp = cfg.setdefault("world_model_params", {})
    
    # P0.1: Soft cap weighting (replaces binary masking)
    wmp["soft_cap_enable"] = True
    wmp["soft_cap_power"] = 0.5  # sqrt gives gradual ramp
    wmp["soft_cap_floor"] = 0.05  # minimum weight for capped timesteps
    wmp["cap_mask_enable"] = False  # Disable binary masking (replaced by soft)
    
    # P0.2: Stricter informative sampling
    wmp["informative_sampling_enable"] = True
    wmp["informative_sampling_mode"] = "uncapped_frac"
    wmp["informative_uncapped_frac_threshold"] = 0.3  # 30% of future must be uncapped
    wmp["keep_prob_noninformative"] = 0.05  # Reduce from 0.1 to further filter capped windows
    
    # Keep late weighting (from parent config)
    wmp["late_weight_enable"] = True
    wmp["late_weight_factor"] = 10.0
    
    # Enable horizon masking for padding-aware RUL loss (removes floor at ~24-30)
    # This ensures padded future timesteps do NOT contribute to RUL future loss.
    wmp["use_horizon_mask"] = True
    
    # Logging
    wmp["debug_wiring_enable"] = True
    wmp["debug_wiring_epochs"] = 1
    wmp["log_informative_stats"] = True
    
    return cfg


def get_fd004_wm_v1_p0_softcap_k3_hm_pad_config() -> ExperimentConfig:
    """
    P0 cap-collapse fix (ADR-0010) + Stage-1 horizon padding:
      - P0.1: soft_cap_enable=True (distance-based soft weighting, replaces binary masking)
      - P0.2: informative_sampling_mode="uncapped_frac" with threshold=0.3
      - keeps late weighting from previous experiments
      - NEW: use_padded_horizon_targets=True (enables near-EOL windows in Stage-1)
      - NEW: use_horizon_mask=True (masks padded timesteps in RUL loss)
    
    Based on p0_softcap_k3, but with Stage-1 padding enabled to remove floor at ~24-30.
    
    Go/No-Go criteria:
      - Bias_LAST: +70 -> <+30 cycles
      - R²_LAST: ~0 -> >0.3
      - pred_rul_seq_std: ~0.02 -> >=0.10
      - Stage-1 pad_frac > 0.0 (confirms padding is active)
      - Stage-1 y_eol min <= 5 (confirms near-EOL windows included)
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_p0_softcap_k3_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3_hm_pad"
    
    wmp = cfg.setdefault("world_model_params", {})
    
    # Enable Stage-1 padded horizon targets (allows near-EOL windows with RUL < H)
    # This sets require_full_horizon=False in Stage-1 window builder
    wmp["use_padded_horizon_targets"] = True
    
    # Ensure horizon masking is enabled (should already be set from parent, but be explicit)
    wmp["use_horizon_mask"] = True
    
    return cfg


def get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_rulonly_v1_config() -> ExperimentConfig:
    """
    Ablation to isolate collapse source: RUL-only (no HI anchor, no HI loss) + lower LR + earlier unfreeze.

    Base: fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659
    Changes:
      - use_hi_anchor: False
      - hi_future_loss_weight: 0.0
      - rul_future_loss_weight: 3.0
      - learning_rate: 2e-4
      - freeze_encoder_epochs: 5
    """
    cfg = copy.deepcopy(get_fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659_config())
    cfg["experiment_name"] = "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_rulonly_v1"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["use_hi_anchor"] = False
    wmp["hi_future_loss_weight"] = 0.0
    wmp["rul_future_loss_weight"] = 3.0
    wmp["learning_rate"] = 2e-4
    wmp["freeze_encoder_epochs"] = 5
    # RUL trajectory + monotonic + anti-saturation stabilizers (WM-V1)
    # Keys must match run_experiments.py wiring.
    wmp["rul_traj_weight"] = 6.0
    wmp["rul_traj_late_ramp"] = True
    # rul_linear_decay already enforces monotonicity
    wmp["rul_mono_future_weight"] = 0.0
    wmp["rul_saturation_weight"] = 1.0
    wmp["rul_saturation_margin"] = 0.05
    wmp["rul_cap_threshold"] = 0.999999
    # Optional physics-consistent linear-decay construction from pred_rul0
    wmp["rul_linear_decay"] = True
    # Optional WM-V1 hardening knobs (do not affect other experiments)
    wmp["rul_train_max_cycles"] = 95.0
    wmp["rul_r0_only"] = True
    wmp["rul_r0_points"] = [0, 15, 29]
    wmp["rul_sample_weight_power"] = 1.0
    wmp["rul_sample_weight_min"] = 0.2
    wmp["rul_sample_weight_max"] = 3.0
    return cfg

def get_fd003_transformer_encoder_ms_dt_v1_config() -> ExperimentConfig:
    """
    Multi-Scale + Digital-Twin Transformer-Encoder experiment on FD003.

    - Uses the base Transformer-Encoder EOL+HI config for FD003
    - Enables explicit multi-scale temporal windows via `features`
    - Enables continuous condition vector + digital-twin residuals via `phys_features`
    """
    cfg = get_fd003_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd003_transformer_encoder_ms_dt_v1"

    cfg["features"] = {
        "use_multiscale_features": True,
        "multiscale": {
            "windows_short": [10],
            "windows_medium": [30],
            "windows_long": [60, 120],
        },
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "condition_vector_version": 3,
        "use_digital_twin_residuals": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd004_transformer_encoder_ms_dt_v1_config() -> ExperimentConfig:
    """
    Multi-Scale + Digital-Twin Transformer-Encoder experiment on FD004.

    - Uses the base Transformer-Encoder EOL+HI config for FD004
    - Enables explicit multi-scale temporal windows via `features`
    - Enables continuous condition vector + digital-twin residuals via `phys_features`
    """
    cfg = get_fd004_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v1"

    cfg["features"] = {
        "use_multiscale_features": True,
        "multiscale": {
            "windows_short": [10],
            "windows_medium": [30],
            "windows_long": [60, 120],
        },
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "condition_vector_version": 3,
        "use_digital_twin_residuals": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_config() -> ExperimentConfig:
    """
    FD004 – Transformer-Encoder ms+DT V2 with continuous condition encoder.

    - Starts from fd004_transformer_encoder_ms_dt_v1_config
    - Enables a per-timestep continuous condition encoder (Cond_* vector)
    - Adds an auxiliary condition reconstruction head as a mild regulariser
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v1_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2"

    # Slightly larger encoder capacity for V2 (optional)
    enc = cfg["encoder_kwargs"]
    enc["d_model"] = 128
    enc["num_layers"] = 4
    enc["n_heads"] = enc.get("n_heads", 4)
    enc["dim_feedforward"] = enc.get("dim_feedforward", 256)

    # NEW: continuous condition encoder flags.
    # cond_in_dim will be inferred at runtime from Cond_* features.
    enc["use_cond_encoder"] = True
    enc["cond_in_dim"] = None
    enc["cond_encoder_dim"] = enc["d_model"]
    enc["use_cond_recon_head"] = True

    # Loss weights – add a small condition reconstruction term.
    loss = cfg["loss_params"]
    loss["cond_recon_weight"] = 0.1

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v1_config() -> ExperimentConfig:
    """
    FD004 – Transformer-Encoder ms+DT V2 with cumulative damage head.

    - Uses the same ms+DT+residual+Cond_*+Twin_* feature pipeline as ms_dt_v2
    - Keeps the original EOL/RUL/HI/monotonic losses
    - Adds a small extra loss on a cumulative damage-based HI trajectory.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v1"

    # Enable cumulative damage head in the encoder kwargs
    enc = cfg["encoder_kwargs"]
    enc["use_damage_head"] = True
    enc.setdefault("damage_L_ref", 300.0)
    enc.setdefault("damage_alpha_base", 0.1)
    enc.setdefault("damage_hidden_dim", 64)

    # Loss params: reuse base settings and add a small damage-HI weight
    loss = cfg["loss_params"]
    loss.setdefault("damage_hi_weight", 0.2)

    return cfg


def get_fd004_state_encoder_v3_physics_C1_msdt_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (C1):
    - clones the ms_dt_v2 feature configuration (349 features: raw+ms+residual+cond+twin)
    - uses transformer_state_encoder_v3_physics as encoder
    - trains with HI as main target, RUL as auxiliary, no alignment loss.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_state_encoder_v3_physics_C1_msdt"
    cfg["dataset"] = "FD004"
    cfg["encoder_type"] = "transformer_state_encoder_v3_physics"

    # Data section mirrors the other state-encoder physics configs
    data_cfg = cfg.get("data", {})
    data_cfg.update(
        {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        }
    )
    cfg["data"] = data_cfg

    # Model: compact Transformer encoder as in other state-encoder variants
    cfg["model"] = {
        "input_dim": None,
        "d_model": 96,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "cond_in_dim": None,
    }

    # Training hyperparameters
    cfg["training"] = {
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 80,
        "patience": 10,
    }

    # Loss: HI main task, RUL auxiliary, no alignment term
    cfg["loss"] = {
        "hi_weight": 1.0,
        "rul_weight": 0.3,
        "align_weight": 0.0,
        "use_rul_hi_alignment_loss": False,
    }

    # IMPORTANT: do NOT touch cfg["features"] / cfg["phys_features"];
    # they remain exactly as in ms_dt_v2 so the feature space matches.

    cfg["paths"] = {
        "result_dir": "results/fd004/fd004_state_encoder_v3_physics_C1_msdt",
    }
    cfg["seed"] = base.get("training_params", {}).get("random_seed", 42)

    return cfg


def get_fd004_state_encoder_v3_physics_C2_msdt_align_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (C2):
    - clones the ms_dt_v2 feature configuration (349 features)
    - uses transformer_state_encoder_v3_physics as encoder
    - trains with HI as main task, RUL as auxiliary + small RUL–HI alignment loss.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_state_encoder_v3_physics_C2_msdt_align"
    cfg["dataset"] = "FD004"
    cfg["encoder_type"] = "transformer_state_encoder_v3_physics"

    data_cfg = cfg.get("data", {})
    data_cfg.update(
        {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        }
    )
    cfg["data"] = data_cfg

    cfg["model"] = {
        "input_dim": None,
        "d_model": 96,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "cond_in_dim": None,
    }

    cfg["training"] = {
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 80,
        "patience": 10,
    }

    cfg["loss"] = {
        "hi_weight": 1.0,
        "rul_weight": 0.3,
        "align_weight": 0.1,
        "use_rul_hi_alignment_loss": True,
    }

    cfg["paths"] = {
        "result_dir": "results/fd004/fd004_state_encoder_v3_physics_C2_msdt_align",
    }
    cfg["seed"] = base.get("training_params", {}).get("random_seed", 42)

    return cfg


def get_fd004_state_encoder_v3_physics_C3_msdt_rulonly_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (C3, Debug):
    - Same ms+DT v2 feature configuration as C1/C2 (361D encoder input).
    - RUL-only training (HI disabled) to debug encoder capacity without HI.
    """
    base = get_fd004_state_encoder_v3_physics_C1_msdt_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_state_encoder_v3_physics_C3_msdt_rulonly"
    # Loss: RUL-only
    cfg["loss"] = {
        "hi_weight": 0.0,
        "rul_weight": 1.0,
        "align_weight": 0.0,
        "use_rul_hi_alignment_loss": False,
    }

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v2_config() -> ExperimentConfig:
    """
    FD004 ms+DT EOLFullTransformerEncoder with cumulative DamageHead,
    trained against a physics-based HI_phys_v2 trajectory (HI_phys_seq).

    This is an evolution of _damage_v1:
      - same ms_dt_v2 feature pipeline,
      - same EOL losses,
      - PLUS: stronger supervision of the damage head on HI_phys_seq.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v2"

    enc_kwargs = cfg.setdefault("encoder_kwargs", {})
    enc_kwargs["use_damage_head"] = True
    enc_kwargs.setdefault("damage_L_ref", 300.0)  # Note: parameter name is damage_L_ref in model
    enc_kwargs.setdefault("damage_alpha_base", 0.1)  # Note: parameter name is damage_alpha_base in model
    enc_kwargs.setdefault("damage_hidden_dim", 64)

    loss_params = cfg.setdefault("loss_params", {})
    # Keep existing RUL + standard HI weights, but reduce standard HI a bit
    loss_params["rul_weight"] = loss_params.get("rul_weight", 1.0)
    loss_params["health_loss_weight"] = loss_params.get("health_loss_weight", 0.1)
    # New: physics-based damage HI weight (stronger, main supervision for damage head)
    loss_params["damage_hi_weight"] = 1.0

    # keep mono_late_weight, mono_global_weight, etc. as in base config
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v3_config() -> ExperimentConfig:
    """
    FD004 ms+DT EOLFullTransformerEncoder with cumulative DamageHead,
    trained against a physics-based HI_phys_v3 trajectory (HI_phys_seq).

    This is a sibling of _damage_v2:
      - same ms_dt_v2 feature pipeline,
      - same EOL losses,
      - BUT: supervision of the damage head on HI_phys_v3 instead of HI_phys_v2.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3"

    enc_kwargs = cfg.setdefault("encoder_kwargs", {})
    enc_kwargs["use_damage_head"] = True
    enc_kwargs.setdefault("damage_L_ref", 300.0)
    enc_kwargs.setdefault("damage_alpha_base", 0.1)
    enc_kwargs.setdefault("damage_hidden_dim", 64)

    loss_params = cfg.setdefault("loss_params", {})
    # Keep existing RUL + standard HI weights, but reduce standard HI a bit
    loss_params["rul_weight"] = loss_params.get("rul_weight", 1.0)
    loss_params["health_loss_weight"] = loss_params.get("health_loss_weight", 0.1)
    # Physics-based damage HI weight (main supervision for damage head)
    loss_params["damage_hi_weight"] = 1.0

    # Optional: record that this experiment is meant to use HI_phys_v3 as target
    # (currently used implicitly via presence of the HI_phys_v3 column).
    cfg.setdefault("hi_target_type", "phys_v3")

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v3b_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder with cumulative DamageHead,
    trained against HI_phys_v3 with a stronger damage loss weight.

    This variant is identical to _damage_v3 except for:
      - experiment_name suffix `_v3b`
      - increased damage_hi_weight to emphasize the damage head.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3b"

    # Stronger emphasis on damage head while keeping RUL/HI weights unchanged.
    loss_params = cfg.setdefault("loss_params", {})
    loss_params["rul_weight"] = loss_params.get("rul_weight", 1.0)
    loss_params["health_loss_weight"] = loss_params.get("health_loss_weight", 0.1)
    loss_params["damage_hi_weight"] = 5.0

    # Document that this experiment is meant to use HI_phys_v3 as target
    cfg["hi_target_type"] = "phys_v3"

    return cfg


def get_fd004_decoder_v1_from_encoder_v3d_config() -> ExperimentConfig:
    """
    FD004 RUL Trajectory Decoder v1 on top of a frozen encoder v3d.

    This is a thin wrapper that allows launching the standalone
    decoder training script via run_experiments.py using:

        --experiments fd004_decoder_v1_from_encoder_v3d
    """
    cfg: ExperimentConfig = {
        "experiment_name": "fd004_decoder_v1_from_encoder_v3d",
        "dataset": "FD004",
        # Custom encoder_type used as a switch in run_experiments.run_single_experiment
        "encoder_type": "decoder_v1_from_encoder_v3d",
        # Minimal training params for the decoder; can be overridden if needed.
        "training_params": {
            "num_epochs": 50,
            "batch_size": 256,
        },
    }
    return cfg


def get_fd004_decoder_v3_from_encoder_v3d_config() -> ExperimentConfig:
    """
    FD004 RUL Trajectory Decoder v3 on top of a frozen encoder v3d.

    This config allows launching the standalone decoder v3 training script via
    run_experiments.py using:

        --experiments fd004_decoder_v3_from_encoder_v3d
    """
    encoder_experiment = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
    dataset = "FD004"

    cfg: ExperimentConfig = {
        "experiment_name": "fd004_decoder_v3_from_encoder_v3d",
        "dataset": dataset,
        # Custom encoder_type used as a switch in run_experiments.run_single_experiment
        "encoder_type": "decoder_v3",
        # Reference to the frozen encoder run
        "encoder_experiment": encoder_experiment,
        # Path to the global HI calibrator (fit via src.analysis.hi_calibration)
        "hi_calibrator_path": (
            f"results/{dataset.lower()}/{encoder_experiment}/hi_calibrator_{dataset}.pkl"
        ),
        # Decoder v3 hyperparameters and training settings
        "past_len": 30,
        "max_rul": 125.0,
        "training_params": {
            "num_epochs": 80,
            "batch_size": 256,
            "engine_train_ratio": 0.8,
            "random_seed": 42,
        },
        "decoder_hidden_dim": 128,
        "decoder_num_layers": 2,
        "decoder_dropout": 0.1,
        # Loss weights
        "w_traj": 1.0,
        "w_eol": 0.2,
        "w_mono": 0.1,
        "w_smooth": 0.01,
        "w_slope": 0.2,
        # Stage-1: curriculum + coupling (defaults are conservative; adjust in ablations)
        "three_phase_schedule": True,
        "phase_a_frac": 0.2,
        "phase_b_end_frac": 0.8,
        # IMPORTANT: keep coupling OFF by default until KPIs confirm saturation/flatline.
        "w_couple": 0.0,
        "coupling_alpha": 1.0,
        "coupling_hi_to_rul_scale": 125.0,
        "coupling_hi_delta_threshold": 0.0,
        "coupling_hi_source": "hi_damage",
    }
    return cfg


def get_fd004_decoder_v3_uncertainty_from_encoder_v3d_config() -> ExperimentConfig:
    """
    Decoder v3 with per-timestep sigma(t) uncertainty head and weighted Gaussian NLL.
    """
    cfg = get_fd004_decoder_v3_from_encoder_v3d_config()
    cfg["experiment_name"] = "fd004_decoder_v3_uncertainty_from_encoder_v3d"
    # New loss term for trajectory uncertainty
    cfg.setdefault("w_traj_nll", 0.5)
    cfg.setdefault("sigma_floor", 1e-3)
    return cfg

def get_fd004_decoder_v2_from_encoder_v3d_config() -> ExperimentConfig:
    """
    FD004 RUL Trajectory Decoder v2 on top of a frozen encoder v3d.

    This config allows launching the standalone decoder v2 training script via
    run_experiments.py using:

        --experiments fd004_decoder_v2_from_encoder_v3d
    """
    encoder_experiment = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
    dataset = "FD004"

    cfg: ExperimentConfig = {
        "experiment_name": "fd004_decoder_v2_from_encoder_v3d",
        "dataset": dataset,
        # Custom encoder_type used as a switch in run_experiments.run_single_experiment
        "encoder_type": "decoder_v2",
        # Reference to the frozen encoder run
        "encoder_experiment": encoder_experiment,
        # Path to the global HI calibrator (fit via src.analysis.hi_calibration)
        "hi_calibrator_path": (
            f"results/{dataset.lower()}/{encoder_experiment}/hi_calibrator_{dataset}.pkl"
        ),
        # Decoder v2 hyperparameters and training settings
        "past_len": 30,
        "max_rul": 125.0,
        "training_params": {
            "num_epochs": 60,
            "batch_size": 256,
            "engine_train_ratio": 0.8,
            "random_seed": 42,
        },
        "decoder_hidden_dim": 128,
        "decoder_num_layers": 2,
        "decoder_dropout": 0.1,
        # Loss weights for trajectory-centric training
        "w_traj": 1.0,
        "w_eol": 0.2,
        "w_mono": 0.1,
        "w_smooth": 0.01,
    }
    return cfg

def get_fd004_decoder_v1_from_encoder_v3e_config() -> ExperimentConfig:
    """
    FD004 RUL Trajectory Decoder v1 on top of a frozen encoder v3e.
    """
    cfg: ExperimentConfig = {
        "experiment_name": "fd004_decoder_v1_from_encoder_v3e",
        "dataset": "FD004",
        "encoder_type": "decoder_v1_from_encoder_v3e",
        "training_params": {
            "num_epochs": 50,
            "batch_size": 256,
        },
    }
    return cfg

def get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder with MLP-based DamageHead and two-phase training.

    - Starts from _damage_v3b (same ms_dt_v2 feature pipeline and HI_phys_v3 target)
    - Uses an MLP-based damage head on top of encoder states
    - Trains with a two-phase schedule:
        Phase 1: focus on Damage-HI (HI_phys_v3), no RUL/standard HI loss
        Phase 2: full multi-task (RUL + HI + Damage-HI) with slightly reduced damage weight
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3b_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase"

    # Encoder: enable MLP-based damage head
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_damage_head"] = True
    enc.setdefault("damage_L_ref", 300.0)
    enc.setdefault("damage_alpha_base", 0.1)
    enc.setdefault("damage_hidden_dim", 64)
    enc["damage_use_mlp"] = True
    enc["damage_mlp_hidden_factor"] = 2
    enc["damage_mlp_num_layers"] = 2
    enc["damage_mlp_dropout"] = 0.1

    # Training: two-phase schedule for damage HI loss
    train_params = cfg.setdefault("training_params", {})
    train_params["damage_two_phase"] = True
    train_params["damage_warmup_epochs"] = 5
    train_params["damage_phase1_damage_weight"] = 5.0  # strong focus in warmup
    train_params["damage_phase2_damage_weight"] = 3.0  # slightly reduced in full multi-task

    # Keep loss_params (including damage_hi_weight=5.0) from v3b as base
    loss_params = cfg.setdefault("loss_params", {})
    loss_params["rul_weight"] = loss_params.get("rul_weight", 1.0)
    loss_params["health_loss_weight"] = loss_params.get("health_loss_weight", 0.1)
    loss_params["damage_hi_weight"] = loss_params.get("damage_hi_weight", 5.0)

    cfg["hi_target_type"] = "phys_v3"

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder with Delta-Cumsum DamageHead (v3d) and two-phase training.
    
    - Starts from v3c (MLP base), but activates use_delta_cumsum=True
    - Adds smoothness loss on delta_damage increments
    - Adjusts loss balance to favor RUL trajectory stability
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"

    # Modell / Encoder
    # Config keys may be under "encoder_kwargs" (from v3c base)
    model_cfg = cfg.get("model", {})
    enc = cfg.setdefault("encoder_kwargs", model_cfg.get("encoder_kwargs", {}))
    # Note: v3c puts them in "encoder_kwargs" directly in previous function
    
    enc["use_damage_head"] = True
    enc["damage_use_mlp"] = True
    enc["damage_use_delta_cumsum"] = True
    enc["damage_delta_alpha"] = 1.0  # ggf. 0.5–2.0 tunable
    enc["damage_L_ref"] = 300.0
    enc["damage_hidden_dim"] = 64

    # Training / Two-Phase
    train = cfg.setdefault("training_params", {})
    train["damage_two_phase"] = True
    train["damage_warmup_epochs"] = 10
    train["damage_phase1_damage_weight"] = 10.0
    train["damage_phase2_damage_weight"] = 3.0
    # NEW: Smoothness weights
    train["damage_phase1_smooth_weight"] = 0.1
    train["damage_phase2_smooth_weight"] = 0.03

    # Loss-Params: RUL-Trajektorie stärker gewichten
    loss = cfg.setdefault("loss_params", {})
    loss["damage_hi_weight"] = 5.0   # Basis
    loss["rul_traj_weight"] = 1.5    # NEU, falls unterstützt

    cfg["hi_target_type"] = "phys_v3"

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v3e_smooth_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder with Delta-Cumsum DamageHead (v3e).
    
    - Based on v3d (delta-cumsum + two-phase)
    - Adds temporal convolution smoother on delta_damage
    - Increases smoothness penalties and adjusts RUL trajectory weights
    - Adds gentle alignment to physical HI at start/end
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3e_smooth"

    # Encoder / Damage Head params
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["damage_use_delta_cumsum"] = True
    enc["damage_delta_alpha"] = 0.5  # Smaller steps -> smoother

    # New Temporal Conv Smoother flags
    enc["damage_use_temporal_conv"] = True
    enc["damage_temporal_conv_kernel_size"] = 5
    enc["damage_temporal_conv_num_layers"] = 1

    # Training params
    train = cfg.setdefault("training_params", {})
    # Adjusted weights for smoothness
    train["damage_phase1_damage_weight"] = 10.0
    train["damage_phase2_damage_weight"] = 4.0  # Slightly increased from v3d (3.0)
    train["damage_phase1_smooth_weight"] = 0.10 # Increased
    train["damage_phase2_smooth_weight"] = 0.04 # Increased from v3d (0.03)

    # Loss params
    loss = cfg.setdefault("loss_params", {})
    loss["rul_traj_weight"] = 2.0  # Increased from v3d (1.5)
    
    # New Alignment Weights
    loss["damage_hi_align_start_weight"] = 0.01
    loss["damage_hi_align_end_weight"] = 0.02

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v4_hi_cal_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder v4 with cumulative DamageHead and HI_cal_v2 head.

    - Starts from v3d (delta-cumsum + two-phase) configuration.
    - Adds an additional calibrated HI head (HI_cal_v2) trained against the global
      HI_phys_v3 -> HI_cal_v1/v2 calibrator.
    - Adds encoder-level monotonicity and slope-regularisation losses on HI_cal_v2.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v4_hi_cal"

    # Encoder kwargs: enable HI_cal head
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_hi_cal_head"] = True

    # Loss weights for HI_cal_v2 and its regularisation
    loss = cfg.setdefault("loss_params", {})
    loss.setdefault("w_rul_eol", 1.0)
    loss.setdefault("w_hi_phys", loss.get("health_loss_weight", 0.3))
    loss.setdefault("w_hi_damage", loss.get("damage_hi_weight", 5.0))
    loss.setdefault("w_hi_cal", 0.5)
    loss.setdefault("w_mono_damage", 0.05)
    loss.setdefault("w_mono_hi_cal", 0.05)
    loss.setdefault("w_slope_hi_cal", 0.1)

    # Training hyperparameters (inherit and, if needed, override)
    train = cfg.setdefault("training_params", {})
    train.setdefault("num_epochs", 120)
    train.setdefault("batch_size", 256)
    train.setdefault("engine_train_ratio", 0.8)
    train.setdefault("random_seed", 42)

    # Path to global HI calibrator (fit beforehand via src.analysis.hi_calibration)
    dataset = cfg.get("dataset", "FD004")
    encoder_run_base = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
    cfg["hi_calibrator_path"] = (
        f"results/{dataset.lower()}/{encoder_run_base}/hi_calibrator_{dataset}.pkl"
    )

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder v5:
      - cumulative DamageHead (as in v4)
      - HI_cal_v2 head + fusion into RUL head
      - ConditionNormalizer on selected degradation-sensitive sensors
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v4_hi_cal_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm"

    # Encoder kwargs: enable HI_cal head, HI_cal fusion and condition normaliser
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_hi_cal_head"] = True
    enc["use_hi_cal_fusion_for_rul"] = True
    enc["use_condition_normalizer"] = True
    enc.setdefault("condition_normalizer_hidden_dim", 64)

    # Loss weights remain as in v4 by default; ensure they are present.
    loss = cfg.setdefault("loss_params", {})
    loss.setdefault("w_hi_cal", 0.5)
    loss.setdefault("w_mono_hi_cal", 0.05)
    loss.setdefault("w_slope_hi_cal", 0.1)

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_uncertainty_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder v5 + uncertainty head:
      - same as v5_cond_norm
      - additionally predicts sigma for RUL at last observed cycle
      - trains with an additional Gaussian NLL term (configurable)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_uncertainty"

    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_rul_uncertainty_head"] = True
    enc.setdefault("rul_uncertainty_min_sigma", 1e-3)

    loss = cfg.setdefault("loss_params", {})
    # Additional loss term: Gaussian NLL for EOL/last-observed RUL
    loss.setdefault("rul_nll_weight", 0.5)
    # Prevent NLL from shifting the mean predictor (mu) too optimistic/biased.
    loss.setdefault("rul_nll_detach_mu", True)

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_config() -> ExperimentConfig:
    """
    FD004 ms+DT Transformer-Encoder v5 (cond_norm) with RUL quantile head:
      - Predicts RUL quantiles at last observed cycle (default P10/P50/P90)
      - Uses P50 as point prediction
      - Trains with pinball loss + non-crossing penalty
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles"

    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_rul_quantiles_head"] = True
    enc["rul_quantiles"] = [0.1, 0.5, 0.9]
    # Ensure sigma head/NLL are off for this run
    enc.setdefault("use_rul_uncertainty_head", False)
    # Quantile head run should not depend on HI_cal calibrator by default.
    # (HI_cal supervision requires a valid hi_calibrator_*.pkl which may be missing/corrupted in Colab.)
    enc["use_hi_cal_fusion_for_rul"] = False
    enc["use_hi_cal_head"] = False

    loss = cfg.setdefault("loss_params", {})
    loss["rul_nll_weight"] = 0.0
    loss["rul_nll_detach_mu"] = False
    loss["rul_quantiles"] = [0.1, 0.5, 0.9]
    loss["rul_quantile_weight"] = 0.5
    loss["rul_quantile_cross_weight"] = 0.1
    # Optional stabilizer (start at 0.0; increase if P50 drifts)
    loss.setdefault("rul_quantile_p50_mse_weight", 0.0)
    # Disable HI_cal supervision losses to avoid loading calibrator at training start.
    loss["w_hi_cal"] = 0.0
    loss["w_mono_hi_cal"] = 0.0
    loss["w_slope_hi_cal"] = 0.0

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_tuned_p50mse_config() -> ExperimentConfig:
    """
    Tuned quantiles run aiming to preserve P50 RMSE while learning quantiles:
      - smaller quantile loss weight (auxiliary)
      - explicit P50 MSE stabilizer
    Keeps HI_cal disabled to avoid calibrator dependency in Colab.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_tuned_p50mse"

    loss = cfg.setdefault("loss_params", {})
    loss["rul_quantile_weight"] = 0.1
    loss["rul_quantile_cross_weight"] = 0.1
    loss["rul_quantile_p50_mse_weight"] = 1.0

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_hi_cal_tuned_p50mse_config() -> ExperimentConfig:
    """
    Quantiles run that is closer to the "v5 cond_norm + HI_cal" family:
      - enables HI_cal head + HI_cal fusion for RUL head
      - enables HI_cal supervision losses (requires valid hi_calibrator_*.pkl)
      - keeps quantile head active, but treats quantile loss as auxiliary
      - stabilizes P50 with an explicit MSE term
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_tuned_p50mse_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_hi_cal_tuned_p50mse"

    # Re-enable HI_cal features in the encoder
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_hi_cal_head"] = True
    enc["use_hi_cal_fusion_for_rul"] = True

    # Re-enable HI_cal supervision losses (weights chosen to match v4/v5 defaults)
    loss = cfg.setdefault("loss_params", {})
    loss["w_hi_cal"] = float(loss.get("w_hi_cal", 0.5))
    loss["w_mono_hi_cal"] = float(loss.get("w_mono_hi_cal", 0.05))
    loss["w_slope_hi_cal"] = float(loss.get("w_slope_hi_cal", 0.1))

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_censoring_aware_config() -> ExperimentConfig:
    """
    FD004 Transformer-Encoder v5 (cond_norm) with censoring-aware training:
      - dynamic truncation sampling (K truncations per engine per epoch)
      - pairwise ranking hinge loss on mu
      - auxiliary bucket head (classification)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_censoring_aware"

    training = cfg.setdefault("training_params", {})
    training["censoring_aware_training"] = True
    training.setdefault("num_truncations_per_engine", 5)
    training.setdefault("trunc_p_full", 0.25)
    training.setdefault("trunc_r_min", 0.4)
    training.setdefault("trunc_r_max", 1.0)
    # Ensure we actually run the intended two-phase damage warmup (v3d/v4/v5 base),
    # even if a caller overrides/strips training_params.
    training.setdefault("damage_two_phase", True)
    training.setdefault("damage_warmup_epochs", 10)

    # Enable bucket head in the encoder
    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_bucket_head"] = True
    enc.setdefault("rul_bucket_edges", [25.0, 50.0, 75.0, 100.0, 125.0])

    # Loss params
    loss = cfg.setdefault("loss_params", {})
    loss.setdefault("use_ranking_loss", True)
    loss.setdefault("lambda_rank", 0.1)
    loss.setdefault("rank_margin", 1.0)

    loss.setdefault("use_bucket_head", True)
    loss.setdefault("lambda_bucket", 0.1)
    loss.setdefault("rul_bucket_edges", [25.0, 50.0, 75.0, 100.0, 125.0])
    # Ensure damage HI supervision is enabled (required for meaningful Phase-1 warmup).
    loss.setdefault("damage_hi_weight", 5.0)

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring_config() -> ExperimentConfig:
    """
    FD004 v5 (cond_norm) with multiview censoring samples:
      - keeps full regular sliding-window training intact
      - adds auxiliary censored-view samples (K cutpoints per engine, M windows per cut)
      - mixes regular/aux streams via aux_sample_ratio
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring"

    train = cfg.setdefault("training_params", {})
    train["use_multiview_censoring"] = True
    train.setdefault("num_truncations_per_engine", 5)      # K
    train.setdefault("num_windows_per_truncation", 8)      # M
    train.setdefault("trunc_ratio_min", 0.4)
    train.setdefault("trunc_ratio_max", 1.0)
    train.setdefault("trunc_p_full", 0.25)
    train.setdefault("aux_sample_ratio", 0.3)
    train.setdefault("run_seed", train.get("random_seed", 42))

    # Keep two-phase damage warmup as in v3d/v5 base
    train.setdefault("damage_two_phase", True)
    train.setdefault("damage_warmup_epochs", 10)

    # Enable bucket head (optional) and keep ranking loss enabled but applied only to aux samples
    enc = cfg.setdefault("encoder_kwargs", {})
    enc.setdefault("use_bucket_head", True)
    enc.setdefault("rul_bucket_edges", [25.0, 50.0, 75.0, 100.0, 125.0])

    loss = cfg.setdefault("loss_params", {})
    loss.setdefault("use_ranking_loss", True)
    loss.setdefault("lambda_rank", 0.1)
    loss.setdefault("rank_margin", 1.0)
    loss.setdefault("use_bucket_head", True)
    loss.setdefault("lambda_bucket", 0.1)
    loss.setdefault("rul_bucket_edges", [25.0, 50.0, 75.0, 100.0, 125.0])
    loss.setdefault("damage_hi_weight", 5.0)

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_quantiles_risk_config() -> ExperimentConfig:
    """
    FD004 v5 (cond_norm) + multiview censoring + quantile head with risk/bias penalties.
    Focus: reduce optimistic tail risk (q_upper > true) and then systematic bias.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_quantiles_risk"

    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_rul_quantiles_head"] = True
    enc["rul_quantiles"] = [0.1, 0.5, 0.9]

    loss = cfg.setdefault("loss_params", {})
    # Base quantile loss (pinball)
    loss["rul_quantiles"] = [0.1, 0.5, 0.9]
    loss["rul_quantile_weight"] = 1.0
    loss.setdefault("rul_quantile_cross_weight", 0.1)
    loss.setdefault("rul_quantile_p50_mse_weight", 0.0)

    # NEW: risk + bias calibration (advisor naming)
    loss["lambda_risk"] = 0.2
    loss["risk_margin"] = 0.0
    loss["lambda_q50_bias"] = 0.05
    loss["bias_calibration_mode"] = "batch_abs"  # off|batch_abs|ema
    loss["bias_ema_beta"] = 0.98

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config() -> ExperimentConfig:
    """
    FD004 v5 (cond_norm) + multiview censoring + residual risk head.

    Strategy:
      - Keep μ as the primary head (MSE training unchanged)
      - Add a separate head predicting an upper quantile of overshoot residual
      - Derive safe_RUL = clamp(μ - relu(risk_q), 0, max_rul) in inference
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk"

    enc = cfg.setdefault("encoder_kwargs", {})
    enc["use_residual_risk_head"] = True
    enc.setdefault("residual_risk_hidden_dim", 128)

    # Ensure direct RUL quantiles are OFF for this experiment (explicit safety layer instead)
    enc["use_rul_quantiles_head"] = False

    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.90
    loss["lambda_residual_risk"] = 0.10

    # Diagnostics knobs (inference-only; safe defaults)
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)

    # Critical: this run should NOT early-stop on val_loss (dominated by HI/damage),
    # otherwise μ can collapse while val_loss looks "good".
    training = cfg.setdefault("training_params", {})
    training.setdefault("monitor_metric", "val_rmse")
    # Training budget: make risk-head calibration stable across seeds/runs.
    training.setdefault("num_epochs", 100)

    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau90_w1_config() -> ExperimentConfig:
    """
    Residual-risk sweep A1:
      - risk_tau = 0.90 (baseline quantile)
      - lambda_residual_risk = 1.0 (stronger learning signal for risk head)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau90_w1"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.90
    loss["lambda_residual_risk"] = 1.0
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_config() -> ExperimentConfig:
    """
    Residual-risk sweep A2:
      - risk_tau = 0.95 (target ~95% coverage)
      - lambda_residual_risk = 1.0
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 1.0
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w2_config() -> ExperimentConfig:
    """
    Residual-risk sweep A3:
      - risk_tau = 0.99 (target ~99% coverage; very conservative)
      - lambda_residual_risk = 2.0 (stronger weight to learn extreme quantile)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w2"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.99
    loss["lambda_residual_risk"] = 2.0
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low10_config() -> ExperimentConfig:
    """
    Residual-risk sweep A4 (low-RUL focused):
      - risk_tau = 0.95
      - lambda_residual_risk = 3.0
      - risk_low_weight = 10.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low10"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 3.0
    loss["risk_low_weight"] = 10.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20_config() -> ExperimentConfig:
    """
    Residual-risk sweep A5 (very conservative, low-RUL focused):
      - risk_tau = 0.99
      - lambda_residual_risk = 5.0
      - risk_low_weight = 20.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.99
    loss["lambda_residual_risk"] = 5.0
    loss["risk_low_weight"] = 20.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    # Training budget: align with the project default for comparable sweeps.
    train = cfg.setdefault("training_params", {})
    train["num_epochs"] = 100
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_config() -> ExperimentConfig:
    """
    Residual-risk sweep (τ=0.95 target):
      - risk_tau = 0.95
      - lambda_residual_risk = 1.0
      - risk_low_weight = 20.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 1.0
    loss["risk_low_weight"] = 20.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    train = cfg.setdefault("training_params", {})
    train["num_epochs"] = 100
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_harmonized_windows_config() -> ExperimentConfig:
    """
    Same as `..._tau95_w1_low20`, but with explicit harmonized window/target policy
    persisted to summary.json and consumed by the training pipeline:
      - past_len=30
      - horizon=40
      - pad_mode="clamp"
      - max_rul=125 (capped targets/eval)

    This is meant to align windowing with the WorldModel v3 policy.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_harmwin"
    cfg["window_cfg"] = {
        "past_len": 30,
        "horizon": 40,
        "stride": 1,
        "pad_mode": "clamp",
        "require_full_horizon": False,
    }
    cfg["target_cfg"] = {
        "max_rul": 125,
        "cap_targets": True,
        # Transformer/EOL models train on the scalar at the window end (current RUL).
        "eol_target_mode": "current_from_df",
        # Eval should be NASA-style capped for comparability.
        "clip_eval_y_true": True,
    }
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low20_config() -> ExperimentConfig:
    """
    Residual-risk sweep (τ=0.95 target):
      - risk_tau = 0.95
      - lambda_residual_risk = 3.0
      - risk_low_weight = 20.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low20"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 3.0
    loss["risk_low_weight"] = 20.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    train = cfg.setdefault("training_params", {})
    train["num_epochs"] = 100
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w5_low20_config() -> ExperimentConfig:
    """
    Residual-risk sweep (τ=0.95 target):
      - risk_tau = 0.95
      - lambda_residual_risk = 5.0
      - risk_low_weight = 20.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w5_low20"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 5.0
    loss["risk_low_weight"] = 20.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    train = cfg.setdefault("training_params", {})
    train["num_epochs"] = 100
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w8_low20_config() -> ExperimentConfig:
    """
    Residual-risk sweep (τ=0.95 target):
      - risk_tau = 0.95
      - lambda_residual_risk = 8.0
      - risk_low_weight = 20.0 for rul_true <= low_rul_threshold (default 20)
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w8_low20"
    loss = cfg.setdefault("loss_params", {})
    loss["risk_tau"] = 0.95
    loss["lambda_residual_risk"] = 8.0
    loss["risk_low_weight"] = 20.0
    loss.setdefault("low_rul_threshold", 20.0)
    loss.setdefault("overshoot_threshold", 20.0)
    train = cfg.setdefault("training_params", {})
    train["num_epochs"] = 100
    return cfg


def get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_tuned_config() -> ExperimentConfig:
    """
    Tuned version of v3c: stronger Phase-1 damage warmup and slightly higher
    Phase-2 damage weight. Same architecture and feature setup as v3c.
    """
    cfg = get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_config()

    cfg["experiment_name"] = "fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_tuned"

    training = cfg.setdefault("training_params", {})

    # Stronger warmup on HI_phys_v3
    training["damage_two_phase"] = True
    training["damage_warmup_epochs"] = 10  # Phase 1: epochs 0–9

    # Stronger focus on damage in Phase 1, slightly increased in Phase 2
    loss_params = cfg.setdefault("loss_params", {})
    base_damage_w = loss_params.get("damage_hi_weight", 5.0)

    training["damage_phase1_damage_weight"] = 10.0  # was 5.0
    training["damage_phase2_damage_weight"] = 4.0   # was 3.0

    # Keep base damage_hi_weight as reference (for non-two-phase runs)
    loss_params["damage_hi_weight"] = base_damage_w

    return cfg

def get_fd004_state_encoder_v3_damage_msdt_v1_config() -> ExperimentConfig:
    """
    FD004 – State Encoder V3 with cumulative damage head (ms+DT v2 features).

    - Reuses the full ms_dt_v2 feature pipeline (raw + ms + residual + Cond_* + Twin_*)
    - Enables cumulative damage head to produce HI_phys(t) as primary trajectory
    - Keeps scalar HI / RUL heads as auxiliary outputs.
    """
    base = get_fd004_transformer_encoder_ms_dt_v2_config()
    cfg: ExperimentConfig = copy.deepcopy(base)

    cfg["experiment_name"] = "fd004_state_encoder_v3_damage_msdt_v1"
    cfg["dataset"] = "FD004"
    cfg["encoder_type"] = "transformer_state_encoder_v3_physics"

    # Data section mirrors the other state-encoder physics configs
    data_cfg = cfg.get("data", {})
    data_cfg.update(
        {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        }
    )
    cfg["data"] = data_cfg

    # Model: compact Transformer encoder with cumulative damage head enabled
    cfg["model"] = {
        "input_dim": None,
        "d_model": 96,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "cond_in_dim": None,
        "use_damage_head": True,
        "L_ref": 300.0,
        "alpha_base": 0.1,
    }

    cfg["training"] = {
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 80,
        "patience": 10,
    }

    # Loss configuration for damage-based HI_phys trajectory
    cfg["loss"] = {
        "hi_phys_weight": 1.0,
        "hi_aux_weight": 0.3,
        "rul_weight": 0.3,
        "mono_weight": 0.01,
        "smooth_weight": 0.01,
        # Keep alignment weights present but disabled for this run
        "hi_weight": 1.0,
        "align_weight": 0.0,
        "use_rul_hi_alignment_loss": False,
    }

    # IMPORTANT: do NOT touch cfg["features"] / cfg["phys_features"];
    # they remain exactly as in ms_dt_v2 so the feature space matches.
    cfg["paths"] = {
        "result_dir": "results/fd004/fd004_state_encoder_v3_damage_msdt_v1",
    }
    cfg["seed"] = base.get("training_params", {}).get("random_seed", 42)

    return cfg

def get_fd004_transformer_state_encoder_v3_config() -> ExperimentConfig:
    """
    FD004 – Transformer State Encoder V3 (ms+DT features + continuous condition vector).

    This config mirrors the ms+DT feature pipeline and is consumed by
    train_state_encoder_v3 via a SimpleNamespace-style view.
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()

    cfg: ExperimentConfig = {
        "experiment_name": "fd004_transformer_state_encoder_v3",
        "dataset": "FD004",
        "encoder_type": "transformer_state_encoder_v3",
        "data": {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            # feature_cols will be determined at runtime from the same pipeline
            # used for the ms+DT encoder; we keep this key for clarity.
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        },
        "model": {
            "input_dim": None,  # to be set at runtime once feature_cols are known
            "d_model": 96,
            "num_layers": 3,
            "num_heads": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "use_cond_encoder": True,
            # cond_in_dim is inferred inside the training script from Cond_* columns
            "cond_in_dim": None,
        },
        "hi": {
            "plateau_threshold": 80.0,
            "eol_threshold": 25.0,
        },
        "training": {
            "batch_size": 256,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 80,
            "patience": 10,
        },
        "loss": {
            "hi_weight": 1.0,
            "rul_weight": 0.1,
        },
        "paths": {
            "result_dir": "results/fd004/fd004_transformer_state_encoder_v3",
        },
        "seed": 42,
    }

    return cfg


def get_fd004_transformer_state_encoder_v3_physics_config() -> ExperimentConfig:
    """
    FD004 – Physics-informed Transformer State Encoder V3.

    Uses the full ms+DT feature set (raw + ms + residuals + Cond_* + Twin_*)
    and trains on physics-informed HI labels (residuals + EGT_Drift + Effizienz_HPC_Proxy)
    plus a weak auxiliary RUL_norm loss.
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()

    cfg: ExperimentConfig = {
        "experiment_name": "fd004_transformer_state_encoder_v3_physics",
        "dataset": "FD004",
        "encoder_type": "transformer_state_encoder_v3_physics",
        "data": {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            # feature_cols will be determined at runtime from the same pipeline
            # used for the ms+DT encoder; we keep this key for clarity.
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        },
        "model": {
            "input_dim": None,  # to be set at runtime once feature_cols are known
            "d_model": 96,
            "num_layers": 3,
            "num_heads": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            # cond_in_dim is inferred inside the training script from Cond_* columns
            "cond_in_dim": None,
        },
        "training": {
            "batch_size": 256,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 80,
            "patience": 10,
        },
        # Loss parameters for the dedicated state-encoder training loop.
        # These are read via cfg["loss"] in the experiment registry, and then
        # routed into the SimpleNamespace cfg.loss used in
        # train_state_encoder_v3_physics.
        "loss": {
            "hi_weight": 1.0,
            "rul_weight": 0.1,
            "align_weight": 0.0,
            "use_rul_hi_alignment_loss": False,
        },
        "paths": {
            # Match the directory name for the physics state encoder experiment
            "result_dir": "results/fd004/fd004_transformer_state_encoder_v3_physics",
        },
        "seed": 42,
    }

    return cfg


def get_fd004_state_encoder_v3_physics_A1_ms_only_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (A1):
    - Multi-scale temporal + physics features only (no Cond_* / Twin_* / residuals).
    - Stronger auxiliary RUL loss (rul_weight = 0.3), no alignment loss.
    """
    base = get_fd004_transformer_state_encoder_v3_physics_config()

    cfg: ExperimentConfig = {
        **base,
        "experiment_name": "fd004_state_encoder_v3_physics_A1_ms_only",
    }

    # Explicitly disable continuous condition vector and twin/residual features
    # to keep this run purely "ms-only" in terms of inputs.
    cfg["phys_features"] = {
        "use_condition_vector": False,
        "use_digital_twin_residuals": False,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    loss_cfg = cfg.get("loss", {}).copy()
    loss_cfg["hi_weight"] = 1.0
    loss_cfg["rul_weight"] = 0.3
    loss_cfg["use_rul_hi_alignment_loss"] = False
    loss_cfg["align_weight"] = 0.0
    cfg["loss"] = loss_cfg

    return cfg


def get_fd004_state_encoder_v3_physics_A2_ms_only_align_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (A2):
    - Same ms-only input setup as A1.
    - Stronger RUL weight (0.3) + weak RUL–HI alignment loss.
    """
    base = get_fd004_transformer_state_encoder_v3_physics_config()

    cfg: ExperimentConfig = {
        **base,
        "experiment_name": "fd004_state_encoder_v3_physics_A2_ms_only_align",
    }

    cfg["phys_features"] = {
        "use_condition_vector": False,
        "use_digital_twin_residuals": False,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    loss_cfg = cfg.get("loss", {}).copy()
    loss_cfg["hi_weight"] = 1.0
    loss_cfg["rul_weight"] = 0.3
    loss_cfg["use_rul_hi_alignment_loss"] = True
    loss_cfg["align_weight"] = 0.1
    cfg["loss"] = loss_cfg

    return cfg


def get_fd004_state_encoder_v3_physics_B1_msdt_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (B1):
    - Full ms+DT feature block (Cond_* + Twin_* + residuals + multiscale).
    - Physics-based HI target, no alignment loss.
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()

    cfg: ExperimentConfig = {
        **base,
        "experiment_name": "fd004_state_encoder_v3_physics_B1_msdt",
        "dataset": "FD004",
        "encoder_type": "transformer_state_encoder_v3_physics",
    }

    # Data section mirrors the state-encoder physics config
    data_cfg = cfg.get("data", {})
    data_cfg.update(
        {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        }
    )
    cfg["data"] = data_cfg

    # Model section: small Transformer encoder as in the V3 physics baseline
    cfg["model"] = {
        "input_dim": None,
        "d_model": 96,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "cond_in_dim": None,
    }

    # Training hyperparameters: reuse the physics state-encoder defaults
    cfg["training"] = {
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 80,
        "patience": 10,
    }

    loss_cfg = {
        "hi_weight": 1.0,
        "rul_weight": 0.3,
        "align_weight": 0.0,
        "use_rul_hi_alignment_loss": False,
    }
    cfg["loss"] = loss_cfg

    cfg["paths"] = {
        "result_dir": "results/fd004/fd004_state_encoder_v3_physics_B1_msdt",
    }

    cfg["seed"] = base.get("training_params", {}).get("random_seed", 42)

    return cfg


def get_fd004_state_encoder_v3_physics_B2_msdt_align_config() -> ExperimentConfig:
    """
    FD004 – Physics State Encoder V3 (B2):
    - Same full ms+DT feature block as B1.
    - Adds a weak RUL–HI alignment loss.
    """
    base = get_fd004_transformer_encoder_ms_dt_v1_config()

    cfg: ExperimentConfig = {
        **base,
        "experiment_name": "fd004_state_encoder_v3_physics_B2_msdt_align",
        "dataset": "FD004",
        "encoder_type": "transformer_state_encoder_v3_physics",
    }

    data_cfg = cfg.get("data", {})
    data_cfg.update(
        {
            "fd_name": "FD004",
            "max_rul": base.get("max_rul", 125.0),
            "past_len": base.get("past_len", 30),
            "feature_cols": None,
            "train_frac": base.get("training_params", {}).get("engine_train_ratio", 0.8),
        }
    )
    cfg["data"] = data_cfg

    cfg["model"] = {
        "input_dim": None,
        "d_model": 96,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "cond_in_dim": None,
    }

    cfg["training"] = {
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 80,
        "patience": 10,
    }

    loss_cfg = {
        "hi_weight": 1.0,
        "rul_weight": 0.3,
        "align_weight": 0.1,
        "use_rul_hi_alignment_loss": True,
    }
    cfg["loss"] = loss_cfg

    cfg["paths"] = {
        "result_dir": "results/fd004/fd004_state_encoder_v3_physics_B2_msdt_align",
    }

    cfg["seed"] = base.get("training_params", {}).get("random_seed", 42)

    return cfg


def get_fd004_transformer_worldmodel_v1_config() -> ExperimentConfig:
    """
    Transformer World Model V1 on FD004 using the ms+DT feature block.

    - Reuses the fd004_transformer_encoder_ms_dt_v1 encoder hyperparameters
      (EOLFullTransformerEncoder with 349D ms+DT inputs).
    - World model will be built in the training loop from these settings.
    - This config primarily marks the experiment as a world_model_universal_v3
      style run with a Transformer-based encoder.
    """
    # Start from the ms+DT encoder config to keep encoder_kwargs / training_params aligned.
    base = get_fd004_transformer_encoder_ms_dt_v1_config()

    cfg: ExperimentConfig = ExperimentConfig(
        experiment_name="fd004_transformer_worldmodel_v1",
        dataset="FD004",
        # We reuse the existing world model type flag so that run_experiments.py
        # routes into the world-model training block.
        encoder_type="world_model_universal_v3",
        encoder_kwargs={
            # d_model etc. from the base Transformer-Encoder
            "d_model": base["encoder_kwargs"]["d_model"],
            "num_layers": base["encoder_kwargs"]["num_layers"],
            "nhead": base["encoder_kwargs"]["n_heads"],
            "dim_feedforward": base["encoder_kwargs"]["dim_feedforward"],
            "dropout": base["encoder_kwargs"]["dropout"],
            # We will ignore kernel_sizes/seq_encoder_type inside the new
            # transformer world-model path and instead construct
            # EOLFullTransformerEncoder directly.
            "kernel_sizes": base["encoder_kwargs"].get("kernel_sizes", [3, 5, 9]),
            "seq_encoder_type": "transformer",
            "decoder_num_layers": 1,
        },
        loss_params={
            # We keep the same loss hyperparams as the encoder-only experiment;
            # world-model specific weights go into world_model_params below.
            **base["loss_params"],
        },
        optimizer_params={
            **base["optimizer_params"],
        },
        training_params={
            **base["training_params"],
        },
        phase_2_params={
            # Use the same condition embedding setup as the encoder-only TE model.
            **base["phase_2_params"],
        },
        # Forward the ms+DT feature configuration and phys_features block,
        # so run_experiments.py builds exactly the same 349D feature vectors.
        features=base.get("features", {}),
        phys_features=base.get("phys_features", {}),
        # World model specific parameters (horizon, loss weights, etc.)
        world_model_params={
            "horizon": 20,
            "traj_loss_weight": 1.0,
            "eol_loss_weight": 0.0,  # focus first version on sensor trajectory
            "hi_loss_weight": 0.3,
            "mono_late_weight": 0.0,
            "mono_global_weight": 0.0,
            "past_len": 30,
            "max_rul": 125,
            "use_condition_wise_scaling": True,
            # No HI-fusion into EOL yet; we can refine this later.
            "use_hi_in_eol": False,
            "use_hi_slope_in_eol": False,
            "eol_tail_rul_threshold": None,
            "eol_tail_weight": 1.0,
            # World-model specific sensor/HI/RUL loss weights for the decoder
            "sensor_loss_weight": 1.0,
            "hi_future_loss_weight": 0.3,
            "rul_future_loss_weight": 0.3,
        },
    )

    return cfg


def get_fd001_transformer_encoder_phys_v3_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD001 (v3).

    - Uses extended continuous condition vector (condition_vector_version=3)
    - Uses HealthyTwinRegressor for twin predictions + residuals
    - Uses ImprovedRULHead with piecewise RUL mapping.
    """
    cfg = get_fd001_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd001_transformer_encoder_phys_v3"

    # FD001 is effectively single-condition -> no discrete embeddings
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "improved",
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "dropout": 0.1,
        "use_skip": True,
        "use_hi_fusion": True,
        "use_piecewise_mapping": True,
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd001_transformer_encoder_phys_v4_config() -> ExperimentConfig:
    """
    Transformer V4 on FD001 (single-condition, physics-informed).

    - Extended continuous condition vector (condition_vector_version=3)
    - HealthyTwinRegressor residuals
    - RULHeadV4
    """
    cfg = get_fd001_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd001_transformer_encoder_phys_v4"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "v4",
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd002_transformer_encoder_phys_v3_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD002 (v3).

    - Replaces ConditionID embeddings with extended continuous condition vector.
    - Uses HealthyTwinRegressor for twin predictions + residuals.
    - Uses ImprovedRULHead with piecewise RUL mapping.
    """
    cfg = get_fd002_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd002_transformer_encoder_phys_v3"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "improved",
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "dropout": 0.1,
        "use_skip": True,
        "use_hi_fusion": True,
        "use_piecewise_mapping": True,
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd002_transformer_encoder_phys_v4_config() -> ExperimentConfig:
    """
    Transformer V4 on FD002 (multi-condition, physics-informed).

    - Extended continuous condition vector (condition_vector_version=3)
    - HealthyTwinRegressor residuals
    - RULHeadV4
    """
    cfg = get_fd002_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd002_transformer_encoder_phys_v4"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "v4",
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd003_transformer_encoder_phys_v3_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD003 (v3).

    - Single-condition dataset with extended continuous condition vector.
    - HealthyTwinRegressor residuals.
    - Improved RUL head with piecewise mapping.
    """
    cfg = get_fd003_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd003_transformer_encoder_phys_v3"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "improved",
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "dropout": 0.1,
        "use_skip": True,
        "use_hi_fusion": True,
        "use_piecewise_mapping": True,
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd003_transformer_encoder_phys_v4_config() -> ExperimentConfig:
    """
    Transformer V4 on FD003 (single-condition, physics-informed).

    - Extended continuous condition vector (condition_vector_version=3)
    - HealthyTwinRegressor residuals
    - RULHeadV4
    """
    cfg = get_fd003_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd003_transformer_encoder_phys_v4"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
        "condition_vector_version": 3,
    }

    cfg["rul_head_params"] = {
        "rul_head_type": "v4",
    }
    cfg["max_rul"] = 125.0

    return cfg


def get_fd001_transformer_encoder_phys_v2_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD001.

    - Uses continuous condition vector (Cond_* features)
    - Uses a global HealthyTwinRegressor for twin predictions + residuals
    - Disables discrete ConditionID embeddings (single-condition dataset)
    """
    cfg = get_fd001_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd001_transformer_encoder_phys_v2"

    # Ensure condition embeddings are disabled (FD001 has effectively one condition)
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    # Physics / twin feature options used in run_experiments.py
    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd002_transformer_encoder_phys_v2_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD002.

    - Replaces discrete ConditionID embeddings with a continuous condition vector
    - Uses HealthyTwinRegressor for twin predictions + residuals
    """
    cfg = get_fd002_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd002_transformer_encoder_phys_v2"

    # Disable discrete condition embeddings – rely on continuous condition vector instead
    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd003_transformer_encoder_phys_v2_config() -> ExperimentConfig:
    """
    Physics-informed Transformer-Encoder EOL+HI experiment on FD003.

    - Uses continuous condition vector (Cond_* features)
    - Uses HealthyTwinRegressor for twin predictions + residuals
    - Disables discrete ConditionID embeddings (single-condition dataset)
    """
    cfg = get_fd003_transformer_encoder_v1_config()
    cfg["experiment_name"] = "fd003_transformer_encoder_phys_v2"

    cfg["phase_2_params"] = {
        **cfg["phase_2_params"],
        "use_condition_embedding": False,
        "cond_emb_dim": 0,
        "num_conditions": 1,
    }

    cfg["phys_features"] = {
        "use_condition_vector": True,
        "use_twin_features": True,
        "twin_baseline_len": 30,
    }

    return cfg


def get_fd001_transformer_encoder_resid_v1_config() -> ExperimentConfig:
    """
    Transformer-Encoder EOL+HI Experiment auf FD001 mit Residual-Features.
    Identisch zu fd001_transformer_encoder_v1, aber Residuals via Namenssuffix \"resid\".
    """
    base_cfg = get_fd001_transformer_encoder_v1_config()
    base_cfg["experiment_name"] = "fd001_transformer_encoder_resid_v1"
    return base_cfg


def get_fd002_transformer_encoder_resid_v1_config() -> ExperimentConfig:
    """
    Transformer-Encoder EOL+HI Experiment auf FD002 mit Residual-Features.
    Identisch zu fd002_transformer_encoder_v1, aber Residuals via Namenssuffix \"resid\".
    """
    base_cfg = get_fd002_transformer_encoder_v1_config()
    base_cfg["experiment_name"] = "fd002_transformer_encoder_resid_v1"
    return base_cfg


def get_fd003_transformer_encoder_resid_v1_config() -> ExperimentConfig:
    """
    Transformer-Encoder EOL+HI Experiment auf FD003 mit Residual-Features.
    Identisch zu fd003_transformer_encoder_v1, aber Residuals via Namenssuffix \"resid\".
    """
    base_cfg = get_fd003_transformer_encoder_v1_config()
    base_cfg["experiment_name"] = "fd003_transformer_encoder_resid_v1"
    return base_cfg

# ===================================================================
# Experiment Groups
# ===================================================================

def get_experiment_group(group_name: str) -> List[ExperimentConfig]:
    """
    Get a list of experiment configs for a named group.
    
    Args:
        group_name: "A" (LSTM baselines), "B" (Transformer baselines), 
                   "C" (FD004 Transformer sweep), or "all"
    
    Returns:
        List of experiment configurations
    """
    if group_name == "A" or group_name == "lstm_baselines":
        # Group A: LSTM Baselines for all datasets
        return [get_lstm_baseline_config(d) for d in ["FD001", "FD002", "FD003", "FD004"]]
    
    elif group_name == "B" or group_name == "transformer_baselines":
        # Group B: Transformer Baselines for all datasets
        return [get_transformer_baseline_config(d) for d in ["FD001", "FD002", "FD003", "FD004"]]
    
    elif group_name == "C" or group_name == "fd004_sweep":
        # Group C: FD004 Transformer Hyperparameter Sweep
        return [
            get_fd004_transformer_hi_strong(),
            get_fd004_transformer_hi_condcalib(),
            get_fd004_transformer_small_regularized(),
        ]
    
    elif group_name == "P3" or group_name == "phase3" or group_name == "universal_v1":
        # Group P3: UniversalEncoderV1 for all datasets
        return [get_universal_v1_config(d) for d in ["FD001", "FD002", "FD003", "FD004"]]
    
    elif group_name == "P3_2" or group_name == "phase3_2" or group_name == "universal_v2":
        # Group P3_2: UniversalEncoderV2 for FD002 and FD004
        return [get_universal_v2_config(d) for d in ["FD002", "FD004"]]
    
    elif group_name == "P3_2_FD004" or group_name == "phase3_2_fd004" or group_name == "fd004_tuning":
        # Phase 3.2 Mini-Tuning Set for FD004
        return [
            get_universal_v2_config("FD004"),  # Experiment A: Baseline
            get_fd004_universal_v2_d96(),      # Experiment B: Higher capacity
            get_fd004_universal_v2_strongmono(), # Experiment C: Stronger monotonicity
        ]
    
    elif group_name == "P4" or group_name == "phase4" or group_name == "fd_all_phase4_residual":
        # Phase 4: Residual features for all datasets
        return [get_phase4_residual_config(d) for d in ["FD001", "FD002", "FD003", "FD004"]]
    
    elif group_name == "world_phase5" or group_name == "world_p5" or group_name == "world_v3":
        # Default: World Model v3 experiments for FD002 & FD004 (original setup)
        return [get_world_model_phase5_universal_v3_residual_config(d) for d in ["FD002", "FD004"]]
    elif group_name in ["world_v3_fd001_fd002", "world_phase5_fd001_fd002"]:
        # World Model v3 experiments for FD001 & FD002
        return [get_world_model_phase5_universal_v3_residual_config(d) for d in ["FD001", "FD002"]]
    elif group_name in ["world_v3_all", "world_phase5_all"]:
        # World Model v3 experiments for all four datasets
        return [get_world_model_phase5_universal_v3_residual_config(d) for d in ["FD001", "FD002", "FD003", "FD004"]]
    elif group_name == "world_phase4" or group_name == "world_p4":
        # Return default world model experiments
        return [get_world_model_phase4_residual_config(d, variant="default") for d in ["FD002", "FD004"]]
    elif group_name == "world_eol_heavy":
        # EOL-heavy experiments
        return [get_world_model_phase4_residual_config(d, variant="eol_heavy") for d in ["FD002", "FD004"]]
    elif group_name == "world_h10_eol_heavy":
        # Shorter horizon + EOL-heavy experiments
        return [get_world_model_phase4_residual_config(d, variant="h10_eol_heavy") for d in ["FD002", "FD004"]]
    elif group_name in ["TE", "transformer_encoder_all"]:
        # Transformer-Encoder EOL+HI experiments for all datasets
        return [
            get_fd001_transformer_encoder_v1_config(),
            get_fd002_transformer_encoder_v1_config(),
            get_fd003_transformer_encoder_v1_config(),
            get_fd004_transformer_encoder_v1_config(),
        ]
    elif group_name in ["TE_MS_DT", "transformer_encoder_ms_dt_all"]:
        # Multi-Scale + Digital-Twin Transformer-Encoder experiments for all datasets
        return [
            get_fd001_transformer_encoder_ms_dt_v1_config(),
            get_fd002_transformer_encoder_ms_dt_v1_config(),
            get_fd003_transformer_encoder_ms_dt_v1_config(),
            get_fd004_transformer_encoder_ms_dt_v1_config(),
            get_fd004_transformer_encoder_ms_dt_v2_config(),
        ]
    
    elif group_name == "all":
        # All experiments
        return (
            get_experiment_group("A") +
            get_experiment_group("B") +
            get_experiment_group("C") +
            get_experiment_group("P3") +
            get_experiment_group("P3_2") +
            get_experiment_group("P4")
        )
    
    else:
        raise ValueError(
            f"Unknown experiment group: {group_name}. "
            f"Valid options: 'A', 'B', 'C', 'P3', 'P3_2', 'P3_2_FD004', 'P4', 'phase4', "
            f"'fd_all_phase4_residual', 'all', 'lstm_baselines', 'transformer_baselines', "
            f"'fd004_sweep', 'universal_v1', 'universal_v2', 'fd004_tuning', "
            f"'world_phase4', 'world_p4', 'world_eol_heavy', 'world_h10_eol_heavy', "
            f"'world_phase5', 'world_p5', 'world_v3'"
        )


# ===================================================================
# Group P3-U: UniversalEncoderV1 per Dataset (Phase 3)
# ===================================================================

def get_universal_v1_config(dataset: str) -> ExperimentConfig:
    """Get UniversalEncoderV1 config for a dataset."""
    # Determine if condition fusion should be enabled
    use_cond_fusion = dataset in ["FD002", "FD004"]  # Multi-condition datasets
    num_conditions = 7 if use_cond_fusion else None  # None means single condition
    
    return ExperimentConfig(
        experiment_name=f"{dataset.lower()}_phase3_universal_v1",
        dataset=dataset,
        encoder_type="universal_v1",
        encoder_kwargs={
            "d_model": 48,
            "cnn_channels": None,  # Will default to d_model // 3 = 16
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.35,
            "mono_late_weight": 0.03,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 8,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_fusion,  # For compatibility with training code
            "cond_emb_dim": 4 if use_cond_fusion else 0,
            "num_conditions": num_conditions if use_cond_fusion else 1,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_universal_v2_config(dataset: str) -> ExperimentConfig:
    """
    Get UniversalEncoderV2 config for FD002 or FD004.
    
    Phase 3.2: Enhanced encoder with higher capacity (d_model=64) and
    stronger multi-scale CNN (kernels [3, 5, 9]) to improve FD004 performance.
    """
    if dataset not in ["FD002", "FD004"]:
        raise ValueError(f"UniversalEncoderV2 config only available for FD002 and FD004, got {dataset}")
    
    use_cond_fusion = True  # Both FD002 and FD004 have multiple conditions
    num_conditions = 7
    
    return ExperimentConfig(
        experiment_name=f"{dataset.lower()}_phase3_universal_v2_ms_cnn",
        dataset=dataset,
        encoder_type="universal_v2",
        encoder_kwargs={
            "d_model": 64,  # Higher capacity than V1 (32)
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": None,  # Will default to 4*d_model = 256
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],  # Stronger multi-scale CNN
            "seq_encoder_type": "transformer",  # Use Transformer for better sequence modeling
            "use_layer_norm": True,
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,  # Keep Phase 3 tuned settings
            "mono_late_weight": 0.02,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,  # Keep Phase 3 tuned patience
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_fusion,
            "cond_emb_dim": 4,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_universal_v2_d96() -> ExperimentConfig:
    """
    Phase 3.2 Experiment B: Higher-capacity encoder (d_model=96).
    
    Goal: See if RMSE goes down 1-2 points with higher capacity.
    """
    return ExperimentConfig(
        experiment_name="fd004_phase3_universal_v2_ms_cnn_d96",
        dataset="FD004",
        encoder_type="universal_v2",
        encoder_kwargs={
            "d_model": 96,  # Higher capacity
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": None,  # Will default to 4*d_model = 384
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "seq_encoder_type": "transformer",
            "use_layer_norm": True,
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,
            "mono_late_weight": 0.02,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_universal_v2_strongmono() -> ExperimentConfig:
    """
    Phase 3.2 Experiment C: Stronger EOL monotonicity (mono_late_weight=0.03).
    
    Goal: Make predicted RUL decay more cleanly near EOL without hurting mid-trajectory accuracy.
    """
    return ExperimentConfig(
        experiment_name="fd004_phase3_universal_v2_ms_cnn_strongmono",
        dataset="FD004",
        encoder_type="universal_v2",
        encoder_kwargs={
            "d_model": 64,
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": None,  # Will default to 4*d_model = 256
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "seq_encoder_type": "transformer",
            "use_layer_norm": True,
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,
            "mono_late_weight": 0.03,  # Increased from 0.02
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_universal_v1_tuned_config(dataset: str) -> ExperimentConfig:
    """
    Get tuned UniversalEncoderV1 config for FD002 or FD004.
    
    Phase 3.1: Tuned hyperparameters to approach Phase 2 Transformer performance.
    Smaller model (d_model=32) with adjusted loss weights.
    """
    if dataset not in ["FD002", "FD004"]:
        raise ValueError(f"Tuned config only available for FD002 and FD004, got {dataset}")
    
    use_cond_fusion = True  # Both FD002 and FD004 have multiple conditions
    num_conditions = 7
    
    return ExperimentConfig(
        experiment_name=f"{dataset.lower()}_phase3_universal_v1_tuned",
        dataset=dataset,
        encoder_type="universal_v1",
        encoder_kwargs={
            "d_model": 32,  # Reduced from 48
            "cnn_channels": None,  # Will default to d_model // 3 = 10 (min 8)
            "num_layers": 2,  # Reduced from 3
            "nhead": 4,
            "dim_feedforward": 192,  # Reduced from 256
            "dropout": 0.2,  # Increased from 0.1
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,  # Reduced from 0.35
            "mono_late_weight": 0.02,  # Reduced from 0.03
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,  # Increased from 8
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": use_cond_fusion,
            "cond_emb_dim": 4,
            "num_conditions": num_conditions,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_fd004_phase4_residual_config() -> ExperimentConfig:
    """
    Phase 4: FD004 with residual features enabled.
    Based on fd004_phase3_universal_v2_ms_cnn_d96, but with residual features.
    """
    return ExperimentConfig(
        experiment_name="fd004_phase4_universal_v2_ms_cnn_d96_residual",
        dataset="FD004",
        encoder_type="universal_v2",
        encoder_kwargs={
            "d_model": 96,  # Same as d96 config
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": None,  # Will default to 4*d_model = 384
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "seq_encoder_type": "transformer",
            "use_layer_norm": True,
            "max_seq_len": 300,
        },
        loss_params={
            "rul_beta": 45.0,
            "health_loss_weight": 0.30,
            "mono_late_weight": 0.02,
            "mono_global_weight": 0.003,
            "hi_condition_calib_weight": 0.0,
        },
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "use_mixed_precision": True,
            "random_seed": 42,
            "engine_train_ratio": 0.8,
            "shuffle_engines": True,
        },
        phase_2_params={
            "use_condition_embedding": True,
            "cond_emb_dim": 4,
            "num_conditions": 7,
            "smooth_hi_weight": 0.02,
            "smooth_hi_plateau_threshold": 80.0,
        },
    )


def get_phase4_residual_config(dataset: str) -> ExperimentConfig:
    """
    Phase 4: Residual features for all datasets.
    Based on the best phase 3 config for each dataset, but with residual features enabled.
    
    For FD001/FD003: Uses UniversalEncoderV1 (since they don't have phase 3.2 configs)
    For FD002/FD004: Uses UniversalEncoderV2 d96 config
    """
    if dataset in ["FD002", "FD004"]:
        # Use UniversalEncoderV2 d96 config
        base_config = get_fd004_universal_v2_d96() if dataset == "FD004" else get_universal_v2_config(dataset)
        return ExperimentConfig(
            experiment_name=f"{dataset.lower()}_phase4_universal_v2_ms_cnn_d96_residual",
            dataset=dataset,
            encoder_type=base_config["encoder_type"],
            encoder_kwargs=base_config["encoder_kwargs"],
            loss_params=base_config["loss_params"],
            optimizer_params=base_config["optimizer_params"],
            training_params=base_config["training_params"],
            phase_2_params=base_config["phase_2_params"],
        )
    else:
        # FD001/FD003: Use UniversalEncoderV1
        base_config = get_universal_v1_config(dataset)
        return ExperimentConfig(
            experiment_name=f"{dataset.lower()}_phase4_universal_v1_residual",
            dataset=dataset,
            encoder_type=base_config["encoder_type"],
            encoder_kwargs=base_config["encoder_kwargs"],
            loss_params=base_config["loss_params"],
            optimizer_params=base_config["optimizer_params"],
            training_params=base_config["training_params"],
            phase_2_params=base_config["phase_2_params"],
        )


def get_world_model_phase4_residual_config(
    dataset: str,
    variant: str = "default",
) -> ExperimentConfig:
    """
    Phase 4: World Model with UniversalEncoderV2 and residual features.
    
    For FD004 and FD002: Uses UniversalEncoderV2 d96 config with world model architecture.
    
    Args:
        dataset: Dataset name ("FD002" or "FD004")
        variant: Experiment variant:
            - "default": Standard config (horizon=20, traj_weight=1.0, eol_weight=1.0)
            - "eol_heavy": EOL-heavy loss (horizon=20, traj_weight=1.0, eol_weight=5.0)
            - "h10_eol_heavy": Shorter horizon + EOL-heavy (horizon=10, traj_weight=1.0, eol_weight=3.0)
    """
    if dataset not in ["FD002", "FD004"]:
        raise ValueError(f"World model Phase 4 experiments currently only support FD002 and FD004, got {dataset}")
    
    # Base config from Phase 4 EOL experiments
    base_config = get_fd004_universal_v2_d96() if dataset == "FD004" else get_universal_v2_config(dataset)
    
    # Determine experiment name and world_model_params based on variant
    if variant == "default":
        exp_name = f"{dataset.lower()}_world_phase4_universal_v2_residual"
        world_params = {
            "past_len": 30,
            "horizon": 20,
            "max_rul": 125,
            "use_condition_wise_scaling": True,
            "traj_loss_weight": 1.0,
            "eol_loss_weight": 1.0,
            "traj_step_weighting": None,
        }
    elif variant == "eol_heavy":
        exp_name = f"{dataset.lower()}_world_phase4_universal_v2_residual_eol_heavy"
        world_params = {
            "past_len": 30,
            "horizon": 20,
            "max_rul": 125,
            "use_condition_wise_scaling": True,
            "traj_loss_weight": 1.0,
            "eol_loss_weight": 5.0,
            "traj_step_weighting": None,
        }
    elif variant == "h10_eol_heavy":
        exp_name = f"{dataset.lower()}_world_phase4_universal_v2_residual_h10_eol_heavy"
        world_params = {
            "past_len": 30,
            "horizon": 10,
            "max_rul": 125,
            "use_condition_wise_scaling": True,
            "traj_loss_weight": 1.0,
            "eol_loss_weight": 3.0,
            "traj_step_weighting": None,
        }
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'default', 'eol_heavy', or 'h10_eol_heavy'")
    
    return ExperimentConfig(
        experiment_name=exp_name,
        dataset=dataset,
        encoder_type="world_model_universal_v2",  # Special type for world model
        encoder_kwargs={
            "d_model": 96,
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": 384,
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "seq_encoder_type": "transformer",
            "decoder_num_layers": 2,
        },
        loss_params=base_config["loss_params"],  # Not used for world model, but kept for consistency
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "engine_train_ratio": 0.8,
            "random_seed": 42,
            "use_mixed_precision": False,
        },
        phase_2_params=base_config["phase_2_params"],
        world_model_params=world_params,
    )


def get_world_model_phase5_universal_v3_residual_config(
    dataset: str,
) -> ExperimentConfig:
    """
    Phase 5: World Model v3 with Health Index head and residual features.
    
    For FD002 and FD004: Uses UniversalEncoderV2 d96 config with world model v3 architecture.
    For FD001 and FD003: Uses the corresponding Phase 4 residual base config as backbone
    (loss/training params), but still instantiates a WorldModelUniversalV3 encoder/decoder.
    
    Features:
    - Horizon: 40 (longer than v2)
    - Loss weights: traj=1.0, eol=5.0, hi=2.0
    - Monotonicity: late=0.1, global=0.1
    - Health Index head for physical HI prediction
    """
    if dataset not in ["FD001", "FD002", "FD003", "FD004"]:
        raise ValueError(
            f"World model Phase 5 experiments currently only support FD001–FD004, got {dataset}"
        )
    
    # Base config from best available Phase 3/4 EOL experiments
    if dataset in ["FD002", "FD004"]:
        # Multi-condition datasets: use UniversalEncoderV2-based configs
        base_config = get_fd004_universal_v2_d96() if dataset == "FD004" else get_universal_v2_config(dataset)
    else:
        # FD001/FD003: use Phase 4 residual config (UniversalEncoderV1 backbone)
        # We mainly inherit loss/optimizer/training/phase_2 params; the world model
        # itself will still use the V3 encoder/decoder architecture.
        base_config = get_phase4_residual_config(dataset)
    
    exp_name = f"{dataset.lower()}_world_phase5_universal_v3_residual"
    
    world_params = {
        "past_len": 30,
        # Horizon is now a first-class config parameter and will be swept
        # across {40, 30, 20} in dedicated experiments.
        "horizon": 40,
        "max_rul": 125,
        "use_condition_wise_scaling": True,
        "traj_loss_weight": 1.0,
        "eol_loss_weight": 5.0,  # Strongly weighted EOL
        "hi_loss_weight": 2.0,   # Health Index loss
        "mono_late_weight": 0.1,  # Late monotonicity
        "mono_global_weight": 0.1,  # Global trend
        "traj_step_weighting": None,  # Uniform weighting for now

        # New: HI-fusion into EOL head
        "use_hi_in_eol": True,
        "use_hi_slope_in_eol": True,

        # New: tail-weighted EOL loss (emphasize small RUL region)
        "eol_tail_rul_threshold": 40.0,
        "eol_tail_weight": 3.0,

        # Stage-1: 3-phase curriculum schedule (A/B/C) to improve early/mid-life dynamics
        # while preserving EOL performance.
        "three_phase_schedule": True,
        "phase_a_frac": 0.2,       # 0–20%: no EOL loss
        "phase_b_end_frac": 0.8,   # 20–80%: ramp EOL loss in
        "schedule_type": "linear",
        "eol_w_max": 1.0,

        # Stage-1: HI shape regularizers (small weights; tune per ablation)
        "hi_early_slope_weight": 0.05,
        "hi_early_slope_epsilon": 1e-3,
        "hi_early_slope_rul_threshold": 90.0,
        "hi_curvature_weight": 0.01,
        "hi_curvature_abs": True,

        # Stage-2 (optional): WorldModel HI→EOL consistency (OFF by default; enable in ablations)
        "w_eol_hi": 0.0,
        "eol_hi_threshold": 0.2,
        "eol_hi_temperature": 0.05,
        "eol_hi_p_min": 0.2,
    }

    # ------------------------------------------------------------------
    # FD004 stabilization defaults (ONLY for FD004)
    # ------------------------------------------------------------------
    if dataset == "FD004":
        # Stabilize EOL ramp-in by normalizing EOL targets/preds inside loss
        world_params.update(
            {
                "normalize_eol": True,
                "eol_scale": "rul_cap",
                "eol_loss_type": "huber",
                "eol_huber_beta": 0.1,
                "clip_grad_norm": 1.0,
                "freeze_encoder_epochs_after_eol_on": 3,
                # Temporarily disable HI-fusion into EOL head for stability; can be re-enabled later
                "use_hi_in_eol": False,
                "use_hi_slope_in_eol": False,
                # Keep max multiplier; normalization handles scale
                "eol_w_max": 1.0,
                # NEW: ensure we select best checkpoint only after EOL turns on
                "select_best_after_eol_active": True,
                "eol_active_min_mult": 0.02,
                "best_metric": "val_total",
                # NEW: align scalar EOL target + evaluation units to [0, max_rul]
                "cap_rul_targets_to_max_rul": True,
                "eol_target_mode": "current",
                "eval_clip_y_true_to_max_rul": True,
                # Optional: initialize EOL head bias near mean target
                "init_eol_bias_to_target_mean": True,
                # Horizon targets: include near-EOL windows via padded/clamped targets
                "use_padded_horizon_targets": True,
                "target_clamp_min": 0.0,
                "use_horizon_mask": True,
            }
        )
    
    return ExperimentConfig(
        experiment_name=exp_name,
        dataset=dataset,
        encoder_type="world_model_universal_v3",  # Special type for world model v3
        encoder_kwargs={
            "d_model": 96,
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": 384,
            "dropout": 0.1,
            "kernel_sizes": [3, 5, 9],
            "seq_encoder_type": "transformer",
            "decoder_num_layers": 2,
        },
        loss_params=base_config["loss_params"],  # Not used for world model, but kept for consistency
        optimizer_params={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        training_params={
            "num_epochs": 80,
            "batch_size": 256,
            "patience": 10,
            "engine_train_ratio": 0.8,
            "random_seed": 42,
            "use_mixed_precision": False,
        },
        phase_2_params=base_config["phase_2_params"],
        world_model_params=world_params,
    )


def get_experiment_by_name(experiment_name: str) -> ExperimentConfig:
    """
    Get a single experiment config by name.
    
    Args:
        experiment_name: Name of the experiment (e.g., "fd004_phase2_transformer_baseline")
    
    Returns:
        Experiment configuration
    """
    # Explicit check for the new Transformer World Model V1 experiments,
    # and dedicated FD004 State Encoder variants, which are not covered
    # by the generic pattern-based logic below.
    if experiment_name == "fd004_transformer_worldmodel_v1":
        return get_fd004_transformer_worldmodel_v1_config()
    if experiment_name == "fd004_transformer_worldmodel_v1_h10_eolfocus":
        return get_fd004_transformer_worldmodel_v1_h10_eolfocus_config()
    if experiment_name == "fd004_transformer_worldmodel_v1_h20_residuals_eolfocus":
        return get_fd004_transformer_worldmodel_v1_h20_residuals_eolfocus_config()
    # Explicit state-encoder experiments (not covered by generic patterns below)
    if experiment_name == "fd004_transformer_state_encoder_v3":
        return get_fd004_transformer_state_encoder_v3_config()
    if experiment_name == "fd004_transformer_state_encoder_v3_physics":
        return get_fd004_transformer_state_encoder_v3_physics_config()
    if experiment_name == "fd004_state_encoder_v3_physics_A1_ms_only":
        return get_fd004_state_encoder_v3_physics_A1_ms_only_config()
    if experiment_name == "fd004_state_encoder_v3_physics_A2_ms_only_align":
        return get_fd004_state_encoder_v3_physics_A2_ms_only_align_config()
    if experiment_name == "fd004_state_encoder_v3_physics_B1_msdt":
        return get_fd004_state_encoder_v3_physics_B1_msdt_config()
    if experiment_name == "fd004_state_encoder_v3_physics_B2_msdt_align":
        return get_fd004_state_encoder_v3_physics_B2_msdt_align_config()
    if experiment_name == "fd004_state_encoder_v3_physics_C1_msdt":
        return get_fd004_state_encoder_v3_physics_C1_msdt_config()
    if experiment_name == "fd004_state_encoder_v3_physics_C2_msdt_align":
        return get_fd004_state_encoder_v3_physics_C2_msdt_align_config()
    if experiment_name == "fd004_state_encoder_v3_physics_C3_msdt_rulonly":
        return get_fd004_state_encoder_v3_physics_C3_msdt_rulonly_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v1":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v1_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v2":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v2_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3b":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3b_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_tuned":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3c_mlp_two_phase_tuned_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v3e_smooth":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v3e_smooth_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v4_hi_cal":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v4_hi_cal_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_uncertainty":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_uncertainty_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_tuned_p50mse":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_tuned_p50mse_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_hi_cal_tuned_p50mse":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_quantiles_hi_cal_tuned_p50mse_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_censoring_aware":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_censoring_aware_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_censoring_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_quantiles_risk":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_quantiles_risk_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau90_w1":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau90_w1_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w2":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w2_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low10":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low10_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau99_w5_low20_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_harmwin":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20_harmonized_windows_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low20":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low20_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w5_low20":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w5_low20_config()
    if experiment_name == "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w8_low20":
        return get_fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w8_low20_config()
    if experiment_name == "fd004_decoder_v1_from_encoder_v3d":
        return get_fd004_decoder_v1_from_encoder_v3d_config()
    if experiment_name == "fd004_decoder_v1_from_encoder_v3e":
        return get_fd004_decoder_v1_from_encoder_v3e_config()
    if experiment_name == "fd004_decoder_v2_from_encoder_v3d":
        return get_fd004_decoder_v2_from_encoder_v3d_config()
    if experiment_name == "fd004_decoder_v3_from_encoder_v3d":
        return get_fd004_decoder_v3_from_encoder_v3d_config()
    if experiment_name == "fd004_decoder_v3_uncertainty_from_encoder_v3d":
        return get_fd004_decoder_v3_uncertainty_from_encoder_v3d_config()
    if experiment_name == "fd004_state_encoder_v3_damage_msdt_v1":
        return get_fd004_state_encoder_v3_damage_msdt_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_v1":
        return get_fd004_transformer_latent_worldmodel_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_msfreeze_v1":
        return get_fd004_transformer_latent_worldmodel_msfreeze_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_dynamic_v1":
        return get_fd004_transformer_latent_worldmodel_dynamic_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_dynamic_freeze_v1":
        return get_fd004_transformer_latent_worldmodel_dynamic_freeze_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_dynamic_delta_v2":
        return get_fd004_transformer_latent_worldmodel_dynamic_delta_v2_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659":
        return get_fd004_transformer_latent_worldmodel_dynamic_v1_from_encoder_v5_659_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1":
        return get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_latew10":
        return get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_latew10_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_infwin":
        return get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_lossbalance_v1_infwin_config()
    if experiment_name == "fd004_wm_v1_infwin_wiringcheck_k0":
        return get_fd004_wm_v1_infwin_wiringcheck_k0_config()
    if experiment_name == "fd004_wm_v1_infwin_capweight_k1":
        return get_fd004_wm_v1_infwin_capweight_k1_config()
    if experiment_name == "fd004_wm_v1_infwin_capmask_k2":
        return get_fd004_wm_v1_infwin_capmask_k2_config()
    if experiment_name == "fd004_wm_v1_p0_softcap_k3":
        return get_fd004_wm_v1_p0_softcap_k3_config()
    if experiment_name == "fd004_wm_v1_p0_softcap_k3_hm_pad":
        return get_fd004_wm_v1_p0_softcap_k3_hm_pad_config()
    if experiment_name == "fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_rulonly_v1":
        return get_fd004_transformer_latent_worldmodel_v1_from_encoder_v5_659_rulonly_v1_config()
    # Check for world model phase 5 v3 experiments first
    if "world" in experiment_name and "phase5" in experiment_name and "v3" in experiment_name:
        for dataset in ["FD001", "FD002", "FD003", "FD004"]:
            if dataset.lower() in experiment_name:
                # Start from the default Phase 5 V3 config for this dataset
                cfg = get_world_model_phase5_universal_v3_residual_config(dataset)

                # Allow simple hyper-parameter sweeps to be encoded in the name,
                # e.g. "..._h30_w2_universal_v3_residual" or "..._h20_w1p5_..."
                wm_params = cfg.get("world_model_params", {})

                # Parse horizon: look for "_h<integer>"
                m_h = re.search(r"_h(?P<h>\d+)", experiment_name)
                if m_h:
                    try:
                        h_val = int(m_h.group("h"))
                        wm_params["horizon"] = h_val
                    except ValueError:
                        pass

                # Parse tail weight: look for "_w<value>", where value may be "3.0" or "1p5"
                m_w = re.search(r"_w(?P<w>[0-9]+(?:p[0-9]+)?|\d+(?:\.\d+)?)", experiment_name)
                if m_w:
                    raw = m_w.group("w")
                    try:
                        if "p" in raw:
                            # e.g. "1p5" -> 1.5
                            int_part, frac_part = raw.split("p", 1)
                            w_val = float(f"{int_part}.{frac_part}")
                        else:
                            w_val = float(raw)
                        wm_params["eol_tail_weight"] = w_val
                    except ValueError:
                        pass

                # Parse HI→EOL consistency coupling weight (WorldModel-only, optional):
                # - If name contains "_eolhi" => enable with default 0.1
                # - Or "_eolhi0p1" / "_eolhi0.1" to specify weight explicitly
                if "_eolhi" in experiment_name:
                    m_eh = re.search(r"_eolhi(?P<v>[0-9]+(?:p[0-9]+)?|\d+(?:\.\d+)?)?", experiment_name)
                    w_default = 0.1
                    if m_eh:
                        raw = m_eh.group("v")
                        if raw:
                            try:
                                if "p" in raw:
                                    ip, fp = raw.split("p", 1)
                                    w_default = float(f"{ip}.{fp}")
                                else:
                                    w_default = float(raw)
                            except ValueError:
                                w_default = 0.1
                    wm_params["w_eol_hi"] = float(w_default)

                cfg["world_model_params"] = wm_params
                # Ensure the experiment_name in the config matches the requested name
                cfg["experiment_name"] = experiment_name
                return cfg
    
    # Check for world model phase 4 residual experiments
    if "world" in experiment_name and "phase4" in experiment_name and "residual" in experiment_name:
        for dataset in ["FD002", "FD004"]:
            if dataset.lower() in experiment_name:
                # Determine variant from experiment name
                if "eol_heavy" in experiment_name and "h10" in experiment_name:
                    variant = "h10_eol_heavy"
                elif "eol_heavy" in experiment_name:
                    variant = "eol_heavy"
                else:
                    variant = "default"
                return get_world_model_phase4_residual_config(dataset, variant=variant)
    
    # Check for phase 4 residual experiments
    if "phase4" in experiment_name and "residual" in experiment_name:
        for dataset in ["FD001", "FD002", "FD003", "FD004"]:
            if dataset.lower() in experiment_name:
                return get_phase4_residual_config(dataset)
    
    # Try to extract dataset from name
    for dataset in ["FD001", "FD002", "FD003", "FD004"]:
        if dataset.lower() in experiment_name:
            # Dataset-specific Transformer-Encoder runs (with/without residuals/phys_v2/ms_dt)
            if "transformer_encoder" in experiment_name:
                # Multi-Scale + Digital-Twin variants (ms_dt)
                # NEW: explicit ms_dt_v2 routing (currently only FD004 defined)
                if "ms_dt_v2" in experiment_name:
                    if dataset == "FD004":
                        return get_fd004_transformer_encoder_ms_dt_v2_config()
                if "ms_dt_v1" in experiment_name:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_ms_dt_v1_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_ms_dt_v1_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_ms_dt_v1_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_ms_dt_v1_config()
                # Physics-informed variants (continuous condition vector + digital twin)
                if "phys_v4" in experiment_name:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_phys_v4_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_phys_v4_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_phys_v4_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_phys_v4_config()
                if "phys_v3" in experiment_name:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_phys_v3_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_phys_v3_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_phys_v3_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_phys_v3_config()
                if "phys_v2" in experiment_name:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_phys_v2_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_phys_v2_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_phys_v2_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_phys_v2_config()
                # Residual variants
                if "resid" in experiment_name or "residual" in experiment_name:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_resid_v1_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_resid_v1_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_resid_v1_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_resid_v1_config()
                # Plain transformer_encoder_v1 variants
                else:
                    if dataset == "FD001":
                        return get_fd001_transformer_encoder_v1_config()
                    elif dataset == "FD002":
                        return get_fd002_transformer_encoder_v1_config()
                    elif dataset == "FD003":
                        return get_fd003_transformer_encoder_v1_config()
                    else:  # FD004
                        return get_fd004_transformer_encoder_v1_config()

            if "phase3" in experiment_name and "universal" in experiment_name:
                if "v2" in experiment_name or "universal_v2" in experiment_name:
                    return get_universal_v2_config(dataset)
                elif "tuned" in experiment_name:
                    return get_universal_v1_tuned_config(dataset)
                else:
                    return get_universal_v1_config(dataset)
            elif "lstm" in experiment_name and "baseline" in experiment_name:
                return get_lstm_baseline_config(dataset)
            elif "transformer" in experiment_name and "baseline" in experiment_name:
                return get_transformer_baseline_config(dataset)
    
    # Check for FD004 sweep experiments and Transformer+Attention/Encoder
    if "fd004" in experiment_name:
        if "transformer_encoder_phys_v3" in experiment_name:
            return get_fd004_transformer_encoder_phys_v3_config()
        if "transformer_encoder_phys_v2" in experiment_name:
            return get_fd004_transformer_encoder_phys_v2_config()
        if "transformer_encoder_resid" in experiment_name:
            return get_fd004_transformer_encoder_resid_v1_config()
        if "transformer_encoder" in experiment_name:
            return get_fd004_transformer_encoder_v1_config()
        if "transformer_attention" in experiment_name:
            return get_fd004_transformer_attention_v1_config()
        if "universal_v2_ms_cnn_d96" in experiment_name or "d96" in experiment_name:
            if "residual" not in experiment_name:  # Don't override phase 4
                return get_fd004_universal_v2_d96()
        elif "universal_v2_ms_cnn_strongmono" in experiment_name or "strongmono" in experiment_name:
            return get_fd004_universal_v2_strongmono()
        elif "hi_strong" in experiment_name:
            return get_fd004_transformer_hi_strong()
        elif "hi_condcalib" in experiment_name or "condcalib" in experiment_name:
            return get_fd004_transformer_hi_condcalib()
        elif "small_regularized" in experiment_name or "small" in experiment_name:
            return get_fd004_transformer_small_regularized()
    
    raise ValueError(f"Could not find experiment config for: {experiment_name}")

