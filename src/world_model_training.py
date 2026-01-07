import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for training routines. Please install torch."
    ) from exc

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for preprocessing. Please install scikit-learn."
    ) from exc

from .config import (
    CMAPSS_DATASETS,
    MAX_RUL,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    GLOBAL_FEATURE_COLS,
    GLOBAL_DROP_COLS 
)

from .data_loading import load_cmapps_subset, get_feature_drop_cols, load_cmapps_global
from .additional_features import create_physical_features
from .model import LSTMRULPredictor
from .loss import rul_asymmetric_weighted_loss
from .uncertainty import mc_dropout_predict
from .models.lstm_rul_mcdo import LSTMRULPredictorMCDropout
from .models.world_model import WorldModelEncoderDecoder, WorldModelEncoderDecoderMultiTask, WorldModelEncoderDecoderUniversalV2
from .training import build_eol_sequences_from_df
from src.metrics import (
    compute_eol_errors_and_nasa,
    compute_last_per_unit_metrics,
    compute_all_samples_metrics,
)
from src.models.eol_regressor import EOLRegressor
from src.models.tail_lstm import TailLSTMRegressor, TailLSTMConfig


# -------------------------------------------------------------------
# Helper: Sequence building
# -------------------------------------------------------------------

def build_seq2seq_samples_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    past_len: int = 30,
    horizon: int = 20,
):
    """
    Build (X_past, Y_future) samples for a seq2seq world model.

    - df should be a single-FD, single-UnitNumber DataFrame or already grouped.
    - feature_cols = input features (sensors + settings + physics).
    - target_cols  = target features to predict (e.g. ['RUL'] or subset of sensors).
    """
    X_list, Y_list = [], []

    # ensure numeric and float32
    values_in = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    values_out = df[target_cols].to_numpy(dtype=np.float32, copy=True)
    T = len(df)

    for t_past_end in range(past_len - 1, T - horizon):
        t_past_start = t_past_end + 1 - past_len
        t_future_start = t_past_end + 1
        t_future_end = t_future_start + horizon

        X_past = values_in[t_past_start : t_past_end + 1]     # (L_past, F)
        Y_future = values_out[t_future_start : t_future_end]  # (H, F_out)

        X_list.append(X_past)
        Y_list.append(Y_future)

    if not X_list:
        # wichtig: leerer Rückgabefall
        return np.empty((0, past_len, len(feature_cols)), dtype=np.float32), \
               np.empty((0, horizon, len(target_cols)), dtype=np.float32)

    return np.stack(X_list), np.stack(Y_list)


# ===================================================================
# Phase 4: UniversalEncoderV2-based World Model with Residual Features
# ===================================================================

from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class CycleBranchConfig:
    """Configuration for the auxiliary physics cycle branch (Mode 1 Factorized).
    
    This enables a differentiable 0D thermodynamic cycle model that predicts
    sensor outputs from operating conditions and degradation parameters.
    
    Mode 1 Factorized: η_eff = η_nom(ops) × m(t)
    where m(t) are slow-varying degradation modifiers.
    """
    # Master switch
    enable: bool = False
    
    # Semantic target sensor names (resolved via cmapss_sensors.yaml)
    targets: List[str] = field(default_factory=lambda: ["T24", "T30", "P30", "T50"])
    optional_witnesses: List[str] = field(default_factory=lambda: ["Nf", "Nc", "Ps30"])  # optional
    
    # --------------------------------------------------
    # Loss weights
    # --------------------------------------------------
    lambda_cycle: float = 0.1
    lambda_theta_smooth: float = 0.05
    lambda_theta_mono: float = 0.01
    lambda_power_balance: float = 0.0  # optional power balance penalty
    cycle_loss_type: Literal["mse", "huber"] = "huber"
    cycle_huber_beta: float = 0.1
    cycle_ramp_epochs: int = 10  # Curriculum: ramp lambda_cycle from 0 to full
    
    # --------------------------------------------------
    # Monotonicity settings (separate flags, both default OFF)
    # --------------------------------------------------
    mono_on_eta_mods: bool = False  # Apply mono to eta modifiers (m_fan..m_lpt)
    mono_on_dp_mod: bool = False    # Apply mono to dp modifier (m_dp_comb)
    mono_eps: float = 1e-4          # Tolerance before penalizing increases
    
    # --------------------------------------------------
    # Degradation modifier bounds (sigmoid-scaled)
    # --------------------------------------------------
    m_bounds_eta: tuple = (0.85, 1.00)  # [m_fan, m_lpc, m_hpc, m_hpt, m_lpt]
    m_bounds_dp: tuple = (0.90, 1.00)   # m_dp_comb
    
    # --------------------------------------------------
    # NominalHead configuration
    # --------------------------------------------------
    nominal_head_type: Literal["table", "mlp"] = "table"  # 'table' preferred for multi-cond
    nominal_head_hidden: int = 16
    eta_nom_bounds: tuple = (0.80, 0.99)
    
    # --------------------------------------------------
    # dp_nom constant (combustor pressure drop ratio)
    # --------------------------------------------------
    dp_nom_constant: float = 0.95
    
    # --------------------------------------------------
    # Pressure ratio handling (MVP: per-condition constants)
    # --------------------------------------------------
    pr_mode: Literal["per_cond", "head"] = "per_cond"
    pr_head_hidden: int = 16
    
    # --------------------------------------------------
    # Optional degradation warmup (freeze m=1 for k epochs)
    # --------------------------------------------------
    deg_warmup_epochs: int = 0
    
    # --------------------------------------------------
    # ParamHeadTheta6 architecture
    # --------------------------------------------------
    param_head_hidden: int = 64
    param_head_num_layers: int = 2


@dataclass
class WorldModelTrainingConfig:
    """
    Configuration for World Model training with configurable loss weights and horizon.
    
    Attributes:
        forecast_horizon: Future horizon for trajectory prediction (default: 20)
        traj_loss_weight: Weight for trajectory loss (default: 1.0)
        eol_loss_weight: Weight for EOL loss (default: 1.0)
        hi_loss_weight: Weight for Health Index loss (default: 0.0, only for v3)
        mono_late_weight: Weight for late monotonicity loss (default: 0.0, only for v3)
        mono_global_weight: Weight for global trend loss (default: 0.0, only for v3)
        traj_step_weighting: Optional step weighting for trajectory loss
            - None: uniform weighting
            - "late_heavy": linear weighting increasing with time step
        past_len: Past window length (default: 30)
        max_rul: Maximum RUL value (default: 125)
        use_condition_wise_scaling: Whether to use condition-wise scaling (default: True)
        two_phase_training: Whether to use two-phase training (default: False)
        phase2_eol_weight: EOL loss weight for phase 2 (if two_phase_training=True)
    """
    forecast_horizon: int = 20
    traj_loss_weight: float = 1.0
    eol_loss_weight: float = 1.0
    hi_loss_weight: float = 0.0  # For v3: Health Index loss weight
    mono_late_weight: float = 0.0  # For v3: Late monotonicity weight
    mono_global_weight: float = 0.0  # For v3: Global trend weight
    traj_step_weighting: Optional[Literal["late_heavy"]] = None
    past_len: int = 30
    max_rul: int = 125
    use_condition_wise_scaling: bool = True
    two_phase_training: bool = False
    phase2_eol_weight: float = 10.0  # Only used if two_phase_training=True

    # World Model v3 extensions (HI fusion + tail-weighted EOL)
    use_hi_in_eol: bool = False           # if True, fuse HI into EOL head
    use_hi_slope_in_eol: bool = False     # if True, also include local HI slope
    eol_tail_rul_threshold: Optional[float] = None  # RUL < threshold => tail region
    eol_tail_weight: float = 1.0          # multiplicative weight for tail samples

    # Stage-1: 3-phase curriculum schedule (A/B/C) to improve HI/RUL dynamics
    # - Phase A: dynamics warmup (EOL weight = 0)
    # - Phase B: joint training (EOL weight ramps to full)
    # - Phase C: EOL focus (EOL weight = full; keep dynamics losses on as stabilizers)
    three_phase_schedule: bool = False
    phase_a_frac: float = 0.2
    # Backward-compatible naming:
    # - phase_b_end_frac is the end of the ramp-in phase (default 0.8)
    # - phase_b_frac is an alias (preferred name in newer prompts)
    phase_b_end_frac: float = 0.8
    phase_b_frac: Optional[float] = None
    schedule_type: Literal["linear", "cosine"] = "linear"
    eol_w_max: float = 1.0

    # Stabilization knobs for EOL ramp-in (default OFF for backward compatibility)
    normalize_eol: bool = False
    # eol_scale can be: "rul_cap" | "max_cycle" | numeric (float)
    eol_scale: object = "rul_cap"
    eol_loss_type: Literal["huber", "mse", "mae"] = "mse"
    eol_huber_beta: float = 0.1
    clip_grad_norm: Optional[float] = None
    freeze_encoder_epochs_after_eol_on: int = 0

    # --------------------------------------------------
    # Selection / alignment knobs (default OFF for backward compatibility)
    # --------------------------------------------------
    # If True, only allow "best checkpoint" updates once EOL is active
    # (prevents selecting a pre-EOL checkpoint in 3-phase schedules).
    select_best_after_eol_active: bool = False
    # EOL becomes "active" once eol_mult >= this threshold.
    eol_active_min_mult: float = 0.01
    # Which metric to use for best checkpoint selection.
    # - "val_total" (default): uses epoch_val_loss (current behavior)
    # - "val_eol_weighted": uses epoch_val_eol_weighted
    # - "val_eol": uses epoch_val_eol_loss
    best_metric: str = "val_total"

    # Align scalar EOL target definition (default preserves current behavior).
    # - "future0": use Y_future[:, 0] (current behavior in v3 training loop)
    # - "current": approximate current-step RUL from future0 (+1 with capping)
    # - "future_last": use Y_future[:, -1]
    eol_target_mode: str = "future0"
    # If True, clamp seq2seq RUL targets to [0, max_rul] inside training/eval.
    cap_rul_targets_to_max_rul: bool = False
    # If True, clip eval y_true to [0, max_rul] (fixes mismatch when loader returns raw RUL).
    eval_clip_y_true_to_max_rul: bool = False
    # Optional: initialize EOL head bias close to mean target for faster stable ramp-in.
    init_eol_bias_to_target_mean: bool = False

    # --------------------------------------------------
    # Horizon target building (padded/clamped) + optional masking
    # --------------------------------------------------
    # If True, build horizon targets with padding/clamping at the end of life
    # so the last windows (near RUL=0) are included even when no full future
    # horizon exists in the raw dataframe.
    use_padded_horizon_targets: bool = False
    # Clamp minimum for horizon targets (usually 0.0).
    target_clamp_min: float = 0.0
    # If True, return and use a horizon mask (1=valid, 0=padded) for losses.
    use_horizon_mask: bool = False

    # --------------------------------------------------
    # Decoder type selection (World Model v3)
    # --------------------------------------------------
    # Decoder architecture: "lstm" (default), "tf_ar" (Transformer AR self-attn), "tf_ar_xattn" (Transformer AR cross-attn)
    decoder_type: str = "lstm"

    # --------------------------------------------------
    # Transformer World Model V1 (Dynamic Latent WM A+) knobs
    # Stored on the shared config for convenience; default OFF.
    # --------------------------------------------------
    # Stage A/B training for TransformerWorldModelV1 encoder
    freeze_encoder_epochs: int = 0
    unfreeze_encoder_layers: int = 0
    encoder_lr_mult: float = 0.1
    # EOL fusion / latent transformer decoder knobs (used by TransformerWorldModelV1)
    use_eol_fusion: bool = False
    eol_fusion_mode: Literal["token", "feature"] = "token"
    predict_latent: bool = False
    latent_decoder_type: Literal["gru", "transformer"] = "gru"

    # --------------------------------------------------
    # WM-V1: Cap/plateau reweighting (default OFF)
    # --------------------------------------------------
    # Down-weight samples whose entire future RUL horizon is capped at 1.0 (normalized).
    cap_reweight_enable: bool = False
    cap_reweight_eps: float = 1e-6
    cap_reweight_weight: float = 0.05
    cap_reweight_apply_to: Literal["rul", "hi", "both"] = "rul"

    # --------------------------------------------------
    # WM-V1: Cap-aware loss masking (default ON)
    # --------------------------------------------------
    # Mask out time steps where the target is fully capped at 1.0 (normalized).
    # This prevents plateau windows from dominating the regression loss.
    cap_mask_enable: bool = True
    cap_mask_eps: float = 1e-6
    # Apply to: ["rul"] or ["hi"] or ["rul","hi"]
    cap_mask_apply_to: list[str] = field(default_factory=lambda: ["rul"])

    # --------------------------------------------------
    # WM-V1: Soft cap weighting (ADR-0010, replaces binary masking)
    # --------------------------------------------------
    # If True, use soft per-timestep weights based on distance from cap.
    # Replaces binary cap_mask when enabled for RUL future loss ONLY.
    soft_cap_enable: bool = False  # Default OFF for backwards compatibility
    soft_cap_power: float = 0.5  # Weight = distance^power (0.5=sqrt gives gradual ramp)
    soft_cap_floor: float = 0.05  # Minimum weight for capped timesteps (never zero)

    latent_decoder_num_layers: int = 2
    latent_decoder_nhead: int = 4
    eol_scalar_loss_weight: float = 0.0

    # --------------------------------------------------
    # Transformer World Model V1: RUL trajectory stabilization losses
    # (default OFF for backwards compatibility)
    # --------------------------------------------------
    # If set, use full-horizon MSE trajectory loss with this weight.
    # If None, legacy behavior uses `rul_future_loss_weight` (L1).
    rul_traj_weight: Optional[float] = None
    # If True, apply a late-horizon ramp weight to the MSE loss.
    rul_traj_late_ramp: bool = False
    # Penalize increases across the predicted future horizon (monotone decreasing).
    rul_mono_future_weight: float = 0.0
    # Penalize saturation near 1.0 when the target is clearly below 1.
    rul_saturation_weight: float = 0.0
    rul_saturation_margin: float = 0.05

    # If True, construct a physics-consistent linear-decay RUL sequence from pred_rul0.
    # Default OFF for backwards compatibility.
    rul_linear_decay: bool = False

    # Cap-threshold for masking non-informative (fully capped) RUL targets in WM-V1.
    # If true_rul >= rul_cap_threshold, we treat it as "capped/plateau" and exclude from traj loss.
    rul_cap_threshold: float = 0.999999

    # --------------------------------------------------
    # Transformer World Model V1: Late-window weighting (anti "always healthy" collapse)
    # (default OFF for backwards compatibility)
    # --------------------------------------------------
    # If enabled: upweight samples whose FUTURE horizon contains failure (RUL ~ 0).
    # This is applied per-sample after reducing the horizon loss.
    late_weight_enable: bool = False
    late_weight_factor: float = 5.0
    # - "future_has_zero": true if any future step is <= eps (in normalized space)
    # - "future_min_below_eps": same as above but explicitly uses min over horizon (identical for eps>=0)
    late_weight_mode: Literal["future_has_zero", "future_min_below_eps"] = "future_has_zero"
    # Threshold used for normalized detection (RUL in [0,1]).
    late_weight_eps_norm: float = 1e-6
    # Optional: also apply the same per-sample late-weight to HI future loss (default False).
    late_weight_apply_hi: bool = False

    # --------------------------------------------------
    # Transformer World Model V1: Informative window sampling / filtering
    # (default OFF for backwards compatibility)
    # --------------------------------------------------
    # Motivation: many windows have fully capped future RUL targets (all ones in normalized space),
    # which encourages a trivial "always healthy" solution. This sampler keeps all informative windows
    # and only a fraction of non-informative windows.
    informative_sampling_enable: bool = False
    # Modes:
    # - "future_min_lt_cap": informative if ANY timestep is below cap (legacy)
    # - "future_has_zero": informative if future contains RUL=0
    # - "uncapped_frac": informative if >= X% of future timesteps are uncapped (ADR-0010)
    informative_sampling_mode: Literal["future_min_lt_cap", "future_has_zero", "uncapped_frac"] = "future_min_lt_cap"
    informative_eps_norm: float = 1e-6
    # Probability to keep a non-informative window (mixing). 0.0 => strict filter.
    keep_prob_noninformative: float = 0.1
    log_informative_stats: bool = True
    # Threshold for "uncapped_frac" mode: fraction of future timesteps that must be uncapped (ADR-0010)
    informative_uncapped_frac_threshold: float = 0.3

    # --------------------------------------------------
    # WM-V1: "wiring proof" debug instrumentation
    # (default OFF; prints only a small number of batches/epochs)
    # --------------------------------------------------
    debug_wiring_enable: bool = False
    debug_wiring_batches: int = 1
    debug_wiring_epochs: int = 1
    debug_wiring_save_json: bool = True

    # --- WM-V1 RUL training hardening (defaults OFF) ---
    # If set: ignore (mask out) targets with true_rul_cycles >= this.
    rul_train_max_cycles: Optional[float] = None
    # If True: supervise only RUL at t=0 (horizon start) or selected points.
    rul_r0_only: bool = False
    # Optional extra indices (e.g. [0, 15, 29]); if None and r0_only -> [0]
    rul_r0_points: Optional[list[int]] = None
    # 0 => off; >0 enables per-sample weighting (late-life emphasis)
    rul_sample_weight_power: float = 0.0
    # clamp weights
    rul_sample_weight_min: float = 0.2
    rul_sample_weight_max: float = 3.0

    # Stage-1: additional HI shape losses (default off; enable via experiment config)
    hi_early_slope_weight: float = 0.0
    hi_early_slope_epsilon: float = 1e-3
    hi_early_slope_rul_threshold: Optional[float] = None
    hi_curvature_weight: float = 0.0
    hi_curvature_abs: bool = True

    # Stage-1/2: optional WorldModel coupling (no RUL_seq): HI → EOL consistency
    # Default MUST be off to avoid silent behavior changes.
    w_eol_hi: float = 0.0
    eol_hi_threshold: float = 0.2
    eol_hi_temperature: float = 0.05
    eol_hi_p_min: float = 0.2

    # --------------------------------------------------
    # B2.2: Leakage-Free HI-Dynamics Self-Supervision
    # --------------------------------------------------
    use_hi_dynamics: bool = False
    w_hi_dyn: float = 0.5
    w_hi_dyn_mono: float = 0.03
    w_hi_dyn_smooth: float = 0.02
    hi_dyn_huber_beta: float = 0.1

    # --------------------------------------------------
    # B-Track: Differentiable Cycle Branch (Option B)
    # --------------------------------------------------
    cycle_branch: CycleBranchConfig = field(default_factory=CycleBranchConfig)


def compute_trajectory_step_weights(
    horizon: int,
    weighting: Optional[Literal["late_heavy"]] = None,
) -> torch.Tensor:
    """
    Compute step weights for trajectory loss.
    
    Args:
        horizon: Forecast horizon length
        weighting: Weighting scheme ("late_heavy" for linear increase, None for uniform)
    
    Returns:
        Weight tensor of shape (horizon,) normalized to sum to horizon
    """
    if weighting is None:
        # Uniform weighting
        return torch.ones(horizon, dtype=torch.float32)
    elif weighting == "late_heavy":
        # Linear weighting: w[t] = t + 1, normalized so sum = horizon
        weights = torch.arange(1, horizon + 1, dtype=torch.float32)
        # Normalize so sum = horizon (keeps loss scale similar)
        weights = weights / weights.sum() * horizon
        return weights
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")

def build_world_model_dataset_with_cond_ids(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    past_len: int = 30,
    horizon: int = 20,
    unit_col: str = "UnitNumber",
    cond_col: str = "ConditionID",
    *,
    use_padded_horizon_targets: bool = False,
    target_clamp_min: float = 0.0,
    return_horizon_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build world model dataset with condition IDs.
    
    Returns:
        X: (N, past_len, F) - Past sequences
        Y: (N, horizon, 1) - Future RUL trajectories
        cond_ids: (N,) - Condition IDs for each sample
    """
    from src.data.targets import build_rul_horizon_targets

    X_list, Y_list, cond_id_list = [], [], []
    M_list = []  # optional horizon mask (N, H, 1)

    # Prefer an uncapped source if available; fall back to the requested target_col.
    rul_source_col = "RUL_raw" if ("RUL_raw" in df.columns) else target_col

    for unit_id, df_unit in df.groupby(unit_col):
        # Get condition ID for this unit (should be constant per unit)
        cond_id = int(df_unit[cond_col].iloc[0])

        df_unit = df_unit.sort_values("TimeInCycles")

        if not use_padded_horizon_targets:
            # Old behavior: only windows with a full future horizon from the dataframe.
            X_np, Y_np = build_seq2seq_samples_from_df(
                df=df_unit,
                feature_cols=feature_cols,
                target_cols=[target_col],
                past_len=past_len,
                horizon=horizon,
            )
            if X_np.shape[0] == 0:
                continue
            X_list.append(X_np)
            Y_list.append(Y_np)
            cond_id_list.extend([cond_id] * X_np.shape[0])
            if return_horizon_mask:
                # Full horizon => mask is all ones
                m = np.ones((X_np.shape[0], horizon, 1), dtype=np.float32)
                M_list.append(m)
            continue

        # New behavior: include the last windows near EOL by deterministically
        # building horizon targets from the current RUL value.
        values_in = df_unit[feature_cols].to_numpy(dtype=np.float32, copy=True)
        rul_now = df_unit[rul_source_col].to_numpy(dtype=np.float32, copy=True).reshape(-1)
        T = len(df_unit)

        if T < past_len:
            continue

        X_u = []
        Y_u = []
        M_u = []
        # allow t_past_end to reach the last row (near EOL)
        for t_past_end in range(past_len - 1, T):
            t_past_start = t_past_end + 1 - past_len
            X_past = values_in[t_past_start : t_past_end + 1]  # (past_len, F)

            r0 = float(rul_now[t_past_end])
            y_h, m_h = build_rul_horizon_targets(
                np.array([r0], dtype=np.float32),
                horizon=horizon,
                clamp_min=float(target_clamp_min),
                return_mask=True,
            )
            # (1, H) -> (H, 1)
            y_h = y_h.reshape(horizon, 1)
            m_h = m_h.reshape(horizon, 1) if m_h is not None else np.ones((horizon, 1), dtype=np.float32)

            X_u.append(X_past)
            Y_u.append(y_h)
            M_u.append(m_h)

        if not X_u:
            continue

        X_np = np.stack(X_u, axis=0).astype(np.float32)
        Y_np = np.stack(Y_u, axis=0).astype(np.float32)
        M_np = np.stack(M_u, axis=0).astype(np.float32)  # (N_u, H, 1)

        X_list.append(X_np)
        Y_list.append(Y_np)
        cond_id_list.extend([cond_id] * X_np.shape[0])
        if return_horizon_mask:
            M_list.append(M_np)

    if not X_list:
        raise ValueError("No seq2seq samples could be built from the given DataFrame.")

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    cond_ids_all = np.array(cond_id_list, dtype=np.int64)
    M_all = np.concatenate(M_list, axis=0) if (return_horizon_mask and M_list) else None

    X = torch.tensor(X_all, dtype=torch.float32)
    Y = torch.tensor(Y_all, dtype=torch.float32)
    cond_ids = torch.tensor(cond_ids_all, dtype=torch.long)

    if return_horizon_mask and M_all is not None:
        M = torch.tensor(M_all, dtype=torch.float32)
        # Dataset-level logging (critical debug): padding fraction etc.
        try:
            rul0 = Y_all[:, 0, 0]
            pad_frac = 1.0 - float(M_all.mean())
            near_eol_frac = float(np.mean(rul0 <= float(horizon)))
            print(
                f"[targets] rul0: min={float(np.min(rul0)):.2f} mean={float(np.mean(rul0)):.2f} max={float(np.max(rul0)):.2f} "
                f"| pad_frac={pad_frac:.3f} | near_eol_frac(rul0<=H)={near_eol_frac:.3f} "
                f"| clamp_min={float(target_clamp_min):.2f} padded={use_padded_horizon_targets}"
            )
        except Exception:
            pass
        return X, Y, cond_ids, M

    # Logging (even if mask is not returned)
    try:
        rul0 = Y_all[:, 0, 0]
        print(
            f"[targets] rul0: min={float(np.min(rul0)):.2f} mean={float(np.mean(rul0)):.2f} max={float(np.max(rul0)):.2f} "
            f"| clamp_min={float(target_clamp_min):.2f} padded={use_padded_horizon_targets}"
        )
    except Exception:
        pass

    return X, Y, cond_ids


def train_world_model_universal_v2_residual(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    dataset_name: str,
    experiment_name: str,
    d_model: int = 96,
    num_layers: int = 3,
    nhead: int = 4,
    dim_feedforward: int = 384,
    dropout: float = 0.1,
    kernel_sizes: List[int] = None,
    seq_encoder_type: str = "transformer",
    decoder_num_layers: int = 2,
    batch_size: int = 256,
    num_epochs: int = 80,
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    patience: int = 10,
    engine_train_ratio: float = 0.8,
    random_seed: int = 42,
    world_model_config: Optional[WorldModelTrainingConfig] = None,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train UniversalEncoderV2-based world model with Phase 4 residual features.
    
    This function:
    - Uses Phase 4 residual feature pipeline (464 features)
    - Uses UniversalEncoderV2 as encoder
    - Handles condition-wise scaling
    - Supports configurable loss weights and horizon
    - Evaluates EOL metrics
    - Saves model, metrics, and diagnostics
    
    Args:
        df_train: Training DataFrame with features and RUL
        df_test: Test DataFrame with features
        y_test_true: True RUL at EOL for test engines
        feature_cols: List of feature column names
        dataset_name: Dataset name (e.g., "FD004")
        experiment_name: Experiment name
        d_model: Model dimension for UniversalEncoderV2
        num_layers: Number of encoder layers
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        kernel_sizes: CNN kernel sizes
        seq_encoder_type: "transformer" or "lstm"
        decoder_num_layers: Number of decoder LSTM layers
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        engine_train_ratio: Ratio of engines for training
        random_seed: Random seed
        world_model_config: WorldModelTrainingConfig with loss weights, horizon, etc.
            If None, uses defaults (horizon=20, traj_weight=1.0, eol_weight=1.0)
        results_dir: Directory to save results
        device: PyTorch device
    
    Returns:
        Dictionary with training results and metrics
    """
    import os
    import json
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if results_dir is None:
        results_dir = Path("results") / dataset_name.lower() / experiment_name
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 9]
    
    # Use provided config or create default
    if world_model_config is None:
        world_model_config = WorldModelTrainingConfig()
    
    past_len = world_model_config.past_len
    horizon = world_model_config.forecast_horizon
    max_rul = world_model_config.max_rul
    use_condition_wise_scaling = world_model_config.use_condition_wise_scaling
    
    print(f"\n{'='*80}")
    print(f"Training World Model: {experiment_name}")
    print(f"Dataset: {dataset_name}, Features: {len(feature_cols)}")
    print(f"  Horizon: {horizon}, Past len: {past_len}")
    print(f"  Loss weights: traj={world_model_config.traj_loss_weight:.2f}, eol={world_model_config.eol_loss_weight:.2f}")
    if world_model_config.traj_step_weighting:
        print(f"  Trajectory step weighting: {world_model_config.traj_step_weighting}")
    print(f"{'='*80}\n")
    
    # Compute trajectory step weights if needed
    traj_step_weights = compute_trajectory_step_weights(
        horizon=horizon,
        weighting=world_model_config.traj_step_weighting,
    ).to(device)
    
    # ===================================================================
    # 1. Engine-based train/val split (TRUNCATED-AWARE reporting needs per-unit splits)
    # ===================================================================
    print("[1] Creating engine-based train/val split...")
    unit_ids_all = sorted(df_train["UnitNumber"].unique().tolist())
    n_units_total = len(unit_ids_all)
    if n_units_total == 0:
        raise ValueError("No UnitNumber found in df_train.")

    g = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(n_units_total, generator=g).tolist()
    n_units_train = max(1, int(engine_train_ratio * n_units_total))
    train_units = set([unit_ids_all[i] for i in perm[:n_units_train]])
    val_units = set([unit_ids_all[i] for i in perm[n_units_train:]])
    if len(val_units) == 0:
        # Ensure non-empty val split
        moved = next(iter(train_units))
        train_units.remove(moved)
        val_units.add(moved)

    df_train_split_df = df_train[df_train["UnitNumber"].isin(train_units)].copy()
    df_val_split_df = df_train[df_train["UnitNumber"].isin(val_units)].copy()
    print(f"  Units total: {n_units_total} | train: {len(train_units)} | val: {len(val_units)}")

    # Determine number of conditions on the full training dataframe
    unique_conditions = np.unique(df_train["ConditionID"].to_numpy(dtype=int, copy=False))
    num_conditions = int(len(unique_conditions))
    print(f"  Found {num_conditions} unique conditions: {unique_conditions}")

    # ===================================================================
    # 2. Build sequences with condition IDs (train/val separately; avoids leakage)
    # ===================================================================
    print("\n[2] Building seq2seq sequences...")
    X_train_split, Y_train_split, cond_ids_train_split = build_world_model_dataset_with_cond_ids(
        df=df_train_split_df,
        feature_cols=feature_cols,
        target_col="RUL",
        past_len=past_len,
        horizon=horizon,
        unit_col="UnitNumber",
        cond_col="ConditionID",
    )
    X_val, Y_val, cond_ids_val = build_world_model_dataset_with_cond_ids(
        df=df_val_split_df,
        feature_cols=feature_cols,
        target_col="RUL",
        past_len=past_len,
        horizon=horizon,
        unit_col="UnitNumber",
        cond_col="ConditionID",
    )

    print(f"  Train sequences: {X_train_split.shape[0]}, Val sequences: {X_val.shape[0]}")
    print(f"  Input shape: {X_train_split.shape}, Target shape: {Y_train_split.shape}")
    
    # ===================================================================
    # 3. Condition-wise scaling
    # ===================================================================
    print("\n[3] Applying condition-wise feature scaling...")
    scaler_dict = {}
    X_train_scaled = X_train_split.clone()
    X_val_scaled = X_val.clone()
    
    if use_condition_wise_scaling:
        X_train_np = X_train_split.numpy()
        X_val_np = X_val.numpy()
        cond_ids_train_np = cond_ids_train_split.numpy()
        cond_ids_val_np = cond_ids_val.numpy()
        
        for cond in unique_conditions:
            cond = int(cond)
            train_mask = (cond_ids_train_np == cond)
            val_mask = (cond_ids_val_np == cond)
            
            scaler = StandardScaler()
            # Fit on train data for this condition
            X_train_cond_flat = X_train_np[train_mask].reshape(-1, X_train_np.shape[-1])
            scaler.fit(X_train_cond_flat)
            scaler_dict[cond] = scaler
            
            # Transform train
            if train_mask.any():
                X_train_scaled[train_mask] = torch.tensor(
                    scaler.transform(X_train_cond_flat).reshape(-1, past_len, len(feature_cols)),
                    dtype=torch.float32
                )
            
            # Transform val
            if val_mask.any():
                X_val_cond_flat = X_val_np[val_mask].reshape(-1, X_val_np.shape[-1])
                X_val_scaled[val_mask] = torch.tensor(
                    scaler.transform(X_val_cond_flat).reshape(-1, past_len, len(feature_cols)),
                    dtype=torch.float32
                )
        
        print(f"  Fitted {len(scaler_dict)} condition-wise scalers")
    else:
        # Global scaling
        scaler = StandardScaler()
        X_train_flat = X_train_split.numpy().reshape(-1, len(feature_cols))
        scaler.fit(X_train_flat)
        scaler_dict[0] = scaler
        
        X_train_scaled = torch.tensor(
            scaler.transform(X_train_flat).reshape(-1, past_len, len(feature_cols)),
            dtype=torch.float32
        )
        X_val_flat = X_val.numpy().reshape(-1, len(feature_cols))
        X_val_scaled = torch.tensor(
            scaler.transform(X_val_flat).reshape(-1, past_len, len(feature_cols)),
            dtype=torch.float32
        )
    
    # Save scaler
    import pickle
    scaler_path = results_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_dict, f)
    print(f"  Saved scaler to {scaler_path}")
    
    # ===================================================================
    # 4. Create dataloaders
    # ===================================================================
    print("\n[4] Creating dataloaders...")
    train_dataset = TensorDataset(X_train_scaled, Y_train_split, cond_ids_train_split)
    val_dataset = TensorDataset(X_val_scaled, Y_val, cond_ids_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ===================================================================
    # 5. Initialize model
    # ===================================================================
    print("\n[5] Initializing UniversalEncoderV2-based world model...")
    model = WorldModelEncoderDecoderUniversalV2(
        input_size=len(feature_cols),
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=1,
        num_conditions=num_conditions if num_conditions > 1 else None,
        cond_emb_dim=4,
        kernel_sizes=kernel_sizes,
        seq_encoder_type=seq_encoder_type,
        use_layer_norm=True,
        max_seq_len=300,
        decoder_num_layers=decoder_num_layers,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Encoder: UniversalEncoderV2 (d_model={d_model}, num_layers={num_layers})")
    print(f"  Decoder: LSTM (num_layers={decoder_num_layers})")
    
    # ===================================================================
    # 6. Training loop
    # ===================================================================
    print("\n[6] Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_traj_loss": [],
        "val_eol_loss": [],
        "val_traj_weighted": [],
        "val_eol_weighted": [],
    }
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = results_dir / "world_model_best.pt"
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        n_train_samples = 0
        
        for X_batch, Y_batch, cond_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            optimizer.zero_grad()
            
            traj_pred, eol_pred = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                teacher_forcing_ratio=0.5,
                horizon=horizon,
                cond_ids=cond_batch if num_conditions > 1 else None,
            )
            
            # Loss: trajectory + EOL (with configurable weights and step weighting)
            # Trajectory loss with optional step weighting
            if world_model_config.traj_step_weighting:
                # Weighted MSE: (pred - target)^2 * step_weights, then mean
                squared_errors = (traj_pred - Y_batch) ** 2  # (B, H, 1)
                weighted_errors = squared_errors.squeeze(-1) * traj_step_weights.unsqueeze(0)  # (B, H)
                loss_traj = weighted_errors.mean()
            else:
                loss_traj = mse_loss(traj_pred, Y_batch)
            
            # EOL loss
            target_eol = Y_batch[:, 0, 0]
            eol_pred_flat = eol_pred.squeeze(-1)
            loss_eol = mse_loss(eol_pred_flat, target_eol)
            
            # Weighted total loss
            weighted_traj = world_model_config.traj_loss_weight * loss_traj
            weighted_eol = world_model_config.eol_loss_weight * loss_eol
            loss = weighted_traj + weighted_eol
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
            n_train_samples += X_batch.size(0)
        
        epoch_train_loss = running_train_loss / n_train_samples
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_traj_loss = 0.0
        running_val_eol_loss = 0.0
        running_val_traj_weighted = 0.0
        running_val_eol_weighted = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for X_batch, Y_batch, cond_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                cond_batch = cond_batch.to(device)
                
                traj_pred, eol_pred = model(
                    encoder_inputs=X_batch,
                    decoder_targets=Y_batch,
                    teacher_forcing_ratio=0.0,
                    horizon=horizon,
                    cond_ids=cond_batch if num_conditions > 1 else None,
                )
                
                # Trajectory loss with optional step weighting
                if world_model_config.traj_step_weighting:
                    squared_errors = (traj_pred - Y_batch) ** 2  # (B, H, 1)
                    weighted_errors = squared_errors.squeeze(-1) * traj_step_weights.unsqueeze(0)  # (B, H)
                    loss_traj = weighted_errors.mean()
                else:
                    loss_traj = mse_loss(traj_pred, Y_batch)
                
                # EOL loss
                target_eol = Y_batch[:, 0, 0]
                eol_pred_flat = eol_pred.squeeze(-1)
                loss_eol = mse_loss(eol_pred_flat, target_eol)
                
                # Weighted total loss
                weighted_traj = world_model_config.traj_loss_weight * loss_traj
                weighted_eol = world_model_config.eol_loss_weight * loss_eol
                loss = weighted_traj + weighted_eol
                
                running_val_loss += loss.item() * X_batch.size(0)
                running_val_traj_loss += loss_traj.item() * X_batch.size(0)
                running_val_eol_loss += loss_eol.item() * X_batch.size(0)
                running_val_traj_weighted += weighted_traj.item() * X_batch.size(0)
                running_val_eol_weighted += weighted_eol.item() * X_batch.size(0)
                n_val_samples += X_batch.size(0)
        
        epoch_val_loss = running_val_loss / n_val_samples
        epoch_val_traj_loss = running_val_traj_loss / n_val_samples
        epoch_val_eol_loss = running_val_eol_loss / n_val_samples
        epoch_val_traj_weighted = running_val_traj_weighted / n_val_samples
        epoch_val_eol_weighted = running_val_eol_weighted / n_val_samples
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_traj_loss"].append(epoch_val_traj_loss)
        history["val_eol_loss"].append(epoch_val_eol_loss)
        history["val_traj_weighted"].append(epoch_val_traj_weighted)
        history["val_eol_weighted"].append(epoch_val_eol_weighted)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"train: {epoch_train_loss:.4f}, val: {epoch_val_loss:.4f}, "
            f"val_traj: {epoch_val_traj_loss:.4f} (weighted: {epoch_val_traj_weighted:.4f}), "
            f"val_eol: {epoch_val_eol_loss:.4f} (weighted: {epoch_val_eol_weighted:.4f})"
        )
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": epoch_val_loss,
                "input_dim": len(feature_cols),
                "d_model": d_model,
                "num_conditions": num_conditions,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"\n[7] Best model loaded from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")

    # ===================================================================
    # 7a. Evaluate on train/val splits (LAST + ALL; truncated-aware)
    # ===================================================================
    def _evaluate_last_all_on_dataframe(
        df_split: pd.DataFrame,
        *,
        split_name: str,
    ) -> Dict[str, Any]:
        y_col = "RUL_raw" if ("RUL_raw" in df_split.columns) else "RUL"
        if y_col not in df_split.columns:
            raise KeyError(f"Expected '{y_col}' in df_split for split={split_name}")

        X_list: list[np.ndarray] = []
        y_true_list: list[float] = []
        unit_ids_list: list[int] = []
        cond_ids_list: list[int] = []

        for unit_id, df_unit in df_split.groupby("UnitNumber"):
            unit_id = int(unit_id)
            df_u = df_unit.sort_values("TimeInCycles").reset_index(drop=True)

            feats = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
            y_now = df_u[y_col].to_numpy(dtype=np.float32, copy=True).reshape(-1)
            y_now = np.maximum(y_now, 0.0)
            if max_rul is not None:
                y_now = np.minimum(y_now, float(max_rul))

            # Pad short units to allow at least one LAST window
            if feats.shape[0] < past_len:
                pad_len = past_len - feats.shape[0]
                pad = np.repeat(feats[0:1], pad_len, axis=0)
                feats = np.vstack([pad, feats])
                y_pad = np.repeat(y_now[0:1], pad_len, axis=0)
                y_now = np.concatenate([y_pad, y_now], axis=0)

            cond_id = int(df_u["ConditionID"].iloc[0]) if ("ConditionID" in df_u.columns) else 0
            scaler = scaler_dict.get(cond_id, scaler_dict.get(0))
            feats_scaled = scaler.transform(feats)

            # Create windows for ALL timepoints (window end at each observed t_end)
            for t_end in range(past_len - 1, feats_scaled.shape[0]):
                x = feats_scaled[t_end - past_len + 1 : t_end + 1]
                X_list.append(x.astype(np.float32, copy=False))
                y_true_list.append(float(y_now[t_end]))
                unit_ids_list.append(unit_id)
                cond_ids_list.append(cond_id)

        if not X_list:
            return {"dataset_split": split_name, "n_samples_all": 0, "n_units": 0}

        X_all = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32).to(device)
        cond_all = torch.tensor(np.asarray(cond_ids_list, dtype=np.int64), dtype=torch.long).to(device) if num_conditions > 1 else None

        preds: list[float] = []
        bs = 1024
        model.eval()
        with torch.no_grad():
            for i0 in range(0, X_all.size(0), bs):
                xb = X_all[i0 : i0 + bs]
                cb = cond_all[i0 : i0 + bs] if cond_all is not None else None
                _, eol_pred = model(
                    encoder_inputs=xb,
                    decoder_targets=None,
                    teacher_forcing_ratio=0.0,
                    horizon=1,
                    cond_ids=cb,
                )
                p = eol_pred.view(-1).detach().cpu().numpy().astype(float)
                preds.extend(p.tolist())

        unit_ids_np = np.asarray(unit_ids_list, dtype=np.int64)
        y_true_np = np.asarray(y_true_list, dtype=float)
        y_pred_np = np.asarray(preds, dtype=float)

        clip = (0.0, float(max_rul)) if max_rul is not None else None
        metrics_all = compute_all_samples_metrics(y_true_np, y_pred_np, unit_ids=unit_ids_np, clip=clip)
        metrics_last = compute_last_per_unit_metrics(unit_ids_np, y_true_np, y_pred_np, clip=clip)
        merged = {
            **metrics_last,
            **metrics_all,
            "dataset_split": split_name,
            "last_definition": metrics_last.get("note_last_definition", "LAST_AVAILABLE_PER_UNIT (truncated-aware)"),
        }
        return merged

    print("\n[7a] Evaluating on train/val splits...")
    metrics_train = _evaluate_last_all_on_dataframe(df_train_split_df, split_name="train")
    metrics_val = _evaluate_last_all_on_dataframe(df_val_split_df, split_name="val")

    # Save split metrics (standalone artifacts)
    try:
        with open(results_dir / "metrics_train.json", "w") as f:
            json.dump(metrics_train, f, indent=2)
        with open(results_dir / "metrics_val.json", "w") as f:
            json.dump(metrics_val, f, indent=2)
        print(f"  Saved metrics_train.json and metrics_val.json to {results_dir}")
    except Exception as e:
        print(f"  Warning: could not save train/val metrics json: {e}")
    
    # ===================================================================
    # 8. Evaluate on test set
    # ===================================================================
    print("\n[8] Evaluating on test set...")
    test_metrics = evaluate_world_model_eol_universal_v2(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler_dict=scaler_dict,
        past_len=past_len,
        max_rul=max_rul,
        num_conditions=num_conditions,
        device=device,
        eval_all_windows=True,
    )
    
    # Standardized reporting: LAST + ALL
    print("\n--- LAST (literature-style, truncated-aware) ---")
    print(f"  rmse_last: {test_metrics.get('rmse_last', float('nan')):.2f}")
    print(f"  mae_last:  {test_metrics.get('mae_last', float('nan')):.2f}")
    print(f"  bias_last: {test_metrics.get('bias_last', float('nan')):.2f}")
    print(f"  r2_last:   {test_metrics.get('r2_last', 0.0):.4f}")
    print(f"  nasa_last_mean: {test_metrics.get('nasa_last_mean', float('nan')):.4f}")
    print(f"  nasa_last_sum:  {test_metrics.get('nasa_last_sum', float('nan')):.2f}")
    print(f"  n_units: {test_metrics.get('n_units', 0)}")
    print(f"  max_rul_used: {test_metrics.get('max_rul_used', max_rul)}")

    print("\n--- ALL (all windows/timepoints) ---")
    print(f"  rmse_all: {test_metrics.get('rmse_all', float('nan')):.2f}")
    print(f"  mae_all:  {test_metrics.get('mae_all', float('nan')):.2f}")
    print(f"  bias_all: {test_metrics.get('bias_all', float('nan')):.2f}")
    print(f"  r2_all:   {test_metrics.get('r2_all', 0.0):.4f}")
    print(f"  nasa_all_mean: {test_metrics.get('nasa_all_mean', float('nan')):.4f}")
    print(f"  nasa_all_sum:  {test_metrics.get('nasa_all_sum', float('nan')):.2f}")
    print(f"  n_samples_all: {test_metrics.get('n_samples_all', 0)}")
    
    # ===================================================================
    # 8. Compute per-condition metrics
    # ===================================================================
    print("\n[9] Computing per-condition metrics...")
    condition_metrics = {}
    
    # Get condition IDs for test engines
    df_test_cond = df_test.groupby("UnitNumber")["ConditionID"].first()
    unit_ids_test = sorted(df_test["UnitNumber"].unique())
    
    # Per-condition metrics are defined on LAST (1 value per engine)
    y_pred_eol = np.array(test_metrics.get("y_pred_last", test_metrics.get("y_pred_eol", [])))
    y_true_eol = np.array(test_metrics.get("y_true_last", test_metrics.get("y_true_eol", y_test_true)))
    
    if len(y_pred_eol) > 0 and len(y_pred_eol) == len(unit_ids_test):
        for cond_id in unique_conditions:
            cond_id = int(cond_id)
            # Find engines with this condition
            cond_engines = df_test_cond[df_test_cond == cond_id].index.values
            if len(cond_engines) == 0:
                continue
            
            # Map engine IDs to indices (unit_ids_test is sorted list)
            cond_indices = [i for i, uid in enumerate(unit_ids_test) if uid in cond_engines]
            if len(cond_indices) == 0:
                continue
            
            cond_y_true = y_true_eol[cond_indices]
            cond_y_pred = y_pred_eol[cond_indices]
            
            # Compute metrics
            errors = cond_y_pred - cond_y_true
            rmse = np.sqrt(np.mean(errors ** 2))
            mae = np.mean(np.abs(errors))
            bias = np.mean(errors)
            r2 = 1 - np.sum(errors ** 2) / np.sum((cond_y_true - np.mean(cond_y_true)) ** 2) if np.var(cond_y_true) > 0 else 0.0
            
            condition_metrics[cond_id] = {
                "num_engines": len(cond_indices),
                "rmse": float(rmse),
                "mae": float(mae),
                "bias": float(bias),
                "r2": float(r2),
            }
        
        condition_metrics_path = results_dir / "condition_metrics.json"
        with open(condition_metrics_path, "w") as f:
            json.dump(condition_metrics, f, indent=2)
        print(f"  Saved per-condition metrics to {condition_metrics_path}")
        print(f"  Conditions analyzed: {list(condition_metrics.keys())}")
    else:
        print("  Warning: Could not compute per-condition metrics (missing predictions)")
    
    # ===================================================================
    # 9. Save results
    # ===================================================================
    print("\n[10] Saving results...")
    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model_type": "world_model_universal_v2",
        "encoder_type": "universal_v2",
        "d_model": d_model,
        "num_layers": num_layers,
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "kernel_sizes": kernel_sizes,
        "seq_encoder_type": seq_encoder_type,
        "decoder_num_layers": decoder_num_layers,
        "num_features": len(feature_cols),
        "num_conditions": num_conditions,
        "past_len": past_len,
        "horizon": horizon,
        "max_rul": max_rul,
        "world_model_config": {
            "forecast_horizon": horizon,
            "traj_loss_weight": world_model_config.traj_loss_weight,
            "eol_loss_weight": world_model_config.eol_loss_weight,
            "traj_step_weighting": world_model_config.traj_step_weighting,
            "use_condition_wise_scaling": use_condition_wise_scaling,
        },
        "train_metrics": metrics_train,
        "val_metrics": metrics_val,
        "test_metrics": test_metrics,
        "condition_metrics": condition_metrics,
        "training_history": {
            "best_epoch": checkpoint["epoch"] + 1,
            "best_val_loss": float(checkpoint["val_loss"]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
        },
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    # Save split metrics as standalone artifact (standard name)
    metrics_test_path = results_dir / "metrics_test.json"
    try:
        with open(metrics_test_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_test_path}")
    except Exception as e:
        print(f"  Warning: Could not save metrics_test.json: {e}")
    
    # Save full training history
    training_history_path = results_dir / "training_history.json"
    with open(training_history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history to {training_history_path}")
    
    # Plot training curves
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="red")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trajectory vs EOL losses (raw)
        axes[0, 1].plot(epochs, history["val_traj_loss"], label="Val Traj Loss", color="green")
        axes[0, 1].plot(epochs, history["val_eol_loss"], label="Val EOL Loss", color="orange")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Trajectory vs EOL Loss (Raw)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Weighted losses
        axes[1, 0].plot(epochs, history["val_traj_weighted"], label=f"Val Traj Weighted (×{world_model_config.traj_loss_weight:.1f})", color="green", linestyle="--")
        axes[1, 0].plot(epochs, history["val_eol_weighted"], label=f"Val EOL Weighted (×{world_model_config.eol_loss_weight:.1f})", color="orange", linestyle="--")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Weighted Loss")
        axes[1, 0].set_title("Validation Weighted Losses")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view: raw and weighted
        axes[1, 1].plot(epochs, history["val_traj_loss"], label="Traj (raw)", color="green", alpha=0.5)
        axes[1, 1].plot(epochs, history["val_eol_loss"], label="EOL (raw)", color="orange", alpha=0.5)
        axes[1, 1].plot(epochs, history["val_traj_weighted"], label="Traj (weighted)", color="green", linestyle="--")
        axes[1, 1].plot(epochs, history["val_eol_weighted"], label="EOL (weighted)", color="orange", linestyle="--")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Trajectory vs EOL: Raw and Weighted")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_curves_path = results_dir / "training_curves.png"
        plt.savefig(training_curves_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved training curves to {training_curves_path}")
    except Exception as e:
        print(f"  Warning: Could not generate training curves: {e}")
    
    # ===================================================================
    # 10. Generate diagnostics
    # ===================================================================
    print("\n[11] Generating diagnostics...")
    try:
        from src.analysis.diagnostics import run_diagnostics_for_run
        
        print(f"Using diagnostics (sliding-window HI, degraded engines) for {dataset_name}...")
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
    
    return summary


def evaluate_world_model_eol_universal_v2(
    model: nn.Module,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    scaler_dict: Dict[int, StandardScaler],
    past_len: int = 30,
    max_rul: int = 125,
    num_conditions: int = 1,
    device: torch.device = None,
    *,
    eval_all_windows: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate UniversalEncoderV2-based world model on test set (EOL metrics).
    
    Args:
        model: Trained WorldModelEncoderDecoderUniversalV2
        df_test: Test DataFrame
        y_test_true: True RUL at EOL (capped)
        feature_cols: Feature column names
        scaler_dict: Dictionary of condition-wise scalers
        past_len: Past window length
        max_rul: Maximum RUL value
        num_conditions: Number of conditions
        device: PyTorch device
    
    Returns:
        Dictionary with LAST (per-unit last-available) and ALL (all windows/timepoints) metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # --- LAST (one window per unit; last available) ---
    y_pred_last_list: list[float] = []
    y_true_last_list: list[float] = []
    unit_ids_last_list: list[int] = []
    
    # Helper function to build EOL input
    def _build_eol_input_for_unit(df_unit: pd.DataFrame, feature_cols: List[str], past_len: int) -> np.ndarray:
        """Build past_len window for EOL evaluation."""
        df_unit = df_unit.sort_values("TimeInCycles")
        feats = df_unit[feature_cols].values.astype(np.float32)
        
        if len(feats) < past_len:
            # Pad with first row
            padding = np.tile(feats[0:1], (past_len - len(feats), 1))
            feats = np.vstack([padding, feats])
        else:
            # Take last past_len rows
            feats = feats[-past_len:]
        
        return feats
    
    # Create mapping from unit_id to index in y_test_true
    # y_test_true is ordered by UnitNumber (1, 2, 3, ...)
    unit_id_to_idx = {i + 1: i for i in range(len(y_test_true))}
    
    with torch.no_grad():
        for unit_id, df_unit in df_test.groupby("UnitNumber"):
            unit_id = int(unit_id)

            # Build input (last window)
            X_past_np = _build_eol_input_for_unit(df_unit, feature_cols, past_len)

            # Condition ID
            cond_id = int(df_unit["ConditionID"].iloc[0])

            # Scale
            scaler = scaler_dict.get(cond_id, scaler_dict.get(0))
            X_past_scaled = scaler.transform(X_past_np.reshape(-1, len(feature_cols))).reshape(past_len, len(feature_cols))

            X_past = torch.tensor(X_past_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # (1, past_len, F)
            cond_ids = torch.tensor([cond_id], dtype=torch.long).to(device) if num_conditions > 1 else None

            # Predict scalar head (EOL-style)
            _, eol_pred = model(
                encoder_inputs=X_past,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
                cond_ids=cond_ids,
            )

            pred_rul = float(eol_pred[0, 0].cpu().item())
            pred_rul = float(np.clip(pred_rul, 0.0, max_rul))

            # Ground truth at LAST OBSERVED time (NASA test target)
            idx = unit_id_to_idx.get(unit_id, unit_id - 1)
            true_rul = float(y_test_true[idx]) if idx < len(y_test_true) else float(y_test_true[-1])

            y_pred_last_list.append(pred_rul)
            y_true_last_list.append(true_rul)
            unit_ids_last_list.append(unit_id)

    y_true_last = np.asarray(y_true_last_list, dtype=float)
    y_pred_last = np.asarray(y_pred_last_list, dtype=float)
    unit_ids_last = np.asarray(unit_ids_last_list, dtype=np.int64)

    clip = (0.0, float(max_rul)) if max_rul is not None else None
    metrics_last = compute_last_per_unit_metrics(unit_ids_last, y_true_last, y_pred_last, clip=clip)

    # --- ALL (all available windows/timepoints; truncated-aware via y_test_true + remaining-to-last) ---
    metrics_all: Dict[str, Any] = {}
    y_true_all = None
    y_pred_all = None
    unit_ids_all = None
    if eval_all_windows:
        X_all_list: list[np.ndarray] = []
        y_true_all_list: list[float] = []
        unit_ids_all_list: list[int] = []
        cond_ids_all_list: list[int] = []

        # Build per-timepoint truth from (last-observed RUL) + remaining cycles to last observed.
        for unit_id, df_unit in df_test.groupby("UnitNumber"):
            unit_id = int(unit_id)
            df_u = df_unit.sort_values("TimeInCycles").reset_index(drop=True)
            feats = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
            times = df_u["TimeInCycles"].to_numpy(dtype=np.float32, copy=True).reshape(-1)
            if feats.shape[0] < past_len:
                continue

            cond_id = int(df_u["ConditionID"].iloc[0])
            scaler = scaler_dict.get(cond_id, scaler_dict.get(0))
            feats_scaled = scaler.transform(feats)

            t_last = float(np.max(times))
            idx = unit_id_to_idx.get(unit_id, unit_id - 1)
            rul_last_obs = float(y_test_true[idx]) if idx < len(y_test_true) else float(y_test_true[-1])

            # Window ends at each observed timepoint t_end (>= past_len-1)
            for t_end in range(past_len - 1, feats_scaled.shape[0]):
                x = feats_scaled[t_end - past_len + 1 : t_end + 1]  # (P,F)
                # True RUL at this observed timepoint (remaining to failure)
                rul_t = rul_last_obs + (t_last - float(times[t_end]))
                y_true_all_list.append(float(rul_t))
                X_all_list.append(x.astype(np.float32, copy=False))
                unit_ids_all_list.append(unit_id)
                cond_ids_all_list.append(cond_id)

        if X_all_list:
            X_all = torch.tensor(np.stack(X_all_list, axis=0), dtype=torch.float32).to(device)
            cond_all = torch.tensor(np.asarray(cond_ids_all_list, dtype=np.int64), dtype=torch.long).to(device) if num_conditions > 1 else None

            preds: list[float] = []
            bs = 1024
            with torch.no_grad():
                for i0 in range(0, X_all.size(0), bs):
                    xb = X_all[i0 : i0 + bs]
                    cb = cond_all[i0 : i0 + bs] if cond_all is not None else None
                    _, eol_pred = model(
                        encoder_inputs=xb,
                        decoder_targets=None,
                        teacher_forcing_ratio=0.0,
                        horizon=1,
                        cond_ids=cb,
                    )
                    p = eol_pred.view(-1).detach().cpu().numpy().astype(float)
                    preds.extend(p.tolist())

            unit_ids_all = np.asarray(unit_ids_all_list, dtype=np.int64)
            y_true_all = np.asarray(y_true_all_list, dtype=float)
            y_pred_all = np.asarray(preds, dtype=float)

            metrics_all = compute_all_samples_metrics(y_true_all, y_pred_all, unit_ids=unit_ids_all, clip=clip)
        else:
            metrics_all = {"n_samples_all": 0}

    # Merge and keep a few legacy aliases (deprecated)
    merged: Dict[str, Any] = {
        **metrics_last,
        **metrics_all,
        "dataset_split": "test",
        "last_definition": metrics_last.get("note_last_definition", "LAST_AVAILABLE_PER_UNIT (truncated-aware)"),
    }
    # Provide LAST arrays for downstream diagnostics/per-condition logic
    merged["unit_ids_last"] = unit_ids_last.tolist()
    merged["y_true_last"] = y_true_last.tolist()
    merged["y_pred_last"] = y_pred_last.tolist()
    # Backward-compat aliases (deprecated names)
    merged["y_true_eol"] = merged["y_true_last"]
    merged["y_pred_eol"] = merged["y_pred_last"]
    if eval_all_windows and unit_ids_all is not None and y_true_all is not None and y_pred_all is not None:
        merged["unit_ids_all"] = unit_ids_all.tolist()
        merged["y_true_all_raw"] = y_true_all.tolist()
        merged["y_pred_all_raw"] = y_pred_all.tolist()

    # Legacy aliases (deprecated): previous behavior was effectively LAST on test.
    merged["RMSE"] = merged.get("rmse_last")
    merged["MAE"] = merged.get("mae_last")
    merged["Bias"] = merged.get("bias_last")
    merged["R2"] = merged.get("r2_last")
    merged["nasa_score_sum"] = merged.get("nasa_last_sum")
    merged["nasa_score_mean"] = merged.get("nasa_last_mean")
    merged["num_engines"] = merged.get("n_units")

    return merged
