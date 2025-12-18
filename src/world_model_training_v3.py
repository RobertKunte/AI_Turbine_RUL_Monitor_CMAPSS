"""
World Model v3 Training Functions.

This module contains training functions for World Model v3 (UniversalEncoderV2 + HI Head).
Separated from world_model_training.py to keep file sizes manageable.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# region agent log
def _agent_dbg(payload: dict) -> None:
    """Minimal NDJSON debug logger (safe/no-throw)."""
    try:
        import json as _json
        with open(
            r"c:\Users\rober\OneDrive\Dokumente\GitHub\AI_Turbine_RUL_Monitor_CMAPSS\.cursor\debug.log",
            "a",
            encoding="utf-8",
        ) as _f:
            _f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for training routines. Please install torch."
    ) from exc

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for preprocessing. Please install scikit-learn."
    ) from exc

from src.world_model_training import (
    compute_trajectory_step_weights,
    WorldModelTrainingConfig,
)
from src.data.windowing import WindowConfig, TargetConfig, build_sliding_windows, build_test_windows_last
from src.models.world_model import WorldModelUniversalV3
from src.loss import (
    monotonic_health_loss,
    hi_early_slope_regularizer,
    hi_curvature_loss,
    hi_eol_consistency_loss,
)
from src.training_utils import compute_global_trend_loss
from src.eval.eol_eval import evaluate_eol_metrics
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.models.transformer_world_model_v1 import TransformerWorldModelV1


def train_world_model_universal_v3(
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
    Train UniversalEncoderV2-based world model v3 with Health Index head.
    
    This function:
    - Uses Phase 4 residual feature pipeline (464 features)
    - Uses UniversalEncoderV2 as encoder
    - Handles condition-wise scaling
    - Supports configurable loss weights (traj, eol, hi, monotonicity)
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
            If None, uses defaults (horizon=40, traj_weight=1.0, eol_weight=5.0, hi_weight=2.0)
        results_dir: Directory to save results
        device: PyTorch device
    
    Returns:
        Dictionary with training results and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if results_dir is None:
        results_dir = Path("results") / dataset_name.lower() / experiment_name
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 9]
    
    # Use provided config or create default for v3
    if world_model_config is None:
        world_model_config = WorldModelTrainingConfig(
            forecast_horizon=40,
            traj_loss_weight=1.0,
            eol_loss_weight=5.0,
            hi_loss_weight=2.0,
            mono_late_weight=0.1,
            mono_global_weight=0.1,
        )

    # region agent log
    _agent_dbg(
        {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "A",
            "location": "src/world_model_training_v3.py:train_world_model_universal_v3:config_ready",
            "message": "world_model_config schedule fields snapshot",
            "data": {
                "world_model_config_type": type(world_model_config).__name__,
                "three_phase_schedule": getattr(world_model_config, "three_phase_schedule", "<missing>"),
                "phase_a_frac": getattr(world_model_config, "phase_a_frac", "<missing>"),
                "phase_b_frac": getattr(world_model_config, "phase_b_frac", "<missing>"),
                "phase_b_end_frac": getattr(world_model_config, "phase_b_end_frac", "<missing>"),
                "schedule_type": getattr(world_model_config, "schedule_type", "<missing>"),
                "num_epochs": num_epochs,
            },
            "timestamp": int(__import__("time").time() * 1000),
        }
    )
    # endregion
    
    past_len = world_model_config.past_len
    # Allow an override for the decoder forecast horizon specific to WorldModelV1
    horizon = int(getattr(world_model_config, "future_horizon", world_model_config.forecast_horizon))
    max_rul = world_model_config.max_rul
    use_condition_wise_scaling = world_model_config.use_condition_wise_scaling
    
    print(f"\n{'='*80}")
    print(f"Training World Model v3: {experiment_name}")
    print(f"Dataset: {dataset_name}, Features: {len(feature_cols)}")
    print(f"  Horizon: {horizon}, Past len: {past_len}")
    print(f"  Loss weights: traj={world_model_config.traj_loss_weight:.2f}, "
          f"eol={world_model_config.eol_loss_weight:.2f}, "
          f"hi={world_model_config.hi_loss_weight:.2f}")
    print(f"  Monotonicity: late={world_model_config.mono_late_weight:.2f}, "
          f"global={world_model_config.mono_global_weight:.2f}")
    print(f"  EOL tail weighting: thr={world_model_config.eol_tail_rul_threshold}, "
          f"w={world_model_config.eol_tail_weight:.2f}")
    print(f"  HI fusion into EOL: use_hi_in_eol={world_model_config.use_hi_in_eol}, "
          f"use_hi_slope_in_eol={world_model_config.use_hi_slope_in_eol}")
    print(f"{'='*80}\n")
    
    # Compute trajectory step weights if needed
    traj_step_weights = compute_trajectory_step_weights(
        horizon=horizon,
        weighting=world_model_config.traj_step_weighting,
    ).to(device)
    
    # ===================================================================
    # 1. Build sequences with condition IDs
    # ===================================================================
    print("[1] Building seq2seq sequences...")
    use_padded_horizon_targets = bool(getattr(world_model_config, "use_padded_horizon_targets", False))
    use_horizon_mask = bool(getattr(world_model_config, "use_horizon_mask", False))
    cap_targets = bool(getattr(world_model_config, "cap_rul_targets_to_max_rul", False))
    eol_target_mode = str(getattr(world_model_config, "eol_target_mode", "future0"))

    window_cfg = WindowConfig(
        past_len=int(past_len),
        horizon=int(horizon),
        stride=1,
        require_full_horizon=not bool(use_padded_horizon_targets),
        pad_mode="clamp",
    )
    target_cfg = TargetConfig(
        max_rul=int(max_rul if max_rul is not None else 125),
        cap_targets=bool(cap_targets),
        eol_target_mode=str(eol_target_mode),
        clip_eval_y_true=bool(getattr(world_model_config, "eval_clip_y_true_to_max_rul", False)),
    )

    built = build_sliding_windows(
        df_train,
        feature_cols,
        target_col="RUL",
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=window_cfg,
        target_cfg=target_cfg,
        return_mask=bool(use_horizon_mask),
    )

    X_train = torch.tensor(built["X"], dtype=torch.float32)
    Y_train = torch.tensor(built["Y_seq"], dtype=torch.float32)
    unit_ids_train = torch.tensor(built["unit_ids"], dtype=torch.long)
    cond_ids_train = torch.tensor(built["cond_ids"], dtype=torch.long)
    horizon_mask_train = torch.tensor(built["mask"], dtype=torch.float32) if (use_horizon_mask and built["mask"] is not None) else None
    
    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Input shape: {X_train.shape}, Target shape: {Y_train.shape}")
    if horizon_mask_train is not None:
        print(f"  Horizon mask shape: {horizon_mask_train.shape} (use_horizon_mask={use_horizon_mask})")
    
    # Determine number of conditions
    unique_conditions = torch.unique(cond_ids_train).cpu().numpy()
    num_conditions = len(unique_conditions)
    print(f"  Found {num_conditions} unique conditions: {unique_conditions}")
    
    # ===================================================================
    # 2. Engine-based train/val split
    # ===================================================================
    print("\n[2] Creating engine-based train/val split...")
    # Proper engine split: split by unique unit IDs, then select samples by membership.
    rng = torch.Generator().manual_seed(random_seed)
    unique_units = torch.unique(unit_ids_train)
    perm = unique_units[torch.randperm(len(unique_units), generator=rng)]
    n_val_units = max(1, int((1.0 - float(engine_train_ratio)) * float(len(unique_units))))
    val_units = perm[:n_val_units]
    train_units = perm[n_val_units:]
    train_mask = torch.isin(unit_ids_train, train_units)
    val_mask = torch.isin(unit_ids_train, val_units)

    train_indices = torch.nonzero(train_mask, as_tuple=False).view(-1)
    val_indices = torch.nonzero(val_mask, as_tuple=False).view(-1)
    
    X_train_split = X_train[train_indices]
    Y_train_split = Y_train[train_indices]
    cond_ids_train_split = cond_ids_train[train_indices]
    horizon_mask_train_split = horizon_mask_train[train_indices] if horizon_mask_train is not None else None
    
    X_val = X_train[val_indices]
    Y_val = Y_train[val_indices]
    cond_ids_val = cond_ids_train[val_indices]
    horizon_mask_val = horizon_mask_train[val_indices] if horizon_mask_train is not None else None
    
    print(f"  Train samples: {len(X_train_split)}, Val samples: {len(X_val)}")
    
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
    if horizon_mask_train_split is not None:
        train_dataset = TensorDataset(X_train_scaled, Y_train_split, cond_ids_train_split, horizon_mask_train_split)
        val_dataset = TensorDataset(X_val_scaled, Y_val, cond_ids_val, horizon_mask_val)
    else:
        train_dataset = TensorDataset(X_train_scaled, Y_train_split, cond_ids_train_split)
        val_dataset = TensorDataset(X_val_scaled, Y_val, cond_ids_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ===================================================================
    # 5. Initialize model
    # ===================================================================
    print("\n[5] Initializing World Model v3...")
    model = WorldModelUniversalV3(
        input_size=len(feature_cols),
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_conditions=num_conditions if num_conditions > 1 else None,
        cond_emb_dim=4,
        kernel_sizes=kernel_sizes,
        seq_encoder_type=seq_encoder_type,
        use_layer_norm=True,
        max_seq_len=300,
        decoder_num_layers=decoder_num_layers,
        horizon=horizon,
        use_hi_in_eol=world_model_config.use_hi_in_eol,
        use_hi_slope_in_eol=world_model_config.use_hi_slope_in_eol,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Encoder: UniversalEncoderV2 (d_model={d_model}, num_layers={num_layers})")
    print(f"  Decoder: LSTM (num_layers={decoder_num_layers}, horizon={horizon})")
    print(f"  Heads: Trajectory, EOL, HI")
    
    # ===================================================================
    # 6. Training loop
    # ===================================================================
    print("\n[6] Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()

    # -------------------------------
    # EOL loss stabilization options
    # -------------------------------
    normalize_eol = bool(getattr(world_model_config, "normalize_eol", False))
    eol_loss_type = str(getattr(world_model_config, "eol_loss_type", "mse")).lower()
    eol_huber_beta = float(getattr(world_model_config, "eol_huber_beta", 0.1))

    # -------------------------------
    # Selection / target alignment knobs (default OFF)
    # -------------------------------
    select_best_after_eol_active = bool(getattr(world_model_config, "select_best_after_eol_active", False))
    eol_active_min_mult = float(getattr(world_model_config, "eol_active_min_mult", 0.01))
    best_metric = str(getattr(world_model_config, "best_metric", "val_total"))
    cap_rul_targets = bool(getattr(world_model_config, "cap_rul_targets_to_max_rul", False))
    eol_target_mode = str(getattr(world_model_config, "eol_target_mode", "future0")).lower()
    init_eol_bias = bool(getattr(world_model_config, "init_eol_bias_to_target_mean", False))

    # Resolve EOL normalization scale once (used only inside loss)
    def _resolve_eol_scale() -> float:
        scale_cfg = getattr(world_model_config, "eol_scale", "rul_cap")
        if isinstance(scale_cfg, (int, float)):
            return float(scale_cfg)
        if isinstance(scale_cfg, str):
            key = scale_cfg.lower()
            if key == "rul_cap":
                return float(getattr(world_model_config, "max_rul", 125.0) or 125.0)
            if key == "max_cycle":
                try:
                    # global max last-cycle length in training set
                    mx = float(df_train.groupby("UnitNumber")["TimeInCycles"].max().max())
                    return mx if mx > 0 else float(getattr(world_model_config, "max_rul", 125.0) or 125.0)
                except Exception:
                    return float(getattr(world_model_config, "max_rul", 125.0) or 125.0)
        # fallback
        return float(getattr(world_model_config, "max_rul", 125.0) or 125.0)

    eol_scale_value = _resolve_eol_scale() if normalize_eol else 1.0

    # Default grad clipping only when normalization is enabled (opt-in stability)
    clip_grad_norm_cfg = getattr(world_model_config, "clip_grad_norm", None)
    if normalize_eol and clip_grad_norm_cfg is None:
        clip_grad_norm_cfg = 1.0

    freeze_encoder_n = int(getattr(world_model_config, "freeze_encoder_epochs_after_eol_on", 0) or 0)
    eol_on_epoch: Optional[int] = None
    eol_became_active_epoch: Optional[int] = None

    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _eol_loss_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample EOL loss on normalized scale, shape (B,).
        """
        if eol_loss_type == "huber":
            fn = nn.SmoothL1Loss(beta=eol_huber_beta, reduction="none")
            return fn(pred, target)
        if eol_loss_type == "mae":
            return torch.abs(pred - target)
        # default mse
        return (pred - target) ** 2

    # Optional: initialize EOL head bias close to mean target (raw units).
    # Note: even when normalize_eol=True, the model output is still in raw units;
    # normalization happens only inside the loss.
    if init_eol_bias:
        try:
            if hasattr(model, "fc_rul") and isinstance(model.fc_rul, nn.Linear) and model.fc_rul.bias is not None:
                max_rul_f = float(max_rul if max_rul is not None else 125.0)
                # df_train["RUL"] may be raw; clamp if requested
                if "RUL" in df_train.columns:
                    v = df_train["RUL"].to_numpy(dtype=float)
                    if cap_rul_targets:
                        v = np.clip(v, 0.0, max_rul_f)
                    mu = float(np.nanmean(v)) if v.size else float(max_rul_f)
                else:
                    mu = float(max_rul_f)
                model.fc_rul.bias.data.fill_(mu)
                print(f"[init] Initialized EOL head bias to mean target ~= {mu:.2f} (cap_targets={cap_rul_targets})")
        except Exception as e:
            print(f"  ⚠️  Warning: init_eol_bias_to_target_mean failed: {e}")
    
    history = {
        "train_loss": [],
        "train_grad_norm": [],
        "val_loss": [],
        "val_traj_loss": [],
        "val_eol_loss": [],
        "val_hi_loss": [],
        "val_mono_late_loss": [],
        "val_mono_global_loss": [],
        "val_hi_early_slope_loss": [],
        "val_hi_curvature_loss": [],
        "val_traj_weighted": [],
        "val_eol_weighted": [],
        "val_hi_weighted": [],
        "val_mono_weighted": [],
        "val_shape_weighted": [],
        "val_eol_hi_loss": [],
        "val_eol_hi_weighted": [],
    }
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = results_dir / "world_model_v3_best.pt"
    abort_path = results_dir / "world_model_v3_abort_nan.pt"
    
    def _eol_mult_for_epoch(epoch_idx: int) -> float:
        # region agent log
        _agent_dbg(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "src/world_model_training_v3.py:_eol_mult_for_epoch:entry",
                "message": "entered _eol_mult_for_epoch",
                "data": {
                    "epoch_idx": epoch_idx,
                    "num_epochs": num_epochs,
                    "three_phase_schedule": getattr(world_model_config, "three_phase_schedule", "<missing>"),
                },
                "timestamp": int(__import__("time").time() * 1000),
            }
        )
        # endregion
        if not bool(getattr(world_model_config, "three_phase_schedule", False)):
            # region agent log
            _agent_dbg(
                {
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "C",
                    "location": "src/world_model_training_v3.py:_eol_mult_for_epoch:no_schedule",
                    "message": "three_phase_schedule disabled => eol_mult=1.0",
                    "data": {"epoch_idx": epoch_idx},
                    "timestamp": int(__import__("time").time() * 1000),
                }
            )
            # endregion
            return 1.0
        # progress in (0,1]
        p = float(epoch_idx + 1) / float(max(1, num_epochs))
        raw_a = getattr(world_model_config, "phase_a_frac", 0.2)
        raw_b_frac = getattr(world_model_config, "phase_b_frac", "<missing>")
        raw_b_end = getattr(world_model_config, "phase_b_end_frac", "<missing>")
        # region agent log
        _agent_dbg(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "src/world_model_training_v3.py:_eol_mult_for_epoch:pre_float_cast",
                "message": "raw schedule values before float() cast",
                "data": {
                    "epoch_idx": epoch_idx,
                    "p": p,
                    "raw_phase_a_frac": raw_a,
                    "raw_phase_b_frac": raw_b_frac,
                    "raw_phase_b_end_frac": raw_b_end,
                    "types": {
                        "phase_a_frac": type(raw_a).__name__,
                        "phase_b_frac": type(raw_b_frac).__name__,
                        "phase_b_end_frac": type(raw_b_end).__name__,
                    },
                },
                "timestamp": int(__import__("time").time() * 1000),
            }
        )
        # endregion
        a = float(raw_a)
        # phase_b_frac may exist but be None -> treat as missing and fall back
        b_raw = getattr(world_model_config, "phase_b_frac", None)
        if b_raw is None:
            b_raw = getattr(world_model_config, "phase_b_end_frac", 0.8)
        b = float(b_raw)
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        if b <= a:
            # Degenerate schedule: treat as immediate joint training
            return 1.0
        if p <= a:
            return 0.0
        if p >= b:
            return 1.0
        # Ramp in phase B
        ramp = (p - a) / (b - a)
        schedule_type = str(getattr(world_model_config, "schedule_type", getattr(world_model_config, "eol_ramp", "linear"))).lower()
        if schedule_type == "cosine":
            # cosine from 0..1
            import math
            return float(0.5 * (1.0 - math.cos(math.pi * ramp)))
        return float(ramp)

    for epoch in range(num_epochs):
        eol_mult = _eol_mult_for_epoch(epoch)
        eol_weight_eff = float(world_model_config.eol_loss_weight) * float(getattr(world_model_config, "eol_w_max", 1.0)) * float(eol_mult)
        eol_active = bool(eol_mult >= eol_active_min_mult)
        if eol_active and eol_on_epoch is None:
            eol_on_epoch = int(epoch)
        if eol_active and eol_became_active_epoch is None:
            eol_became_active_epoch = int(epoch)

        # Optional: freeze encoder briefly when EOL starts to protect representation
        if freeze_encoder_n > 0 and eol_on_epoch is not None and (epoch - eol_on_epoch) < freeze_encoder_n:
            _set_requires_grad(model.encoder, False)
            encoder_frozen = True
        else:
            _set_requires_grad(model.encoder, True)
            encoder_frozen = False

        # Training
        model.train()
        # If we freeze the encoder during early EOL ramp-in, also put it in eval mode
        # (reduces dropout noise while still training decoder + heads).
        try:
            if encoder_frozen:
                model.encoder.eval()
            else:
                model.encoder.train()
        except Exception:
            pass
        running_train_loss = 0.0
        n_train_samples = 0
        printed_epoch_stats = False
        running_grad_norm = 0.0
        n_grad_norm = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Backward-compatible unpacking: (X,Y,cond) or (X,Y,cond,mask)
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                X_batch, Y_batch, cond_batch, mask_batch = batch
            else:
                X_batch, Y_batch, cond_batch = batch
                mask_batch = None

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            cond_batch = cond_batch.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                teacher_forcing_ratio=0.5,
                horizon=horizon,
                cond_ids=cond_batch if num_conditions > 1 else None,
            )
            
            traj_pred = outputs["traj"]  # (B, H, 1) - predicted HI trajectory
            eol_pred = outputs["eol"].squeeze(-1)  # (B,)   - predicted EOL RUL
            hi_pred = outputs["hi"].squeeze(-1)    # (B,)   - predicted HI at current step

            # ------------------------------------------------------------------
            # Targets
            # ------------------------------------------------------------------
            target_traj_rul = Y_batch              # (B, H, 1) - future RUL trajectory (may be raw/padded)
            if cap_rul_targets:
                max_rul_f = float(max_rul if max_rul is not None else 125.0)
                target_traj_rul = target_traj_rul.clamp(0.0, max_rul_f)

            # Scalar EOL target (explicit definition; must match eval units)
            if eol_target_mode in {"future0", "t0", "start"}:
                eol_scalar_target = target_traj_rul[:, 0, 0]
                target_def = "target_traj[:,0] (future0)"
            elif eol_target_mode in {"future_last", "last"}:
                eol_scalar_target = target_traj_rul[:, -1, 0]
                target_def = "target_traj[:,-1] (future_last)"
            elif eol_target_mode in {"current", "now"}:
                # Approximate current-step RUL from the next-step target (+1),
                # then re-apply the cap (plateau region stays at max_rul).
                max_rul_f = float(max_rul if max_rul is not None else 125.0)
                eol_scalar_target = (target_traj_rul[:, 0, 0] + 1.0).clamp(0.0, max_rul_f)
                target_def = "min(max_rul, target_traj[:,0] + 1) (current approx)"
            else:
                eol_scalar_target = target_traj_rul[:, 0, 0]
                target_def = f"unknown({eol_target_mode}) -> fallback future0"

            # Runtime range check (first epoch / first batch)
            if epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    te = eol_scalar_target.detach()
                    te_max = float(te.max().cpu().item())
                    te_min = float(te.min().cpu().item())
                    mr = float(max_rul if max_rul is not None else 125.0)
                    print(f"[target] eol_target_mode={eol_target_mode} ({target_def}) cap_targets={cap_rul_targets}")
                    print(f"[target] eol_true_scalar: min={te_min:.2f} max={te_max:.2f} (max_rul={mr:.2f})")
                    if te_max > mr + 1e-3:
                        print("  ⚠️  Warning: eol_true_scalar exceeds max_rul -> likely unit/clip mismatch.")

            # Physics-informed Health Index targets derived from RUL
            MAX_VISIBLE_RUL = float(max_rul if max_rul is not None else 125.0)

            # HI sequence target for each horizon step (B, H)
            rul_future = target_traj_rul.squeeze(-1)          # (B, H)
            hi_linear_seq = (rul_future / MAX_VISIBLE_RUL).clamp(0.0, 1.0)
            target_hi_seq = torch.where(
                rul_future > MAX_VISIBLE_RUL,
                torch.ones_like(hi_linear_seq),               # healthy plateau
                hi_linear_seq,                                # linear decay region
            )  # (B, H)

            # HI target at current step (use first horizon step)
            hi_linear = (eol_scalar_target / MAX_VISIBLE_RUL).clamp(0.0, 1.0)
            target_hi_last = torch.where(
                eol_scalar_target > MAX_VISIBLE_RUL,
                torch.ones_like(hi_linear),
                hi_linear,
            )  # (B,)

            # ------------------------------------------------------------------
            # Losses per head
            # ------------------------------------------------------------------
            # 1) EOL head: RUL at current step (optionally tail-weighted)
            # Sample weights (tail emphasis) based on raw cycles
            if (
                world_model_config.eol_tail_rul_threshold is not None
                and world_model_config.eol_tail_weight is not None
                and world_model_config.eol_tail_weight != 1.0
            ):
                thr = float(world_model_config.eol_tail_rul_threshold)
                tail_w = float(world_model_config.eol_tail_weight)
                weights_eol = torch.where(
                    eol_scalar_target < thr,
                    torch.full_like(eol_scalar_target, tail_w),
                    torch.ones_like(eol_scalar_target),
                )
            else:
                weights_eol = torch.ones_like(eol_scalar_target)

            if normalize_eol:
                # Normalize only inside loss
                scale = float(max(1e-6, eol_scale_value))
                eol_pred_n = eol_pred / scale
                eol_true_n = eol_scalar_target / scale
                per = _eol_loss_per_sample(eol_pred_n, eol_true_n)
                loss_eol = (per * weights_eol).mean()
            else:
                # Old behavior (raw scale, pure MSE)
                loss_eol = ((eol_pred - eol_scalar_target) ** 2 * weights_eol).mean()

            # Logging once per epoch (first batch)
            if not printed_epoch_stats and batch_idx == 0:
                with torch.no_grad():
                    te = eol_scalar_target.detach()
                    pe = eol_pred.detach()
                    msg = (
                        f"[EOL stats][epoch {epoch+1}] active={eol_active} mult={eol_mult:.3f} "
                        f"w_eff={eol_weight_eff:.3f} normalize={normalize_eol} scale={eol_scale_value:.2f} "
                        f"loss_type={eol_loss_type} huber_beta={eol_huber_beta} "
                        f"encoder_frozen={encoder_frozen}"
                    )
                    print(msg)
                    print(
                        f"  eol_true: mean={te.mean().item():.2f} std={te.std().item():.2f} "
                        f"min={te.min().item():.2f} max={te.max().item():.2f}"
                    )
                    print(
                        f"  eol_pred: mean={pe.mean().item():.2f} std={pe.std().item():.2f} "
                        f"min={pe.min().item():.2f} max={pe.max().item():.2f}"
                    )
                    if normalize_eol:
                        te_n = te / float(max(1e-6, eol_scale_value))
                        pe_n = pe / float(max(1e-6, eol_scale_value))
                        print(
                            f"  eol_true_n: mean={te_n.mean().item():.3f} std={te_n.std().item():.3f} "
                            f"min={te_n.min().item():.3f} max={te_n.max().item():.3f}"
                        )
                        print(
                            f"  eol_pred_n: mean={pe_n.mean().item():.3f} std={pe_n.std().item():.3f} "
                            f"min={pe_n.min().item():.3f} max={pe_n.max().item():.3f}"
                        )
                printed_epoch_stats = True

            # 2) HI head (scalar): MSE to current-step HI target
            loss_hi_last = mse_loss(hi_pred, target_hi_last)

            # 3) Trajectory head: full HI trajectory with monotonic + smoothness regularization
            hi_seq_pred = traj_pred.squeeze(-1)  # (B, H)
            valid_mask_seq = mask_batch.squeeze(-1) if (use_horizon_mask and mask_batch is not None) else None
            loss_traj, base_hi_mse, mono_raw, smooth_raw = monotonic_health_loss(
                pred_hi=hi_seq_pred,
                target_hi=target_hi_seq,
                alpha=world_model_config.mono_late_weight,
                beta=world_model_config.mono_global_weight,
                valid_mask=valid_mask_seq,
                return_components=True,
            )

            # Additional Stage-1 HI shape losses (optional, default off)
            early_slope_raw = hi_early_slope_regularizer(
                pred_hi=hi_seq_pred,
                rul_seq=rul_future,
                valid_mask=valid_mask_seq,
                epsilon=float(getattr(world_model_config, "hi_early_slope_epsilon", 1e-3)),
                early_rul_threshold=getattr(world_model_config, "hi_early_slope_rul_threshold", None),
            )
            curv_raw = hi_curvature_loss(
                pred_hi=hi_seq_pred,
                abs_mode=bool(getattr(world_model_config, "hi_curvature_abs", True)),
            )
            shape_loss = (
                float(getattr(world_model_config, "hi_early_slope_weight", 0.0)) * early_slope_raw
                + float(getattr(world_model_config, "hi_curvature_weight", 0.0)) * curv_raw
            )

            # Optional WorldModel coupling without RUL trajectory: HI → EOL consistency
            eol_hi_raw = hi_eol_consistency_loss(
                eol_pred=eol_pred,
                hi_seq=hi_seq_pred,
                hi_threshold=float(getattr(world_model_config, "eol_hi_threshold", 0.2)),
                temperature=float(getattr(world_model_config, "eol_hi_temperature", 0.05)),
                p_min=float(getattr(world_model_config, "eol_hi_p_min", 0.2)),
                valid_mask=valid_mask_seq,
            )
            eol_hi_loss = float(getattr(world_model_config, "w_eol_hi", 0.0)) * eol_hi_raw

            # Weighted total loss
            weighted_traj = world_model_config.traj_loss_weight * loss_traj
            weighted_eol = eol_weight_eff * loss_eol
            weighted_hi = world_model_config.hi_loss_weight * loss_hi_last
            weighted_shape = shape_loss
            weighted_eol_hi = eol_hi_loss

            loss = weighted_traj + weighted_eol + weighted_hi + weighted_shape + weighted_eol_hi

            # NaN/Inf guard
            if not torch.isfinite(loss):
                print(f"\n❌ Non-finite loss detected at epoch {epoch+1}, batch {batch_idx}. Aborting training.")
                print(f"  loss={loss.item() if torch.is_tensor(loss) else loss}")
                try:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "loss": float(loss.detach().cpu().item()) if torch.is_tensor(loss) else None,
                        },
                        abort_path,
                    )
                    print(f"  Saved abort checkpoint to {abort_path}")
                except Exception as e:
                    print(f"  ⚠️  Could not save abort checkpoint: {e}")
                return {"error": "non_finite_loss", "abort_checkpoint": str(abort_path)}
            
            loss.backward()

            # Gradient clipping (optional)
            if clip_grad_norm_cfg is not None:
                try:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm_cfg))
                    # gn is total norm BEFORE clipping
                    if torch.is_tensor(gn):
                        running_grad_norm += float(gn.detach().cpu().item())
                        n_grad_norm += 1
                except Exception as e:
                    print(f"  ⚠️  Warning: clip_grad_norm failed: {e}")

            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
            n_train_samples += X_batch.size(0)
        
        epoch_train_loss = running_train_loss / n_train_samples
        epoch_grad_norm = (running_grad_norm / n_grad_norm) if n_grad_norm > 0 else float("nan")
        history["train_grad_norm"].append(epoch_grad_norm)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_traj_loss = 0.0
        running_val_eol_loss = 0.0
        running_val_hi_loss = 0.0
        running_val_mono_late_loss = 0.0
        running_val_mono_global_loss = 0.0
        running_val_hi_early_slope_loss = 0.0
        running_val_hi_curvature_loss = 0.0
        running_val_traj_weighted = 0.0
        running_val_eol_weighted = 0.0
        running_val_hi_weighted = 0.0
        running_val_mono_weighted = 0.0
        running_val_shape_weighted = 0.0
        running_val_eol_hi_loss = 0.0
        running_val_eol_hi_weighted = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 4:
                    X_batch, Y_batch, cond_batch, mask_batch = batch
                else:
                    X_batch, Y_batch, cond_batch = batch
                    mask_batch = None

                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                cond_batch = cond_batch.to(device)
                if mask_batch is not None:
                    mask_batch = mask_batch.to(device)
                
                # Forward pass
                outputs = model(
                    encoder_inputs=X_batch,
                    decoder_targets=Y_batch,
                    teacher_forcing_ratio=0.0,
                    horizon=horizon,
                    cond_ids=cond_batch if num_conditions > 1 else None,
                )
                
                traj_pred = outputs["traj"]  # (B, H, 1) - predicted HI trajectory
                eol_pred = outputs["eol"].squeeze(-1)  # (B,)
                hi_pred = outputs["hi"].squeeze(-1)    # (B,)
                
                # Targets
                target_traj_rul = Y_batch              # (B, H, 1)
                if cap_rul_targets:
                    max_rul_f = float(max_rul if max_rul is not None else 125.0)
                    target_traj_rul = target_traj_rul.clamp(0.0, max_rul_f)

                if eol_target_mode in {"future0", "t0", "start"}:
                    eol_scalar_target = target_traj_rul[:, 0, 0]
                elif eol_target_mode in {"future_last", "last"}:
                    eol_scalar_target = target_traj_rul[:, -1, 0]
                elif eol_target_mode in {"current", "now"}:
                    max_rul_f = float(max_rul if max_rul is not None else 125.0)
                    eol_scalar_target = (target_traj_rul[:, 0, 0] + 1.0).clamp(0.0, max_rul_f)
                else:
                    eol_scalar_target = target_traj_rul[:, 0, 0]

                # Physics-informed HI targets (same as training loop)
                MAX_VISIBLE_RUL = float(max_rul if max_rul is not None else 125.0)
                rul_future = target_traj_rul.squeeze(-1)          # (B, H)
                hi_linear_seq = (rul_future / MAX_VISIBLE_RUL).clamp(0.0, 1.0)
                target_hi_seq = torch.where(
                    rul_future > MAX_VISIBLE_RUL,
                    torch.ones_like(hi_linear_seq),
                    hi_linear_seq,
                )  # (B, H)

                hi_linear = (eol_scalar_target / MAX_VISIBLE_RUL).clamp(0.0, 1.0)
                target_hi_last = torch.where(
                    eol_scalar_target > MAX_VISIBLE_RUL,
                    torch.ones_like(hi_linear),
                    hi_linear,
                )  # (B,)
                
                # Losses
                # EOL loss (same logic as training)
                if (
                    world_model_config.eol_tail_rul_threshold is not None
                    and world_model_config.eol_tail_weight is not None
                    and world_model_config.eol_tail_weight != 1.0
                ):
                    thr = float(world_model_config.eol_tail_rul_threshold)
                    tail_w = float(world_model_config.eol_tail_weight)
                    weights_eol = torch.where(
                        eol_scalar_target < thr,
                        torch.full_like(eol_scalar_target, tail_w),
                        torch.ones_like(eol_scalar_target),
                    )
                else:
                    weights_eol = torch.ones_like(eol_scalar_target)

                if normalize_eol:
                    scale = float(max(1e-6, eol_scale_value))
                    eol_pred_n = eol_pred / scale
                    eol_true_n = eol_scalar_target / scale
                    per = _eol_loss_per_sample(eol_pred_n, eol_true_n)
                    loss_eol = (per * weights_eol).mean()
                else:
                    loss_eol = ((eol_pred - eol_scalar_target) ** 2 * weights_eol).mean()
                loss_hi_last = mse_loss(hi_pred, target_hi_last)

                hi_seq_pred = traj_pred.squeeze(-1)  # (B, H)
                valid_mask_seq = mask_batch.squeeze(-1) if (use_horizon_mask and mask_batch is not None) else None
                loss_traj, base_hi_mse, mono_raw, smooth_raw = monotonic_health_loss(
                    pred_hi=hi_seq_pred,
                    target_hi=target_hi_seq,
                    alpha=world_model_config.mono_late_weight,
                    beta=world_model_config.mono_global_weight,
                    valid_mask=valid_mask_seq,
                    return_components=True,
                )

                early_slope_raw = hi_early_slope_regularizer(
                    pred_hi=hi_seq_pred,
                    rul_seq=rul_future,
                    valid_mask=valid_mask_seq,
                    epsilon=float(getattr(world_model_config, "hi_early_slope_epsilon", 1e-3)),
                    early_rul_threshold=getattr(world_model_config, "hi_early_slope_rul_threshold", None),
                )
                curv_raw = hi_curvature_loss(
                    pred_hi=hi_seq_pred,
                    abs_mode=bool(getattr(world_model_config, "hi_curvature_abs", True)),
                )
                shape_loss = (
                    float(getattr(world_model_config, "hi_early_slope_weight", 0.0)) * early_slope_raw
                    + float(getattr(world_model_config, "hi_curvature_weight", 0.0)) * curv_raw
                )

                eol_hi_raw = hi_eol_consistency_loss(
                    eol_pred=eol_pred,
                    hi_seq=hi_seq_pred,
                    hi_threshold=float(getattr(world_model_config, "eol_hi_threshold", 0.2)),
                    temperature=float(getattr(world_model_config, "eol_hi_temperature", 0.05)),
                    p_min=float(getattr(world_model_config, "eol_hi_p_min", 0.2)),
                    valid_mask=valid_mask_seq,
                )
                eol_hi_loss = float(getattr(world_model_config, "w_eol_hi", 0.0)) * eol_hi_raw
                
                # Weighted losses
                weighted_traj = world_model_config.traj_loss_weight * loss_traj
                weighted_eol = eol_weight_eff * loss_eol
                weighted_hi = world_model_config.hi_loss_weight * loss_hi_last
                weighted_mono = world_model_config.traj_loss_weight * (
                    world_model_config.mono_late_weight * mono_raw
                    + world_model_config.mono_global_weight * smooth_raw
                )
                weighted_shape = shape_loss
                weighted_eol_hi = eol_hi_loss
                
                loss = weighted_traj + weighted_eol + weighted_hi + weighted_shape + weighted_eol_hi
                
                running_val_loss += loss.item() * X_batch.size(0)
                running_val_traj_loss += base_hi_mse.item() * X_batch.size(0)
                running_val_eol_loss += loss_eol.item() * X_batch.size(0)
                running_val_hi_loss += loss_hi_last.item() * X_batch.size(0)
                running_val_mono_late_loss += mono_raw.item() * X_batch.size(0)
                running_val_mono_global_loss += smooth_raw.item() * X_batch.size(0)
                running_val_hi_early_slope_loss += early_slope_raw.item() * X_batch.size(0)
                running_val_hi_curvature_loss += curv_raw.item() * X_batch.size(0)
                running_val_traj_weighted += weighted_traj.item() * X_batch.size(0)
                running_val_eol_weighted += weighted_eol.item() * X_batch.size(0)
                running_val_hi_weighted += weighted_hi.item() * X_batch.size(0)
                running_val_mono_weighted += weighted_mono.item() * X_batch.size(0)
                running_val_shape_weighted += weighted_shape.item() * X_batch.size(0)
                running_val_eol_hi_loss += eol_hi_raw.item() * X_batch.size(0)
                running_val_eol_hi_weighted += weighted_eol_hi.item() * X_batch.size(0)
                n_val_samples += X_batch.size(0)
        
        epoch_val_loss = running_val_loss / n_val_samples
        epoch_val_traj_loss = running_val_traj_loss / n_val_samples
        epoch_val_eol_loss = running_val_eol_loss / n_val_samples
        epoch_val_hi_loss = running_val_hi_loss / n_val_samples
        epoch_val_mono_late_loss = running_val_mono_late_loss / n_val_samples
        epoch_val_mono_global_loss = running_val_mono_global_loss / n_val_samples
        epoch_val_hi_early_slope_loss = running_val_hi_early_slope_loss / n_val_samples
        epoch_val_hi_curvature_loss = running_val_hi_curvature_loss / n_val_samples
        epoch_val_traj_weighted = running_val_traj_weighted / n_val_samples
        epoch_val_eol_weighted = running_val_eol_weighted / n_val_samples
        epoch_val_hi_weighted = running_val_hi_weighted / n_val_samples
        epoch_val_mono_weighted = running_val_mono_weighted / n_val_samples
        epoch_val_shape_weighted = running_val_shape_weighted / n_val_samples
        epoch_val_eol_hi_loss = running_val_eol_hi_loss / n_val_samples
        epoch_val_eol_hi_weighted = running_val_eol_hi_weighted / n_val_samples
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_traj_loss"].append(epoch_val_traj_loss)
        history["val_eol_loss"].append(epoch_val_eol_loss)
        history["val_hi_loss"].append(epoch_val_hi_loss)
        history["val_mono_late_loss"].append(epoch_val_mono_late_loss)
        history["val_mono_global_loss"].append(epoch_val_mono_global_loss)
        history["val_hi_early_slope_loss"].append(epoch_val_hi_early_slope_loss)
        history["val_hi_curvature_loss"].append(epoch_val_hi_curvature_loss)
        history["val_traj_weighted"].append(epoch_val_traj_weighted)
        history["val_eol_weighted"].append(epoch_val_eol_weighted)
        history["val_hi_weighted"].append(epoch_val_hi_weighted)
        history["val_mono_weighted"].append(epoch_val_mono_weighted)
        history["val_shape_weighted"].append(epoch_val_shape_weighted)
        history["val_eol_hi_loss"].append(epoch_val_eol_hi_loss)
        history["val_eol_hi_weighted"].append(epoch_val_eol_hi_weighted)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"train: {epoch_train_loss:.4f}, val: {epoch_val_loss:.4f}, "
            f"val_traj: {epoch_val_traj_loss:.4f} (w: {epoch_val_traj_weighted:.4f}), "
            f"val_eol: {epoch_val_eol_loss:.4f} (w: {epoch_val_eol_weighted:.4f}, mult={eol_mult:.2f}), "
            f"val_hi: {epoch_val_hi_loss:.4f} (w: {epoch_val_hi_weighted:.4f}), "
            f"val_mono: {epoch_val_mono_late_loss:.4f}+{epoch_val_mono_global_loss:.4f} (w: {epoch_val_mono_weighted:.4f}), "
            f"val_shape: {epoch_val_hi_early_slope_loss:.4f}+{epoch_val_hi_curvature_loss:.4f} (w: {epoch_val_shape_weighted:.4f}), "
            f"val_eol_hi: {epoch_val_eol_hi_loss:.4f} (w: {epoch_val_eol_hi_weighted:.4f})"
        )

        # --------------------------------------------------
        # Best checkpoint selection + early stopping (optionally EOL-aware)
        # --------------------------------------------------
        allow_best_update = (not select_best_after_eol_active) or bool(eol_active)

        # Choose which metric to optimize for best checkpoint
        if best_metric == "val_total":
            metric_val = float(epoch_val_loss)
        elif best_metric == "val_eol_weighted":
            metric_val = float(epoch_val_eol_weighted)
        elif best_metric == "val_eol":
            metric_val = float(epoch_val_eol_loss)
        else:
            metric_val = float(epoch_val_loss)

        print(
            f"[checkpoint] eol_active={eol_active} mult={eol_mult:.3f} "
            f"allow_best_update={allow_best_update} best_metric={best_metric} metric={metric_val:.4f}"
        )

        # If gating is enabled, don't start patience until EOL becomes active
        if select_best_after_eol_active and (not eol_active):
            continue

        # If EOL just turned on, reset patience/best so we don't keep the pre-EOL best.
        if select_best_after_eol_active and eol_became_active_epoch is not None and epoch == eol_became_active_epoch:
            best_val_loss = float("inf")
            epochs_no_improve = 0

        if allow_best_update and metric_val < best_val_loss:
            best_val_loss = metric_val
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": metric_val,
                    "best_metric": best_metric,
                    "input_dim": len(feature_cols),
                    "d_model": d_model,
                    "num_conditions": num_conditions,
                    "encoder_type": "world_model_universal_v3",
                    "eol_mult": float(eol_mult),
                    "eol_active_min_mult": float(eol_active_min_mult),
                },
                best_model_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(
        f"\n[7] Best model loaded from epoch {checkpoint['epoch']+1} "
        f"(best_metric={checkpoint.get('best_metric','val_total')}, val={checkpoint['val_loss']:.4f})"
    )
    
    # ===================================================================
    # 7. Evaluate on test set
    # ===================================================================
    print("\n[8] Evaluating on test set...")
    test_metrics = evaluate_world_model_v3_eol(
        model=model,
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        scaler_dict=scaler_dict,
        past_len=past_len,
        max_rul=max_rul,
        num_conditions=num_conditions,
        device=device,
        clip_y_true_to_max_rul=bool(getattr(world_model_config, "eval_clip_y_true_to_max_rul", False)),
        window_cfg=window_cfg,
        target_cfg=target_cfg,
    )
    
    print(f"  Test RMSE: {test_metrics['RMSE']:.2f} cycles")
    print(f"  Test MAE:  {test_metrics['MAE']:.2f} cycles")
    print(f"  Test Bias: {test_metrics['Bias']:.2f} cycles")
    print(f"  Test R²:   {test_metrics.get('R2', 0.0):.4f}")
    print(f"  NASA Score (mean): {test_metrics['nasa_score_mean']:.2f}")
    
    # ===================================================================
    # 8. Compute per-condition metrics
    # ===================================================================
    print("\n[9] Computing per-condition metrics...")
    condition_metrics = {}
    
    df_test_cond = df_test.groupby("UnitNumber")["ConditionID"].first()
    unit_ids_test = sorted(df_test["UnitNumber"].unique())
    
    y_pred_eol = np.array(test_metrics.get("y_pred_eol", []))
    y_true_eol = np.array(test_metrics.get("y_true_eol", y_test_true))
    
    if len(y_pred_eol) > 0 and len(y_pred_eol) == len(unit_ids_test):
        for cond_id in unique_conditions:
            cond_id = int(cond_id)
            cond_engines = df_test_cond[df_test_cond == cond_id].index.values
            if len(cond_engines) == 0:
                continue
            
            cond_indices = [i for i, uid in enumerate(unit_ids_test) if uid in cond_engines]
            if len(cond_indices) == 0:
                continue
            
            cond_y_true = y_true_eol[cond_indices]
            cond_y_pred = y_pred_eol[cond_indices]
            
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
        "model_version": "v3",
        "model_type": "world_model_universal_v3",
        "encoder_type": "universal_v3",
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
        "window_cfg": {
            "past_len": int(window_cfg.past_len),
            "horizon": int(window_cfg.horizon),
            "stride": int(window_cfg.stride),
            "pad_mode": str(window_cfg.pad_mode),
            "require_full_horizon": bool(window_cfg.require_full_horizon),
        },
        "target_cfg": {
            "max_rul": int(target_cfg.max_rul),
            "cap_targets": bool(target_cfg.cap_targets),
            "eol_target_mode": str(target_cfg.eol_target_mode),
            "clip_eval_y_true": bool(target_cfg.clip_eval_y_true),
        },
        "world_model_config": {
            "forecast_horizon": horizon,
            "traj_loss_weight": world_model_config.traj_loss_weight,
            "eol_loss_weight": world_model_config.eol_loss_weight,
            "hi_loss_weight": world_model_config.hi_loss_weight,
            "mono_late_weight": world_model_config.mono_late_weight,
            "mono_global_weight": world_model_config.mono_global_weight,
            "traj_step_weighting": world_model_config.traj_step_weighting,
            "use_condition_wise_scaling": use_condition_wise_scaling,
            "use_hi_in_eol": world_model_config.use_hi_in_eol,
            "use_hi_slope_in_eol": world_model_config.use_hi_slope_in_eol,
            "eol_tail_rul_threshold": world_model_config.eol_tail_rul_threshold,
            "eol_tail_weight": world_model_config.eol_tail_weight,
            # Stage-1 schedule + shape losses (may be absent in older configs)
            "three_phase_schedule": bool(getattr(world_model_config, "three_phase_schedule", False)),
            "phase_a_frac": float(getattr(world_model_config, "phase_a_frac", 0.2)),
            "phase_b_end_frac": float(getattr(world_model_config, "phase_b_end_frac", 0.8)),
            "phase_b_frac": getattr(world_model_config, "phase_b_frac", None),
            "schedule_type": str(getattr(world_model_config, "schedule_type", getattr(world_model_config, "eol_ramp", "linear"))),
            "eol_w_max": float(getattr(world_model_config, "eol_w_max", 1.0)),
            # EOL ramp stabilization (FD004 default-on via experiment config)
            "normalize_eol": bool(getattr(world_model_config, "normalize_eol", False)),
            "eol_scale": getattr(world_model_config, "eol_scale", "rul_cap"),
            "eol_loss_type": str(getattr(world_model_config, "eol_loss_type", "mse")),
            "eol_huber_beta": float(getattr(world_model_config, "eol_huber_beta", 0.1)),
            "clip_grad_norm": getattr(world_model_config, "clip_grad_norm", None),
            "freeze_encoder_epochs_after_eol_on": int(getattr(world_model_config, "freeze_encoder_epochs_after_eol_on", 0) or 0),
            # Selection / alignment
            "select_best_after_eol_active": bool(getattr(world_model_config, "select_best_after_eol_active", False)),
            "eol_active_min_mult": float(getattr(world_model_config, "eol_active_min_mult", 0.01)),
            "best_metric": str(getattr(world_model_config, "best_metric", "val_total")),
            "cap_rul_targets_to_max_rul": bool(getattr(world_model_config, "cap_rul_targets_to_max_rul", False)),
            "eol_target_mode": str(getattr(world_model_config, "eol_target_mode", "future0")),
            "eval_clip_y_true_to_max_rul": bool(getattr(world_model_config, "eval_clip_y_true_to_max_rul", False)),
            "init_eol_bias_to_target_mean": bool(getattr(world_model_config, "init_eol_bias_to_target_mean", False)),
            # Horizon targets
            "use_padded_horizon_targets": bool(getattr(world_model_config, "use_padded_horizon_targets", False)),
            "target_clamp_min": float(getattr(world_model_config, "target_clamp_min", 0.0)),
            "use_horizon_mask": bool(getattr(world_model_config, "use_horizon_mask", False)),
            "hi_early_slope_weight": float(getattr(world_model_config, "hi_early_slope_weight", 0.0)),
            "hi_early_slope_epsilon": float(getattr(world_model_config, "hi_early_slope_epsilon", 1e-3)),
            "hi_early_slope_rul_threshold": getattr(world_model_config, "hi_early_slope_rul_threshold", None),
            "hi_curvature_weight": float(getattr(world_model_config, "hi_curvature_weight", 0.0)),
            "hi_curvature_abs": bool(getattr(world_model_config, "hi_curvature_abs", True)),
            # Stage-2 optional: HI→EOL consistency (WorldModel)
            "w_eol_hi": float(getattr(world_model_config, "w_eol_hi", 0.0)),
            "eol_hi_threshold": float(getattr(world_model_config, "eol_hi_threshold", 0.2)),
            "eol_hi_temperature": float(getattr(world_model_config, "eol_hi_temperature", 0.05)),
            "eol_hi_p_min": float(getattr(world_model_config, "eol_hi_p_min", 0.2)),
        },
        "test_metrics": {
            "rmse": test_metrics["RMSE"],
            "mae": test_metrics["MAE"],
            "bias": test_metrics["Bias"],
            "r2": test_metrics.get("R2", 0.0),
            "nasa_mean": test_metrics["nasa_score_mean"],
            "nasa_sum": test_metrics["nasa_score_sum"],
            "num_engines": test_metrics["num_engines"],
        },
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
    
    # Save full training history
    training_history_path = results_dir / "training_history.json"
    with open(training_history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history to {training_history_path}")
    
    # Plot training curves
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="red")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component losses (raw)
        axes[0, 1].plot(epochs, history["val_traj_loss"], label="Traj", color="green")
        axes[0, 1].plot(epochs, history["val_eol_loss"], label="EOL", color="orange")
        axes[0, 1].plot(epochs, history["val_hi_loss"], label="HI", color="purple")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Component Losses (Raw)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Weighted losses
        axes[0, 2].plot(epochs, history["val_traj_weighted"], label=f"Traj (×{world_model_config.traj_loss_weight:.1f})", color="green", linestyle="--")
        axes[0, 2].plot(epochs, history["val_eol_weighted"], label=f"EOL (×{world_model_config.eol_loss_weight:.1f})", color="orange", linestyle="--")
        axes[0, 2].plot(epochs, history["val_hi_weighted"], label=f"HI (×{world_model_config.hi_loss_weight:.1f})", color="purple", linestyle="--")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Weighted Loss")
        axes[0, 2].set_title("Validation Weighted Losses")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Monotonicity losses
        axes[1, 0].plot(epochs, history["val_mono_late_loss"], label="Late Mono", color="red")
        axes[1, 0].plot(epochs, history["val_mono_global_loss"], label="Global Mono", color="blue")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Validation Monotonicity Losses")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined weighted view
        axes[1, 1].plot(epochs, history["val_traj_weighted"], label="Traj", color="green", alpha=0.7)
        axes[1, 1].plot(epochs, history["val_eol_weighted"], label="EOL", color="orange", alpha=0.7)
        axes[1, 1].plot(epochs, history["val_hi_weighted"], label="HI", color="purple", alpha=0.7)
        axes[1, 1].plot(epochs, history["val_mono_weighted"], label="Mono", color="red", alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Weighted Loss")
        axes[1, 1].set_title("All Weighted Losses")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Total loss breakdown
        axes[1, 2].plot(epochs, history["val_loss"], label="Total Val Loss", color="black", linewidth=2)
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Loss")
        axes[1, 2].set_title("Total Validation Loss")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
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


def train_transformer_world_model_v1(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    dataset_name: str,
    experiment_name: str,
    world_model_config: WorldModelTrainingConfig,
    encoder_d_model: int = 64,
    encoder_num_layers: int = 3,
    encoder_nhead: int = 4,
    encoder_dim_feedforward: int = 256,
    encoder_dropout: float = 0.1,
    num_sensors_out: int = 21,
    cond_dim: int = 9,
    batch_size: int = 128,
    num_epochs: int = 60,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train Transformer World Model V1 on FD004 (or compatible datasets).

    - Encoder: EOLFullTransformerEncoder with ms+DT feature vectors (e.g. 349D).
    - Decoder: GRU-based world model (TransformerWorldModelV1) predicting
      future sensor trajectories (and optionally future HI/RUL).

    This is an initial V1 implementation focused primarily on future sensor
    forecasting; HI/RUL future heads are optional and can be refined later.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if results_dir is None:
        results_dir = Path("results") / dataset_name.lower() / experiment_name
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    past_len = world_model_config.past_len
    horizon = world_model_config.forecast_horizon

    print(f"\n{'='*80}")
    print(f"Training Transformer World Model V1: {experiment_name}")
    print(f"Dataset: {dataset_name}, Features: {len(feature_cols)}")
    print(f"  Horizon: {horizon}, Past len: {past_len}")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # Build seq2seq dataset for future sensor, RUL and HI prediction
    # ------------------------------------------------------------------
    sensor_cols = [c for c in feature_cols if c.startswith("Sensor")]
    if len(sensor_cols) < num_sensors_out:
        # Fallback: use whatever sensor columns are available
        num_sensors_out = len(sensor_cols)
    target_sensor_cols = sensor_cols[:num_sensors_out]

    # ------------------------------------------------------------------
    # Optional sensor scaling: put sensor trajectories into a normalized
    # space (same scaler used for inputs X_past and targets Y_future_sens).
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler

    sensor_scaler = StandardScaler()
    sensor_values = df_train[target_sensor_cols].to_numpy(dtype=np.float32, copy=True)
    sensor_scaler.fit(sensor_values)

    # Apply scaling to the training DataFrame for sensor columns only.
    df_train_scaled = df_train.copy()
    df_train_scaled[target_sensor_cols] = sensor_scaler.transform(sensor_values)

    from src.data.windowing import WindowConfig, TargetConfig, build_sliding_windows

    max_rul = world_model_config.max_rul
    max_visible_rul = float(max_rul if max_rul is not None else 125.0)

    # Continuous condition-feature columns (Cond_*) used for optional future_conds.
    cond_cols = [c for c in feature_cols if c.startswith("Cond_")]
    cond_dim_actual = len(cond_cols)
    cond_dim_model = int(cond_dim_actual) if cond_dim_actual > 0 else int(cond_dim)
    if cond_dim_actual > 0 and cond_dim_actual != cond_dim:
        print(
            f"[WorldModelV1] Warning: cond_dim={cond_dim} but found {cond_dim_actual} Cond_* columns "
            f"in feature_cols. Using {cond_dim_actual} for future_conds."
        )

    # Central window builder with end-padding (includes near-EOL samples)
    wc = WindowConfig(
        past_len=int(past_len),
        horizon=int(horizon),
        stride=1,
        require_full_horizon=False,
        pad_mode="clamp",
    )
    tc = TargetConfig(
        max_rul=int(max_rul if max_rul is not None else 125),
        cap_targets=True,
        # For WM-V1 training, the scalar used for init/anchors comes from future[0]
        # (consistent with the pre-existing code).
        eol_target_mode="future0",
        clip_eval_y_true=False,
    )

    # One call returns padded future RUL targets AND (optionally) future sensor + cond sequences.
    future_cols = target_sensor_cols + (cond_cols if cond_dim_actual > 0 else [])
    built = build_sliding_windows(
        df=df_train_scaled,
        feature_cols=feature_cols,
        target_col="RUL",
        future_feature_cols=future_cols if future_cols else None,
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=wc,
        target_cfg=tc,
        return_mask=True,
    )

    X_train_np = built["X"]  # (N,past_len,F)
    # Future RUL (capped/padded) in cycles:
    future_rul = built["Y_seq"].squeeze(-1)  # (N,H)
    # Normalize RUL future trajectory to [0,1]
    Y_rul_np = (future_rul / float(max_rul if max_rul is not None else 1.0)).astype(np.float32)

    # HI mapping from future_rul (cycles)
    hi_linear = np.clip(future_rul / max_visible_rul, 0.0, 1.0)
    Y_hi_np = np.where(future_rul > max_visible_rul, 1.0, hi_linear).astype(np.float32)

    # Future sensor/cond sequences
    fut = built.get("future_features")
    if fut is None:
        raise ValueError("[WorldModelV1] build_sliding_windows did not return future_features; check future_feature_cols.")
    n_sens = len(target_sensor_cols)
    Y_sens_np = fut[:, :, :n_sens].astype(np.float32)
    if cond_dim_actual > 0:
        future_cond_np = fut[:, :, n_sens:].astype(np.float32)
    else:
        future_cond_np = np.zeros((X_train_np.shape[0], horizon, cond_dim_model), dtype=np.float32)

    cond_id_list = built["cond_ids"].tolist()
    unit_ids_np = built["unit_ids"]

    # ------------------------------------------------------------------
    # Debug: dataset-level normalization statistics
    # ------------------------------------------------------------------
    print("=== WorldModelV1 Debug: Dataset stats ===")
    print(
        "X_train_np:  mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
            float(X_train_np.mean()),
            float(X_train_np.std()),
            float(X_train_np.min()),
            float(X_train_np.max()),
        )
    )
    print(
        "Y_sens_np:   mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
            float(Y_sens_np.mean()),
            float(Y_sens_np.std()),
            float(Y_sens_np.min()),
            float(Y_sens_np.max()),
        )
    )
    print(
        "Y_rul_np:    mean {:.3f}, min {:.3f}, max {:.3f}".format(
            float(Y_rul_np.mean()),
            float(Y_rul_np.min()),
            float(Y_rul_np.max()),
        )
    )
    print(
        "Y_hi_np:     mean {:.3f}, min {:.3f}, max {:.3f}".format(
            float(Y_hi_np.mean()),
            float(Y_hi_np.min()),
            float(Y_hi_np.max()),
        )
    )
    if future_cond_np.size > 0:
        print(
            "future_cond_np: mean {:.3f}, std {:.3f}".format(
                float(future_cond_np.mean()), float(future_cond_np.std())
            )
        )
    print("=========================================")

    X_train = torch.from_numpy(X_train_np).float()
    Y_sens_train = torch.from_numpy(Y_sens_np).float()
    Y_rul_train = torch.from_numpy(Y_rul_np).float()
    Y_hi_train = torch.from_numpy(Y_hi_np).float()
    future_cond_train = torch.from_numpy(future_cond_np).float()
    cond_ids = torch.tensor(cond_id_list, dtype=torch.long)

    # Engine-level train/val split (no leakage across engines)
    N = X_train.shape[0]
    unit_ids_t = torch.from_numpy(unit_ids_np.astype(np.int64))
    uniq = torch.unique(unit_ids_t)
    gen = torch.Generator().manual_seed(42)
    perm = uniq[torch.randperm(len(uniq), generator=gen)]
    n_val_units = max(1, int(0.2 * len(uniq)))
    val_units = perm[:n_val_units]
    train_units = perm[n_val_units:]
    train_mask = torch.isin(unit_ids_t, train_units)
    val_mask = torch.isin(unit_ids_t, val_units)
    train_indices = torch.nonzero(train_mask, as_tuple=False).view(-1).cpu().numpy()
    val_indices = torch.nonzero(val_mask, as_tuple=False).view(-1).cpu().numpy()

    X_tr = X_train[train_indices]
    Y_sens_tr = Y_sens_train[train_indices]
    Y_rul_tr = Y_rul_train[train_indices]
    Y_hi_tr = Y_hi_train[train_indices]
    future_cond_tr = future_cond_train[train_indices]
    cond_tr = cond_ids[train_indices]

    X_val = X_train[val_indices]
    Y_sens_val = Y_sens_train[val_indices]
    Y_rul_val = Y_rul_train[val_indices]
    Y_hi_val = Y_hi_train[val_indices]
    future_cond_val = future_cond_train[val_indices]
    cond_val = cond_ids[val_indices]

    train_ds = torch.utils.data.TensorDataset(
        X_tr, Y_sens_tr, Y_rul_tr, Y_hi_tr, cond_tr, future_cond_tr
    )
    val_ds = torch.utils.data.TensorDataset(
        X_val, Y_sens_val, Y_rul_val, Y_hi_val, cond_val, future_cond_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # Build encoder and world model
    # ------------------------------------------------------------------
    input_dim = len(feature_cols)
    print(f"[WorldModelV1] Using input_dim={input_dim}, num_sensors_out={num_sensors_out}, cond_dim={cond_dim}")

    encoder = EOLFullTransformerEncoder(
        input_dim=input_dim,
        d_model=encoder_d_model,
        num_layers=encoder_num_layers,
        n_heads=encoder_nhead,
        dim_feedforward=encoder_dim_feedforward,
        dropout=encoder_dropout,
        use_condition_embedding=True,
        num_conditions=7,
        cond_emb_dim=4,
        max_seq_len=300,
    ).to(device)

    # Optional: load pretrained encoder weights (e.g. from ms+DT encoder run)
    encoder_ckpt_path = getattr(world_model_config, "encoder_checkpoint", None)
    if encoder_ckpt_path:
        try:
            print(f"[WorldModelV1] Loading encoder weights from: {encoder_ckpt_path}")
            state = torch.load(encoder_ckpt_path, map_location=device)
            # Accept either a full checkpoint or a plain state_dict
            if isinstance(state, dict) and "model_state_dict" in state:
                encoder.load_state_dict(state["model_state_dict"], strict=False)
            else:
                encoder.load_state_dict(state, strict=False)
            print("[WorldModelV1] Encoder weights loaded successfully.")
        except Exception as e:
            print(f"[WorldModelV1] Warning: failed to load encoder checkpoint '{encoder_ckpt_path}': {e}")
    else:
        print("[WorldModelV1] No encoder_checkpoint provided – training encoder from scratch.")

    # World-model specific flags/params
    target_mode = getattr(world_model_config, "target_mode", "sensors")
    init_from_rul_hi = getattr(world_model_config, "init_from_rul_hi", False)
    decoder_hidden_dim = int(getattr(world_model_config, "decoder_hidden_dim", 256))
    num_layers_decoder = int(getattr(world_model_config, "num_layers_decoder", 1))
    freeze_encoder = bool(getattr(world_model_config, "freeze_encoder", False))

    # Dynamic latent flags for Branch A+ (default False for backwards compatibility)
    use_latent_history = bool(getattr(world_model_config, "use_latent_history", False))
    use_hi_anchor = bool(getattr(world_model_config, "use_hi_anchor", False))
    use_future_conds = bool(getattr(world_model_config, "use_future_conds", False))
    use_eol_fusion = bool(getattr(world_model_config, "use_eol_fusion", False))
    eol_fusion_mode = str(getattr(world_model_config, "eol_fusion_mode", "token"))
    predict_latent = bool(getattr(world_model_config, "predict_latent", False))
    latent_decoder_type = str(getattr(world_model_config, "latent_decoder_type", "gru"))
    latent_decoder_num_layers = int(getattr(world_model_config, "latent_decoder_num_layers", 2))
    latent_decoder_nhead = int(getattr(world_model_config, "latent_decoder_nhead", 4))

    # Staged training (default OFF)
    freeze_encoder_epochs = int(getattr(world_model_config, "freeze_encoder_epochs", 0) or 0)
    unfreeze_encoder_layers = int(getattr(world_model_config, "unfreeze_encoder_layers", 0) or 0)
    encoder_lr_mult = float(getattr(world_model_config, "encoder_lr_mult", 0.1) or 0.1)
    eol_scalar_loss_weight = float(getattr(world_model_config, "eol_scalar_loss_weight", 0.0) or 0.0)

    world_model = TransformerWorldModelV1(
        encoder=encoder,
        input_dim=input_dim,
        num_sensors_out=num_sensors_out,
        cond_dim=cond_dim_model,
        future_horizon=horizon,
        decoder_hidden_dim=decoder_hidden_dim,
        num_layers_decoder=num_layers_decoder,
        dropout=encoder_dropout,
        predict_hi=True,
        predict_rul=True,
        target_mode=target_mode,
        init_from_rul_hi=init_from_rul_hi,
        use_latent_history=use_latent_history,
        use_hi_anchor=use_hi_anchor,
        use_future_conds=use_future_conds,
        use_eol_fusion=use_eol_fusion,
        eol_fusion_mode=eol_fusion_mode,
        predict_latent=predict_latent,
        latent_decoder_type=latent_decoder_type,
        latent_decoder_num_layers=latent_decoder_num_layers,
        latent_decoder_nhead=latent_decoder_nhead,
    ).to(device)

    def _set_encoder_trainability(*, frozen: bool, unfreeze_last_k: int = 0) -> None:
        # Freeze everything first
        for p in world_model.encoder.parameters():
            p.requires_grad = not frozen
        if frozen:
            world_model.encoder.eval()
            return

        # Partial unfreeze: freeze all then unfreeze last K transformer layers if possible
        if unfreeze_last_k > 0:
            for p in world_model.encoder.parameters():
                p.requires_grad = False
            # Common structure: encoder.transformer.layers is a ModuleList
            layers = None
            if hasattr(world_model.encoder, "transformer") and hasattr(world_model.encoder.transformer, "layers"):
                layers = world_model.encoder.transformer.layers
            if layers is not None:
                k = min(int(unfreeze_last_k), len(layers))
                for layer in list(layers)[-k:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            else:
                # Fallback: unfreeze whole encoder if we can't identify layers
                for p in world_model.encoder.parameters():
                    p.requires_grad = True
        world_model.encoder.train()

    # Stage A: freeze encoder (either explicit freeze_encoder, or for first N epochs)
    if freeze_encoder or freeze_encoder_epochs > 0:
        print(
            f"[WorldModelV1] Stage-A encoder freeze enabled (freeze_encoder={freeze_encoder}, "
            f"freeze_encoder_epochs={freeze_encoder_epochs})"
        )
        _set_encoder_trainability(frozen=True)
    print(
        f"[WorldModelV1] Encoder configuration -> "
        f"checkpoint: {encoder_ckpt_path if encoder_ckpt_path else 'None'}, "
        f"freeze_encoder: {freeze_encoder}, freeze_encoder_epochs: {freeze_encoder_epochs}, "
        f"unfreeze_encoder_layers: {unfreeze_encoder_layers}, encoder_lr_mult: {encoder_lr_mult}"
    )

    def _make_optimizer() -> torch.optim.Optimizer:
        enc_params = [p for p in world_model.encoder.parameters() if p.requires_grad]
        other_params = [p for n, p in world_model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad]
        if enc_params:
            return torch.optim.Adam(
                [
                    {"params": other_params, "lr": lr},
                    {"params": enc_params, "lr": lr * float(encoder_lr_mult)},
                ],
                weight_decay=weight_decay,
            )
        return torch.optim.Adam(other_params, lr=lr, weight_decay=weight_decay)

    optimizer = _make_optimizer()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    sensor_w = world_model_config.__dict__.get("sensor_loss_weight", 1.0)
    hi_w = world_model_config.__dict__.get("hi_future_loss_weight", 0.0)
    rul_w = world_model_config.__dict__.get("rul_future_loss_weight", 0.0)

    for epoch in range(num_epochs):
        # Stage B: unfreeze encoder after N epochs (if configured and not permanently frozen)
        if (not freeze_encoder) and freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            print(f"[WorldModelV1] Stage-B unfreeze at epoch {epoch+1}: unfreeze_last_k={unfreeze_encoder_layers}")
            _set_encoder_trainability(frozen=False, unfreeze_last_k=unfreeze_encoder_layers)
            optimizer = _make_optimizer()

        world_model.train()
        running_train = 0.0
        n_train_samples = 0

        for batch_idx, (X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b) in enumerate(train_loader):
            X_b = X_b.to(device)
            Y_sens_b = Y_sens_b.to(device)
            Y_rul_b = Y_rul_b.to(device)
            Y_hi_b = Y_hi_b.to(device)
            cond_b = cond_b.to(device)
            future_cond_b = future_cond_b.to(device)

            if epoch == 0 and batch_idx == 0:
                print("=== WorldModelV1 Debug: First batch stats ===")
                print(
                    "X_b:       mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
                        float(X_b.mean()), float(X_b.std()),
                        float(X_b.min()), float(X_b.max())
                    )
                )
                print(
                    "Y_sens_b:  mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
                        float(Y_sens_b.mean()), float(Y_sens_b.std()),
                        float(Y_sens_b.min()), float(Y_sens_b.max())
                    )
                )
                print(
                    "Y_rul_b:   mean {:.3f}, min {:.3f}, max {:.3f}".format(
                        float(Y_rul_b.mean()), float(Y_rul_b.min()), float(Y_rul_b.max())
                    )
                )
                print(
                    "Y_hi_b:    mean {:.3f}, min {:.3f}, max {:.3f}".format(
                        float(Y_hi_b.mean()), float(Y_hi_b.min()), float(Y_hi_b.max())
                    )
                )
                print("============================================")

            optimizer.zero_grad()

            # For now we don't pass a continuous condition vector – caller should
            # precompute Cond_* and pass here; as a first version we use zeros.
            cond_vec = torch.zeros(X_b.size(0), cond_dim_model, device=device)

            # Use the first future step as "current" RUL/HI for decoder init
            current_rul_b = Y_rul_b[:, 0]  # (B,)
            current_hi_b = Y_hi_b[:, 0]    # (B,)

            if target_mode in ["latent_hi_rul", "latent_hi_rul_dynamic_delta_v2"]:
                # Dynamic latent world model: no sensor teacher forcing, optional future_conds + HI anchor.
                out = world_model(
                    past_seq=X_b,
                    cond_vec=cond_vec,
                    cond_ids=cond_b,
                    future_horizon=horizon,
                    teacher_forcing_targets=None,
                    current_rul=current_rul_b if use_hi_anchor else None,
                    current_hi=current_hi_b if use_hi_anchor else None,
                    future_conds=future_cond_b if use_future_conds else None,
                )
                if isinstance(out, (tuple, list)) and len(out) == 4:
                    pred_sensors, pred_hi, pred_rul, pred_eol = out
                else:
                    pred_sensors, pred_hi, pred_rul = out
                    pred_eol = None
                if epoch == 0 and batch_idx == 0:
                    print(
                        f"[WorldModelV1][shapes] X_b={tuple(X_b.shape)} future_cond_b={tuple(future_cond_b.shape)} "
                        f"pred_hi={(tuple(pred_hi.shape) if pred_hi is not None else None)} "
                        f"pred_rul={(tuple(pred_rul.shape) if pred_rul is not None else None)} "
                        f"pred_eol={(tuple(pred_eol.shape) if pred_eol is not None else None)}"
                    )
            else:
                # Original sensor-autoregressive world model path.
                out = world_model(
                    past_seq=X_b,
                    cond_vec=cond_vec,
                    cond_ids=cond_b,
                    future_horizon=horizon,
                    teacher_forcing_targets=Y_sens_b,
                    current_rul=current_rul_b,
                    current_hi=current_hi_b,
                )
                if isinstance(out, (tuple, list)) and len(out) == 4:
                    pred_sensors, pred_hi, pred_rul, pred_eol = out
                else:
                    pred_sensors, pred_hi, pred_rul = out
                    pred_eol = None

            # Sensor trajectory loss
            loss_sensors = F.mse_loss(pred_sensors, Y_sens_b)
            loss = sensor_w * loss_sensors

            # HI future loss (piecewise HI targets already built into Y_hi_b)
            if hi_w > 0.0 and pred_hi is not None:
                # pred_hi: (B, H, 1), Y_hi_b: (B, H)
                hi_loss = F.mse_loss(pred_hi.squeeze(-1), Y_hi_b)
                loss = loss + hi_w * hi_loss

            # RUL future loss (L1 on cycles)
            if rul_w > 0.0 and pred_rul is not None:
                # pred_rul: (B, H, 1), Y_rul_b: (B, H)
                rul_loss = F.l1_loss(pred_rul.squeeze(-1), Y_rul_b)
                loss = loss + rul_w * rul_loss

            # Optional: supervise the predicted EOL scalar (in [0,1]) against normalized current RUL
            if eol_scalar_loss_weight > 0.0 and pred_eol is not None:
                # current_rul_b is already normalized in Y_rul_b[:,0]
                eol_loss = F.mse_loss(pred_eol.view(-1), current_rul_b.view(-1))
                loss = loss + eol_scalar_loss_weight * eol_loss

            loss.backward()
            optimizer.step()

            running_train += loss.item() * X_b.size(0)
            n_train_samples += X_b.size(0)

        train_loss = running_train / max(1, n_train_samples)

        # Validation
        world_model.eval()
        running_val = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b in val_loader:
                X_b = X_b.to(device)
                Y_sens_b = Y_sens_b.to(device)
                Y_rul_b = Y_rul_b.to(device)
                Y_hi_b = Y_hi_b.to(device)
                cond_b = cond_b.to(device)
                future_cond_b = future_cond_b.to(device)
                cond_vec = torch.zeros(X_b.size(0), cond_dim, device=device)

                current_rul_b = Y_rul_b[:, 0]
                current_hi_b = Y_hi_b[:, 0]

                if target_mode in ["latent_hi_rul", "latent_hi_rul_dynamic_delta_v2"]:
                    out = world_model(
                        past_seq=X_b,
                        cond_vec=cond_vec,
                        cond_ids=cond_b,
                        future_horizon=horizon,
                        teacher_forcing_targets=None,
                        current_rul=current_rul_b if use_hi_anchor else None,
                        current_hi=current_hi_b if use_hi_anchor else None,
                        future_conds=future_cond_b if use_future_conds else None,
                    )
                    if isinstance(out, (tuple, list)) and len(out) == 4:
                        pred_sensors, pred_hi, pred_rul, pred_eol = out
                    else:
                        pred_sensors, pred_hi, pred_rul = out
                        pred_eol = None
                else:
                    out = world_model(
                        past_seq=X_b,
                        cond_vec=cond_vec,
                        cond_ids=cond_b,
                        future_horizon=horizon,
                        teacher_forcing_targets=Y_sens_b,
                        current_rul=current_rul_b,
                        current_hi=current_hi_b,
                    )
                    if isinstance(out, (tuple, list)) and len(out) == 4:
                        pred_sensors, pred_hi, pred_rul, pred_eol = out
                    else:
                        pred_sensors, pred_hi, pred_rul = out
                        pred_eol = None

                loss_sensors = F.mse_loss(pred_sensors, Y_sens_b)
                loss = sensor_w * loss_sensors

                if hi_w > 0.0 and pred_hi is not None:
                    hi_loss = F.mse_loss(pred_hi.squeeze(-1), Y_hi_b)
                    loss = loss + hi_w * hi_loss

                if rul_w > 0.0 and pred_rul is not None:
                    rul_loss = F.l1_loss(pred_rul.squeeze(-1), Y_rul_b)
                    loss = loss + rul_w * rul_loss

                if eol_scalar_loss_weight > 0.0 and pred_eol is not None:
                    eol_loss = F.mse_loss(pred_eol.view(-1), current_rul_b.view(-1))
                    loss = loss + eol_scalar_loss_weight * eol_loss

                running_val += loss.item() * X_b.size(0)
                n_val_samples += X_b.size(0)

        val_loss = running_val / max(1, n_val_samples)

        print(
            f"[WorldModelV1] Epoch {epoch+1}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in world_model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[WorldModelV1] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        world_model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint_path = results_dir / f"transformer_world_model_v1_best_{experiment_name}.pt"
    torch.save(
        {
            "model_state_dict": world_model.state_dict(),
            "val_loss": best_val_loss,
            "input_dim": input_dim,
            "num_sensors_out": num_sensors_out,
            "cond_dim": cond_dim,
            "horizon": horizon,
        },
        checkpoint_path,
    )
    print(f"[WorldModelV1] Saved best model to {checkpoint_path}")

    # Save sensor scaler used for normalizing sensor trajectories so that
    # diagnostics / rollouts can de-normalize back to physical units.
    try:
        import pickle

        scaler_path = results_dir / "world_model_v1_sensor_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(
                {
                    "sensor_cols": target_sensor_cols,
                    "scaler_state": {
                        "mean_": sensor_scaler.mean_,
                        "scale_": sensor_scaler.scale_,
                    },
                },
                f,
            )
        print(f"[WorldModelV1] Saved sensor scaler to {scaler_path}")
    except Exception as e:
        print(f"[WorldModelV1] Warning: could not save sensor scaler: {e}")

    # ------------------------------------------------------------------
    # EOL-style test evaluation on FD004 test set (literature-style RUL metrics)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"World Model Experiment Complete: {experiment_name}")
    print("=" * 80)
    print("Test Metrics: (literature-style EOL metrics for RUL)")

    from typing import Dict

    # Ensure df_test has an RUL column for evaluation (per-unit, per-cycle),
    # using the same convention as training: RUL_raw = MaxTime - TimeInCycles,
    # clipped to [0, max_rul]. This yields a monotonically decreasing RUL
    # towards 0 at the last observed cycle.
    df_test_for_eval = df_test.copy()
    if "RUL" not in df_test_for_eval.columns:
        df_max_time = (
            df_test_for_eval.groupby("UnitNumber")["TimeInCycles"]
            .max()
            .reset_index()
            .rename(columns={"TimeInCycles": "MaxTime"})
        )
        df_test_for_eval = df_test_for_eval.merge(df_max_time, on="UnitNumber", how="left")
        df_test_for_eval["RUL_raw"] = df_test_for_eval["MaxTime"] - df_test_for_eval["TimeInCycles"]
        df_test_for_eval["RUL"] = np.minimum(
            df_test_for_eval["RUL_raw"],
            float(getattr(world_model_config, "max_rul", 125.0)),
        )

    try:
        test_metrics = evaluate_transformer_world_model_v1_on_test(
            model=world_model,
            df_test=df_test_for_eval,
            feature_cols=feature_cols,
            world_model_config=world_model_config,
            device=device,
        )
    except Exception as exc:
        print(f"  Error while computing test metrics: {exc}")
        test_metrics: Dict[str, Any] = {}

    if test_metrics:
        print(f"  RMSE: {test_metrics['rmse']:.2f} cycles")
        print(f"  MAE:  {test_metrics['mae']:.2f} cycles")
        print(f"  Bias: {test_metrics['bias']:.2f} cycles")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  NASA Mean: {test_metrics['nasa_mean']:.2f}")
    else:
        print("  (no test metrics computed)")
    print("=" * 80)

    # Summary with validation loss and optional test metrics
    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model_type": "transformer_world_model_v1",
        "num_features": input_dim,
        "past_len": past_len,
        "horizon": horizon,
        "target_mode": target_mode,
        "val_loss": float(best_val_loss),
        # Encoder configuration is important for latent world-model runs
        "encoder_checkpoint": encoder_ckpt_path if encoder_ckpt_path else None,
        "freeze_encoder": freeze_encoder,
    }
    if test_metrics:
        summary["test_metrics"] = test_metrics

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WorldModelV1] Saved summary to {summary_path}")

    return summary


def evaluate_transformer_world_model_v1_on_test(
    model: nn.Module,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    world_model_config: WorldModelTrainingConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate TransformerWorldModelV1 on FD004 test set and compute
    literature-style EOL metrics (RMSE, MAE, Bias, R², NASA score).

    Strategy:
    - Build seq2seq samples (past_len, future_horizon) from test data.
    - Run model in eval mode without teacher forcing (free rollouts).
    - Use the RUL forecast at the last horizon step as EOL-like prediction.
    - Compute metrics using compute_eol_errors_and_nasa.
    """
    model.eval()

    past_len = int(getattr(world_model_config, "past_len", 30))
    horizon = int(getattr(world_model_config, "future_horizon", world_model_config.forecast_horizon))
    max_rul = float(getattr(world_model_config, "max_rul", 125.0))

    X_list: List[np.ndarray] = []
    hi_seq_list: List[np.ndarray] = []
    true_rul_last_list: List[float] = []
    cond_id_list: List[int] = []

    # Ensure we have a RUL column clipped to [0, max_rul]
    if "RUL" not in df_test.columns:
        raise KeyError("evaluate_transformer_world_model_v1_on_test expects 'RUL' column in df_test.")

    for unit_id, df_unit in df_test.groupby("UnitNumber"):
        df_unit = df_unit.sort_values("TimeInCycles").reset_index(drop=True)
        num_rows = len(df_unit)
        if num_rows < past_len + horizon:
            continue

        cond_id_unit = int(df_unit["ConditionID"].iloc[0])

        for start in range(0, num_rows - past_len - horizon + 1):
            past = df_unit.iloc[start : start + past_len]
            future = df_unit.iloc[start + past_len : start + past_len + horizon]

            X_list.append(past[feature_cols].to_numpy(dtype=np.float32, copy=True))

            # RUL future in cycles for metrics (clipped)
            rul_future = future["RUL"].clip(lower=0.0, upper=max_rul).to_numpy(dtype=np.float32)
            true_rul_last_list.append(float(rul_future[-1]))

            # HI future (normalized) as in training: HI = clip(RUL / max_rul, 0, 1)
            hi_future = np.clip(rul_future / max_rul, 0.0, 1.0)
            hi_seq_list.append(hi_future.astype(np.float32))

            cond_id_list.append(cond_id_unit)

    if not X_list:
        print("[WorldModelV1-Test] No valid test samples could be built.")
        return {}

    X_np = np.stack(X_list, axis=0)  # (N, past_len, F)
    Y_hi_np = np.stack(hi_seq_list, axis=0)  # (N, horizon)
    cond_ids_np = np.array(cond_id_list, dtype=np.int64)  # (N,)

    X = torch.from_numpy(X_np).float().to(device)
    Y_hi = torch.from_numpy(Y_hi_np).float().to(device)
    cond_ids = torch.from_numpy(cond_ids_np).long().to(device)

    batch_size = int(getattr(world_model_config, "batch_size", 256))
    ds = TensorDataset(X, Y_hi, cond_ids)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_true_rul_last: List[float] = []
    all_pred_rul_last: List[float] = []
    idx_offset = 0

    cond_dim = getattr(world_model_config, "cond_dim", 9)

    with torch.no_grad():
        for X_b, Y_hi_b, cond_b in loader:
            B, _, _ = X_b.shape
            cond_vec = torch.zeros(B, cond_dim, device=device)

            # Approximate current RUL from HI (since HI ≈ RUL/max_rul in our mapping)
            current_hi_b = Y_hi_b[:, 0]  # (B,)
            current_rul_b = current_hi_b  # normalized RUL in [0,1]

            pred_sensors, pred_hi, pred_rul = model(
                past_seq=X_b,
                cond_vec=cond_vec,
                cond_ids=cond_b,
                future_horizon=horizon,
                teacher_forcing_targets=None,
                current_rul=current_rul_b,
                current_hi=current_hi_b,
            )

            if pred_rul is None:
                continue

            # Model predicts normalized RUL; take last horizon step and denormalize
            pred_rul_last_norm = pred_rul[:, -1, 0]  # (B,)
            pred_rul_last = (pred_rul_last_norm * max_rul).cpu().numpy()

            true_rul_last_batch = np.array(true_rul_last_list[idx_offset : idx_offset + B], dtype=np.float32)
            idx_offset += B

            all_true_rul_last.append(true_rul_last_batch)
            all_pred_rul_last.append(pred_rul_last)

    if not all_true_rul_last:
        print("[WorldModelV1-Test] No RUL predictions were produced.")
        return {}

    y_true = np.concatenate(all_true_rul_last, axis=0)
    y_pred = np.concatenate(all_pred_rul_last, axis=0)

    m = evaluate_eol_metrics(
        y_true=y_true,
        y_pred=y_pred,
        max_rul=float(max_rul),
        clip_y_true=False,
        clip_y_pred=True,
        log_prefix="[WorldModelV1-eval]",
    )

    metrics = {
        "rmse": float(m["RMSE"]),
        "mae": float(m["MAE"]),
        "bias": float(m["Bias"]),
        "r2": float(m["R2"]),
        "nasa_mean": float(m["nasa_score_mean"]),
        "nasa_sum": float(m["nasa_score_sum"]),
        "num_samples": int(len(m["y_true"])),
    }

    return metrics


def evaluate_world_model_v3_eol(
    model: nn.Module,
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: List[str],
    scaler_dict: Dict[int, StandardScaler],
    past_len: int = 30,
    max_rul: int = 125,
    num_conditions: int = 1,
    device: torch.device = None,
    # Optional: clip y_true to max_rul (fixes unit mismatch when loader returns raw RUL)
    clip_y_true_to_max_rul: bool = False,
    # Optional: provide the exact window/target policy used in training
    window_cfg: Optional[WindowConfig] = None,
    target_cfg: Optional[TargetConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate World Model v3 on test set (EOL metrics).
    
    Args:
        model: Trained WorldModelUniversalV3
        df_test: Test DataFrame
        y_test_true: True RUL at EOL (capped)
        feature_cols: Feature column names
        scaler_dict: Dictionary of condition-wise scalers
        past_len: Past window length
        max_rul: Maximum RUL value
        num_conditions: Number of conditions
        device: PyTorch device
    
    Returns:
        Dictionary with EOL metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    # Use the central builder for "last window per unit"
    if window_cfg is None:
        window_cfg = WindowConfig(past_len=int(past_len), horizon=1, pad_mode="clamp")
    if target_cfg is None:
        target_cfg = TargetConfig(max_rul=int(max_rul), cap_targets=True)

    built = build_test_windows_last(
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=window_cfg,
        target_cfg=target_cfg,
    )

    X = built["X"]  # (N, P, F)
    y_true = built["y_true"]  # (N,)
    cond_ids_np = built["cond_ids"]

    # Condition-wise scaling (same as training)
    X_scaled = np.empty_like(X, dtype=np.float32)
    for cond in np.unique(cond_ids_np):
        cond = int(cond)
        idxs = np.where(cond_ids_np == cond)[0]
        scaler = scaler_dict.get(cond, scaler_dict.get(0))
        flat = X[idxs].reshape(-1, len(feature_cols))
        X_scaled[idxs] = scaler.transform(flat).reshape(-1, int(past_len), len(feature_cols)).astype(np.float32)

    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    cond_t = torch.tensor(cond_ids_np, dtype=torch.long).to(device) if num_conditions > 1 else None

    y_pred_vals: list[float] = []
    with torch.no_grad():
        outputs = model(
            encoder_inputs=X_t,
            decoder_targets=None,
            teacher_forcing_ratio=0.0,
            horizon=1,
            cond_ids=cond_t,
        )
        if isinstance(outputs, dict):
            eol_pred = outputs.get("eol", outputs.get("rul"))
            if eol_pred is None:
                raise ValueError("evaluate_world_model_v3_eol: could not find 'eol' or 'rul' in model outputs dict")
        else:
            eol_pred = outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) >= 2 else outputs

        e = eol_pred
        if torch.is_tensor(e):
            e = e.view(-1).detach().cpu().numpy()
        e = np.asarray(e, dtype=float).reshape(-1)
        y_pred = np.clip(e, 0.0, float(max_rul)).astype(float)

    metrics = evaluate_eol_metrics(
        y_true=y_true,
        y_pred=y_pred,
        max_rul=float(max_rul),
        clip_y_true=bool(clip_y_true_to_max_rul),
        clip_y_pred=True,
        log_prefix="[eval]",
    )

    return {
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "Bias": metrics["Bias"],
        "R2": metrics["R2"],
        "nasa_score_sum": metrics["nasa_score_sum"],
        "nasa_score_mean": metrics["nasa_score_mean"],
        "num_engines": int(len(metrics["y_true"])),
        "y_pred_eol": metrics["y_pred"].tolist(),
        "y_true_eol": metrics["y_true"].tolist(),
        # Expose nasa_scores for diagnostics plots
        "nasa_scores": metrics["nasa_scores"].tolist(),
    }

