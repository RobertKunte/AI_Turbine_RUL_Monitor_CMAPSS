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
from src.metrics import compute_last_per_unit_metrics, compute_all_samples_metrics
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.models.transformer_world_model_v1 import TransformerWorldModelV1
from src.utils.feature_pipeline_contract import (
    derive_feature_pipeline_config_from_feature_cols,
    validate_feature_pipeline_config,
)


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
    feature_pipeline_config: Optional[Dict[str, Any]] = None,
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
    eval_clip_y_true = bool(getattr(world_model_config, "eval_clip_y_true_to_max_rul", False))
    
    # Stage -1: FD004 hard enforcement of capped targets (no opt-out)
    # Since WorldModelTrainingConfig is a dataclass with defaults, all attributes always exist.
    # For FD004, we always enforce True regardless of user config.
    if dataset_name.upper() == "FD004":
        # Force all capped semantics flags to True for FD004
        cap_targets = True
        eval_clip_y_true = True
        
        # Also set on the config object for consistency (used later in training loop)
        world_model_config.cap_rul_targets_to_max_rul = True
        world_model_config.eval_clip_y_true_to_max_rul = True
        
        print(f"[Stage-1] FD004: Enforced cap_targets=True, cap_rul_targets_to_max_rul=True, eval_clip_y_true_to_max_rul=True")

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
        clip_eval_y_true=bool(eval_clip_y_true),
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
    
    # Stage -1: Enforce clipping at source for FD004 (before training loop)
    if dataset_name.upper() == "FD004":
        max_rul_f = float(max_rul if max_rul is not None else 125.0)
        # Clip Y_seq (future RUL trajectories) to [0, max_rul]
        Y_train = Y_train.clamp(0.0, max_rul_f)
        # Clip Y_eol if present
        Y_eol_check = built.get("Y_eol", None)
        if Y_eol_check is not None:
            Y_eol_clipped = np.clip(Y_eol_check, 0.0, max_rul_f)
            built["Y_eol"] = Y_eol_clipped
            y_eol_max = float(np.max(Y_eol_clipped))
            assert y_eol_max <= max_rul_f + 1e-6, (
                f"[Stage-1] FD004 Y_eol clipping failed: max={y_eol_max:.2f} > {max_rul_f:.2f}"
            )
            print(f"[Stage-1] FD004: Clipped Y_eol at source, max={y_eol_max:.2f} <= max_rul={max_rul_f:.2f}")
    
    # Stage -1: Verify capped targets consistency (ADR)
    if cap_targets:
        Y_eol_check = built.get("Y_eol", None)
        if Y_eol_check is not None:
            y_eol_max = float(np.max(Y_eol_check))
            max_rul_check = float(max_rul if max_rul is not None else 125.0)
            if y_eol_max > max_rul_check + 1e-3:
                raise ValueError(
                    f"[Stage-1 Consistency] Y_eol from windowing exceeds max_rul: max={y_eol_max:.2f} > {max_rul_check:.2f}. "
                    f"cap_targets={cap_targets} but windowing produced uncapped values. Check windowing.py _cap() logic."
                )
            print(f"[Stage-1] Verified Y_eol capped: max={y_eol_max:.2f} <= max_rul={max_rul_check:.2f}")
    
    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Input shape: {X_train.shape}, Target shape: {Y_train.shape}")
    if horizon_mask_train is not None:
        print(f"  Horizon mask shape: {horizon_mask_train.shape} (use_horizon_mask={use_horizon_mask})")
    
    # Log y_eol min and pad_frac for verification (Stage-1 padding check)
    Y_eol_built = built.get("Y_eol", None)
    if Y_eol_built is not None:
        y_eol_min = float(np.min(Y_eol_built))
        y_eol_max = float(np.max(Y_eol_built))
        print(f"  [Stage-1] Y_eol range: min={y_eol_min:.2f}, max={y_eol_max:.2f}")
    if horizon_mask_train is not None:
        pad_frac = 1.0 - float(horizon_mask_train.mean().item())
        print(f"  [Stage-1] pad_frac={pad_frac:.4f} (use_padded_horizon_targets={use_padded_horizon_targets})")
        if pad_frac < 1e-4:
            print("  [Stage-1] WARNING: No padding detected. Check use_padded_horizon_targets flag.")
    else:
        print(f"  [Stage-1] pad_frac=NA (use_horizon_mask={use_horizon_mask}, use_padded_horizon_targets={use_padded_horizon_targets})")
    
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
    # Decoder type selection (default: "lstm")
    decoder_type = str(getattr(world_model_config, "decoder_type", "lstm"))
    
    # Debug log: show decoder_type from config
    print(f"[config] decoder_type={decoder_type} (from world_model_config.decoder_type={getattr(world_model_config, 'decoder_type', 'NOT_SET')})")

    # Extract decoder_num_layers from config (default to parameter value)
    decoder_num_layers = int(getattr(world_model_config, "decoder_num_layers", decoder_num_layers))

    # Get tf_cross decoder parameters
    future_max_len = int(getattr(world_model_config, "future_max_len", 256))
    cond_dim = int(getattr(world_model_config, "cond_dim", 0))
    quantiles = getattr(world_model_config, "quantiles", [0.1, 0.5, 0.9])
    dec_ff = int(getattr(world_model_config, "dec_ff", dim_feedforward))
    
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
        decoder_type=decoder_type,
        future_max_len=future_max_len,
        cond_dim=cond_dim,
        quantiles=quantiles,
        dec_ff=dec_ff,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    encoder_d_model = model.encoder.d_model
    print(f"  Model parameters: {num_params:,}")
    print(f"  Encoder: UniversalEncoderV2 (d_model={encoder_d_model}, num_layers={num_layers})")
    if decoder_type == "lstm":
        print(f"  Decoder: LSTM (num_layers={decoder_num_layers}, horizon={horizon})")
    elif decoder_type == "tf_ar":
        print(f"  Decoder: Transformer AR (self-attn, num_layers={decoder_num_layers}, horizon={horizon})")
    elif decoder_type == "tf_ar_xattn":
        print(f"  Decoder: Transformer AR (cross-attn, num_layers={decoder_num_layers}, horizon={horizon})")
    elif decoder_type == "tf_cross":
        print(f"  Decoder: Transformer Cross-Attention (non-AR, d_model={encoder_d_model}, num_layers={decoder_num_layers}, horizon={horizon})")
        print(f"    Quantiles: {quantiles}, Future max len: {future_max_len}")
    print(f"  Heads: Trajectory, EOL, HI" + (", Quantile RUL" if decoder_type == "tf_cross" else ""))
    
    # Hard-fail guard: verify decoder matches config
    if decoder_type == "tf_cross":
        # Check that model actually has tf_cross decoder
        if not hasattr(model, 'tf_cross_decoder') or model.tf_cross_decoder is None:
            raise RuntimeError(
                f"Decoder type mismatch: config specifies decoder_type='{decoder_type}' "
                f"but model was not initialized with tf_cross decoder. "
                f"Experiment: {experiment_name}, "
                f"model.decoder_type={model.decoder_type}"
            )
        decoder_class_name = model.tf_cross_decoder.__class__.__name__
        print(f"  [VERIFIED] Transformer cross-attention decoder instantiated: {decoder_class_name}")
    elif decoder_type != "lstm":
        # Check that model actually has Transformer decoder
        if not hasattr(model, 'tf_decoder') or model.tf_decoder is None:
            raise RuntimeError(
                f"Decoder type mismatch: config specifies decoder_type='{decoder_type}' "
                f"but model was initialized with LSTM decoder. "
                f"Experiment: {experiment_name}, "
                f"model.decoder_type={model.decoder_type}, "
                f"model.decoder class={type(model.decoder).__name__ if hasattr(model, 'decoder') else 'N/A'}"
            )
        # Verify decoder class name contains "Transformer"
        decoder_class_name = model.tf_decoder.__class__.__name__
        if "Transformer" not in decoder_class_name:
            raise RuntimeError(
                f"Decoder type mismatch: config specifies decoder_type='{decoder_type}' "
                f"but decoder class is '{decoder_class_name}' (expected TransformerARDecoder). "
                f"Experiment: {experiment_name}, "
                f"model.decoder_type={model.decoder_type}"
            )
        print(f"  [VERIFIED] Transformer decoder instantiated: {decoder_class_name}")
    else:
        # For LSTM, verify it's actually LSTM
        if hasattr(model, 'decoder') and model.decoder is not None:
            decoder_class_name = model.decoder.__class__.__name__
            if decoder_class_name != "LSTM":
                raise RuntimeError(
                    f"Decoder type mismatch: config specifies decoder_type='lstm' "
                    f"but decoder class is '{decoder_class_name}' (expected LSTM). "
                    f"Experiment: {experiment_name}, "
                    f"model.decoder_type={model.decoder_type}"
                )
    
    # ===================================================================
    # 6. Training loop
    # ===================================================================
    print("\n[6] Training model...")

    # Encoder freezing parameters
    freeze_encoder = bool(getattr(world_model_config, "freeze_encoder", False))
    freeze_encoder_epochs = int(getattr(world_model_config, "freeze_encoder_epochs", 0) or 0)
    # Support unfreeze_encoder_layers as int, "all", or -1
    unfreeze_encoder_layers_raw = getattr(world_model_config, "unfreeze_encoder_layers", 0)
    if unfreeze_encoder_layers_raw == "all" or unfreeze_encoder_layers_raw == -1:
        unfreeze_encoder_layers = -1  # Use -1 internally for "all layers"
    else:
        unfreeze_encoder_layers = int(unfreeze_encoder_layers_raw) if unfreeze_encoder_layers_raw else 0
    encoder_lr_mult = float(getattr(world_model_config, "encoder_lr_mult", 0.1) or 0.1)

    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _freeze_encoder() -> None:
        """Freeze all encoder parameters."""
        _set_requires_grad(model.encoder, False)
        model.encoder.eval()

    def count_trainable_params(module: nn.Module) -> Tuple[int, int]:
        """
        Count trainable parameters in a module.
        Returns: (num_params, num_tensors)
        """
        trainable_params = [p for p in module.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in trainable_params)
        num_tensors = len(trainable_params)
        return num_params, num_tensors

    def list_trainable_param_names(module: nn.Module, k: int = 10) -> List[str]:
        """
        List names of trainable parameters (up to k).
        """
        names = [name for name, param in module.named_parameters() if param.requires_grad]
        return names[:k]

    def summarize_trainability(model: nn.Module) -> dict:
        """Summarize parameter trainability statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Encoder stats
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

        # Decoder/Heads stats (everything that's not encoder)
        decoder_params = total_params - encoder_params
        decoder_trainable = trainable_params - encoder_trainable

        # Per-encoder-block stats if available
        encoder_blocks = {}
        if hasattr(model.encoder, "transformer") and hasattr(model.encoder.transformer, "layers"):
            layers = model.encoder.transformer.layers
            if layers is not None:
                for i, layer in enumerate(layers):
                    block_params = sum(p.numel() for p in layer.parameters())
                    block_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    encoder_blocks[f"layer_{i}"] = {
                        "total": block_params,
                        "trainable": block_trainable,
                        "frozen": block_trainable == 0
                    }

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder": {
                "total": encoder_params,
                "trainable": encoder_trainable,
                "frozen": encoder_trainable == 0
            },
            "decoder_heads": {
                "total": decoder_params,
                "trainable": decoder_trainable,
                "frozen": decoder_trainable == 0
            },
            "encoder_blocks": encoder_blocks
        }

    def _discover_transformer_layers(encoder: nn.Module) -> Optional[list]:
        """
        Discover transformer encoder layers from UniversalEncoderV2 or similar structures.
        Returns list of layer modules or None if not found.
        """
        # Try UniversalEncoderV2 structure: encoder.transformer.layers
        if hasattr(encoder, "transformer"):
            transformer = encoder.transformer
            if isinstance(transformer, nn.TransformerEncoder):
                # nn.TransformerEncoder stores layers in .layers (ModuleList)
                if hasattr(transformer, "layers"):
                    layers = transformer.layers
                    if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                        return list(layers)
            # Fallback: check if transformer has .layers directly
            elif hasattr(transformer, "layers"):
                layers = transformer.layers
                if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                    return list(layers)

        # Try alternative structures
        if hasattr(encoder, "seq_encoder"):
            seq_enc = encoder.seq_encoder
            if isinstance(seq_enc, nn.TransformerEncoder) and hasattr(seq_enc, "layers"):
                layers = seq_enc.layers
                if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                    return list(layers)

        # Try encoder.encoder or encoder.layers
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
            layers = encoder.encoder.layers
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return list(layers)

        if hasattr(encoder, "layers"):
            layers = encoder.layers
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return list(layers)

        # Try blocks/block pattern
        if hasattr(encoder, "blocks"):
            blocks = encoder.blocks
            if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
                return list(blocks)

        return None

    def _partial_unfreeze_encoder(*, unfreeze_last_k: int) -> list[str]:
        """
        Unfreeze encoder layers based on unfreeze_last_k:
        - unfreeze_last_k > 0: unfreeze last K transformer layers
        - unfreeze_last_k == -1: unfreeze all layers (full encoder)
        Returns list of unfrozen module names for logging.
        """
        # First freeze everything
        _set_requires_grad(model.encoder, False)

        unfrozen_modules: list[str] = []

        # Handle unfreeze all layers (-1)
        if unfreeze_last_k == -1:
            _set_requires_grad(model.encoder, True)
            unfrozen_modules.append("encoder.* (all layers)")
            model.encoder.train()
            return unfrozen_modules

        # Strategy 1: Try to unfreeze last K transformer layers
        layers = _discover_transformer_layers(model.encoder)
        if layers is not None and unfreeze_last_k > 0:
            k = min(int(unfreeze_last_k), len(layers))
            # Unfreeze last k layers (from end of list)
            for idx in range(len(layers) - k, len(layers)):
                layer = layers[idx]
                _set_requires_grad(layer, True)
                unfrozen_modules.append(f"encoder.transformer.layers[{idx}]")

        # Strategy 2: Fallback to name-pattern matching if layer discovery failed
        if unfreeze_last_k > 0 and len(unfrozen_modules) == 0:
            # Patterns to try for unfreezing by name
            patterns = []
            # Try to identify which submodule contains layers
            if hasattr(model.encoder, "transformer"):
                patterns.extend(["transformer.layers", "transformer"])
            if hasattr(model.encoder, "seq_encoder"):
                patterns.extend(["seq_encoder.layers", "seq_encoder"])
            patterns.extend(["layers", "blocks", "block", "encoder.layers"])

            # Also always try to unfreeze projection/head modules
            head_patterns = ["proj", "out_proj", "head", "mlp", "fc_out", "output"]

            # Unfreeze params matching patterns
            unfrozen_by_pattern = []
            for name, param in model.encoder.named_parameters():
                # Check if name matches any pattern
                matches_pattern = any(pattern in name for pattern in patterns)
                matches_head = any(pattern in name for pattern in head_patterns)
                
                if matches_pattern or matches_head:
                    param.requires_grad = True
                    if name not in unfrozen_by_pattern:
                        unfrozen_by_pattern.append(name)
                        unfrozen_modules.append(f"encoder.{name}")

            if unfrozen_by_pattern:
                print(f"[EncoderFreeze] Fallback: Unfroze {len(unfrozen_by_pattern)} params by name pattern")

        # Always unfreeze shared_head if present (part of encoder)
        if hasattr(model.encoder, "shared_head") and model.encoder.shared_head is not None:
            _set_requires_grad(model.encoder.shared_head, True)
            if "encoder.shared_head" not in unfrozen_modules:
                unfrozen_modules.append("encoder.shared_head")

        model.encoder.train()
        return unfrozen_modules

    # Initialize encoder freezing state
    did_initial_unfreeze = False

    # Stage A: Initial encoder freeze (if requested)
    if freeze_encoder or freeze_encoder_epochs > 0:
        print(f"[EncoderFreeze] Initial freeze enabled: freeze_encoder={freeze_encoder}, freeze_encoder_epochs={freeze_encoder_epochs}")
        _freeze_encoder()
    else:
        print(f"[EncoderFreeze] No initial freeze: freeze_encoder={freeze_encoder}, freeze_encoder_epochs={freeze_encoder_epochs}")

    # Safety check: ensure decoder/heads are trainable when encoder is frozen
    trainability_check = summarize_trainability(model)
    encoder_trainable = trainability_check["encoder"]["trainable"]
    decoder_trainable = trainability_check["decoder_heads"]["trainable"]
    
    if (freeze_encoder or freeze_encoder_epochs > 0) and encoder_trainable == 0:
        if decoder_trainable == 0:
            raise RuntimeError(
                f"[EncoderFreeze] CRITICAL: Encoder is frozen but decoder/heads also have 0 trainable params! "
                f"This would prevent any training. Check model structure."
            )
        print(f"[EncoderFreeze] Safety check passed: encoder_trainable={encoder_trainable:,}, decoder_trainable={decoder_trainable:,}")

    print(f"[EncoderFreeze] Config: unfreeze_encoder_layers={unfreeze_encoder_layers}, encoder_lr_mult={encoder_lr_mult}")

    def _make_optimizer() -> torch.optim.Optimizer:
        """
        Create optimizer with separate param groups for encoder (with lr_mult) and decoder/heads.
        This ensures encoder params get correct learning rate when unfrozen.
        """
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        if enc_params:
            param_groups.append({"params": enc_params, "lr": lr * encoder_lr_mult})

        # If no trainable params, create empty optimizer (shouldn't happen, but safety)
        if not param_groups:
            print("[EncoderFreeze] WARNING: No trainable parameters found! Creating optimizer with all params.")
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)

    optimizer = _make_optimizer()
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
    
    # Freeze-aware checkpointing (prevents degenerate solutions during freeze phase)
    # Support both new names (best_ckpt_*) and old names (checkpoint_*) for backwards compatibility
    best_ckpt_min_epoch = getattr(world_model_config, "best_ckpt_min_epoch", None)
    if best_ckpt_min_epoch is None:
        best_ckpt_min_epoch = getattr(world_model_config, "checkpoint_min_epoch", None)
    
    best_ckpt_min_epoch_after_unfreeze = int(getattr(world_model_config, "best_ckpt_min_epoch_after_unfreeze", 
        getattr(world_model_config, "checkpoint_unfreeze_warmup_epochs", 2)))
    
    best_ckpt_require_pred_std_min = getattr(world_model_config, "best_ckpt_require_pred_std_min", None)
    if best_ckpt_require_pred_std_min is None:
        # Check old name, default to 1.0 if not set
        best_ckpt_require_pred_std_min = getattr(world_model_config, "checkpoint_pred_std_min", 1.0)
    if best_ckpt_require_pred_std_min is not None:
        best_ckpt_require_pred_std_min = float(best_ckpt_require_pred_std_min)
    
    # Legacy: checkpoint_pred_sanity_gate (for backwards compatibility)
    checkpoint_pred_sanity_gate = bool(getattr(world_model_config, "checkpoint_pred_sanity_gate", True))
    
    # Stage -1: FD004 hard enforcement (sync with earlier enforcement)
    if dataset_name.upper() == "FD004":
        cap_rul_targets = True
        # Override best_metric to use capped LAST metric
        best_metric = "val_rmse_last"
        print(f"[Stage-1] FD004: Enforced cap_rul_targets=True, best_metric={best_metric}")

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
        # Quantile metrics for tf_cross decoder
        "train_pinball_loss": [],
        "train_crossing_penalty": [],
        "train_crossing_rate_last": [],
        "train_crossing_rate_seq": [],
        "train_coverage_10_90_last": [],
        "train_coverage_10_90_seq": [],
        "val_pinball_loss": [],
        "val_crossing_penalty": [],
        "val_crossing_rate_last": [],
        "val_crossing_rate_seq": [],
        "val_coverage_10_90_last": [],
        "val_coverage_10_90_seq": [],
        # Checkpoint tracking (freeze-aware)
        "checkpoint_allowed": [],
        "checkpoint_min_epoch": [],
        "checkpoint_reason_blocked": [],
        "checkpoint_pred_range_last": [],
        "checkpoint_pred_std_last": [],
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

        # Stage B: Partial unfreeze after initial freeze period
        # Encoder freeze semantics:
        #   Epochs 0 to (freeze_encoder_epochs-1): encoder frozen
        #   Epoch freeze_encoder_epochs: unfreezing happens HERE, encoder trainable for this epoch
        #   Epochs (freeze_encoder_epochs+1) onwards: encoder remains unfrozen
        # Example: freeze_encoder_epochs=10 means frozen [0-9], unfrozen starting epoch 10
        if (not did_initial_unfreeze) and freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            unfrozen = _partial_unfreeze_encoder(unfreeze_last_k=unfreeze_encoder_layers)
            did_initial_unfreeze = True
            
            # Fail-fast check: verify unfreeze actually worked
            enc_params, enc_tensors = count_trainable_params(model.encoder)
            total_params, total_tensors = count_trainable_params(model)
            trainable_names = list_trainable_param_names(model.encoder, k=10)
            
            print(f"[EncoderFreeze] Epoch {epoch}: Partial unfreeze - unfrozen modules: {unfrozen}")
            print(f"[EncoderFreeze] Epoch {epoch}: encoder trainable tensors: {enc_tensors}, params: {enc_params:,}")
            print(f"[EncoderFreeze] Epoch {epoch}: total trainable params: {total_params:,}")
            if trainable_names:
                print(f"[EncoderFreeze] Epoch {epoch}: sample trainable names: {trainable_names[:5]}")
            
            # Hard assertion: encoder must have trainable params after unfreeze (unless unfreeze_last_k==0)
            if unfreeze_encoder_layers != 0 and enc_params == 0:
                encoder_type = type(model.encoder).__name__
                encoder_attrs = [a for a in dir(model.encoder) if not a.startswith('_')][:15]
                raise RuntimeError(
                    f"[EncoderFreeze] CRITICAL: Unfreeze failed! Encoder has 0 trainable params after unfreeze attempt.\n"
                    f"  Configuration: freeze_encoder_epochs={freeze_encoder_epochs}, unfreeze_encoder_layers={unfreeze_encoder_layers}\n"
                    f"  Encoder type: {encoder_type}\n"
                    f"  Encoder attributes: {encoder_attrs}\n"
                    f"  Unfrozen modules reported: {unfrozen}\n"
                    f"  This will cause training to fail (no encoder updates).\n"
                    f"  Suggestion: Check encoder structure or set unfreeze_encoder_layers=-1 to unfreeze all layers."
                )
            
            # Rebuild optimizer to include newly unfrozen encoder params with correct lr_mult
            optimizer = _make_optimizer()
            print(f"[EncoderFreeze] Epoch {epoch}: Rebuilt optimizer with {len(optimizer.param_groups)} param groups")
            for pg_idx, pg in enumerate(optimizer.param_groups):
                n_params = sum(p.numel() for p in pg["params"])
                effective_lr = pg["lr"]
                print(f"  Param group {pg_idx}: {n_params:,} params, lr={effective_lr:.6f} "
                      f"{'(encoder, lr_mult=' + str(encoder_lr_mult) + ')' if effective_lr != lr else '(decoder/heads)'}")

            # Special logging for transition epoch
            enc_params_after = count_trainable_params(model.encoder)[0]
            print(f"[EncoderFreeze] *** TRANSITION EPOCH {epoch} ***")
            print(f"  Encoder unfrozen this epoch: now training with {enc_params_after:,} encoder params")

        # Optional: freeze encoder briefly when EOL starts to protect representation
        eol_freeze_active = False
        if freeze_encoder_n > 0 and eol_on_epoch is not None and (epoch - eol_on_epoch) < freeze_encoder_n:
            _set_requires_grad(model.encoder, False)
            eol_freeze_active = True

        # Get trainability summary for logging
        trainability = summarize_trainability(model)

        # Determine freeze state AFTER unfreezing logic has run
        if freeze_encoder_epochs > 0 and not did_initial_unfreeze:
            # Still in frozen period (epoch < freeze_encoder_epochs, unfreeze hasn't happened yet)
            encoder_should_be_frozen = True
        else:
            # Either: no freeze config, OR already unfrozen, OR this is transition epoch (just unfroze)
            encoder_should_be_frozen = freeze_encoder or eol_freeze_active

        encoder_trainable_params = trainability["encoder"]["trainable"]
        decoder_trainable_params = trainability["decoder_heads"]["trainable"]
        encoder_frozen_flag = trainability["encoder"]["frozen"]

        # Use actual frozen state from trainability (not separate variable)
        encoder_frozen = encoder_frozen_flag

        # Log freeze state with component breakdown
        if epoch == freeze_encoder_epochs and did_initial_unfreeze:
            # Transition epoch already logged above, just show final state
            print(f"[EncoderFreeze] Training epoch {epoch} with unfrozen encoder: "
                  f"{encoder_trainable_params:,} encoder params + {decoder_trainable_params:,} decoder/head params")
        else:
            # Normal logging
            print(f"[EncoderFreeze] epoch={epoch} frozen={encoder_frozen_flag} "
                  f"encoder_trainable={encoder_trainable_params:,} decoder_trainable={decoder_trainable_params:,}")
            if eol_freeze_active:
                print(f"  [EOL temp freeze active: epoch {epoch - eol_on_epoch} of {freeze_encoder_n}]")

        # Take parameter snapshots for update norm calculation
        param_snapshots = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_snapshots[name] = param.data.clone()

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
        # Quantile metrics tracking
        running_pinball_loss = 0.0
        running_crossing_penalty = 0.0
        running_crossing_rate_last = 0.0
        running_crossing_rate_seq = 0.0
        running_coverage_10_90_last = 0.0
        running_coverage_10_90_seq = 0.0
        n_quantile_batches = 0

        # Gradient/Update norm tracking (sample first K batches)
        grad_norm_sample_batches = min(2, len(train_loader))  # Sample first 2 batches or all if fewer
        running_grad_norm_encoder = 0.0
        running_grad_norm_decoder = 0.0
        running_grad_norm_heads = 0.0
        running_update_norm_encoder = 0.0
        running_update_norm_decoder = 0.0
        running_update_norm_heads = 0.0
        n_norm_batches = 0

        # Store parameter snapshots for update norm calculation
        param_snapshots = None
        
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

            # ------------------------------------------------------------------
            # Define target_traj_rul immediately after batch unpacking (ALWAYS available)
            # ------------------------------------------------------------------
            target_traj_rul = Y_batch  # (B, H, 1) - future RUL trajectory (may be raw/padded)
            if cap_rul_targets:
                max_rul_f = float(max_rul if max_rul is not None else 125.0)
                target_traj_rul = target_traj_rul.clamp(0.0, max_rul_f)

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
            # Quantile loss for tf_cross decoder
            # ------------------------------------------------------------------
            loss_pinball = None
            loss_cross = None
            crossing_rate_last = None
            crossing_rate_seq = None
            coverage_10_90_last = None
            coverage_10_90_seq = None
            
            if decoder_type == "tf_cross":
                from src.losses.quantile_losses import (
                    pinball_loss,
                    quantile_crossing_penalty,
                    quantile_crossing_rate,
                    quantile_coverage,
                )

                # Defensive assertions for target_traj_rul
                assert target_traj_rul is not None, "target_traj_rul must be defined"
                assert target_traj_rul.ndim == 3 and target_traj_rul.shape[-1] == 1, f"target_traj_rul must be (B, T, 1), got {target_traj_rul.shape}"

                # Get quantile outputs
                rul_q_seq = outputs.get("rul_q_seq")  # (B, T_future, 3)
                rul_q_last = outputs.get("rul_q_last")  # (B, 3)

                if rul_q_seq is None or rul_q_last is None:
                    raise ValueError("tf_cross decoder must return rul_q_seq and rul_q_last")
                
                # Get config parameters
                b2_loss_mode = str(getattr(world_model_config, "b2_loss_mode", "last")).lower()
                lambda_cross = float(getattr(world_model_config, "lambda_cross", 0.05))
                quantiles_tensor = torch.tensor(
                    getattr(world_model_config, "quantiles", [0.1, 0.5, 0.9]),
                    device=X_batch.device,
                    dtype=torch.float32,
                )
                
                # Get right-censor/horizon mask
                valid_mask_seq = mask_batch.squeeze(-1) if (use_horizon_mask and mask_batch is not None) else None
                
                if b2_loss_mode == "last":
                    # B2.0: LAST-only loss
                    y_true_last = target_traj_rul[:, -1, 0]  # (B,) - last timestep RUL
                    y_pred_last = rul_q_last  # (B, 3)
                    
                    # Mask for last timestep (right-censor aware)
                    mask_last = None
                    if valid_mask_seq is not None:
                        mask_last = valid_mask_seq[:, -1]  # (B,)
                    
                    # Compute losses
                    loss_pinball = pinball_loss(
                        y_true=y_true_last,
                        y_pred=y_pred_last,
                        quantiles=quantiles_tensor,
                        mask=mask_last,
                    )
                    
                    loss_cross = quantile_crossing_penalty(
                        y_pred=y_pred_last,
                        mask=mask_last,
                        margin=0.0,
                    )
                    
                    # Metrics
                    with torch.no_grad():
                        crossing_rate_last = quantile_crossing_rate(
                            y_pred=y_pred_last,
                            mask=mask_last,
                        )
                        coverage_10_90_last = quantile_coverage(
                            y_true=y_true_last,
                            y_pred=y_pred_last,
                            mask=mask_last,
                        )
                
                elif b2_loss_mode == "traj":
                    # B2.1: Trajectory loss
                    y_true_seq = target_traj_rul.squeeze(-1)  # (B, T_future)
                    y_pred_seq = rul_q_seq  # (B, T_future, 3)
                    
                    # Compute losses with sequence mask
                    loss_pinball = pinball_loss(
                        y_true=y_true_seq,
                        y_pred=y_pred_seq,
                        quantiles=quantiles_tensor,
                        mask=valid_mask_seq,
                    )
                    
                    loss_cross = quantile_crossing_penalty(
                        y_pred=y_pred_seq,
                        mask=valid_mask_seq,
                        margin=0.0,
                    )
                    
                    # Metrics
                    with torch.no_grad():
                        crossing_rate_seq = quantile_crossing_rate(
                            y_pred=y_pred_seq,
                            mask=valid_mask_seq,
                        )
                        coverage_10_90_seq = quantile_coverage(
                            y_true=y_true_seq,
                            y_pred=y_pred_seq,
                            mask=valid_mask_seq,
                        )
                else:
                    raise ValueError(f"Unknown b2_loss_mode: {b2_loss_mode}. Must be 'last' or 'traj'")

            # ------------------------------------------------------------------
            # Targets (target_traj_rul already defined above)
            # ------------------------------------------------------------------
            # target_traj_rul is already defined and capped above after batch unpacking

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
                    # Stage -1: Fail-fast assertion for capped targets (ADR consistency)
                    if cap_rul_targets and te_max > mr + 1e-3:
                        raise ValueError(
                            f"[Stage-1 Consistency] eol_true_scalar exceeds max_rul: max={te_max:.2f} > {mr:.2f}. "
                            f"This indicates uncapped targets are being used despite cap_targets=True. "
                            f"Check windowing.py and target_cfg.cap_targets."
                        )
                    elif not cap_rul_targets and te_max > mr + 1e-3:
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
            
            # Quantile loss for tf_cross decoder
            weighted_quantile = 0.0
            if decoder_type == "tf_cross" and loss_pinball is not None:
                lambda_cross = float(getattr(world_model_config, "lambda_cross", 0.05))
                weighted_quantile = loss_pinball + lambda_cross * loss_cross

            loss = weighted_traj + weighted_eol + weighted_hi + weighted_shape + weighted_eol_hi + weighted_quantile

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

            # Calculate gradient and update norms for sampled batches
            if batch_idx < grad_norm_sample_batches and param_snapshots is not None:
                # Calculate gradient norms
                grad_norm_enc = 0.0
                grad_norm_dec = 0.0
                grad_norm_heads = 0.0

                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        grad_norm_sq = torch.sum(param.grad ** 2).item()
                        if name.startswith("encoder."):
                            grad_norm_enc += grad_norm_sq
                        elif name.startswith(("decoder.", "tf_cross_decoder.", "future_query_builder.", "rul_quantile_head.", "fc_rul", "fc_health", "shared_head", "traj_head")):
                            grad_norm_dec += grad_norm_sq
                        else:
                            grad_norm_heads += grad_norm_sq

                # Calculate update norms (difference from snapshot)
                update_norm_enc = 0.0
                update_norm_dec = 0.0
                update_norm_heads = 0.0

                for name, param in model.named_parameters():
                    if param.requires_grad and name in param_snapshots:
                        update_sq = torch.sum((param.data - param_snapshots[name]) ** 2).item()
                        if name.startswith("encoder."):
                            update_norm_enc += update_sq
                        elif name.startswith(("decoder.", "tf_cross_decoder.", "future_query_builder.", "rul_quantile_head.", "fc_rul", "fc_health", "shared_head", "traj_head")):
                            update_norm_dec += update_sq
                        else:
                            update_norm_heads += update_sq

                running_grad_norm_encoder += grad_norm_enc ** 0.5
                running_grad_norm_decoder += grad_norm_dec ** 0.5
                running_grad_norm_heads += grad_norm_heads ** 0.5
                running_update_norm_encoder += update_norm_enc ** 0.5
                running_update_norm_decoder += update_norm_dec ** 0.5
                running_update_norm_heads += update_norm_heads ** 0.5
                n_norm_batches += 1

            running_train_loss += loss.item() * X_batch.size(0)
            n_train_samples += X_batch.size(0)
            
            # Accumulate quantile metrics
            if decoder_type == "tf_cross" and loss_pinball is not None:
                running_pinball_loss += loss_pinball.item() * X_batch.size(0)
                running_crossing_penalty += loss_cross.item() * X_batch.size(0)
                n_quantile_batches += X_batch.size(0)
                if crossing_rate_last is not None:
                    running_crossing_rate_last += crossing_rate_last * X_batch.size(0)
                    running_coverage_10_90_last += coverage_10_90_last * X_batch.size(0)
                if crossing_rate_seq is not None:
                    running_crossing_rate_seq += crossing_rate_seq * X_batch.size(0)
                    running_coverage_10_90_seq += coverage_10_90_seq * X_batch.size(0)
        
        epoch_train_loss = running_train_loss / n_train_samples
        epoch_grad_norm = (running_grad_norm / n_grad_norm) if n_grad_norm > 0 else float("nan")
        history["train_grad_norm"].append(epoch_grad_norm)
        
        # Quantile metrics (for tf_cross decoder)
        if decoder_type == "tf_cross" and n_quantile_batches > 0:
            history["train_pinball_loss"].append(running_pinball_loss / n_quantile_batches)
            history["train_crossing_penalty"].append(running_crossing_penalty / n_quantile_batches)
            if running_crossing_rate_last > 0:
                history["train_crossing_rate_last"].append(running_crossing_rate_last / n_quantile_batches)
                history["train_coverage_10_90_last"].append(running_coverage_10_90_last / n_quantile_batches)
            else:
                history["train_crossing_rate_last"].append(0.0)
                history["train_coverage_10_90_last"].append(0.0)
            if running_crossing_rate_seq > 0:
                history["train_crossing_rate_seq"].append(running_crossing_rate_seq / n_quantile_batches)
                history["train_coverage_10_90_seq"].append(running_coverage_10_90_seq / n_quantile_batches)
            else:
                history["train_crossing_rate_seq"].append(0.0)
                history["train_coverage_10_90_seq"].append(0.0)
        else:
            history["train_pinball_loss"].append(0.0)
            history["train_crossing_penalty"].append(0.0)
            history["train_crossing_rate_last"].append(0.0)
            history["train_crossing_rate_seq"].append(0.0)
            history["train_coverage_10_90_last"].append(0.0)
            history["train_coverage_10_90_seq"].append(0.0)

        # Freeze audit: log gradient/update norms
        if n_norm_batches > 0:
            avg_grad_norm_enc = running_grad_norm_encoder / n_norm_batches
            avg_grad_norm_dec = running_grad_norm_decoder / n_norm_batches
            avg_grad_norm_heads = running_grad_norm_heads / n_norm_batches
            avg_update_norm_enc = running_update_norm_encoder / n_norm_batches
            avg_update_norm_dec = running_update_norm_decoder / n_norm_batches
            avg_update_norm_heads = running_update_norm_heads / n_norm_batches

            print(f"[EncoderFreeze] epoch={epoch} grad_norm_enc={avg_grad_norm_enc:.4f} "
                  f"grad_norm_dec={avg_grad_norm_dec:.4f} grad_norm_heads={avg_grad_norm_heads:.4f} "
                  f"upd_norm_enc={avg_update_norm_enc:.6f} upd_norm_dec={avg_update_norm_dec:.6f} upd_norm_heads={avg_update_norm_heads:.6f}")

            # Add to history for JSON export
            if "freeze_audit" not in history:
                history["freeze_audit"] = []
            history["freeze_audit"].append({
                "epoch": epoch,
                "encoder_should_be_frozen": encoder_should_be_frozen,
                "encoder_trainable_params": encoder_trainable_params,
                "encoder_frozen_flag": encoder_frozen_flag,
                "grad_norm_encoder": avg_grad_norm_enc,
                "grad_norm_decoder": avg_grad_norm_dec,
                "grad_norm_heads": avg_grad_norm_heads,
                "update_norm_encoder": avg_update_norm_enc,
                "update_norm_decoder": avg_update_norm_dec,
                "update_norm_heads": avg_update_norm_heads,
                "trainability": trainability
            })
        else:
            print(f"[EncoderFreeze] epoch={epoch} no_norm_data")

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
        # Quantile metrics tracking for validation
        running_val_pinball_loss = 0.0
        running_val_crossing_penalty = 0.0
        running_val_crossing_rate_last = 0.0
        running_val_crossing_rate_seq = 0.0
        running_val_coverage_10_90_last = 0.0
        running_val_coverage_10_90_seq = 0.0
        n_val_quantile_batches = 0
        
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

                # ------------------------------------------------------------------
                # Define target_traj_rul immediately after batch unpacking (ALWAYS available)
                # ------------------------------------------------------------------
                target_traj_rul = Y_batch  # (B, H, 1) - future RUL trajectory (may be raw/padded)
                if cap_rul_targets:
                    max_rul_f = float(max_rul if max_rul is not None else 125.0)
                    target_traj_rul = target_traj_rul.clamp(0.0, max_rul_f)

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
                
                # Quantile loss for tf_cross decoder (validation)
                val_loss_pinball = None
                val_loss_cross = None
                val_crossing_rate_last = None
                val_crossing_rate_seq = None
                val_coverage_10_90_last = None
                val_coverage_10_90_seq = None
                
                if decoder_type == "tf_cross":
                    from src.losses.quantile_losses import (
                        pinball_loss,
                        quantile_crossing_penalty,
                        quantile_crossing_rate,
                        quantile_coverage,
                    )

                    # Defensive assertions for target_traj_rul
                    assert target_traj_rul is not None, "target_traj_rul must be defined"
                    assert target_traj_rul.ndim == 3 and target_traj_rul.shape[-1] == 1, f"target_traj_rul must be (B, T, 1), got {target_traj_rul.shape}"

                    # Get quantile outputs
                    rul_q_seq = outputs.get("rul_q_seq")  # (B, T_future, 3)
                    rul_q_last = outputs.get("rul_q_last")  # (B, 3)

                    if rul_q_seq is not None and rul_q_last is not None:
                        # Get config parameters
                        b2_loss_mode = str(getattr(world_model_config, "b2_loss_mode", "last")).lower()
                        quantiles_tensor = torch.tensor(
                            getattr(world_model_config, "quantiles", [0.1, 0.5, 0.9]),
                            device=X_batch.device,
                            dtype=torch.float32,
                        )
                        
                        # Get right-censor/horizon mask
                        valid_mask_seq = mask_batch.squeeze(-1) if (use_horizon_mask and mask_batch is not None) else None
                        
                        if b2_loss_mode == "last":
                            # B2.0: LAST-only loss
                            y_true_last = target_traj_rul[:, -1, 0]  # (B,)
                            y_pred_last = rul_q_last  # (B, 3)
                            
                            mask_last = None
                            if valid_mask_seq is not None:
                                mask_last = valid_mask_seq[:, -1]  # (B,)
                            
                            val_loss_pinball = pinball_loss(
                                y_true=y_true_last,
                                y_pred=y_pred_last,
                                quantiles=quantiles_tensor,
                                mask=mask_last,
                            )
                            
                            val_loss_cross = quantile_crossing_penalty(
                                y_pred=y_pred_last,
                                mask=mask_last,
                                margin=0.0,
                            )
                            
                            val_crossing_rate_last = quantile_crossing_rate(
                                y_pred=y_pred_last,
                                mask=mask_last,
                            )
                            val_coverage_10_90_last = quantile_coverage(
                                y_true=y_true_last,
                                y_pred=y_pred_last,
                                mask=mask_last,
                            )
                        
                        elif b2_loss_mode == "traj":
                            # B2.1: Trajectory loss
                            y_true_seq = target_traj_rul.squeeze(-1)  # (B, T_future)
                            y_pred_seq = rul_q_seq  # (B, T_future, 3)
                            
                            val_loss_pinball = pinball_loss(
                                y_true=y_true_seq,
                                y_pred=y_pred_seq,
                                quantiles=quantiles_tensor,
                                mask=valid_mask_seq,
                            )
                            
                            val_loss_cross = quantile_crossing_penalty(
                                y_pred=y_pred_seq,
                                mask=valid_mask_seq,
                                margin=0.0,
                            )
                            
                            val_crossing_rate_seq = quantile_crossing_rate(
                                y_pred=y_pred_seq,
                                mask=valid_mask_seq,
                            )
                            val_coverage_10_90_seq = quantile_coverage(
                                y_true=y_true_seq,
                                y_pred=y_pred_seq,
                                mask=valid_mask_seq,
                            )

                # Targets (target_traj_rul already defined above)

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
                
                # Quantile loss for tf_cross decoder (validation)
                weighted_quantile = 0.0
                if decoder_type == "tf_cross" and val_loss_pinball is not None:
                    lambda_cross = float(getattr(world_model_config, "lambda_cross", 0.05))
                    weighted_quantile = val_loss_pinball + lambda_cross * val_loss_cross
                
                loss = weighted_traj + weighted_eol + weighted_hi + weighted_shape + weighted_eol_hi + weighted_quantile
                
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
                
                # Accumulate quantile metrics (validation)
                if decoder_type == "tf_cross" and val_loss_pinball is not None:
                    running_val_pinball_loss += val_loss_pinball.item() * X_batch.size(0)
                    running_val_crossing_penalty += val_loss_cross.item() * X_batch.size(0)
                    n_val_quantile_batches += X_batch.size(0)
                    if val_crossing_rate_last is not None:
                        running_val_crossing_rate_last += val_crossing_rate_last * X_batch.size(0)
                        running_val_coverage_10_90_last += val_coverage_10_90_last * X_batch.size(0)
                    if val_crossing_rate_seq is not None:
                        running_val_crossing_rate_seq += val_crossing_rate_seq * X_batch.size(0)
                        running_val_coverage_10_90_seq += val_coverage_10_90_seq * X_batch.size(0)
        
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
        
        # Quantile metrics (validation)
        if decoder_type == "tf_cross" and n_val_quantile_batches > 0:
            history["val_pinball_loss"].append(running_val_pinball_loss / n_val_quantile_batches)
            history["val_crossing_penalty"].append(running_val_crossing_penalty / n_val_quantile_batches)
            if running_val_crossing_rate_last > 0:
                history["val_crossing_rate_last"].append(running_val_crossing_rate_last / n_val_quantile_batches)
                history["val_coverage_10_90_last"].append(running_val_coverage_10_90_last / n_val_quantile_batches)
            else:
                history["val_crossing_rate_last"].append(0.0)
                history["val_coverage_10_90_last"].append(0.0)
            if running_val_crossing_rate_seq > 0:
                history["val_crossing_rate_seq"].append(running_val_crossing_rate_seq / n_val_quantile_batches)
                history["val_coverage_10_90_seq"].append(running_val_coverage_10_90_seq / n_val_quantile_batches)
            else:
                history["val_crossing_rate_seq"].append(0.0)
                history["val_coverage_10_90_seq"].append(0.0)
        else:
            history["val_pinball_loss"].append(0.0)
            history["val_crossing_penalty"].append(0.0)
            history["val_crossing_rate_last"].append(0.0)
            history["val_crossing_rate_seq"].append(0.0)
            history["val_coverage_10_90_last"].append(0.0)
            history["val_coverage_10_90_seq"].append(0.0)
        
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
        # Base allow_best_update from EOL-aware logic
        allow_best_update_base = (not select_best_after_eol_active) or bool(eol_active)
        
        # Freeze-aware checkpointing: calculate min_epoch
        if best_ckpt_min_epoch is not None:
            min_epoch = int(best_ckpt_min_epoch)
        elif freeze_encoder or freeze_encoder_epochs > 0:
            # During freeze phase, require warmup after unfreeze
            min_epoch = freeze_encoder_epochs + 1  # Default: allow after freeze ends + 1 epoch
        else:
            # No freeze: allow from epoch 0
            min_epoch = 0
        
        # Compute epochs since unfreeze
        if freeze_encoder_epochs > 0:
            epochs_since_unfreeze = max(0, epoch - freeze_encoder_epochs)
        else:
            epochs_since_unfreeze = epoch
        
        # Check if checkpoint is allowed based on freeze-aware gating
        checkpoint_allowed_min_epoch = epoch >= min_epoch
        checkpoint_allowed_after_unfreeze = epochs_since_unfreeze >= best_ckpt_min_epoch_after_unfreeze
        checkpoint_allowed = checkpoint_allowed_min_epoch and checkpoint_allowed_after_unfreeze
        reason_blocked = None
        
        if not checkpoint_allowed:
            allow_best_update = False
            if not checkpoint_allowed_min_epoch:
                reason_blocked = f"min_epoch={min_epoch} (freeze-aware: freeze_epochs={freeze_encoder_epochs})"
            elif not checkpoint_allowed_after_unfreeze:
                reason_blocked = f"min_after_unfreeze={best_ckpt_min_epoch_after_unfreeze} (epochs_since_unfreeze={epochs_since_unfreeze})"
        else:
            allow_best_update = allow_best_update_base
            reason_blocked = None if allow_best_update_base else f"eol_active={eol_active} (EOL-aware gating)"
        
        # Sanity gate: reject degenerate predictions (only if checkpoint would be allowed)
        pred_range_last = None
        pred_std_last = None
        if allow_best_update and best_ckpt_require_pred_std_min is not None and y_pred_val_for_gate is not None:
            # Get y_pred_val if available (for FD004 val_rmse_last)
            pred_range_last = float(np.max(y_pred_val_for_gate) - np.min(y_pred_val_for_gate))
            pred_std_last = float(np.std(y_pred_val_for_gate))
            
            if pred_std_last < best_ckpt_require_pred_std_min:
                allow_best_update = False
                reason_blocked = f"sanity_gate(std={pred_std_last:.2f}<{best_ckpt_require_pred_std_min})"

        # Choose which metric to optimize for best checkpoint
        # Stage -1: FD004 must use capped LAST metric (val_rmse_last)
        y_pred_val_for_gate = None  # Initialize for sanity gate
        if dataset_name.upper() == "FD004" and best_metric == "val_rmse_last":
            # Compute val_rmse_last during validation for FD004
            # This requires evaluating on validation set (quick evaluation)
            try:
                # Quick evaluation on validation set to get rmse_last
                val_units_set = set(val_units.detach().cpu().numpy().astype(int).tolist())
                df_val_split_df = df_train[df_train["UnitNumber"].isin(val_units_set)].copy()
                if len(df_val_split_df) > 0:
                    # Use the same evaluation function as post-training
                    wc_val = WindowConfig(past_len=int(past_len), horizon=1, stride=1, require_full_horizon=False, pad_mode="clamp")
                    tc_val = TargetConfig(max_rul=int(max_rul if max_rul is not None else 125), cap_targets=True, eol_target_mode="current_from_df")
                    built_val = build_sliding_windows(
                        df_val_split_df,
                        feature_cols,
                        target_col="RUL_raw" if ("RUL_raw" in df_val_split_df.columns) else "RUL",
                        unit_col="UnitNumber",
                        time_col="TimeInCycles",
                        cond_col="ConditionID",
                        window_cfg=wc_val,
                        target_cfg=tc_val,
                        return_mask=False,
                    )
                    X_val_quick = built_val["X"]
                    y_true_val = built_val["Y_eol"].astype(float)
                    # Stage -1: Clip y_true for FD004
                    if dataset_name.upper() == "FD004":
                        y_true_val = np.clip(y_true_val, 0.0, float(max_rul if max_rul is not None else 125.0))
                    unit_ids_val = built_val["unit_ids"].astype(np.int64)
                    
                    # Scale and predict
                    X_val_scaled_quick = np.empty_like(X_val_quick, dtype=np.float32)
                    for cond in np.unique(built_val["cond_ids"]):
                        cond = int(cond)
                        idxs = np.where(built_val["cond_ids"] == cond)[0]
                        scaler = scaler_dict.get(cond, scaler_dict.get(0))
                        flat = X_val_quick[idxs].reshape(-1, len(feature_cols))
                        X_val_scaled_quick[idxs] = scaler.transform(flat).reshape(-1, int(past_len), len(feature_cols)).astype(np.float32)
                    
                    X_val_t_quick = torch.tensor(X_val_scaled_quick, dtype=torch.float32).to(device)
                    cond_val_t_quick = torch.tensor(built_val["cond_ids"], dtype=torch.long).to(device) if num_conditions > 1 else None
                    
                    with torch.no_grad():
                        out_val = model(
                            encoder_inputs=X_val_t_quick,
                            decoder_targets=None,
                            teacher_forcing_ratio=0.0,
                            horizon=1,
                            cond_ids=cond_val_t_quick,
                        )
                        if isinstance(out_val, dict):
                            e_val = out_val.get("eol", out_val.get("rul"))
                        else:
                            e_val = out_val[1] if isinstance(out_val, (tuple, list)) and len(out_val) >= 2 else out_val
                        if torch.is_tensor(e_val):
                            y_pred_val = e_val.view(-1).detach().cpu().numpy().astype(float)
                        else:
                            y_pred_val = np.asarray(e_val, dtype=float).reshape(-1)
                    y_pred_val = np.clip(y_pred_val, 0.0, float(max_rul if max_rul is not None else 125.0)).astype(float)
                    
                    # Compute rmse_last
                    from src.metrics import compute_last_per_unit_metrics
                    clip_val = (0.0, float(max_rul)) if max_rul is not None else None
                    m_val_last = compute_last_per_unit_metrics(unit_ids_val, y_true_val, y_pred_val, clip=clip_val)
                    metric_val = float(m_val_last.get("rmse_last", epoch_val_eol_loss))
                    
                    # Store y_pred_val for sanity gate (only for FD004 val_rmse_last)
                    y_pred_val_for_gate = y_pred_val.copy()
                    print(f"[Stage-1] FD004: Computed val_rmse_last={metric_val:.4f} for checkpoint selection")
                else:
                    # Fallback to val_eol_loss if validation set is empty
                    metric_val = float(epoch_val_eol_loss)
                    print(f"[Stage-1] FD004: val_rmse_last not available, using val_eol_loss={metric_val:.4f}")
            except Exception as e:
                # Fallback to val_eol_loss if computation fails
                metric_val = float(epoch_val_eol_loss)
                print(f"[Stage-1] FD004: val_rmse_last computation failed ({e}), using val_eol_loss={metric_val:.4f}")
        elif best_metric == "val_total":
            if dataset_name.upper() == "FD004":
                raise ValueError(
                    "[Stage-1] FD004 cannot use val_total for checkpoint selection. "
                    "Use val_rmse_last or val_eol_loss (both use capped targets)."
                )
            metric_val = float(epoch_val_loss)
        elif best_metric == "val_eol_weighted":
            metric_val = float(epoch_val_eol_weighted)
        elif best_metric == "val_eol":
            metric_val = float(epoch_val_eol_loss)
        else:
            metric_val = float(epoch_val_loss)

        # Stage -1: Explicit logging of checkpoint metric and capping status
        # encoder_frozen is computed earlier in the epoch loop
        encoder_frozen_state = encoder_frozen if 'encoder_frozen' in locals() else (freeze_encoder or (freeze_encoder_epochs > 0 and epoch < freeze_encoder_epochs))
        log_msg = (
            f"[checkpoint] epoch={epoch} encoder_frozen={encoder_frozen_state} "
            f"checkpoint_allowed={checkpoint_allowed} min_epoch={min_epoch} epochs_since_unfreeze={epochs_since_unfreeze} "
            f"allow_best_update={allow_best_update} best_metric={best_metric} metric={metric_val:.4f}"
        )
        if reason_blocked:
            log_msg += f" reason_blocked={reason_blocked}"
        if pred_range_last is not None:
            log_msg += f" pred_range={pred_range_last:.2f} pred_std={pred_std_last:.2f}"
        log_msg += f" (cap_targets={cap_rul_targets}, max_rul={max_rul if max_rul is not None else 125.0})"
        print(log_msg)
        
        # Track checkpoint decisions in history
        history["checkpoint_allowed"].append(checkpoint_allowed)
        history["checkpoint_min_epoch"].append(min_epoch)
        history["checkpoint_reason_blocked"].append(reason_blocked if reason_blocked else "")
        history["checkpoint_pred_range_last"].append(pred_range_last if pred_range_last is not None else float("nan"))
        history["checkpoint_pred_std_last"].append(pred_std_last if pred_std_last is not None else float("nan"))

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
            
            # Debug: verify we're saving full model state_dict
            state_dict = model.state_dict()
            num_keys = len(state_dict)
            head_keys = [k for k in state_dict.keys() if any(p in k.lower() for p in ["traj_head", "fc_rul", "fc_eol", "fc_health", "hi_head", "eol_head"])]
            has_heads = len(head_keys) > 0
            
            torch.save(
                {
                    "model_state_dict": state_dict,
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
            print(f"[checkpoint-save] saving model.state_dict keys={num_keys} has_heads={has_heads} path={best_model_path.name}")
            if not has_heads:
                print(f"  ⚠️  WARNING: No head keys found in checkpoint! Head patterns: {head_keys[:5] if head_keys else 'none'}")
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
    # 7a. Evaluate on train/val splits (LAST + ALL; truncated-aware)
    # ===================================================================
    def _evaluate_last_all_on_df(
        df_split: pd.DataFrame,
        *,
        split_name: str,
    ) -> Dict[str, Any]:
        y_col = "RUL_raw" if ("RUL_raw" in df_split.columns) else "RUL"
        if y_col not in df_split.columns:
            raise KeyError(f"Expected '{y_col}' in df_split for split={split_name}")

        wc = WindowConfig(past_len=int(past_len), horizon=1, stride=1, require_full_horizon=False, pad_mode="clamp")
        tc = TargetConfig(max_rul=int(max_rul if max_rul is not None else 125), cap_targets=True, eol_target_mode="current_from_df")

        built_split = build_sliding_windows(
            df_split,
            feature_cols,
            target_col=y_col,
            unit_col="UnitNumber",
            time_col="TimeInCycles",
            cond_col="ConditionID",
            window_cfg=wc,
            target_cfg=tc,
            return_mask=False,
        )

        X = built_split["X"]
        y_true = built_split["Y_eol"].astype(float)
        # Stage -1: FD004 evaluation consistency (clip y_true to [0, max_rul])
        if dataset_name.upper() == "FD004":
            max_rul_f = float(max_rul if max_rul is not None else 125.0)
            y_true = np.clip(y_true, 0.0, max_rul_f)
            y_true_max = float(np.max(y_true))
            if y_true_max > max_rul_f + 1e-6:
                raise ValueError(
                    f"[Stage-1] FD004 evaluation clipping failed: y_true max={y_true_max:.2f} > {max_rul_f:.2f}"
                )
        unit_ids = built_split["unit_ids"].astype(np.int64)
        cond_ids = built_split["cond_ids"].astype(np.int64)

        # Condition-wise scaling
        X_scaled = np.empty_like(X, dtype=np.float32)
        for cond in np.unique(cond_ids):
            cond = int(cond)
            idxs = np.where(cond_ids == cond)[0]
            scaler = scaler_dict.get(cond, scaler_dict.get(0))
            flat = X[idxs].reshape(-1, len(feature_cols))
            X_scaled[idxs] = scaler.transform(flat).reshape(-1, int(past_len), len(feature_cols)).astype(np.float32)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        cond_t = torch.tensor(cond_ids, dtype=torch.long).to(device) if num_conditions > 1 else None

        with torch.no_grad():
            out = model(
                encoder_inputs=X_t,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
                cond_ids=cond_t,
            )
            if isinstance(out, dict):
                e = out.get("eol", out.get("rul"))
            else:
                e = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
            if torch.is_tensor(e):
                y_pred = e.view(-1).detach().cpu().numpy().astype(float)
            else:
                y_pred = np.asarray(e, dtype=float).reshape(-1)
        y_pred = np.clip(y_pred, 0.0, float(max_rul)).astype(float)

        clip = (0.0, float(max_rul)) if max_rul is not None else None
        m_all = compute_all_samples_metrics(y_true, y_pred, unit_ids=unit_ids, clip=clip)
        m_last = compute_last_per_unit_metrics(unit_ids, y_true, y_pred, clip=clip)
        return {
            **m_last,
            **m_all,
            "dataset_split": split_name,
            "last_definition": m_last.get("note_last_definition", "LAST_AVAILABLE_PER_UNIT (truncated-aware)"),
        }

    try:
        train_units_set = set(train_units.detach().cpu().numpy().astype(int).tolist())
        val_units_set = set(val_units.detach().cpu().numpy().astype(int).tolist())
        df_train_split_df = df_train[df_train["UnitNumber"].isin(train_units_set)].copy()
        df_val_split_df = df_train[df_train["UnitNumber"].isin(val_units_set)].copy()

        print("\n[7a] Evaluating on train/val splits...")
        metrics_train = _evaluate_last_all_on_df(df_train_split_df, split_name="train")
        metrics_val = _evaluate_last_all_on_df(df_val_split_df, split_name="val")

        with open(results_dir / "metrics_train.json", "w") as f:
            json.dump(metrics_train, f, indent=2)
        with open(results_dir / "metrics_val.json", "w") as f:
            json.dump(metrics_val, f, indent=2)
        print(f"  Saved metrics_train.json and metrics_val.json to {results_dir}")

        # Save freeze audit data
        if "freeze_audit" in history and history["freeze_audit"]:
            with open(results_dir / "freeze_audit.json", "w") as f:
                json.dump(history["freeze_audit"], f, indent=2)
            print(f"  Saved freeze_audit.json to {results_dir}")
        
        # Stage -1: FD004 checkpoint re-selection based on val_rmse_last
        if dataset_name.upper() == "FD004" and "rmse_last" in metrics_val:
            val_rmse_last = float(metrics_val["rmse_last"])
            # Re-load all checkpoints and find best based on val_rmse_last
            # For now, log the metric - full re-selection would require saving all epoch checkpoints
            print(f"[Stage-1] FD004 val_rmse_last={val_rmse_last:.4f} (best checkpoint was selected using val_eol_loss during training)")
    except Exception as e:
        print(f"[7a] Warning: could not compute/save train/val metrics: {e}")
        metrics_train = {}
        metrics_val = {}
    
    # ===================================================================
    # 8. Evaluate on test set
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
    
    df_test_cond = df_test.groupby("UnitNumber")["ConditionID"].first()
    unit_ids_test = sorted(df_test["UnitNumber"].unique())
    
    # Per-condition metrics are defined on LAST (1 value per engine)
    y_pred_eol = np.array(test_metrics.get("y_pred_last", test_metrics.get("y_pred_eol", [])))
    y_true_eol = np.array(test_metrics.get("y_true_last", test_metrics.get("y_true_eol", y_test_true)))
    
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
            # Decoder type (for architecture tracking)
            "decoder_type": str(getattr(world_model_config, "decoder_type", "lstm")),
            # Decoder class name (for debugging)
            "decoder_class": (
                model.tf_decoder.__class__.__name__ if hasattr(model, 'tf_decoder') and model.tf_decoder is not None
                else (model.decoder.__class__.__name__ if hasattr(model, 'decoder') and model.decoder is not None else "unknown")
            ),
            # Freeze-aware checkpointing config
            "freeze_encoder": freeze_encoder,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "unfreeze_encoder_layers": unfreeze_encoder_layers_raw if 'unfreeze_encoder_layers_raw' in locals() else unfreeze_encoder_layers,
            "best_ckpt_min_epoch": best_ckpt_min_epoch,
            "best_ckpt_min_epoch_after_unfreeze": best_ckpt_min_epoch_after_unfreeze,
            "best_ckpt_require_pred_std_min": best_ckpt_require_pred_std_min,
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
    
    # Stage -1: Save feature columns list for diagnostics consistency
    feature_cols_path = results_dir / "feature_cols.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[Stage-1] Saved feature_cols.json ({len(feature_cols)} features) to {feature_cols_path}")
    
    # Stage -1: Save feature pipeline config for diagnostics reconstruction
    # Use deterministic derivation from feature_cols to ensure exact reconstruction
    if feature_pipeline_config is None:
        # Derive config deterministically from actual feature columns
        feature_pipeline_config = derive_feature_pipeline_config_from_feature_cols(
            feature_cols=feature_cols,
            dataset=dataset_name,
            default_twin_baseline_len=30,
            default_condition_vector_version=3,
        )
        ms_cfg = feature_pipeline_config["features"]["multiscale"]
        print(f"[Stage-1] feature_pipeline_config not provided; derived from feature_cols (schema_version=2)")
        print(f"   windows_short={ms_cfg['windows_short']}, windows_medium={ms_cfg['windows_medium']}, windows_long={ms_cfg['windows_long']}")
        print(f"   extra_temporal_base_cols_selected=<{len(ms_cfg.get('extra_temporal_base_cols_selected', []))}>")
    else:
        # Ensure schema_version and dataset are set
        if "schema_version" not in feature_pipeline_config:
            feature_pipeline_config["schema_version"] = 2
        if "dataset" not in feature_pipeline_config:
            feature_pipeline_config["dataset"] = dataset_name
        
        # If provided config lacks extra_temporal_base_cols_selected, derive and fill it
        ms_cfg = feature_pipeline_config.get("features", {}).get("multiscale", {})
        if "extra_temporal_base_cols_selected" not in ms_cfg:
            print(f"[Stage-1] Provided config missing extra_temporal_base_cols_selected; deriving from feature_cols")
            derived_config = derive_feature_pipeline_config_from_feature_cols(
                feature_cols=feature_cols,
                dataset=dataset_name,
                default_twin_baseline_len=feature_pipeline_config.get("phys_features", {}).get("twin_baseline_len", 30),
                default_condition_vector_version=feature_pipeline_config.get("phys_features", {}).get("condition_vector_version", 3),
            )
            ms_cfg["extra_temporal_base_cols_selected"] = derived_config["features"]["multiscale"]["extra_temporal_base_cols_selected"]
            ms_cfg["extra_temporal_base_max_cols"] = len(ms_cfg["extra_temporal_base_cols_selected"])
            feature_pipeline_config["schema_version"] = 2
        
        # Validate config
        issues = validate_feature_pipeline_config(feature_pipeline_config)
        if issues:
            print(f"⚠️ [Stage-1] Config validation issues: {issues}")
            print(f"   Deriving complete config from feature_cols to fix issues")
            feature_pipeline_config = derive_feature_pipeline_config_from_feature_cols(
                feature_cols=feature_cols,
                dataset=dataset_name,
                default_twin_baseline_len=feature_pipeline_config.get("phys_features", {}).get("twin_baseline_len", 30),
                default_condition_vector_version=feature_pipeline_config.get("phys_features", {}).get("condition_vector_version", 3),
            )
    
    # Log key toggles
    features_cfg = feature_pipeline_config.get("features", {})
    phys_features_cfg = feature_pipeline_config.get("phys_features", {})
    use_multiscale = features_cfg.get("use_multiscale_features", True)
    use_twin = phys_features_cfg.get("use_digital_twin_residuals", False)
    use_resid = feature_pipeline_config.get("use_residuals", False)
    use_cond = phys_features_cfg.get("use_condition_vector", False)
    
    feature_pipeline_config_path = results_dir / "feature_pipeline_config.json"
    with open(feature_pipeline_config_path, "w") as f:
        json.dump(feature_pipeline_config, f, indent=2)
    schema_ver = feature_pipeline_config.get("schema_version", 1)
    print(f"[Stage-1] Saved feature_pipeline_config.json (schema_version={schema_ver}) to {feature_pipeline_config_path}")
    print(f"   Config: multiscale={use_multiscale}, twin={use_twin}, resid={use_resid}, cond={use_cond}")
    if schema_ver >= 2:
        ms_cfg = feature_pipeline_config.get("features", {}).get("multiscale", {})
        base_cols_count = len(ms_cfg.get("extra_temporal_base_cols_selected", []))
        print(f"   extra_temporal_base_cols_selected: {base_cols_count} columns")

    # Save split metrics as standalone artifact (standard name)
    try:
        metrics_test_path = results_dir / "metrics_test.json"
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
    t_end_pos_np = built.get("t_end_pos")
    
    # Extract horizon mask if available (for padding-aware loss computation)
    horizon_mask_all = built.get("mask")  # (N, H, 1) or None
    if horizon_mask_all is not None:
        horizon_mask_all = torch.from_numpy(horizon_mask_all.astype(np.float32))
    else:
        horizon_mask_all = None

    # --------------------------------------------------------------
    # Train/val split (engine-level) BEFORE fitting X scaler
    # (so scaler is fit on TRAIN windows only, no leakage)
    # --------------------------------------------------------------
    N_all = int(X_train_np.shape[0])
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

    # --------------------------------------------------------------
    # WM-V1 wiring proof debug instrumentation (toggleable)
    # --------------------------------------------------------------
    debug_wiring_enable = bool(getattr(world_model_config, "debug_wiring_enable", False))
    debug_wiring_batches = int(getattr(world_model_config, "debug_wiring_batches", 1) or 1)
    debug_wiring_epochs = int(getattr(world_model_config, "debug_wiring_epochs", 1) or 1)
    debug_wiring_save_json = bool(getattr(world_model_config, "debug_wiring_save_json", True))

    wiring_debug: Dict[str, Any] = {
        "debug_wiring_enable": debug_wiring_enable,
        "debug_wiring_batches": debug_wiring_batches,
        "debug_wiring_epochs": debug_wiring_epochs,
        "debug_wiring_save_json": debug_wiring_save_json,
        "notes": "WM-V1 wiring proof: masks/weights + per-loss contributions (scalar-only).",
    }

    # --------------------------------------------------------------
    # Informative window sampling (WM-V1 only): keep all informative windows and
    # only a fraction of non-informative (fully capped) windows.
    # Default OFF (backwards compatible).
    # --------------------------------------------------------------
    informative_stats: Dict[str, Any] = {
        "informative_sampling_enable": bool(getattr(world_model_config, "informative_sampling_enable", False)),
        "informative_sampling_mode": str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap")),
        "informative_eps_norm": float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6),
        "keep_prob_noninformative": float(getattr(world_model_config, "keep_prob_noninformative", 0.1) or 0.1),
    }
    if bool(getattr(world_model_config, "informative_sampling_enable", False)) and train_indices.size > 0:
        mode = str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap") or "future_min_lt_cap")
        eps = float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6)
        keep_p = float(getattr(world_model_config, "keep_prob_noninformative", 0.1) or 0.1)
        keep_p = float(np.clip(keep_p, 0.0, 1.0))

        # Use normalized future RUL targets if available; otherwise normalize from cycles.
        y_tr = Y_rul_np[train_indices]  # expected (N_tr, H) normalized
        if y_tr.ndim != 2:
            y_tr = y_tr.reshape(y_tr.shape[0], -1)
        is_norm = bool(float(np.max(y_tr)) <= 1.0 + 1e-3)
        y_tr_norm = np.clip(y_tr, 0.0, 1.0) if is_norm else np.clip(y_tr / float(max_rul if max_rul is not None else 125.0), 0.0, 1.0)

        future_min = y_tr_norm.min(axis=1)  # (N_tr,)
        # ----------------------------------------------------------
        # P0.2: Stricter informative sampling (ADR-0010)
        # ----------------------------------------------------------
        if mode == "future_has_zero":
            is_inf = future_min <= float(eps)
        elif mode == "uncapped_frac":
            # NEW (ADR-0010): Require at least X% of future timesteps to be uncapped
            uncapped_frac_threshold = float(getattr(world_model_config, "informative_uncapped_frac_threshold", 0.3) or 0.3)
            uncapped_mask = y_tr_norm < (1.0 - float(eps))  # (N_tr, H)
            uncapped_frac = uncapped_mask.mean(axis=1)  # (N_tr,) fraction uncapped per sample
            is_inf = uncapped_frac >= uncapped_frac_threshold
        else:
            # Legacy: "future_min_lt_cap" - informative if ANY timestep is below cap
            is_inf = future_min < (1.0 - float(eps))
        inf_frac = float(is_inf.mean()) if is_inf.size > 0 else 0.0

        rng = np.random.default_rng(42)
        keep_rand = rng.random(is_inf.shape[0]) < keep_p
        keep = is_inf | keep_rand
        kept_frac = float(keep.mean()) if keep.size > 0 else 0.0

        # Update train_indices (val stays unchanged)
        train_indices = train_indices[keep]

        # Log + stash for summary
        informative_stats.update(
            {
                "train_informative_frac_before": inf_frac,
                "train_keep_frac": kept_frac,
                "train_n_before": int(is_inf.shape[0]),
                "train_n_after": int(train_indices.shape[0]),
            }
        )
        try:
            informative_stats["train_kept_indices_head"] = train_indices[:20].astype(int).tolist()
        except Exception:
            pass
        if bool(getattr(world_model_config, "log_informative_stats", True)):
            all_one_frac = float((future_min >= (1.0 - float(eps))).mean()) if future_min.size > 0 else 0.0
            print(
                "[WorldModelV1][infwin] "
                f"mode={mode} eps_norm={eps:g} keep_p_noninf={keep_p:.3f} "
                f"train_n={int(is_inf.shape[0])}->{int(train_indices.shape[0])} "
                f"informative_frac={inf_frac:.6f} all_one_future_frac={all_one_frac:.6f}"
            )

    wiring_debug["informative_sampling"] = dict(informative_stats)

    # ------------------------------------------------------------------
    # Debug: dataset-level normalization statistics
    # ------------------------------------------------------------------
    print("=== WorldModelV1 Debug: Dataset stats ===")
    # IMPORTANT: Avoid full-array std() here (can allocate multiple GB).
    # Use a small deterministic subset for stats.
    n_stats = int(min(1024, X_train_np.shape[0]))
    X_stats = X_train_np[:n_stats]
    print(
        "X_train_np(subset):  mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f} (n={})".format(
            float(X_stats.mean()),
            float(X_stats.std()),
            float(X_stats.min()),
            float(X_stats.max()),
            n_stats,
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

    # --------------------------------------------------------------
    # [dbg] Feature-group stats + scaler sanity (once per run)
    # --------------------------------------------------------------
    try:
        from src.additional_features import group_feature_columns
        from src.tools.debug_stats import tensor_stats

        groups = group_feature_columns(feature_cols)
        group_indices: dict[str, list[int]] = {}
        for gname, cols in groups.items():
            idxs = [i for i, c in enumerate(feature_cols) if c in set(cols)]
            group_indices[gname] = idxs

        print("=== [dbg] TRAIN feature group stats (subset) ===")
        # Use a small subset to keep this cheap even for large N
        n_dbg = int(min(512, X_train_np.shape[0]))
        X_dbg = X_train_np[:n_dbg]
        tensor_stats("TRAIN_X_all", X_dbg)
        for gname, idxs in group_indices.items():
            if idxs:
                tensor_stats(f"TRAIN_X_{gname}", X_dbg[:, :, idxs])
        print("==============================================")
    except Exception as e:
        print(f"[dbg] Warning: could not compute feature-group stats: {e}")

    # --------------------------------------------------------------
    # Fit X scaler on TRAIN windows only, apply to train+val (and persist)
    # --------------------------------------------------------------
    try:
        import os
        from src.tools.x_scaler import fit_x_scaler, transform_x, save_scaler, clip_x
        from src.tools.debug_stats import tensor_stats

        # Fit only on TRAIN split
        X_tr_np_raw = X_train_np[train_indices]
        x_scaler = fit_x_scaler(X_tr_np_raw, max_rows=2_000_000, random_state=42)
        x_scaler_path = os.path.join(str(results_dir), "world_model_v1_x_scaler.pkl")
        save_scaler(x_scaler_path, x_scaler)
        print(f"[WorldModelV1] Saved X scaler to {x_scaler_path}")

        # dbg before/after (small subset)
        tensor_stats("TRAIN_X_before_scaler", X_tr_np_raw[:256])
        X_tr_np = transform_x(x_scaler, X_tr_np_raw)
        X_tr_np, frac = clip_x(X_tr_np, clip=10.0)
        print(f"[WorldModelV1] TRAIN X clip frac={frac:.6f}")
        tensor_stats("TRAIN_X_after_scaler", X_tr_np[:256])

        X_val_np_raw = X_train_np[val_indices]
        X_val_np = transform_x(x_scaler, X_val_np_raw)
        X_val_np, frac = clip_x(X_val_np, clip=10.0)
        print(f"[WorldModelV1] VAL   X clip frac={frac:.6f}")

        # Scale future_cond_np consistently using Cond_* stats from the fitted X scaler.
        # (future_cond_np itself is NOT part of X; but its features correspond to Cond_* columns in X)
        cond_idx = np.array([i for i, c in enumerate(feature_cols) if c.startswith("Cond_")], dtype=np.int64)
        future_cond_tr_np = future_cond_np[train_indices]
        future_cond_val_np = future_cond_np[val_indices]
        if cond_idx.size > 0:
            cond_mean = x_scaler.mean_[cond_idx]
            cond_scale = x_scaler.scale_[cond_idx]
            future_cond_tr_np = (future_cond_tr_np - cond_mean[None, None, :]) / cond_scale[None, None, :]
            future_cond_val_np = (future_cond_val_np - cond_mean[None, None, :]) / cond_scale[None, None, :]
            future_cond_tr_np = np.clip(future_cond_tr_np, -10.0, 10.0).astype(np.float32, copy=False)
            future_cond_val_np = np.clip(future_cond_val_np, -10.0, 10.0).astype(np.float32, copy=False)
        else:
            # No Cond_* features in X; keep future conds as-is.
            future_cond_tr_np = future_cond_tr_np.astype(np.float32, copy=False)
            future_cond_val_np = future_cond_val_np.astype(np.float32, copy=False)

    except Exception as e:
        print(f"[WorldModelV1] Warning: failed to fit/apply X scaler; proceeding unscaled: {e}")
        X_tr_np = X_train_np[train_indices]
        X_val_np = X_train_np[val_indices]
        future_cond_tr_np = future_cond_np[train_indices].astype(np.float32, copy=False)
        future_cond_val_np = future_cond_np[val_indices].astype(np.float32, copy=False)

    # Targets + aux stay unscaled; only X is scaled.
    Y_sens_train = torch.from_numpy(Y_sens_np).float()
    Y_rul_train = torch.from_numpy(Y_rul_np).float()
    Y_hi_train = torch.from_numpy(Y_hi_np).float()
    # Use the scaled future condition sequences (if X scaler succeeded), otherwise raw.
    future_cond_tr = torch.from_numpy(future_cond_tr_np).float()
    future_cond_val = torch.from_numpy(future_cond_val_np).float()
    cond_ids = torch.tensor(cond_id_list, dtype=torch.long)

    X_tr = torch.from_numpy(X_tr_np).float()
    X_val = torch.from_numpy(X_val_np).float()

    Y_sens_tr = Y_sens_train[train_indices]
    Y_rul_tr = Y_rul_train[train_indices]
    Y_hi_tr = Y_hi_train[train_indices]
    cond_tr = cond_ids[train_indices]

    Y_sens_val = Y_sens_train[val_indices]
    Y_rul_val = Y_rul_train[val_indices]
    Y_hi_val = Y_hi_train[val_indices]
    cond_val = cond_ids[val_indices]
    
    # Extract horizon masks for train/val splits if available
    use_horizon_mask = bool(getattr(world_model_config, "use_horizon_mask", False))
    if use_horizon_mask and horizon_mask_all is not None:
        horizon_mask_tr = horizon_mask_all[train_indices]  # (N_tr, H, 1)
        horizon_mask_val = horizon_mask_all[val_indices]   # (N_val, H, 1)
        
        # Pre-check: verify padding exists (Step 0.5)
        pad_frac_tr = 1.0 - horizon_mask_tr.mean().item()
        pad_frac_val = 1.0 - horizon_mask_val.mean().item()
        print(f"[Pre-check] train padding fraction: {pad_frac_tr:.4f}")
        print(f"[Pre-check] val padding fraction: {pad_frac_val:.4f}")
        if pad_frac_tr < 1e-4:
            print("[WARNING] No padding detected in training data. Horizon masking may be unnecessary.")
    else:
        horizon_mask_tr = None
        horizon_mask_val = None

    # Optional: include per-sample metadata for wiring audits (unit_id and window end position).
    # Kept behind debug_wiring_enable so normal training data shape stays unchanged.
    debug_wiring_enable = bool(getattr(world_model_config, "debug_wiring_enable", False))
    if debug_wiring_enable and (t_end_pos_np is not None):
        unit_ids_t_all = torch.from_numpy(unit_ids_np.astype(np.int64))
        t_end_pos_t_all = torch.from_numpy(np.asarray(t_end_pos_np, dtype=np.int64))
        unit_ids_tr = unit_ids_t_all[train_indices]
        t_end_tr = t_end_pos_t_all[train_indices]
        unit_ids_val = unit_ids_t_all[val_indices]
        t_end_val = t_end_pos_t_all[val_indices]
        if use_horizon_mask and horizon_mask_tr is not None:
            train_ds = torch.utils.data.TensorDataset(
                X_tr, Y_sens_tr, Y_rul_tr, Y_hi_tr, cond_tr, future_cond_tr, horizon_mask_tr, unit_ids_tr, t_end_tr
            )
            val_ds = torch.utils.data.TensorDataset(
                X_val, Y_sens_val, Y_rul_val, Y_hi_val, cond_val, future_cond_val, horizon_mask_val, unit_ids_val, t_end_val
            )
        else:
            train_ds = torch.utils.data.TensorDataset(
                X_tr, Y_sens_tr, Y_rul_tr, Y_hi_tr, cond_tr, future_cond_tr, unit_ids_tr, t_end_tr
            )
            val_ds = torch.utils.data.TensorDataset(
                X_val, Y_sens_val, Y_rul_val, Y_hi_val, cond_val, future_cond_val, unit_ids_val, t_end_val
            )
    else:
        if use_horizon_mask and horizon_mask_tr is not None:
            train_ds = torch.utils.data.TensorDataset(
                X_tr, Y_sens_tr, Y_rul_tr, Y_hi_tr, cond_tr, future_cond_tr, horizon_mask_tr
            )
            val_ds = torch.utils.data.TensorDataset(
                X_val, Y_sens_val, Y_rul_val, Y_hi_val, cond_val, future_cond_val, horizon_mask_val
            )
        else:
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

            def infer_checkpoint_input_dim(ckpt: object) -> Optional[int]:
                # 1) meta.input_dim if present
                if isinstance(ckpt, dict):
                    meta = ckpt.get("meta", {})
                    if isinstance(meta, dict) and "input_dim" in meta:
                        try:
                            return int(meta["input_dim"])
                        except Exception:
                            pass

                    # 2) state dict inference via common key names
                    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                    if isinstance(sd, dict):
                        for k, v in sd.items():
                            if not hasattr(v, "shape"):
                                continue
                            # EOLFullTransformerEncoder uses `input_proj: Linear(input_dim, d_model)`
                            if str(k).endswith("input_proj.weight") and len(v.shape) == 2:
                                return int(v.shape[1])
                            if "input_projection" in str(k) and str(k).endswith(".weight") and len(v.shape) == 2:
                                return int(v.shape[1])
                            if str(k).endswith("feat_proj.weight") and len(v.shape) == 2:
                                return int(v.shape[1])
                return None

            ckpt_input_dim = infer_checkpoint_input_dim(state)
            if ckpt_input_dim is not None:
                if int(ckpt_input_dim) != int(input_dim):
                    raise RuntimeError(
                        "Encoder checkpoint feature-set mismatch:\n"
                        f"- checkpoint_input_dim={int(ckpt_input_dim)}\n"
                        f"- current_input_dim={int(input_dim)}\n"
                        "This usually happens when features.include_groups is enabled or feature engineering differs.\n"
                        "Fix: use the matching encoder run / same features, or disable include_groups."
                    )
            else:
                print(
                    "[WorldModelV1] Warning: could not infer checkpoint input_dim; "
                    "skipping feature-set mismatch guard."
                )

            # Accept either a full checkpoint or a plain state_dict
            if isinstance(state, dict) and "model_state_dict" in state:
                sd = state["model_state_dict"]
            else:
                sd = state

            def load_encoder_backbone_from_checkpoint(
                *,
                state_dict: dict[str, torch.Tensor],
                model: nn.Module,
                allowed_substrings: Optional[list[str]] = None,
            ) -> None:
                # Conservative backbone loader:
                # 1) Keep shared_head always (even though it contains "head")
                # 2) Else, drop obvious task heads / aux modules
                # 3) Else, keep only known backbone components
                drop_tokens = [
                    "fc_rul",
                    "fc_health",
                    "fc_hi",
                    "fc_eol",
                    "bucket",
                    "damage",
                    "calib",
                    "risk",
                    "sigma",
                    "quantile",
                    # Generic "head" is OK as long as we special-case shared_head above.
                    "head",
                ]
                keep_tokens = allowed_substrings or [
                    "input_proj",
                    "pos_encoding",
                    "pos_emb",
                    "transformer",
                    "attn_pool",
                    "condition_embedding",
                    "cond_proj",
                    "cond_encoder",
                    "cond_recon",
                    "condition_normalizer",
                    "shared_head",
                ]

                selected: dict[str, torch.Tensor] = {}
                selected_names: list[str] = []
                skipped_names: list[str] = []
                for k, v in state_dict.items():
                    ks = str(k)

                    # Special-case: shared_head is part of the representation trunk
                    if "shared_head" in ks:
                        selected[k] = v
                        selected_names.append(ks)
                        continue

                    if any(tok in ks for tok in drop_tokens):
                        skipped_names.append(ks)
                        continue

                    if any(tok in ks for tok in keep_tokens):
                        selected[k] = v
                        selected_names.append(ks)
                        continue

                    skipped_names.append(ks)

                print(
                    f"[WorldModelV1] Encoder ckpt tensors: total={len(state_dict)} | selected(backbone)={len(selected)} | skipped={len(skipped_names)}"
                )
                if selected_names:
                    print("[WorldModelV1] Selected backbone tensors (first 20):")
                    for name in selected_names[:20]:
                        print(f"  + {name}")
                if skipped_names:
                    print("[WorldModelV1] Skipped tensors (first 20):")
                    for name in skipped_names[:20]:
                        print(f"  - {name}")

                model.load_state_dict(selected, strict=False)

            if isinstance(sd, dict):
                load_encoder_backbone_from_checkpoint(
                    state_dict=sd,
                    model=encoder,
                    allowed_substrings=[
                        "input_proj",
                        "pos_encoding",
                        "transformer",
                        "attn_pool",
                        "condition_embedding",
                        "cond_proj",
                        "cond_encoder",
                        "cond_recon",
                        "condition_normalizer",
                        "shared_head",
                    ],
                )
                print("[WorldModelV1] Encoder backbone loaded successfully (heads excluded).")
            else:
                encoder.load_state_dict(sd, strict=False)
                print("[WorldModelV1] Encoder weights loaded (strict=False).")
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

    def count_trainable_params(module: nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

    def _freeze_encoder() -> None:
        for p in world_model.encoder.parameters():
            p.requires_grad = False
        world_model.encoder.eval()

    def _partial_unfreeze_encoder(*, unfreeze_last_k: int) -> list[str]:
        """
        Unfreeze only:
        - last K transformer blocks (if discoverable)
        - shared_head (always, if present)
        """
        # Freeze everything first
        for p in world_model.encoder.parameters():
            p.requires_grad = False

        unfrozen_modules: list[str] = []

        # Unfreeze last K transformer layers if possible
        layers = None
        if hasattr(world_model.encoder, "transformer") and hasattr(world_model.encoder.transformer, "layers"):
            layers = world_model.encoder.transformer.layers
        if layers is not None and unfreeze_last_k > 0:
            k = min(int(unfreeze_last_k), len(layers))
            # Mark last k layers trainable
            for idx, layer in enumerate(list(layers)[-k:], start=len(layers) - k):
                for p in layer.parameters():
                    p.requires_grad = True
                unfrozen_modules.append(f"encoder.transformer.layers[{idx}]")

        # Always unfreeze shared_head if present (part of encoder trunk)
        if hasattr(world_model.encoder, "shared_head") and world_model.encoder.shared_head is not None:
            for p in world_model.encoder.shared_head.parameters():
                p.requires_grad = True
            unfrozen_modules.append("encoder.shared_head")

        world_model.encoder.train()
        return unfrozen_modules

    # Stage A: freeze encoder (either explicit freeze_encoder, or for first N epochs)
    if freeze_encoder or freeze_encoder_epochs > 0:
        print(
            f"[WorldModelV1] Stage-A encoder freeze enabled (freeze_encoder={freeze_encoder}, "
            f"freeze_encoder_epochs={freeze_encoder_epochs})"
        )
        _freeze_encoder()
    print(
        f"[WorldModelV1] Encoder configuration -> "
        f"checkpoint: {encoder_ckpt_path if encoder_ckpt_path else 'None'}, "
        f"freeze_encoder: {freeze_encoder}, freeze_encoder_epochs: {freeze_encoder_epochs}, "
        f"unfreeze_encoder_layers: {unfreeze_encoder_layers}, encoder_lr_mult: {encoder_lr_mult}"
    )

    print(
        f"[WorldModelV1] Trainable params after Stage-A: "
        f"encoder={count_trainable_params(world_model.encoder):,} "
        f"total={count_trainable_params(world_model):,}"
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

    did_stage_b_unfreeze = False
    stage_b_epoch0 = int(freeze_encoder_epochs) if int(freeze_encoder_epochs) > 0 else None
    stage_b_unfrozen_modules: list[str] = []

    for epoch in range(num_epochs):
        # Stage B: unfreeze encoder after N epochs (epoch is 0-based; epoch+1 is human-readable)
        if (not did_stage_b_unfreeze) and freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            unfrozen = _partial_unfreeze_encoder(unfreeze_last_k=unfreeze_encoder_layers)
            stage_b_unfrozen_modules = list(unfrozen)
            did_stage_b_unfreeze = True
            print(
                f"[WorldModelV1] Stage-B unfreeze at epoch {epoch+1} "
                f"(freeze_encoder_epochs={freeze_encoder_epochs}): "
                f"unfroze={unfrozen if unfrozen else ['<none detected>']}"
            )
            print(
                f"[WorldModelV1] Trainable params after Stage-B: "
                f"encoder={count_trainable_params(world_model.encoder):,} "
                f"total={count_trainable_params(world_model):,}"
            )
            optimizer = _make_optimizer()

        # Wiring-probe selection helpers (must be defined regardless of HI/RUL loss presence).
        def _should_wiring_probe(batch_i: int) -> bool:
            if not debug_wiring_enable:
                return False
            if batch_i >= debug_wiring_batches:
                return False
            # Always probe epoch 0, and also probe first Stage-B epoch (after unfreeze).
            if epoch == 0:
                return True
            if stage_b_epoch0 is not None and epoch == stage_b_epoch0:
                return True
            # Optional additional probing for early epochs
            return epoch < debug_wiring_epochs

        def _probe_key(batch_i: int) -> str:
            # Keep backward-compat for epoch0_batch0
            if epoch == 0 and batch_i == 0:
                return "epoch0_batch0"
            # Human-readable epoch index (epoch is 0-based)
            return f"epoch{epoch+1}_batch{batch_i}"

        # Per-epoch sanity log (trainable encoder params + lrs)
        try:
            lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
        except Exception:
            lrs = []
        print(
            f"[WorldModelV1][epoch {epoch+1}/{num_epochs}] "
            f"trainable_encoder_params={count_trainable_params(world_model.encoder):,} "
            f"lrs={lrs}"
        )

        world_model.train()
        running_train = 0.0
        n_train_samples = 0
        # Track saturation-penalty activity (train) once per epoch
        sat_loss_sum = 0.0
        sat_mask_frac_sum = 0.0
        sat_batches = 0
        sat_below_thr_frac_sum = 0.0
        sat_below_thr_batches = 0
        cap_frac_sum = 0.0
        cap_batches = 0
        early_drop_frac_sum = 0.0
        mask_keep_frac_sum = 0.0
        mask_batches = 0
        # Per-epoch prediction stats (normalized RUL seq); helps diagnose sat_mask_frac_mean==0
        pred_rul_sum = 0.0
        pred_rul_count = 0
        pred_rul_min = None
        pred_rul_max = None
        # Late-window (future hits zero) diagnostics
        late_frac_sum = 0.0
        late_batches = 0
        true_future_min_sum = 0.0
        true_future_min_count = 0
        true_future_min_min = None
        true_future_min_max = None
        pred_future_min_sum = 0.0
        pred_future_min_count = 0
        pred_future_min_min = None
        pred_future_min_max = None
        # Informative-window batch diagnostics
        inf_frac_sum = 0.0
        inf_batches = 0
        all_one_frac_sum = 0.0
        all_one_batches = 0
        # Cap/plateau (all-cap future horizon) diagnostics
        all_cap_frac_sum = 0.0
        all_cap_batches = 0
        cap_weight_mean_sum = 0.0
        cap_weight_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Backwards-compatible unpacking (optionally includes horizon_mask, unit_id and t_end_pos for audits)
            unit_ids_b = None
            t_end_b = None
            mask_batch = None
            if isinstance(batch, (tuple, list)):
                if len(batch) == 9:  # With horizon_mask + unit_ids + t_end
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b, mask_batch, unit_ids_b, t_end_b = batch
                elif len(batch) == 8:  # With unit_ids + t_end (no mask)
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b, unit_ids_b, t_end_b = batch
                elif len(batch) == 7:  # With horizon_mask only
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b, mask_batch = batch
                else:  # Standard 6-tuple
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b = batch
            else:
                X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b = batch
            X_b = X_b.to(device)
            Y_sens_b = Y_sens_b.to(device)
            Y_rul_b = Y_rul_b.to(device)
            Y_hi_b = Y_hi_b.to(device)
            cond_b = cond_b.to(device)
            future_cond_b = future_cond_b.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
            if unit_ids_b is not None:
                unit_ids_b = unit_ids_b.to(device)
            if t_end_b is not None:
                t_end_b = t_end_b.to(device)

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

            # Optional: deterministic linear-decay RUL sequence from pred_rul0 (physics-consistent)
            if bool(getattr(world_model_config, "rul_linear_decay", False)) and (pred_rul is not None):
                # pred_rul is normalized in [0,1]; step size is 1/max_rul per horizon step
                max_rul_denom = float(getattr(world_model_config, "max_rul", 125.0))
                H = int(pred_rul.size(1))
                pred_rul0 = pred_rul[:, 0, 0].clamp(0.0, 1.0)  # (B,)
                k = torch.arange(H, device=pred_rul.device, dtype=pred_rul.dtype).view(1, H)
                pred_seq = (pred_rul0.unsqueeze(1) - k / max(max_rul_denom, 1e-6)).clamp(0.0, 1.0)
                pred_rul = pred_seq.unsqueeze(-1)  # (B,H,1)

            # Sensor trajectory loss
            loss_sensors = F.mse_loss(pred_sensors, Y_sens_b)
            loss = sensor_w * loss_sensors

            # HI future loss (piecewise HI targets already built into Y_hi_b)
            if hi_w > 0.0 and pred_hi is not None:
                # pred_hi: (B, H, 1), Y_hi_b: (B, H)
                # Optional WM-V1 late-window weighting (per sample) to emphasize late-life degradation.
                late_weight_enable = bool(getattr(world_model_config, "late_weight_enable", False))
                late_weight_apply_hi = bool(getattr(world_model_config, "late_weight_apply_hi", False))
                late_weight_factor = float(getattr(world_model_config, "late_weight_factor", 5.0) or 5.0)
                late_weight_eps_norm = float(getattr(world_model_config, "late_weight_eps_norm", 1e-6) or 1e-6)
                max_rul_cycles = float(getattr(world_model_config, "max_rul", 125.0))
                # Optional WM-V1 cap/plateau reweighting for HI future loss (default OFF)
                cap_rw_enable = bool(getattr(world_model_config, "cap_reweight_enable", False))
                cap_rw_eps = float(getattr(world_model_config, "cap_reweight_eps", 1e-6) or 1e-6)
                cap_rw_weight = float(getattr(world_model_config, "cap_reweight_weight", 0.05) or 0.05)
                cap_rw_apply_to = str(getattr(world_model_config, "cap_reweight_apply_to", "rul") or "rul").lower()
                # Optional cap-aware masking for HI future loss (default ON, but apply_to defaults to ["rul"])
                cap_mask_enable = bool(getattr(world_model_config, "cap_mask_enable", True))
                cap_mask_eps = float(getattr(world_model_config, "cap_mask_eps", 1e-6) or 1e-6)
                cap_mask_apply_to = getattr(world_model_config, "cap_mask_apply_to", ["rul"])
                if not isinstance(cap_mask_apply_to, (list, tuple, set)):
                    cap_mask_apply_to = [str(cap_mask_apply_to)]
                cap_mask_apply_set = {str(x).lower() for x in cap_mask_apply_to}

                if late_weight_enable and late_weight_apply_hi and late_weight_factor > 1.0:
                    # Detect normalized targets from RUL horizon (unit-safe).
                    y_rul_seq = Y_rul_b
                    if y_rul_seq.dim() == 3 and y_rul_seq.size(-1) == 1:
                        y_rul_seq = y_rul_seq.squeeze(-1)
                    is_norm = bool(float(y_rul_seq.detach().max().cpu()) <= 1.0 + 1e-3)
                    if is_norm:
                        true_rul_norm = y_rul_seq.clamp(0.0, 1.0)
                    else:
                        true_rul_norm = (y_rul_seq / max(max_rul_cycles, 1e-6)).clamp(0.0, 1.0)
                    fut_min = true_rul_norm.min(dim=1).values  # (B,)
                    late_mask = (fut_min <= float(late_weight_eps_norm))
                    w = 1.0 + late_mask.float() * (float(late_weight_factor) - 1.0)  # (B,)
                    if cap_rw_enable and (cap_rw_apply_to in {"hi", "both"}):
                        all_cap = (fut_min >= (1.0 - cap_rw_eps))
                        w_cap = torch.where(
                            all_cap,
                            torch.full_like(all_cap.float(), float(cap_rw_weight)),
                            torch.ones_like(all_cap.float()),
                        )
                        w = w * w_cap

                    hi_err2 = (pred_hi.squeeze(-1) - Y_hi_b) ** 2  # (B,H)
                    if cap_mask_enable and ("hi" in cap_mask_apply_set):
                        m_t = (true_rul_norm < float(1.0 - cap_mask_eps)).float()  # (B,H)
                        num_i = (hi_err2 * m_t).sum(dim=1)
                        den_i = m_t.sum(dim=1).clamp_min(1e-6)
                        hi_loss_per_sample = num_i / den_i
                        hi_loss = (w * hi_loss_per_sample).sum() / (w.sum() + 1e-6)
                    else:
                        hi_loss_per_sample = hi_err2.mean(dim=1)  # (B,)
                        hi_loss = (w * hi_loss_per_sample).mean()
                else:
                    # Optional cap/plateau reweighting without late-weighting.
                    if cap_rw_enable and (cap_rw_apply_to in {"hi", "both"}):
                        y_rul_seq = Y_rul_b
                        if y_rul_seq.dim() == 3 and y_rul_seq.size(-1) == 1:
                            y_rul_seq = y_rul_seq.squeeze(-1)
                        true_rul_norm = y_rul_seq.clamp(0.0, 1.0)
                        fut_min = true_rul_norm.min(dim=1).values  # (B,)
                        all_cap = (fut_min >= (1.0 - cap_rw_eps))
                        w_cap = torch.where(
                            all_cap,
                            torch.full_like(all_cap.float(), float(cap_rw_weight)),
                            torch.ones_like(all_cap.float()),
                        )
                        hi_err2 = (pred_hi.squeeze(-1) - Y_hi_b) ** 2  # (B,H)
                        hi_loss_per_sample = hi_err2.mean(dim=1)  # (B,)
                        hi_loss = (w_cap * hi_loss_per_sample).sum() / (w_cap.sum() + 1e-6)
                    else:
                        if cap_mask_enable and ("hi" in cap_mask_apply_set):
                            y_rul_seq = Y_rul_b
                            if y_rul_seq.dim() == 3 and y_rul_seq.size(-1) == 1:
                                y_rul_seq = y_rul_seq.squeeze(-1)
                            true_rul_norm = y_rul_seq.clamp(0.0, 1.0)
                            m_t = (true_rul_norm < float(1.0 - cap_mask_eps)).float()  # (B,H)
                            hi_err2 = (pred_hi.squeeze(-1) - Y_hi_b) ** 2
                            hi_loss = (hi_err2 * m_t).sum() / (m_t.sum() + 1e-6)
                        else:
                            hi_loss = F.mse_loss(pred_hi.squeeze(-1), Y_hi_b)
                loss = loss + hi_w * hi_loss

                # Wiring proof (HI): print/record only a tiny summary
                if _should_wiring_probe(batch_idx):
                    try:
                        pred_hi_seq_norm = pred_hi.squeeze(-1)  # (B,H)
                        true_hi_seq_norm = Y_hi_b               # (B,H)
                        hi_loss_per_sample_dbg = ((pred_hi_seq_norm - true_hi_seq_norm) ** 2).mean(dim=1)
                        k = _probe_key(batch_idx)
                        wiring_debug.setdefault(k, {})
                        wiring_debug[k]["hi"] = {
                            "pred_hi_seq_norm_shape": list(pred_hi_seq_norm.shape),
                            "true_hi_seq_norm_shape": list(true_hi_seq_norm.shape),
                            "pred_hi_requires_grad": bool(pred_hi_seq_norm.requires_grad),
                            "pred_hi_min_mean_max": [
                                float(pred_hi_seq_norm.detach().min().cpu()),
                                float(pred_hi_seq_norm.detach().mean().cpu()),
                                float(pred_hi_seq_norm.detach().max().cpu()),
                            ],
                            "true_hi_min_mean_max": [
                                float(true_hi_seq_norm.detach().min().cpu()),
                                float(true_hi_seq_norm.detach().mean().cpu()),
                                float(true_hi_seq_norm.detach().max().cpu()),
                            ],
                            "hi_loss_per_sample_mean": float(hi_loss_per_sample_dbg.detach().mean().cpu()),
                        }
                        # Sample-index proof (if metadata is present)
                        if unit_ids_b is not None and t_end_b is not None:
                            ws = (t_end_b - int(past_len) + 1).detach().cpu().numpy().astype(int).tolist()
                            wiring_debug[k].setdefault("sample_indices", {})
                            wiring_debug[k]["sample_indices"].update(
                                {
                                    "unit_ids_first16": unit_ids_b.detach().cpu().numpy().astype(int).tolist()[:16],
                                    "t_end_pos_first16": t_end_b.detach().cpu().numpy().astype(int).tolist()[:16],
                                    "window_start_first16": ws[:16],
                                }
                            )
                        # Human-readable console log (small)
                        print(
                            f"[WorldModelV1][wiring] hi: pred_requires_grad={bool(pred_hi_seq_norm.requires_grad)} "
                            f"pred(min/mean/max)="
                            f"{float(pred_hi_seq_norm.detach().min().cpu()):.3f}/"
                            f"{float(pred_hi_seq_norm.detach().mean().cpu()):.3f}/"
                            f"{float(pred_hi_seq_norm.detach().max().cpu()):.3f} "
                            f"loss_per_sample_mean={float(hi_loss_per_sample_dbg.detach().mean().cpu()):.6f}"
                        )
                    except Exception as e:
                        k = _probe_key(batch_idx)
                        wiring_debug.setdefault(k, {})
                        wiring_debug[k]["hi_error"] = str(e)

            # RUL future loss (L1 on cycles)
            # Backwards compatible:
            # - If new WM-V1 stabilizers are not enabled, keep legacy L1 loss with rul_future_loss_weight.
            # - If enabled, use full-horizon MSE (+ optional late ramp) and add monotonic + anti-sat penalties.
            if pred_rul is not None:
                pred_rul_h = pred_rul.squeeze(-1)  # (B,H)
                y_rul_h = Y_rul_b
                if y_rul_h.dim() == 3 and y_rul_h.size(-1) == 1:
                    y_rul_h = y_rul_h.squeeze(-1)

                # Aggregate pred stats for the epoch (normalized space; unit-safe)
                try:
                    pr_raw = pred_rul_h.detach()
                    max_rul_cycles = float(getattr(world_model_config, "max_rul", 125.0))
                    is_norm = bool(float(pr_raw.max().cpu()) <= 1.0 + 1e-3)
                    pr = pr_raw.clamp(0.0, 1.0) if is_norm else (pr_raw / max(max_rul_cycles, 1e-6)).clamp(0.0, 1.0)
                    pred_rul_sum += float(pr.mean().cpu())
                    pred_rul_count += 1
                    pr_min = float(pr.min().cpu())
                    pr_max = float(pr.max().cpu())
                    pred_rul_min = pr_min if pred_rul_min is None else min(pred_rul_min, pr_min)
                    pred_rul_max = pr_max if pred_rul_max is None else max(pred_rul_max, pr_max)
                except Exception:
                    pass

                # [dbg] Print true RUL sequence stats once (same tensor used for loss)
                if epoch == 0 and batch_idx == 0:
                    from src.tools.debug_stats import tensor_stats, batch_time_std, compare_two_samples

                    y_rul_seq_norm = y_rul_h.unsqueeze(-1)  # (B,H,1)
                    tensor_stats("true_rul_seq_norm", y_rul_seq_norm)
                    batch_time_std("true_rul_seq_norm", y_rul_seq_norm)
                    compare_two_samples("true_rul_seq_norm", y_rul_seq_norm, t_steps=10)

                # Informative-window batch stats (cheap): fraction of non-capped horizons in this batch.
                try:
                    yb = y_rul_h.detach()
                    is_norm_b = bool(float(yb.max().cpu()) <= 1.0 + 1e-3)
                    yb_norm = yb.clamp(0.0, 1.0) if is_norm_b else (yb / max(float(getattr(world_model_config, "max_rul", 125.0)), 1e-6)).clamp(0.0, 1.0)
                    eps_inf = float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6)
                    mode_inf = str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap") or "future_min_lt_cap")
                    fut_min_b = yb_norm.min(dim=1).values  # (B,)
                    if mode_inf == "future_has_zero":
                        is_inf_b = (fut_min_b <= eps_inf)
                    else:
                        is_inf_b = (fut_min_b < (1.0 - eps_inf))
                    inf_frac_sum += float(is_inf_b.float().mean().cpu())
                    inf_batches += 1
                    all_one_frac_sum += float((fut_min_b >= (1.0 - eps_inf)).float().mean().cpu())
                    all_one_batches += 1
                except Exception:
                    pass

                rul_traj_weight = getattr(world_model_config, "rul_traj_weight", None)
                rul_traj_late_ramp = bool(getattr(world_model_config, "rul_traj_late_ramp", False))
                rul_mono_future_weight = float(getattr(world_model_config, "rul_mono_future_weight", 0.0) or 0.0)
                rul_saturation_weight = float(getattr(world_model_config, "rul_saturation_weight", 0.0) or 0.0)
                margin = float(getattr(world_model_config, "rul_saturation_margin", 0.05) or 0.05)
                cap_thr = float(getattr(world_model_config, "rul_cap_threshold", 0.999999) or 0.999999)
                max_rul_cycles = float(getattr(world_model_config, "max_rul", 125.0))
                rul_train_max_cycles = getattr(world_model_config, "rul_train_max_cycles", None)
                rul_r0_only = bool(getattr(world_model_config, "rul_r0_only", False))
                rul_r0_points = getattr(world_model_config, "rul_r0_points", None)
                w_power = float(getattr(world_model_config, "rul_sample_weight_power", 0.0) or 0.0)
                w_min = float(getattr(world_model_config, "rul_sample_weight_min", 0.2) or 0.2)
                w_max = float(getattr(world_model_config, "rul_sample_weight_max", 3.0) or 3.0)

                use_new = (
                    (rul_traj_weight is not None)
                    or rul_traj_late_ramp
                    or (rul_mono_future_weight > 0.0)
                    or (rul_saturation_weight > 0.0)
                )

                if not use_new:
                    if rul_w > 0.0:
                        rul_loss = F.l1_loss(pred_rul_h, y_rul_h)
                        loss = loss + float(rul_w) * rul_loss
                else:
                    # Build masks in normalized + cycles space
                    # Unit-safe normalization:
                    # - If targets are already normalized (<= ~1): use as-is.
                    # - Else assume cycles in [0, max_rul] and normalize by max_rul.
                    true_seq = y_rul_h  # (B,H)
                    pred_seq = pred_rul_h  # (B,H)
                    is_norm = bool(float(true_seq.detach().max().cpu()) <= 1.0 + 1e-3)
                    if is_norm:
                        true_rul_norm = true_seq.clamp(0.0, 1.0)
                        pred_rul_norm = pred_seq.clamp(0.0, 1.0)
                    else:
                        denom = max(max_rul_cycles, 1e-6)
                        true_rul_norm = (true_seq / denom).clamp(0.0, 1.0)
                        pred_rul_norm = (pred_seq / denom).clamp(0.0, 1.0)

                    true_rul_seq_norm = true_rul_norm.unsqueeze(-1)  # (B,H,1)
                    pred_rul_seq_norm = pred_rul_norm.unsqueeze(-1)  # (B,H,1)
                    true_rul_cycles = true_rul_seq_norm * max(max_rul_cycles, 1e-6)

                    # ----------------------------------------------------------
                    # Cap/plateau reweighting (optional; default OFF)
                    # ----------------------------------------------------------
                    cap_rw_enable = bool(getattr(world_model_config, "cap_reweight_enable", False))
                    cap_rw_eps = float(getattr(world_model_config, "cap_reweight_eps", 1e-6) or 1e-6)
                    cap_rw_weight = float(getattr(world_model_config, "cap_reweight_weight", 0.05) or 0.05)
                    cap_rw_apply_to = str(getattr(world_model_config, "cap_reweight_apply_to", "rul") or "rul").lower()
                    # all-cap := entire future target horizon is capped at 1.0 (within eps)
                    all_cap_mask = (true_rul_seq_norm[:, :, 0].min(dim=1).values >= (1.0 - cap_rw_eps))  # (B,)
                    w_cap = torch.where(
                        all_cap_mask,
                        torch.full_like(all_cap_mask.float(), float(cap_rw_weight)),
                        torch.ones_like(all_cap_mask.float()),
                    )  # (B,)
                    try:
                        all_cap_frac_sum += float(all_cap_mask.float().mean().detach().cpu())
                        all_cap_batches += 1
                        cap_weight_mean_sum += float(w_cap.detach().mean().cpu())
                        cap_weight_batches += 1
                    except Exception:
                        pass

                    # ----------------------------------------------------------
                    # Cap-aware loss masking (optional; default ON)
                    # ----------------------------------------------------------
                    # mask_uncapped := (true_rul_seq_norm < 1 - eps) per timestep.
                    # If a sample has all timesteps capped, it contributes 0 (den=0 -> loss_i==0).
                    cap_mask_enable = bool(getattr(world_model_config, "cap_mask_enable", True))
                    cap_mask_eps = float(getattr(world_model_config, "cap_mask_eps", 1e-6) or 1e-6)
                    cap_mask_apply_to = getattr(world_model_config, "cap_mask_apply_to", ["rul"])
                    if not isinstance(cap_mask_apply_to, (list, tuple, set)):
                        cap_mask_apply_to = [str(cap_mask_apply_to)]
                    cap_mask_apply_set = {str(x).lower() for x in cap_mask_apply_to}
                    cap_thr_norm = float(1.0 - cap_mask_eps)
                    if cap_mask_enable and ("rul" in cap_mask_apply_set):
                        cap_mask = (true_rul_seq_norm < cap_thr_norm)  # bool (B,H,1)
                    else:
                        cap_mask = torch.ones_like(true_rul_seq_norm, dtype=torch.bool)
                    early_mask = torch.ones_like(cap_mask, dtype=torch.bool)
                    if rul_train_max_cycles is not None:
                        early_mask = (true_rul_cycles < float(rul_train_max_cycles))
                    mask = cap_mask & early_mask
                    mask_f = mask.float()

                    # ----------------------------------------------------------
                    # Extract and prepare horizon mask (for padding-aware loss)
                    # ----------------------------------------------------------
                    if use_horizon_mask and mask_batch is not None:
                        valid_mask_seq = mask_batch  # (B, H, 1) or (B, H)
                    else:
                        # No-op: create all-ones mask if horizon masking is disabled
                        valid_mask_seq = torch.ones_like(true_rul_seq_norm[:, :, 0])  # (B, H)
                    
                    # Fix shape/broadcasting: ensure valid_mask_seq is (B, H)
                    if valid_mask_seq.dim() == 3 and valid_mask_seq.size(-1) == 1:
                        valid_mask_seq = valid_mask_seq.squeeze(-1)  # (B, H)
                    elif valid_mask_seq.dim() == 2:
                        pass  # Already (B, H)
                    else:
                        raise ValueError(f"Unexpected valid_mask_seq shape: {valid_mask_seq.shape}")
                    
                    # Expand to (B, H, 1) for multiplication with mask_f
                    valid_mask_seq_exp = valid_mask_seq.unsqueeze(-1)  # (B, H, 1)
                    
                    # Combine masks: apply horizon mask to mask_f
                    mask_rul = mask_f * valid_mask_seq_exp  # (B, H, 1)

                    # ----------------------------------------------------------
                    # P0.1: Soft cap weighting (ADR-0010)
                    # Apply to RUL future loss ONLY, not to HI or other losses.
                    # ----------------------------------------------------------
                    soft_cap_enable = bool(getattr(world_model_config, "soft_cap_enable", False))
                    soft_cap_power = float(getattr(world_model_config, "soft_cap_power", 0.5) or 0.5)
                    soft_cap_floor = float(getattr(world_model_config, "soft_cap_floor", 0.05) or 0.05)

                    if soft_cap_enable:
                        # Distance from cap: 0 = fully capped, 1 = RUL=0
                        cap_distance = (1.0 - true_rul_seq_norm).clamp(0.0, 1.0)  # (B, H, 1)
                        
                        # Soft weight: power gives gradual ramp from cap to low RUL
                        soft_cap_weight = cap_distance.pow(soft_cap_power)
                        soft_cap_weight = soft_cap_weight.clamp(soft_cap_floor, 1.0)  # Never fully zero
                        
                        # IMPORTANT: Apply horizon mask to soft cap weights
                        soft_cap_weight = soft_cap_weight * valid_mask_seq_exp  # Zero out padded timesteps
                    else:
                        soft_cap_weight = None  # Fall back to binary masking

                    # Late-window weighting: upweight samples whose FUTURE horizon contains failure (RUL ~ 0).
                    late_weight_enable = bool(getattr(world_model_config, "late_weight_enable", False))
                    late_weight_factor = float(getattr(world_model_config, "late_weight_factor", 5.0) or 5.0)
                    late_weight_mode = str(getattr(world_model_config, "late_weight_mode", "future_has_zero") or "future_has_zero")
                    late_weight_eps_norm = float(getattr(world_model_config, "late_weight_eps_norm", 1e-6) or 1e-6)

                    # Fractions for logging
                    cap_frac = float((~cap_mask).float().mean().detach().cpu())
                    early_drop_frac = float((~early_mask).float().mean().detach().cpu())
                    mask_keep_frac = float(mask_f.mean().detach().cpu())
                    cap_frac_sum += cap_frac
                    cap_batches += 1
                    early_drop_frac_sum += early_drop_frac
                    mask_keep_frac_sum += mask_keep_frac
                    mask_batches += 1
                    
                    # Horizon mask statistics (for padding-aware loss) - log after loss computation
                    valid_frac = float(mask_rul.mean().detach().cpu())
                    pad_frac_batch = 1.0 - float(valid_mask_seq.mean().detach().cpu())
                    
                    # Log on first epoch / first batch (before loss computation)
                    if epoch == 0 and batch_idx == 0:
                        print(f"[HorizonMask] use_horizon_mask={use_horizon_mask}")
                        print(f"[HorizonMask] valid_frac={valid_frac:.4f} pad_frac={pad_frac_batch:.4f}")
                        if pad_frac_batch < 1e-4:
                            print("[HorizonMask] WARNING: No padding detected in this batch")

                    # Sample weighting (late-life emphasis) – based on true RUL at t=0
                    if w_power > 0.0:
                        base = (1.0 - true_rul_seq_norm[:, 0, 0]).clamp(0.0, 1.0)  # (B,)
                        w_b = (base + 1e-6) ** w_power
                        w_b = w_b.clamp(w_min, w_max)
                    else:
                        w_b = torch.ones(true_rul_seq_norm.size(0), device=true_rul_seq_norm.device, dtype=true_rul_seq_norm.dtype)

                    # Apply late-window weighting per sample (after time reduction).
                    # We define "future hits zero" using the normalized target horizon.
                    # IMPORTANT: Use only valid timesteps (exclude padded ones)
                    if late_weight_enable and late_weight_factor > 1.0:
                        # Compute fut_min using only valid timesteps
                        invalid = (valid_mask_seq_exp == 0)
                        true_rul_for_min = true_rul_seq_norm.clone()
                        true_rul_for_min[invalid] = +1e9  # Set invalid timesteps to large value so they're ignored
                        fut_min_valid = true_rul_for_min[:, :, 0].min(dim=1).values  # (B,)
                        
                        if late_weight_mode in {"future_min_below_eps", "future_has_zero"}:
                            late_mask = (fut_min_valid <= float(late_weight_eps_norm))
                        else:
                            late_mask = (fut_min_valid <= float(late_weight_eps_norm))
                        w_late = 1.0 + late_mask.float() * (float(late_weight_factor) - 1.0)  # (B,)
                        w_b = w_b * w_late

                        # Epoch diagnostics (train): fraction + true/pred future-min stats
                        try:
                            late_frac_sum += float(late_mask.float().mean().detach().cpu())
                            late_batches += 1
                            true_min_b = fut_min.detach()
                            pred_min_b = pred_rul_seq_norm[:, :, 0].min(dim=1).values.detach()
                            true_future_min_sum += float(true_min_b.mean().cpu())
                            true_future_min_count += 1
                            tmn = float(true_min_b.min().cpu())
                            tmx = float(true_min_b.max().cpu())
                            true_future_min_min = tmn if true_future_min_min is None else min(true_future_min_min, tmn)
                            true_future_min_max = tmx if true_future_min_max is None else max(true_future_min_max, tmx)
                            pred_future_min_sum += float(pred_min_b.mean().cpu())
                            pred_future_min_count += 1
                            pmn = float(pred_min_b.min().cpu())
                            pmx = float(pred_min_b.max().cpu())
                            pred_future_min_min = pmn if pred_future_min_min is None else min(pred_future_min_min, pmn)
                            pred_future_min_max = pmx if pred_future_min_max is None else max(pred_future_min_max, pmx)
                        except Exception:
                            pass

                    # ----------------------------------------------------------
                    # Wiring proof (RUL): prove masks/weights + loss reductions
                    # ----------------------------------------------------------
                    if _should_wiring_probe(batch_idx):
                        try:
                            # This mirrors the loss reduction path:
                            # per-element -> per-sample (masked mean) -> weighted batch mean.
                            diff2_dbg = (pred_rul_seq_norm - true_rul_seq_norm) ** 2  # (B,H,1)
                            num_i_dbg = (diff2_dbg * mask_f).sum(dim=(1, 2))          # (B,)
                            den_i_dbg = mask_f.sum(dim=(1, 2)).clamp_min(1e-6)        # (B,)
                            loss_i_dbg = num_i_dbg / den_i_dbg                         # (B,)
                            weighted_mean_dbg = float((w_b * loss_i_dbg).mean().detach().cpu())

                            # Late mask (if enabled)
                            if late_weight_enable and late_weight_factor > 1.0:
                                late_mask_dbg = (true_rul_seq_norm[:, :, 0].min(dim=1).values <= float(late_weight_eps_norm))
                                late_frac_dbg = float(late_mask_dbg.float().mean().detach().cpu())
                            else:
                                late_mask_dbg = None
                                late_frac_dbg = 0.0

                            # Informative vs noninformative classification for this batch
                            eps_inf = float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6)
                            mode_inf = str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap") or "future_min_lt_cap")
                            fut_min_b = true_rul_seq_norm[:, :, 0].min(dim=1).values
                            if mode_inf == "future_has_zero":
                                is_inf_b = (fut_min_b <= eps_inf)
                            else:
                                is_inf_b = (fut_min_b < (1.0 - eps_inf))
                            all_one_b = (fut_min_b >= (1.0 - eps_inf))

                            def _masked_mean(v, m):
                                if m is None:
                                    return float("nan")
                                m = m.detach()
                                if int(m.sum().item()) == 0:
                                    return float("nan")
                                return float(v[m].mean().detach().cpu())

                            k = _probe_key(batch_idx)
                            wiring_debug.setdefault(k, {})
                            wiring_debug[k]["rul"] = {
                                "pred_rul_seq_norm_shape": list(pred_rul_seq_norm.shape),
                                "true_rul_seq_norm_shape": list(true_rul_seq_norm.shape),
                                "pred_rul_requires_grad": bool(pred_rul_seq_norm.requires_grad),
                                "pred_rul_min_mean_max": [
                                    float(pred_rul_seq_norm.detach().min().cpu()),
                                    float(pred_rul_seq_norm.detach().mean().cpu()),
                                    float(pred_rul_seq_norm.detach().max().cpu()),
                                ],
                                "true_rul_min_mean_max": [
                                    float(true_rul_seq_norm.detach().min().cpu()),
                                    float(true_rul_seq_norm.detach().mean().cpu()),
                                    float(true_rul_seq_norm.detach().max().cpu()),
                                ],
                                "pred_rul_mean_std": [
                                    float(pred_rul_seq_norm.detach().mean().cpu()),
                                    float(pred_rul_seq_norm.detach().view(-1).std(unbiased=False).cpu()),
                                ],
                                "true_rul_mean_std": [
                                    float(true_rul_seq_norm.detach().mean().cpu()),
                                    float(true_rul_seq_norm.detach().view(-1).std(unbiased=False).cpu()),
                                ],
                                "frac_all_cap_future_batch": float(all_cap_mask.float().mean().detach().cpu()),
                                "mask_keep_frac": float(mask_f.detach().mean().cpu()),
                                "horizon_mask_stats": {
                                    "use_horizon_mask": bool(use_horizon_mask),
                                    "valid_frac": float(mask_rul.mean().detach().cpu()),
                                    "pad_frac": float(1.0 - valid_mask_seq.mean().detach().cpu()),
                                },
                                "w_b_min_mean_max": [
                                    float(w_b.detach().min().cpu()),
                                    float(w_b.detach().mean().cpu()),
                                    float(w_b.detach().max().cpu()),
                                ],
                                "w_first16": [float(x) for x in w_b.detach().cpu().view(-1)[:16].numpy().astype(float).tolist()],
                                "late_weight_enable": bool(late_weight_enable),
                                "late_weight_factor": float(late_weight_factor),
                                "late_weight_eps_norm": float(late_weight_eps_norm),
                                "late_mask_frac": float(late_frac_dbg),
                                "frac_informative_batch": float(is_inf_b.float().mean().detach().cpu()),
                                "frac_all_one_future_batch": float(all_one_b.float().mean().detach().cpu()),
                                "rul_loss_per_sample_mean": float(loss_i_dbg.detach().mean().cpu()),
                                "loss_i_mean_informative": _masked_mean(loss_i_dbg, is_inf_b),
                                "loss_i_mean_noninformative": _masked_mean(loss_i_dbg, ~is_inf_b),
                                "weighted_mean_debug": weighted_mean_dbg,
                                # Show first 2 samples (first 10 steps) to catch accidental broadcasting
                                "true_seq_s0_first10": true_rul_seq_norm[0, :10, 0].detach().cpu().numpy().astype(float).tolist(),
                                "pred_seq_s0_first10": pred_rul_seq_norm[0, :10, 0].detach().cpu().numpy().astype(float).tolist(),
                                "true_seq_s1_first10": true_rul_seq_norm[1, :10, 0].detach().cpu().numpy().astype(float).tolist() if true_rul_seq_norm.size(0) > 1 else None,
                                "pred_seq_s1_first10": pred_rul_seq_norm[1, :10, 0].detach().cpu().numpy().astype(float).tolist() if pred_rul_seq_norm.size(0) > 1 else None,
                            }

                            # Additional target/index proof: last step stats (normalized + cycles)
                            max_rul_cycles_dbg = float(getattr(world_model_config, "max_rul", 125.0))
                            true_last = true_rul_seq_norm[:, -1, 0]
                            pred_last = pred_rul_seq_norm[:, -1, 0]
                            wiring_debug[k]["rul"].update(
                                {
                                    "true_rul_last_norm_min_mean_max": [
                                        float(true_last.detach().min().cpu()),
                                        float(true_last.detach().mean().cpu()),
                                        float(true_last.detach().max().cpu()),
                                    ],
                                    "pred_rul_last_norm_min_mean_max": [
                                        float(pred_last.detach().min().cpu()),
                                        float(pred_last.detach().mean().cpu()),
                                        float(pred_last.detach().max().cpu()),
                                    ],
                                    "true_rul_last_cycles_min_mean_max": [
                                        float((true_last * max_rul_cycles_dbg).detach().min().cpu()),
                                        float((true_last * max_rul_cycles_dbg).detach().mean().cpu()),
                                        float((true_last * max_rul_cycles_dbg).detach().max().cpu()),
                                    ],
                                    "pred_rul_last_cycles_min_mean_max": [
                                        float((pred_last * max_rul_cycles_dbg).detach().min().cpu()),
                                        float((pred_last * max_rul_cycles_dbg).detach().mean().cpu()),
                                        float((pred_last * max_rul_cycles_dbg).detach().max().cpu()),
                                    ],
                                }
                            )

                            # Assertions: informative mask definition sanity
                            # all_one => NOT informative (for future_min_lt_cap mode)
                            if mode_inf != "future_has_zero":
                                mismatch_inf = int((all_one_b & is_inf_b).sum().item())
                            else:
                                mismatch_inf = 0
                            # late_weight_mode=future_has_zero equivalence: True iff any timepoint <= eps
                            late_mode = str(getattr(world_model_config, "late_weight_mode", "future_has_zero") or "future_has_zero")
                            eps_lw = float(late_weight_eps_norm)
                            has_zero_any = (true_rul_seq_norm[:, :, 0] <= eps_lw).any(dim=1)
                            late_mask_cmp = has_zero_any if late_mode == "future_has_zero" else (true_rul_seq_norm[:, :, 0].min(dim=1).values <= eps_lw)
                            late_mask_used = has_zero_any if late_mask_dbg is None else late_mask_dbg
                            mismatch_late = float((late_mask_cmp != late_mask_used).float().mean().detach().cpu())
                            wiring_debug[k]["rul"].update(
                                {
                                    "assert_all_one_implies_not_informative_mismatch_count": mismatch_inf,
                                    "assert_late_mask_equivalence_mismatch_frac": float(mismatch_late),
                                }
                            )

                            # Masks/weights section (computed from SAME tensors used for weighting)
                            wiring_debug[k]["masks_weights"] = {
                                "informative_sampling": {
                                    "informative_sampling_enable": bool(getattr(world_model_config, "informative_sampling_enable", False)),
                                    "informative_sampling_mode": str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap")),
                                    "keep_prob_noninformative": float(getattr(world_model_config, "keep_prob_noninformative", 0.1) or 0.1),
                                    "informative_uncapped_frac_threshold": float(getattr(world_model_config, "informative_uncapped_frac_threshold", 0.3) or 0.3),
                                    "informative_mask_frac_in_batch": float(is_inf_b.float().mean().detach().cpu()),
                                    "frac_all_one_future_targets_in_batch": float(all_one_b.float().mean().detach().cpu()),
                                },
                                "late_weighting": {
                                    "late_weight_enable": bool(late_weight_enable),
                                    "late_weight_mode": str(getattr(world_model_config, "late_weight_mode", "future_has_zero") or "future_has_zero"),
                                    "late_weight_factor": float(late_weight_factor),
                                    "late_weight_mask_frac_in_batch": float(late_frac_dbg),
                                },
                                # P0.1: Soft cap weighting stats (ADR-0010)
                                "soft_cap_weighting": {
                                    "soft_cap_enable": bool(soft_cap_enable),
                                    "soft_cap_power": float(soft_cap_power) if soft_cap_enable else None,
                                    "soft_cap_floor": float(soft_cap_floor) if soft_cap_enable else None,
                                    "soft_cap_weight_min": float(soft_cap_weight.min().detach().cpu()) if (soft_cap_enable and soft_cap_weight is not None) else None,
                                    "soft_cap_weight_mean": float(soft_cap_weight.mean().detach().cpu()) if (soft_cap_enable and soft_cap_weight is not None) else None,
                                    "soft_cap_weight_max": float(soft_cap_weight.max().detach().cpu()) if (soft_cap_enable and soft_cap_weight is not None) else None,
                                },
                                "final_sample_weights": {
                                    "min_weight": float(w_b.detach().min().cpu()),
                                    "mean_weight": float(w_b.detach().mean().cpu()),
                                    "max_weight": float(w_b.detach().max().cpu()),
                                    "first_16_weights": [float(x) for x in w_b.detach().cpu().view(-1)[:16].numpy().astype(float).tolist()],
                                },
                            }

                            # Optional correlation(pred_last, true_last) on debug batch (normalized)
                            try:
                                pl = pred_rul_seq_norm[:, -1, 0].detach()
                                tl = true_rul_seq_norm[:, -1, 0].detach()
                                plc = pl - pl.mean()
                                tlc = tl - tl.mean()
                                denom = (plc.std(unbiased=False) * tlc.std(unbiased=False)).clamp_min(1e-12)
                                corr = float((plc * tlc).mean().cpu() / denom.cpu())
                            except Exception:
                                corr = float("nan")
                            wiring_debug[k]["rul"]["corr_pred_last_true_last_norm"] = float(corr)

                            # Cap reweighting block (even when disabled we include a minimal proof)
                            wiring_debug[k]["cap_reweight"] = {
                                "enable": bool(cap_rw_enable),
                                "eps": float(cap_rw_eps),
                                "weight": float(cap_rw_weight),
                                "apply_to": str(cap_rw_apply_to),
                                "frac_all_cap_future": float(all_cap_mask.float().mean().detach().cpu()),
                                "mean_weight": float(w_cap.detach().mean().cpu()),
                            }

                            # EOL scalar head proof (if present / used)
                            if pred_eol is not None:
                                try:
                                    pe = pred_eol.view(-1)
                                    wiring_debug[k]["eol"] = {
                                        "pred_eol_shape": list(pred_eol.shape),
                                        "pred_eol_requires_grad": bool(pe.requires_grad),
                                        "pred_eol_min_mean_max": [
                                            float(pe.detach().min().cpu()),
                                            float(pe.detach().mean().cpu()),
                                            float(pe.detach().max().cpu()),
                                        ],
                                    }
                                except Exception:
                                    pass

                            # Assertions (record-only; do not crash)
                            wiring_debug[k]["asserts"] = {
                                "pred_rul_requires_grad": bool(pred_rul_seq_norm.requires_grad),
                                "loss_i_finite": bool(torch.isfinite(loss_i_dbg).all().item()),
                                "loss_i_nonzero_mean": bool(float(loss_i_dbg.detach().mean().cpu()) > 0.0),
                                "late_mask_nonzero_if_enabled": bool((late_frac_dbg > 0.0) if (late_weight_enable and late_weight_factor > 1.0) else True),
                            }

                            # Human-readable console log (small)
                            print(
                                f"[WorldModelV1][wiring] rul: mask_keep={float(mask_f.mean().detach().cpu()):.4f} "
                                f"late_frac={late_frac_dbg:.4f} w(min/mean/max)="
                                f"{float(w_b.min().detach().cpu()):.3f}/"
                                f"{float(w_b.mean().detach().cpu()):.3f}/"
                                f"{float(w_b.max().detach().cpu()):.3f} "
                                f"loss_i_mean={float(loss_i_dbg.mean().detach().cpu()):.6f} "
                                f"weighted_mean={weighted_mean_dbg:.6f}"
                            )
                        except Exception as e:
                            k = _probe_key(batch_idx)
                            wiring_debug.setdefault(k, {})
                            wiring_debug[k]["rul_error"] = str(e)

                    # r0-only supervision when linear decay is used
                    if bool(getattr(world_model_config, "rul_linear_decay", False)) and rul_r0_only:
                        points = rul_r0_points
                        if not points:
                            points = [0]
                        idx = torch.tensor(points, device=pred_rul_seq_norm.device, dtype=torch.long)
                        pred_sel = pred_rul_seq_norm[:, idx, :]  # (B,K,1)
                        true_sel = true_rul_seq_norm[:, idx, :]  # (B,K,1)
                        mask_sel = mask_rul[:, idx, :]             # (B,K,1) - use mask_rul (includes horizon mask)
                        w_bk = w_b.view(-1, 1, 1).expand(-1, pred_sel.size(1), -1)
                        diff2 = (pred_sel - true_sel) ** 2
                        # Per-sample normalized MSE on selected points, then apply per-sample weights.
                        num_i = (diff2 * mask_sel).sum(dim=(1, 2))  # (B,)
                        den_i = mask_sel.sum(dim=(1, 2)).clamp_min(1e-6)  # (B,)
                        loss_i = num_i / den_i  # (B,)
                        w_final = w_b
                        if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                            w_final = w_b * w_cap
                        loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                    else:
                        # Full-horizon masked MSE (optionally late-ramped)
                        diff2 = (pred_rul_seq_norm - true_rul_seq_norm) ** 2
                        if rul_traj_late_ramp:
                            w_t = torch.linspace(0.2, 1.0, steps=pred_rul_seq_norm.size(1), device=pred_rul_seq_norm.device)
                            w_t = (w_t / w_t.mean()).view(1, -1, 1)  # (1,H,1)
                            wt = (w_t * mask_rul)  # Use mask_rul (includes horizon mask)
                            num_i = (diff2 * wt).sum(dim=(1, 2))  # (B,)
                            den_i = wt.sum(dim=(1, 2)).clamp_min(1e-6)  # (B,)
                            loss_i = num_i / den_i
                            w_final = w_b
                            if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                                w_final = w_b * w_cap
                            loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                        else:
                            # P0.1: Use soft cap weighting when enabled (ADR-0010)
                            if soft_cap_enable and soft_cap_weight is not None:
                                # Soft-weighted MSE (replaces binary masking for RUL future loss)
                                # soft_cap_weight already has horizon mask applied
                                num_i = (diff2 * soft_cap_weight).sum(dim=(1, 2))  # (B,)
                                den_i = soft_cap_weight.sum(dim=(1, 2)).clamp_min(1e-6)  # (B,)
                            else:
                                # Legacy binary masking path - use mask_rul (includes horizon mask)
                                num_i = (diff2 * mask_rul).sum(dim=(1, 2))  # (B,)
                                den_i = mask_rul.sum(dim=(1, 2)).clamp_min(1e-6)  # (B,)
                            loss_i = num_i / den_i
                            w_final = w_b
                            if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                                w_final = w_b * w_cap
                            loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                            
                            # Log denom stats after computation (first epoch / first batch)
                            if epoch == 0 and batch_idx == 0:
                                print(f"[HorizonMask] denom(min/mean)={float(den_i.min().detach().cpu()):.2f}/{float(den_i.mean().detach().cpu()):.2f}")

                    w_traj = float(rul_traj_weight) if rul_traj_weight is not None else float(rul_w)
                    if w_traj > 0.0:
                        loss = loss + w_traj * loss_rul_traj
                    else:
                        loss_rul_traj = torch.tensor(0.0, device=pred_rul_h.device)

                    # Monotonic decreasing penalty over horizon
                    if rul_mono_future_weight > 0.0:
                        delta = pred_rul_h[:, 1:] - pred_rul_h[:, :-1]
                        # Use mask_rul for adjacent pairs (includes horizon mask)
                        mask_adj = (mask_rul[:, 1:, 0] * mask_rul[:, :-1, 0]).float()  # (B,H-1)
                        loss_mono_future = (torch.relu(delta) ** 2 * mask_adj).sum() / (mask_adj.sum() + 1e-6)
                        loss = loss + rul_mono_future_weight * loss_mono_future
                    else:
                        loss_mono_future = torch.tensor(0.0, device=pred_rul_h.device)

                    # Anti-high-saturation penalty (only where target is below 1-margin)
                    if rul_saturation_weight > 0.0:
                        # IMPORTANT: sat-mask is defined in NORMALIZED units (0..1), even if inputs were cycles.
                        thr_norm = float(1.0 - margin)
                        mask_sat = ((true_rul_seq_norm < thr_norm) & (mask_rul > 0)).float()  # (B,H,1) - use mask_rul
                        sat_err2 = torch.relu(pred_rul_seq_norm - (1.0 - margin)) ** 2
                        loss_sat = (sat_err2 * mask_sat).sum() / (mask_sat.sum() + 1e-6)
                        loss = loss + rul_saturation_weight * loss_sat
                        sat_loss_sum += float(loss_sat.detach().cpu())
                        sat_mask_frac_sum += float(mask_sat.detach().mean().cpu())
                        # Debug: how often does the "below threshold" condition even hold?
                        sat_below_thr_frac_sum += float((true_rul_seq_norm < thr_norm).float().mean().detach().cpu())
                        sat_below_thr_batches += 1
                        sat_batches += 1
                    else:
                        loss_sat = torch.tensor(0.0, device=pred_rul_h.device)

                    if epoch == 0 and batch_idx == 0:
                        true_rul0_cycles = (true_rul_seq_norm[:, 0, 0] * max(max_rul_cycles, 1e-6)).detach()
                        pred_rul0_cycles = (pred_rul_seq_norm[:, 0, 0] * max(max_rul_cycles, 1e-6)).detach()
                        print(
                            "[WorldModelV1][dbg] rul_traj_weighted="
                            f"{float(loss_rul_traj.detach().cpu()):.6f} "
                            f"mono_future={float(loss_mono_future.detach().cpu()):.6f} "
                            f"sat={float(loss_sat.detach().cpu()):.6f} "
                            f"pred_mean={float(pred_rul_h.mean().detach().cpu()):.4f} "
                            f"true_mean={float(y_rul_h.mean().detach().cpu()):.4f}"
                        )
                        print(
                            f"[WorldModelV1][dbg] true_rul0_cycles(min/mean/max)="
                            f"{float(true_rul0_cycles.min().cpu()):.2f}/"
                            f"{float(true_rul0_cycles.mean().cpu()):.2f}/"
                            f"{float(true_rul0_cycles.max().cpu()):.2f} "
                            f"pred_rul0_cycles(min/mean/max)="
                            f"{float(pred_rul0_cycles.min().cpu()):.2f}/"
                            f"{float(pred_rul0_cycles.mean().cpu()):.2f}/"
                            f"{float(pred_rul0_cycles.max().cpu()):.2f}"
                        )
                        print(
                            f"[WorldModelV1][dbg] cap_frac={cap_frac:.6f} "
                            f"early_drop_frac={early_drop_frac:.6f} "
                            f"mask_keep_frac={mask_keep_frac:.6f}"
                        )
                        if w_power > 0.0:
                            print(
                                f"[WorldModelV1][dbg] sample_weight w(min/mean/max)="
                                f"{float(w_b.min().detach().cpu()):.4f}/"
                                f"{float(w_b.mean().detach().cpu()):.4f}/"
                                f"{float(w_b.max().detach().cpu()):.4f}"
                            )

            # Optional: supervise the predicted EOL scalar (in [0,1]) against normalized current RUL
            if eol_scalar_loss_weight > 0.0 and pred_eol is not None:
                # current_rul_b is already normalized in Y_rul_b[:,0]
                eol_loss = F.mse_loss(pred_eol.view(-1), current_rul_b.view(-1))
                loss = loss + eol_scalar_loss_weight * eol_loss

            # AMP/overflow-ish guard (even without AMP): skip non-finite losses
            if not torch.isfinite(loss).all():
                print(f"  ⚠️  Non-finite loss at epoch={epoch+1} batch={batch_idx}: {loss.detach().cpu().item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # Wiring proof: gradient-flow verification (encoder vs decoder).
            # Only runs for a tiny number of batches/epochs when enabled.
            if _should_wiring_probe(batch_idx):
                try:
                    import numpy as _np

                    def _grad_norm(params) -> float:
                        s = 0.0
                        for p in params:
                            if (p is None) or (p.grad is None):
                                continue
                            g = p.grad.detach()
                            if not torch.isfinite(g).all():
                                continue
                            s += float((g * g).sum().cpu())
                        return float(_np.sqrt(max(s, 0.0)))

                    enc_params = [p for p in world_model.encoder.parameters() if p.requires_grad]
                    dec_params = [
                        p for n, p in world_model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad
                    ]
                    gn_enc = _grad_norm(enc_params)
                    gn_dec = _grad_norm(dec_params)

                    gn_inproj = None
                    if hasattr(world_model.encoder, "input_proj") and getattr(world_model.encoder, "input_proj") is not None:
                        try:
                            gn_inproj = _grad_norm(list(world_model.encoder.input_proj.parameters()))
                        except Exception:
                            gn_inproj = None

                    def _module_grad_norm(m) -> float:
                        if m is None:
                            return 0.0
                        ps = [p for p in m.parameters() if p.requires_grad]
                        if not ps:
                            return 0.0
                        return _grad_norm(ps)

                    # Heads (best-effort; different paths use different modules)
                    hi_head_mod = getattr(world_model, "hi_head_latent", None) or getattr(world_model, "hi_head", None)
                    rul_head_mod = getattr(world_model, "rul_head_latent", None) or getattr(world_model, "rul_head", None)
                    eol_head_mod = getattr(world_model, "eol_scalar_head", None)

                    # Encoder trunk parts
                    shared_head_mod = getattr(world_model.encoder, "shared_head", None) if hasattr(world_model.encoder, "shared_head") else None
                    last_layer_mod = None
                    try:
                        if hasattr(world_model.encoder, "transformer") and hasattr(world_model.encoder.transformer, "layers"):
                            layers = world_model.encoder.transformer.layers
                            if layers is not None and len(layers) > 0:
                                last_layer_mod = list(layers)[-1]
                    except Exception:
                        last_layer_mod = None

                    k = _probe_key(batch_idx)
                    wiring_debug.setdefault(k, {})
                    wiring_debug[k]["grad_norms"] = {
                        "encoder_grad_norm": float(gn_enc),
                        "decoder_grad_norm": float(gn_dec),
                        "input_proj_grad_norm": float(gn_inproj) if gn_inproj is not None else None,
                        "hi_head_grad_norm": float(_module_grad_norm(hi_head_mod)),
                        "rul_head_grad_norm": float(_module_grad_norm(rul_head_mod)),
                        "eol_head_grad_norm": float(_module_grad_norm(eol_head_mod)),
                        "encoder_shared_head_grad_norm": float(_module_grad_norm(shared_head_mod)),
                        "encoder_last_block_grad_norm": float(_module_grad_norm(last_layer_mod)),
                        "stage_b_unfrozen_modules": list(stage_b_unfrozen_modules),
                        "encoder_trainable_params": int(sum(p.numel() for p in enc_params)),
                        "decoder_trainable_params": int(sum(p.numel() for p in dec_params)),
                    }
                    print(
                        f"[WorldModelV1][wiring] grad_norms: enc={gn_enc:.6f} dec={gn_dec:.6f}"
                        + (f" input_proj={float(gn_inproj):.6f}" if gn_inproj is not None else "")
                    )
                except Exception as e:
                    k = _probe_key(batch_idx)
                    wiring_debug.setdefault(k, {})
                    wiring_debug[k]["grad_norms_error"] = str(e)
            # Optional: grad clipping for stability (WM-V1 path)
            grad_clip_norm = getattr(world_model_config, "grad_clip_norm", None)
            if grad_clip_norm is not None:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        world_model.parameters(), max_norm=float(grad_clip_norm)
                    )
                except Exception as e:
                    print(f"  ⚠️  Warning: grad_clip_norm failed: {e}")
            optimizer.step()

            running_train += loss.item() * X_b.size(0)
            n_train_samples += X_b.size(0)

        train_loss = running_train / max(1, n_train_samples)
        if sat_batches > 0:
            print(
                f"[WorldModelV1][epoch {epoch+1}] sat_loss_mean={sat_loss_sum / sat_batches:.6f} "
                f"sat_mask_frac_mean={sat_mask_frac_sum / sat_batches:.6f}"
            )
        else:
            print(f"[WorldModelV1][epoch {epoch+1}] sat_loss_mean=0.000000 sat_mask_frac_mean=0.000000")
        if sat_below_thr_batches > 0 and (epoch == 0 or ((epoch + 1) % 5 == 0)):
            thr_norm = float(1.0 - float(getattr(world_model_config, "rul_saturation_margin", 0.05) or 0.05))
            cap_thr = float(getattr(world_model_config, "rul_cap_threshold", 0.999999) or 0.999999)
            print(
                f"[WorldModelV1][epoch {epoch+1}] sat_debug: thr_norm={thr_norm:.6f} cap_thr_norm={cap_thr:.6f} "
                f"true_below_thr_frac_mean={sat_below_thr_frac_sum / sat_below_thr_batches:.6f}"
            )
        if cap_batches > 0:
            print(f"[WorldModelV1][epoch {epoch+1}] true_rul_cap_frac_train={cap_frac_sum / cap_batches:.6f}")
        if mask_batches > 0:
            print(
                f"[WorldModelV1][epoch {epoch+1}] "
                f"rul_early_drop_frac_train={early_drop_frac_sum / mask_batches:.6f} "
                f"rul_mask_keep_frac_train={mask_keep_frac_sum / mask_batches:.6f}"
            )
        if pred_rul_count > 0:
            pm = pred_rul_sum / max(1, pred_rul_count)
            try:
                print(
                    f"[WorldModelV1][epoch {epoch+1}] pred_rul_seq_norm(mean/min/max)="
                    f"{pm:.6f}/"
                    f"{float(pred_rul_min) if pred_rul_min is not None else float('nan'):.6f}/"
                    f"{float(pred_rul_max) if pred_rul_max is not None else float('nan'):.6f}"
                )
            except Exception:
                pass
        if bool(getattr(world_model_config, "log_informative_stats", True)) and inf_batches > 0:
            # Per-epoch batch composition diagnostics (train)
            print(
                f"[WorldModelV1][epoch {epoch+1}] "
                f"informative_frac_in_batches={inf_frac_sum / inf_batches:.6f} "
                f"frac_all_one_future_targets={all_one_frac_sum / all_one_batches:.6f}"
            )
        if all_cap_batches > 0:
            print(
                f"[WorldModelV1][epoch {epoch+1}] "
                f"frac_all_cap_future_train={all_cap_frac_sum / all_cap_batches:.6f} "
                f"cap_weight_mean_train={cap_weight_mean_sum / max(1, cap_weight_batches):.6f}"
            )
        if late_batches > 0 and (epoch == 0 or ((epoch + 1) % 5 == 0)):
            # Print sparsely (every 5 epochs + epoch 1) to keep logs readable.
            tfm = true_future_min_sum / max(1, true_future_min_count)
            pfm = pred_future_min_sum / max(1, pred_future_min_count)
            print(
                f"[WorldModelV1][epoch {epoch+1}] late_frac_train={late_frac_sum / late_batches:.6f} "
                f"true_future_min(norm mean/min/max)={tfm:.6f}/"
                f"{float(true_future_min_min) if true_future_min_min is not None else float('nan'):.6f}/"
                f"{float(true_future_min_max) if true_future_min_max is not None else float('nan'):.6f} "
                f"pred_future_min(norm mean/min/max)={pfm:.6f}/"
                f"{float(pred_future_min_min) if pred_future_min_min is not None else float('nan'):.6f}/"
                f"{float(pred_future_min_max) if pred_future_min_max is not None else float('nan'):.6f}"
            )

        # Validation
        world_model.eval()
        running_val = 0.0
        n_val_samples = 0
        cap_frac_val_sum = 0.0
        cap_val_batches = 0
        # Cap/plateau (all-cap future horizon) diagnostics (val)
        all_cap_frac_val_sum = 0.0
        all_cap_val_batches = 0
        cap_weight_mean_val_sum = 0.0
        cap_weight_val_batches = 0
        early_drop_frac_val_sum = 0.0
        mask_keep_frac_val_sum = 0.0
        mask_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 8:
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b, _unit_ids_b, _t_end_b = batch
                else:
                    X_b, Y_sens_b, Y_rul_b, Y_hi_b, cond_b, future_cond_b = batch
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

                # Optional: deterministic linear-decay RUL sequence from pred_rul0 (physics-consistent)
                if bool(getattr(world_model_config, "rul_linear_decay", False)) and (pred_rul is not None):
                    max_rul_denom = float(getattr(world_model_config, "max_rul", 125.0))
                    H = int(pred_rul.size(1))
                    pred_rul0 = pred_rul[:, 0, 0].clamp(0.0, 1.0)  # (B,)
                    k = torch.arange(H, device=pred_rul.device, dtype=pred_rul.dtype).view(1, H)
                    pred_seq = (pred_rul0.unsqueeze(1) - k / max(max_rul_denom, 1e-6)).clamp(0.0, 1.0)
                    pred_rul = pred_seq.unsqueeze(-1)  # (B,H,1)

                loss_sensors = F.mse_loss(pred_sensors, Y_sens_b)
                loss = sensor_w * loss_sensors

                if hi_w > 0.0 and pred_hi is not None:
                    hi_loss = F.mse_loss(pred_hi.squeeze(-1), Y_hi_b)
                    loss = loss + hi_w * hi_loss

                if pred_rul is not None:
                    # Mirror training: use the same new-loss switch logic
                    pred_rul_h = pred_rul.squeeze(-1)
                    y_rul_h = Y_rul_b
                    if y_rul_h.dim() == 3 and y_rul_h.size(-1) == 1:
                        y_rul_h = y_rul_h.squeeze(-1)

                    rul_traj_weight = getattr(world_model_config, "rul_traj_weight", None)
                    rul_traj_late_ramp = bool(getattr(world_model_config, "rul_traj_late_ramp", False))
                    rul_mono_future_weight = float(getattr(world_model_config, "rul_mono_future_weight", 0.0) or 0.0)
                    rul_saturation_weight = float(getattr(world_model_config, "rul_saturation_weight", 0.0) or 0.0)
                    margin = float(getattr(world_model_config, "rul_saturation_margin", 0.05) or 0.05)
                    cap_thr = float(getattr(world_model_config, "rul_cap_threshold", 0.999999) or 0.999999)
                    max_rul_cycles = float(getattr(world_model_config, "max_rul", 125.0))
                    rul_train_max_cycles = getattr(world_model_config, "rul_train_max_cycles", None)
                    rul_r0_only = bool(getattr(world_model_config, "rul_r0_only", False))
                    rul_r0_points = getattr(world_model_config, "rul_r0_points", None)
                    w_power = float(getattr(world_model_config, "rul_sample_weight_power", 0.0) or 0.0)
                    w_min = float(getattr(world_model_config, "rul_sample_weight_min", 0.2) or 0.2)
                    w_max = float(getattr(world_model_config, "rul_sample_weight_max", 3.0) or 3.0)

                    use_new = (
                        (rul_traj_weight is not None)
                        or rul_traj_late_ramp
                        or (rul_mono_future_weight > 0.0)
                        or (rul_saturation_weight > 0.0)
                    )

                    if not use_new:
                        if float(rul_w) > 0.0:
                            rul_loss = F.l1_loss(pred_rul_h, y_rul_h)
                            loss = loss + float(rul_w) * rul_loss
                    else:
                        # Unit-safe normalization (val): define masks and saturation in normalized space.
                        true_seq = y_rul_h  # (B,H)
                        pred_seq = pred_rul_h  # (B,H)
                        is_norm = bool(float(true_seq.detach().max().cpu()) <= 1.0 + 1e-3)
                        if is_norm:
                            true_rul_norm = true_seq.clamp(0.0, 1.0)
                            pred_rul_norm = pred_seq.clamp(0.0, 1.0)
                        else:
                            denom = max(max_rul_cycles, 1e-6)
                            true_rul_norm = (true_seq / denom).clamp(0.0, 1.0)
                            pred_rul_norm = (pred_seq / denom).clamp(0.0, 1.0)

                        true_rul_seq_norm = true_rul_norm.unsqueeze(-1)  # (B,H,1)
                        pred_rul_seq_norm = pred_rul_norm.unsqueeze(-1)  # (B,H,1)
                        true_rul_cycles = true_rul_seq_norm * max(max_rul_cycles, 1e-6)

                        # Cap/plateau reweighting (val) – mirror train semantics (default OFF).
                        cap_rw_enable = bool(getattr(world_model_config, "cap_reweight_enable", False))
                        cap_rw_eps = float(getattr(world_model_config, "cap_reweight_eps", 1e-6) or 1e-6)
                        cap_rw_weight = float(getattr(world_model_config, "cap_reweight_weight", 0.05) or 0.05)
                        cap_rw_apply_to = str(getattr(world_model_config, "cap_reweight_apply_to", "rul") or "rul").lower()
                        all_cap_mask = (true_rul_seq_norm[:, :, 0].min(dim=1).values >= (1.0 - cap_rw_eps))  # (B,)
                        w_cap = torch.where(
                            all_cap_mask,
                            torch.full_like(all_cap_mask.float(), float(cap_rw_weight)),
                            torch.ones_like(all_cap_mask.float()),
                        )  # (B,)
                        try:
                            all_cap_frac_val_sum += float(all_cap_mask.float().mean().detach().cpu())
                            all_cap_val_batches += 1
                            cap_weight_mean_val_sum += float(w_cap.detach().mean().cpu())
                            cap_weight_val_batches += 1
                        except Exception:
                            pass

                        # Cap-aware loss masking (val) – mirror train semantics (default ON).
                        cap_mask_enable = bool(getattr(world_model_config, "cap_mask_enable", True))
                        cap_mask_eps = float(getattr(world_model_config, "cap_mask_eps", 1e-6) or 1e-6)
                        cap_mask_apply_to = getattr(world_model_config, "cap_mask_apply_to", ["rul"])
                        if not isinstance(cap_mask_apply_to, (list, tuple, set)):
                            cap_mask_apply_to = [str(cap_mask_apply_to)]
                        cap_mask_apply_set = {str(x).lower() for x in cap_mask_apply_to}
                        cap_thr_norm = float(1.0 - cap_mask_eps)
                        if cap_mask_enable and ("rul" in cap_mask_apply_set):
                            cap_mask = (true_rul_seq_norm < cap_thr_norm)
                        else:
                            cap_mask = torch.ones_like(true_rul_seq_norm, dtype=torch.bool)
                        early_mask = torch.ones_like(cap_mask, dtype=torch.bool)
                        if rul_train_max_cycles is not None:
                            early_mask = (true_rul_cycles < float(rul_train_max_cycles))
                        mask = cap_mask & early_mask
                        mask_f = mask.float()

                        cap_frac_val_sum += float((~cap_mask).float().mean().detach().cpu())
                        cap_val_batches += 1
                        early_drop_frac_val_sum += float((~early_mask).float().mean().detach().cpu())
                        mask_keep_frac_val_sum += float(mask_f.mean().detach().cpu())
                        mask_val_batches += 1

                        # P0.1: Soft cap weighting (val) – mirror train semantics (ADR-0010)
                        soft_cap_enable = bool(getattr(world_model_config, "soft_cap_enable", False))
                        soft_cap_power = float(getattr(world_model_config, "soft_cap_power", 0.5) or 0.5)
                        soft_cap_floor = float(getattr(world_model_config, "soft_cap_floor", 0.05) or 0.05)
                        if soft_cap_enable:
                            cap_distance = (1.0 - true_rul_seq_norm).clamp(0.0, 1.0)
                            soft_cap_weight = cap_distance.pow(soft_cap_power)
                            soft_cap_weight = soft_cap_weight.clamp(soft_cap_floor, 1.0)
                        else:
                            soft_cap_weight = None

                        if w_power > 0.0:
                            base = (1.0 - true_rul_seq_norm[:, 0, 0]).clamp(0.0, 1.0)
                            w_b = (base + 1e-6) ** w_power
                            w_b = w_b.clamp(w_min, w_max)
                        else:
                            w_b = torch.ones(true_rul_seq_norm.size(0), device=true_rul_seq_norm.device, dtype=true_rul_seq_norm.dtype)

                        # Late-window weighting (val): keep consistent with train when enabled.
                        late_weight_enable = bool(getattr(world_model_config, "late_weight_enable", False))
                        late_weight_factor = float(getattr(world_model_config, "late_weight_factor", 5.0) or 5.0)
                        late_weight_eps_norm = float(getattr(world_model_config, "late_weight_eps_norm", 1e-6) or 1e-6)
                        if late_weight_enable and late_weight_factor > 1.0:
                            fut_min = true_rul_seq_norm[:, :, 0].min(dim=1).values  # (B,)
                            late_mask = (fut_min <= float(late_weight_eps_norm))
                            w_late = 1.0 + late_mask.float() * (float(late_weight_factor) - 1.0)
                            w_b = w_b * w_late

                        w_traj = float(rul_traj_weight) if rul_traj_weight is not None else float(rul_w)
                        if w_traj > 0.0:
                            # r0-only supervision when linear decay is used
                            if bool(getattr(world_model_config, "rul_linear_decay", False)) and rul_r0_only:
                                points = rul_r0_points
                                if not points:
                                    points = [0]
                                idx = torch.tensor(points, device=pred_rul_seq_norm.device, dtype=torch.long)
                                pred_sel = pred_rul_seq_norm[:, idx, :]
                                true_sel = true_rul_seq_norm[:, idx, :]
                                mask_sel = mask_f[:, idx, :]
                                w_bk = w_b.view(-1, 1, 1).expand(-1, pred_sel.size(1), -1)
                                diff2 = (pred_sel - true_sel) ** 2
                                num_i = (diff2 * mask_sel).sum(dim=(1, 2))
                                den_i = mask_sel.sum(dim=(1, 2)).clamp_min(1e-6)
                                loss_i = num_i / den_i
                                w_final = w_b
                                if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                                    w_final = w_b * w_cap
                                loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                            else:
                                diff2 = (pred_rul_seq_norm - true_rul_seq_norm) ** 2
                                if rul_traj_late_ramp:
                                    w_t = torch.linspace(0.2, 1.0, steps=pred_rul_seq_norm.size(1), device=pred_rul_seq_norm.device)
                                    w_t = (w_t / w_t.mean()).view(1, -1, 1)
                                    wt = (w_t * mask_f)
                                    num_i = (diff2 * wt).sum(dim=(1, 2))
                                    den_i = wt.sum(dim=(1, 2)).clamp_min(1e-6)
                                    loss_i = num_i / den_i
                                    w_final = w_b
                                    if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                                        w_final = w_b * w_cap
                                    loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                                else:
                                    # P0.1: Use soft cap weighting when enabled (val) (ADR-0010)
                                    if soft_cap_enable and soft_cap_weight is not None:
                                        num_i = (diff2 * soft_cap_weight).sum(dim=(1, 2))
                                        den_i = soft_cap_weight.sum(dim=(1, 2)).clamp_min(1e-6)
                                    else:
                                        num_i = (diff2 * mask_f).sum(dim=(1, 2))
                                        den_i = mask_f.sum(dim=(1, 2)).clamp_min(1e-6)
                                    loss_i = num_i / den_i
                                    w_final = w_b
                                    if cap_rw_enable and (cap_rw_apply_to in {"rul", "both"}):
                                        w_final = w_b * w_cap
                                    loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
                            loss = loss + w_traj * loss_rul_traj

                        if rul_mono_future_weight > 0.0:
                            delta = pred_rul_h[:, 1:] - pred_rul_h[:, :-1]
                            mask_adj = (mask[:, 1:, 0] & mask[:, :-1, 0]).float()
                            loss_mono_future = (torch.relu(delta) ** 2 * mask_adj).sum() / (mask_adj.sum() + 1e-6)
                            loss = loss + rul_mono_future_weight * loss_mono_future

                        if rul_saturation_weight > 0.0:
                            thr_norm = float(1.0 - margin)
                            mask_sat = ((true_rul_seq_norm < thr_norm) & mask).float()
                            sat_err2 = torch.relu(pred_rul_seq_norm - (1.0 - margin)) ** 2
                            loss_sat = (sat_err2 * mask_sat).sum() / (mask_sat.sum() + 1e-6)
                            loss = loss + rul_saturation_weight * loss_sat

                if eol_scalar_loss_weight > 0.0 and pred_eol is not None:
                    eol_loss = F.mse_loss(pred_eol.view(-1), current_rul_b.view(-1))
                    loss = loss + eol_scalar_loss_weight * eol_loss

                running_val += loss.item() * X_b.size(0)
                n_val_samples += X_b.size(0)

        val_loss = running_val / max(1, n_val_samples)

        if cap_val_batches > 0:
            print(f"[WorldModelV1][epoch {epoch+1}] rul_cap_frac_val={cap_frac_val_sum / cap_val_batches:.6f}")
        if all_cap_val_batches > 0:
            print(
                f"[WorldModelV1][epoch {epoch+1}] "
                f"frac_all_cap_future_val={all_cap_frac_val_sum / all_cap_val_batches:.6f} "
                f"cap_weight_mean_val={cap_weight_mean_val_sum / max(1, cap_weight_val_batches):.6f}"
            )
        if mask_val_batches > 0:
            print(
                f"[WorldModelV1][epoch {epoch+1}] "
                f"rul_early_drop_frac_val={early_drop_frac_val_sum / mask_val_batches:.6f} "
                f"rul_mask_keep_frac_val={mask_keep_frac_val_sum / mask_val_batches:.6f}"
            )

        print(
            f"[WorldModelV1] Epoch {epoch+1}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )
        
        # Stage -1: Log checkpoint metric and capping status
        cap_targets_wm = bool(getattr(world_model_config, "cap_rul_targets_to_max_rul", False))
        max_rul_wm = float(getattr(world_model_config, "max_rul", 125.0))
        print(
            f"[WorldModelV1][checkpoint] best_metric=val_total (total_loss) "
            f"cap_targets={cap_targets_wm} max_rul={max_rul_wm:.1f}"
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
            results_dir=results_dir,
        )
    except Exception as exc:
        print(f"  Error while computing test metrics: {exc}")
        test_metrics: Dict[str, Any] = {}

    if test_metrics:
        print("\n--- LAST (literature-style, truncated-aware) ---")
        print(f"  rmse_last: {test_metrics.get('rmse_last', float('nan')):.2f}")
        print(f"  mae_last:  {test_metrics.get('mae_last', float('nan')):.2f}")
        print(f"  bias_last: {test_metrics.get('bias_last', float('nan')):.2f}")
        print(f"  r2_last:   {test_metrics.get('r2_last', 0.0):.4f}")
        print(f"  nasa_last_mean: {test_metrics.get('nasa_last_mean', float('nan')):.4f}")
        print(f"  nasa_last_sum:  {test_metrics.get('nasa_last_sum', float('nan')):.2f}")
        print(f"  n_units: {test_metrics.get('n_units', 0)}")

        print("\n--- ALL (all windows/timepoints) ---")
        print(f"  rmse_all: {test_metrics.get('rmse_all', float('nan')):.2f}")
        print(f"  mae_all:  {test_metrics.get('mae_all', float('nan')):.2f}")
        print(f"  bias_all: {test_metrics.get('bias_all', float('nan')):.2f}")
        print(f"  r2_all:   {test_metrics.get('r2_all', 0.0):.4f}")
        print(f"  nasa_all_mean: {test_metrics.get('nasa_all_mean', float('nan')):.4f}")
        print(f"  nasa_all_sum:  {test_metrics.get('nasa_all_sum', float('nan')):.2f}")
        print(f"  n_samples_all: {test_metrics.get('n_samples_all', 0)}")
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
    # Include informative sampling config + measured fractions for reproducibility/debugging
    summary["informative_sampling"] = informative_stats
    if debug_wiring_enable:
        summary["wiring_debug_enabled"] = True
        summary["wiring_debug_path"] = str(results_dir / "wiring_debug.json")
    if test_metrics:
        summary["test_metrics"] = test_metrics

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WorldModelV1] Saved summary to {summary_path}")

    # Optional: save wiring debug JSON (small, scalar-only)
    if debug_wiring_enable and debug_wiring_save_json:
        try:
            wiring_path = results_dir / "wiring_debug.json"
            with open(wiring_path, "w") as f:
                json.dump(wiring_debug, f, indent=2)
            print(f"[WorldModelV1] Saved wiring debug to {wiring_path}")
        except Exception as e:
            print(f"[WorldModelV1] Warning: could not save wiring_debug.json: {e}")

    if test_metrics:
        try:
            metrics_test_path = results_dir / "metrics_test.json"
            with open(metrics_test_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"[WorldModelV1] Saved metrics to {metrics_test_path}")
        except Exception as e:
            print(f"[WorldModelV1] Warning: could not save metrics_test.json: {e}")

    return summary


def evaluate_transformer_world_model_v1_on_test(
    model: nn.Module,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    world_model_config: WorldModelTrainingConfig,
    device: torch.device,
    results_dir: Optional[Path] = None,
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
    rul_seq_norm_list: List[np.ndarray] = []
    true_rul_last_list: List[float] = []
    unit_id_list: List[int] = []
    cond_id_list: List[int] = []
    future_cond_list: List[np.ndarray] = []

    # Ensure we have a RUL column clipped to [0, max_rul]
    if "RUL" not in df_test.columns:
        raise KeyError("evaluate_transformer_world_model_v1_on_test expects 'RUL' column in df_test.")

    for unit_id, df_unit in df_test.groupby("UnitNumber"):
        df_unit = df_unit.sort_values("TimeInCycles").reset_index(drop=True)
        num_rows = len(df_unit)
        if num_rows < past_len + horizon:
            continue

        cond_id_unit = int(df_unit["ConditionID"].iloc[0])
        cond_cols = [c for c in feature_cols if c.startswith("Cond_")]

        for start in range(0, num_rows - past_len - horizon + 1):
            past = df_unit.iloc[start : start + past_len]
            future = df_unit.iloc[start + past_len : start + past_len + horizon]

            X_list.append(past[feature_cols].to_numpy(dtype=np.float32, copy=True))
            unit_id_list.append(int(unit_id))

            # RUL future in cycles for metrics (clipped)
            rul_future = future["RUL"].clip(lower=0.0, upper=max_rul).to_numpy(dtype=np.float32)
            true_rul_last_list.append(float(rul_future[-1]))
            rul_seq_norm_list.append((rul_future / max(max_rul, 1e-6)).astype(np.float32))

            # HI future (normalized) as in training: HI = clip(RUL / max_rul, 0, 1)
            hi_future = np.clip(rul_future / max_rul, 0.0, 1.0)
            hi_seq_list.append(hi_future.astype(np.float32))

            cond_id_list.append(cond_id_unit)
            # Future conditions aligned to horizon (if present; else zeros)
            if cond_cols:
                future_cond_list.append(future[cond_cols].to_numpy(dtype=np.float32, copy=True))
            else:
                future_cond_list.append(np.zeros((horizon, int(getattr(world_model_config, "cond_dim", 9))), dtype=np.float32))

    if not X_list:
        print("[WorldModelV1-Test] No valid test samples could be built.")
        return {}

    X_np = np.stack(X_list, axis=0)  # (N, past_len, F)
    Y_hi_np = np.stack(hi_seq_list, axis=0)  # (N, horizon)
    Y_rul_np = np.stack(rul_seq_norm_list, axis=0)  # (N, horizon)
    unit_ids_np = np.array(unit_id_list, dtype=np.int64)  # (N,)
    cond_ids_np = np.array(cond_id_list, dtype=np.int64)  # (N,)
    future_cond_np = np.stack(future_cond_list, axis=0).astype(np.float32)  # (N, horizon, cond_dim_actual)

    # --------------------------------------------------------------
    # Apply persisted X scaler (preferred: numpy before tensor creation)
    # --------------------------------------------------------------
    try:
        import os
        from src.tools.x_scaler import load_scaler, transform_x, clip_x
        from src.tools.debug_stats import tensor_stats

        if results_dir is None:
            raise RuntimeError("results_dir is None")
        x_scaler_path = os.path.join(str(results_dir), "world_model_v1_x_scaler.pkl")
        x_scaler = load_scaler(x_scaler_path)
        print(f"[WorldModelV1] Loaded X scaler from {x_scaler_path}")

        tensor_stats("TEST_X_before_scaler", X_np[:256])
        X_np = transform_x(x_scaler, X_np)
        X_np, frac = clip_x(X_np, clip=10.0)
        print(f"[WorldModelV1] TEST  X clip frac={frac:.6f}")
        tensor_stats("TEST_X_after_scaler", X_np[:256])

        # Scale future_cond_np consistently using Cond_* stats from the fitted X scaler.
        cond_idx = np.array([i for i, c in enumerate(feature_cols) if c.startswith("Cond_")], dtype=np.int64)
        if cond_idx.size > 0:
            cond_mean = x_scaler.mean_[cond_idx]
            cond_scale = x_scaler.scale_[cond_idx]
            tensor_stats("TEST_future_cond_before_scale", future_cond_np[:256])
            future_cond_np = (future_cond_np - cond_mean[None, None, :]) / cond_scale[None, None, :]
            future_cond_np = np.clip(future_cond_np, -10.0, 10.0).astype(np.float32, copy=False)
            tensor_stats("TEST_future_cond_after_scale", future_cond_np[:256])
    except Exception as e:
        print(f"[WorldModelV1] Warning: could not load/apply X scaler for test eval: {e}")

    X = torch.from_numpy(X_np).float().to(device)
    Y_hi = torch.from_numpy(Y_hi_np).float().to(device)
    Y_rul = torch.from_numpy(Y_rul_np).float().to(device)
    cond_ids = torch.from_numpy(cond_ids_np).long().to(device)
    future_cond = torch.from_numpy(future_cond_np).float().to(device)

    batch_size = int(getattr(world_model_config, "batch_size", 256))
    ds = TensorDataset(X, Y_hi, Y_rul, cond_ids, future_cond)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_true_rul_last: List[float] = []
    all_pred_rul_last: List[float] = []
    all_unit_ids: List[np.ndarray] = []
    idx_offset = 0

    cond_dim = getattr(world_model_config, "cond_dim", 9)

    with torch.no_grad():
        for X_b, Y_hi_b, Y_rul_b, cond_b, future_cond_b in loader:
            B, _, _ = X_b.shape
            cond_vec = torch.zeros(B, cond_dim, device=device)

            # Approximate current RUL from HI (since HI ≈ RUL/max_rul in our mapping)
            current_hi_b = Y_hi_b[:, 0]  # (B,)
            current_rul_b = current_hi_b  # normalized RUL in [0,1]

            out = model(
                past_seq=X_b,
                cond_vec=cond_vec,
                cond_ids=cond_b,
                future_horizon=horizon,
                teacher_forcing_targets=None,
                current_rul=current_rul_b,
                current_hi=current_hi_b,
                future_conds=future_cond_b if bool(getattr(world_model_config, "use_future_conds", False)) else None,
            )
            pred_sensors = None
            pred_hi = None
            pred_rul = None
            pred_eol = None

            # Robust output mapping:
            # - tuple/list: (pred_sensors, pred_hi_seq, pred_rul_seq, pred_eol_scalar?)  (current WM-V1)
            # - dict: keys like pred_rul/pred_hi/pred_eol (future-proof)
            if isinstance(out, dict):
                pred_sensors = out.get("pred_sensors", None)
                pred_hi = out.get("pred_hi", out.get("hi", None))
                pred_rul = out.get("pred_rul", out.get("rul", None))
                pred_eol = out.get("pred_eol", out.get("eol", None))
            elif isinstance(out, (tuple, list)):
                if len(out) >= 3:
                    pred_sensors = out[0]
                    pred_hi = out[1]
                    pred_rul = out[2]
                if len(out) >= 4:
                    pred_eol = out[3]
            else:
                raise RuntimeError(
                    f"[WorldModelV1-Test] Unexpected model output type: {type(out)}"
                )

            if pred_rul is None:
                continue

            # Optional: deterministic linear-decay RUL sequence from pred_rul0 (physics-consistent)
            if bool(getattr(world_model_config, "rul_linear_decay", False)) and isinstance(pred_rul, torch.Tensor):
                max_rul_denom = float(getattr(world_model_config, "max_rul", 125.0))
                H = int(pred_rul.size(1))
                pred_rul0 = pred_rul[:, 0, 0].clamp(0.0, 1.0)  # (B,)
                k = torch.arange(H, device=pred_rul.device, dtype=pred_rul.dtype).view(1, H)
                pred_seq = (pred_rul0.unsqueeze(1) - k / max(max_rul_denom, 1e-6)).clamp(0.0, 1.0)
                pred_rul = pred_seq.unsqueeze(-1)  # (B,H,1)

            # Model predicts normalized RUL; take last horizon step and denormalize
            pred_rul_last_norm = pred_rul[:, -1, 0]  # (B,)
            pred_rul_last = (pred_rul_last_norm * max_rul).cpu().numpy()

            # Eval sanity logging for first batch
            if idx_offset == 0:
                from src.tools.debug_stats import tensor_stats, batch_time_std, compare_two_samples

                tensor_stats("TEST_X_b", X_b)
                tensor_stats("TEST_future_cond_b", future_cond_b)
                tensor_stats("TEST_future_cond_b_scaled", future_cond_b)

                # [dbg] True RUL horizon sequence stats (normalized) – same horizon length H
                y_rul_seq_norm = Y_rul_b.unsqueeze(-1)  # (B,H,1)
                tensor_stats("true_rul_seq_norm", y_rul_seq_norm)
                batch_time_std("true_rul_seq_norm", y_rul_seq_norm)
                compare_two_samples("true_rul_seq_norm", y_rul_seq_norm, t_steps=10)
                cap_thr = float(getattr(world_model_config, "rul_cap_threshold", 0.999999) or 0.999999)
                true_rul_cap_frac_batch = float((y_rul_seq_norm >= cap_thr).float().mean().detach().cpu())
                print(f"[dbg] true_rul_cap_frac_batch={true_rul_cap_frac_batch:.6f}")
                cap_mask_eps = float(getattr(world_model_config, "cap_mask_eps", 1e-6) or 1e-6)
                frac_all_cap_future_batch = float(
                    (y_rul_seq_norm[:, :, 0].min(dim=1).values >= (1.0 - cap_mask_eps)).float().mean().detach().cpu()
                )
                print(f"[dbg] frac_all_cap_future_batch={frac_all_cap_future_batch:.6f}")

                # Also report r0-cycle stats + combined mask keep fraction for visibility
                max_rul_cycles = float(getattr(world_model_config, "max_rul", 125.0))
                true_rul0_cycles = (y_rul_seq_norm[:, 0, 0] * max(max_rul_cycles, 1e-6)).detach()
                pred_rul0_cycles = (pred_rul[:, 0, 0] * max(max_rul_cycles, 1e-6)).detach()
                print(
                    f"[dbg] true_rul0_cycles(min/mean/max)="
                    f"{float(true_rul0_cycles.min().cpu()):.2f}/"
                    f"{float(true_rul0_cycles.mean().cpu()):.2f}/"
                    f"{float(true_rul0_cycles.max().cpu()):.2f} "
                    f"pred_rul0_cycles(min/mean/max)="
                    f"{float(pred_rul0_cycles.min().cpu()):.2f}/"
                    f"{float(pred_rul0_cycles.mean().cpu()):.2f}/"
                    f"{float(pred_rul0_cycles.max().cpu()):.2f}"
                )
                rul_train_max_cycles = getattr(world_model_config, "rul_train_max_cycles", None)
                cap_mask = (y_rul_seq_norm < cap_thr)
                early_mask = torch.ones_like(cap_mask, dtype=torch.bool)
                if rul_train_max_cycles is not None:
                    early_mask = ((y_rul_seq_norm * max(max_rul_cycles, 1e-6)) < float(rul_train_max_cycles))
                mask_keep_frac_test = float((cap_mask & early_mask).float().mean().detach().cpu())
                print(f"[dbg] mask_keep_frac_test={mask_keep_frac_test:.6f}")

                tensor_stats("pred_rul_seq_norm", pred_rul)
                batch_time_std("pred_rul_seq_norm", pred_rul)
                compare_two_samples("pred_rul_seq_norm", pred_rul, t_steps=5)

                if pred_hi is not None:
                    tensor_stats("pred_hi_seq_norm", pred_hi)
                    batch_time_std("pred_hi_seq_norm", pred_hi)
                    compare_two_samples("pred_hi_seq_norm", pred_hi, t_steps=5)

                if pred_eol is not None:
                    tensor_stats("pred_eol", pred_eol)
                    if isinstance(pred_eol, torch.Tensor) and pred_eol.numel() > 1:
                        print(f"[dbg][pred_eol] std={pred_eol.detach().float().std().item():.8f}")

                print(f"[dbg] pred_rul_seq shape={tuple(pred_rul.shape)}")
                print(f"[dbg] pred_rul_last shape={tuple(pred_rul_last.shape)}")

                pr = pred_rul.detach()
                print(
                    "[WorldModelV1-Test][debug] pred_rul_seq (normalized) "
                    f"min={float(pr.min()):.4f} mean={float(pr.mean()):.4f} max={float(pr.max()):.4f}"
                )
                try:
                    pr_flat = pr.reshape(-1)
                    tr_flat = y_rul_seq_norm.detach().reshape(-1)
                    print(
                        "[WorldModelV1-Test][debug] mean/std "
                        f"pred_rul_seq_norm={float(pr_flat.mean()):.6f}/{float(pr_flat.std(unbiased=False)):.6f} "
                        f"true_rul_seq_norm={float(tr_flat.mean()):.6f}/{float(tr_flat.std(unbiased=False)):.6f}"
                    )
                except Exception:
                    pass
                print(
                    "[WorldModelV1-Test][debug] pred_rul_last (cycles, unclipped) "
                    f"min={float(pred_rul_last.min()):.2f} mean={float(pred_rul_last.mean()):.2f} max={float(pred_rul_last.max()):.2f}"
                )
                if pred_eol is not None:
                    pe = pred_eol.detach()
                    print(
                        "[WorldModelV1-Test][debug] pred_eol (NOT used for RUL metrics) "
                        f"shape={tuple(pe.shape)} min={float(pe.min()):.4f} mean={float(pe.mean()):.4f} max={float(pe.max()):.4f}"
                    )

            unit_ids_batch = unit_ids_np[idx_offset : idx_offset + B].astype(np.int64, copy=False)
            true_rul_last_batch = np.array(true_rul_last_list[idx_offset : idx_offset + B], dtype=np.float32)
            idx_offset += B

            all_true_rul_last.append(true_rul_last_batch)
            all_pred_rul_last.append(pred_rul_last)
            all_unit_ids.append(unit_ids_batch)

    if not all_true_rul_last:
        print("[WorldModelV1-Test] No RUL predictions were produced.")
        return {}

    y_true = np.concatenate(all_true_rul_last, axis=0)
    y_pred = np.concatenate(all_pred_rul_last, axis=0)
    unit_ids_all = np.concatenate(all_unit_ids, axis=0)

    # Log unclipped vs clipped stats (clip happens inside evaluate_eol_metrics when clip_y_pred=True)
    y_pred_clip = np.clip(y_pred, 0.0, float(max_rul))
    print(
        "[WorldModelV1-Test][debug] y_pred cycles stats: "
        f"unclipped(min/mean/max)={float(y_pred.min()):.2f}/{float(y_pred.mean()):.2f}/{float(y_pred.max()):.2f} | "
        f"clipped(min/mean/max)={float(y_pred_clip.min()):.2f}/{float(y_pred_clip.mean()):.2f}/{float(y_pred_clip.max()):.2f}"
    )

    clip = (0.0, float(max_rul)) if max_rul is not None else None
    metrics_all = compute_all_samples_metrics(y_true, y_pred, unit_ids=unit_ids_all, clip=clip)
    metrics_last = compute_last_per_unit_metrics(unit_ids_all, y_true, y_pred, clip=clip)

    # --------------------------------------------------------------
    # Split LAST metrics: capped vs uncapped LAST targets (diagnostic)
    # --------------------------------------------------------------
    try:
        # stable last index per unit: last occurrence wins (same as compute_last_per_unit_metrics)
        last_idx: Dict[int, int] = {}
        for i, uid in enumerate(unit_ids_all.tolist()):
            last_idx[int(uid)] = int(i)
        idxs = np.array(sorted(last_idx.values()), dtype=np.int64)
        yt_last = np.asarray(y_true, dtype=float).reshape(-1)[idxs]
        yp_last = np.asarray(y_pred, dtype=float).reshape(-1)[idxs]

        eps = float(getattr(world_model_config, "cap_mask_eps", 1e-6) or 1e-6)
        yt_last_norm = np.clip(yt_last / max(float(max_rul), 1e-6), 0.0, 1.0)
        is_capped = yt_last_norm >= (1.0 - eps)
        frac_last_capped = float(np.mean(is_capped)) if yt_last_norm.size else 0.0

        def _basic_metrics(a_true: np.ndarray, a_pred: np.ndarray) -> Dict[str, float]:
            if a_true.size == 0:
                return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
            err = a_pred - a_true
            return {
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
            }

        m_unc = _basic_metrics(yt_last[~is_capped], yp_last[~is_capped])
        m_cap = _basic_metrics(yt_last[is_capped], yp_last[is_capped])
        split_last = {
            "frac_last_capped": float(frac_last_capped),
            "n_last_uncapped": int((~is_capped).sum()),
            "rmse_last_uncapped": float(m_unc["rmse"]),
            "mae_last_uncapped": float(m_unc["mae"]),
            "bias_last_uncapped": float(m_unc["bias"]),
            "n_last_capped": int(is_capped.sum()),
            "rmse_last_capped": float(m_cap["rmse"]),
            "mae_last_capped": float(m_cap["mae"]),
            "bias_last_capped": float(m_cap["bias"]),
        }
        print(
            "[WorldModelV1-Test] LAST split: "
            f"uncapped n={split_last['n_last_uncapped']} rmse={split_last['rmse_last_uncapped']:.2f} bias={split_last['bias_last_uncapped']:.2f} | "
            f"capped n={split_last['n_last_capped']} rmse={split_last['rmse_last_capped']:.2f} bias={split_last['bias_last_capped']:.2f}"
        )
    except Exception as e:
        split_last = {"split_last_error": str(e)}

    merged = {
        **metrics_last,
        **metrics_all,
        "dataset_split": "test",
        "last_definition": metrics_last.get("note_last_definition", "LAST_AVAILABLE_PER_UNIT (truncated-aware)"),
        **(split_last if isinstance(split_last, dict) else {}),
    }
    return merged


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

    clip = (0.0, float(max_rul)) if max_rul is not None else None

    # Use the central builder for "last window per unit" (LAST)
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
    # Stage -1: FD004 evaluation consistency (clip y_true to [0, max_rul])
    if clip_y_true_to_max_rul and max_rul is not None:
        max_rul_f = float(max_rul)
        y_true = np.clip(y_true, 0.0, max_rul_f)
        y_true_max = float(np.max(y_true))
        if y_true_max > max_rul_f + 1e-6:
            raise ValueError(
                f"[Stage-1] Evaluation clipping failed: y_true max={y_true_max:.2f} > {max_rul_f:.2f}"
            )
    unit_ids_np = built["unit_ids"]
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
        y_pred_last = np.asarray(np.clip(e, 0.0, float(max_rul)), dtype=float).reshape(-1)

    # LAST metrics
    metrics_last = compute_last_per_unit_metrics(unit_ids_np, y_true, y_pred_last, clip=clip)

    # ALL metrics: derive per-timepoint truth for test via y_test_true + remaining-to-last-observed,
    # then build sliding windows with horizon=1 and current_from_df targets.
    metrics_all: Dict[str, Any] = {}
    try:
        df_eval = df_test.copy()
        # Create an eval RUL column that is valid for the truncated test trajectories:
        # RUL(t) = RUL_last_observed + (t_last - t)
        rul_eval = np.zeros((len(df_eval),), dtype=float)
        for uid, df_u in df_eval.groupby("UnitNumber"):
            uid_i = int(uid)
            idx = {i + 1: i for i in range(len(y_test_true))}.get(uid_i, uid_i - 1)
            rul_last_obs = float(y_test_true[idx]) if idx < len(y_test_true) else float(y_test_true[-1])
            t_last = float(df_u["TimeInCycles"].max())
            mask = (df_eval["UnitNumber"].to_numpy() == uid_i)
            times = df_eval.loc[mask, "TimeInCycles"].to_numpy(dtype=float)
            rul_eval[mask] = rul_last_obs + (t_last - times)
        df_eval["RUL_eval"] = rul_eval

        wc_all = WindowConfig(past_len=int(past_len), horizon=1, stride=1, require_full_horizon=False, pad_mode="clamp")
        tc_all = TargetConfig(max_rul=int(max_rul), cap_targets=True, eol_target_mode="current_from_df")
        built_all = build_sliding_windows(
            df_eval,
            feature_cols,
            target_col="RUL_eval",
            unit_col="UnitNumber",
            time_col="TimeInCycles",
            cond_col="ConditionID",
            window_cfg=wc_all,
            target_cfg=tc_all,
            return_mask=False,
        )

        X_all = built_all["X"]
        y_true_all = built_all["Y_eol"].astype(float)
        unit_ids_all = built_all["unit_ids"].astype(np.int64)
        cond_ids_all = built_all["cond_ids"].astype(np.int64)

        # Condition-wise scaling
        X_all_scaled = np.empty_like(X_all, dtype=np.float32)
        for cond in np.unique(cond_ids_all):
            cond = int(cond)
            idxs = np.where(cond_ids_all == cond)[0]
            scaler = scaler_dict.get(cond, scaler_dict.get(0))
            flat = X_all[idxs].reshape(-1, len(feature_cols))
            X_all_scaled[idxs] = scaler.transform(flat).reshape(-1, int(past_len), len(feature_cols)).astype(np.float32)

        X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)
        cond_all_t = torch.tensor(cond_ids_all, dtype=torch.long).to(device) if num_conditions > 1 else None

        with torch.no_grad():
            out_all = model(
                encoder_inputs=X_all_t,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
                cond_ids=cond_all_t,
            )
            if isinstance(out_all, dict):
                eol_all = out_all.get("eol", out_all.get("rul"))
            else:
                eol_all = out_all[1] if isinstance(out_all, (tuple, list)) and len(out_all) >= 2 else out_all
            if torch.is_tensor(eol_all):
                y_pred_all = eol_all.view(-1).detach().cpu().numpy().astype(float)
            else:
                y_pred_all = np.asarray(eol_all, dtype=float).reshape(-1)
        y_pred_all = np.clip(y_pred_all, 0.0, float(max_rul)).astype(float)

        metrics_all = compute_all_samples_metrics(y_true_all, y_pred_all, unit_ids=unit_ids_all, clip=clip)
    except Exception as e:
        print(f"[eval] Warning: could not compute ALL windows metrics for test: {e}")
        metrics_all = {"n_samples_all": 0}

    merged = {
        **metrics_last,
        **metrics_all,
        "dataset_split": "test",
        "last_definition": metrics_last.get("note_last_definition", "LAST_AVAILABLE_PER_UNIT (truncated-aware)"),
        "unit_ids_last": unit_ids_np.tolist(),
        "y_true_last": y_true.tolist(),
        "y_pred_last": y_pred_last.tolist(),
    }

    # Backward-compat aliases for older diagnostics code (deprecated names).
    merged["y_true_eol"] = merged["y_true_last"]
    merged["y_pred_eol"] = merged["y_pred_last"]
    try:
        from src.metrics import compute_eol_errors_and_nasa
        nasa_stats = compute_eol_errors_and_nasa(np.asarray(y_true, dtype=float), np.asarray(y_pred_last, dtype=float), max_rul=float(max_rul))
        merged["nasa_scores"] = nasa_stats["nasa_scores"].tolist()
    except Exception:
        pass

    # Legacy aliases (deprecated): previous behavior was LAST on test.
    merged["RMSE"] = merged.get("rmse_last")
    merged["MAE"] = merged.get("mae_last")
    merged["Bias"] = merged.get("bias_last")
    merged["R2"] = merged.get("r2_last")
    merged["nasa_score_sum"] = merged.get("nasa_last_sum")
    merged["nasa_score_mean"] = merged.get("nasa_last_mean")
    merged["num_engines"] = merged.get("n_units")

    return merged

