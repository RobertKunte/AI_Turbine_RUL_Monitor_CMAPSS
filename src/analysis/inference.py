"""
Inference utilities for analyzing experiment results.

This module provides functions to run inference on trained models and collect
per-engine metrics and trajectories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_loading import load_cmapps_subset
from src.additional_features import (
    create_physical_features,
    create_all_features,
    FeatureConfig,
    TemporalFeatureConfig,
    PhysicsFeatureConfig,
)
from src.feature_safety import remove_rul_leakage, check_feature_dimensions
from src.eol_full_lstm import (
    EOLFullLSTMWithHealth,
    build_full_eol_sequences_from_df,
    build_test_sequences_from_df,
)
from src.metrics import nasa_phm_score_single
from src.eval_utils import forward_rul_only
from sklearn.preprocessing import StandardScaler
from src.config import USE_CONDITION_WISE_SCALING
from src.models.universal_encoder_v1 import (
    UniversalEncoderV1,
    UniversalEncoderV2,
    RULHIUniversalModel,
    RULHIUniversalModelV2,
    UniversalEncoderV3Attention,
)
from src.models.transformer_eol import EOLFullTransformerEncoder


@dataclass
class EngineEOLMetrics:
    """EOL metrics for a single engine."""
    unit_id: int
    true_rul: float
    pred_rul: float
    error: float  # pred - true
    nasa: float  # NASA contribution for this engine


@dataclass
class EngineTrajectory:
    """Full trajectory for a single engine."""
    unit_id: int
    cycles: np.ndarray  # shape (T,)
    hi: np.ndarray  # shape (T,)
    true_rul: np.ndarray  # shape (T,)
    pred_rul: np.ndarray  # shape (T,)
    hi_damage: Optional[np.ndarray] = None  # shape (T,) - optional damage-based HI trajectory
    hi_cal: Optional[np.ndarray] = None     # shape (T,) - optional calibrated HI (v4) trajectory


def load_model_from_experiment(
    experiment_dir: Path,
    device: torch.device | str = "cpu",
    input_dim: Optional[int] = None,
    num_conditions: Optional[int] = None,
) -> Tuple[nn.Module, dict]:
    """
    Load model and config from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (contains summary.json and model.pt)
        device: Device to load model on
        input_dim: Input feature dimension (if None, will try to infer from checkpoint/config)
        num_conditions: Number of conditions (if None, will try to infer from checkpoint/config)
    
    Returns:
        model: Loaded model
        config: Configuration dictionary from summary.json
    """
    # Load summary.json for config
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {experiment_dir}")
    
    with open(summary_path, "r") as f:
        config = json.load(f)
    
    # Find model checkpoint(s)
    model_files = list(experiment_dir.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt model file found in {experiment_dir}")

    # ------------------------------------------------------------------
    # 1) Choose checkpoint file with V3-aware priority
    # ------------------------------------------------------------------
    model_version = config.get("model_version")
    model_type = config.get("model_type")
    encoder_type_cfg = config.get("encoder_type")
    experiment_name = config.get("experiment_name", "")

    # Does the config explicitly request a V3 world model?
    config_requests_v3 = (
        model_version == "v3"
        or model_type == "world_model_universal_v3"
        or encoder_type_cfg == "world_model_universal_v3"
        or ("world" in experiment_name.lower() and "phase5" in experiment_name.lower())
        or ("world" in experiment_name.lower() and "v3" in experiment_name.lower())
    )

    model_path: Path
    if config_requests_v3:
        # Prefer V3-specific filenames first
        preferred_names = [
            "world_model_v3_best.pt",
            "world_model_v3.pt",
        ]
        v3_candidate = None
        for name in preferred_names:
            candidate = experiment_dir / name
            if candidate.exists():
                v3_candidate = candidate
                break
        if v3_candidate is not None:
            model_path = v3_candidate
        else:
            # Fall back to old logic (best*.pt), but later we will *strictly*
            # validate that the checkpoint actually contains V3 heads.
            best_models = [f for f in model_files if "best" in f.name.lower()]
            model_path = best_models[0] if best_models else model_files[0]
    else:
        # Non-V3 experiments: keep original best-model selection
        best_models = [f for f in model_files if "best" in f.name.lower()]
        model_path = best_models[0] if best_models else model_files[0]

    print(f"Loading model from {model_path}")

    # Load checkpoint first to inspect weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict and metadata (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            meta = checkpoint.get("meta", {})
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            meta = checkpoint.get("meta", {})
        else:
            # Checkpoint might be the state dict itself
            state_dict = checkpoint
            meta = {}
    else:
        state_dict = checkpoint
        meta = {}
    
    # Detect if this is a world model experiment
    # Check multiple sources: model_type, encoder_type, experiment_name
    encoder_type = meta.get("encoder_type") or encoder_type_cfg
    
    # Robust world model detection
    # Helper: detect presence of V3 heads regardless of potential 'module.' prefix
    def _has_v3_heads(sd: Dict[str, torch.Tensor]) -> bool:
        keys = list(sd.keys())
        has_health = any(k.endswith("fc_health.weight") for k in keys)
        # EOL head might be named fc_rul or fc_eol depending on version
        has_eol = any(k.endswith("fc_rul.weight") or k.endswith("fc_eol.weight") for k in keys)
        has_traj = any(k.endswith("traj_head.weight") for k in keys)
        return has_health and has_eol and has_traj

    has_v3_keys = _has_v3_heads(state_dict)

    # Check by name/config again (using updated encoder_type)
    is_world_model_v3_by_name = (
        model_version == "v3"
        or model_type == "world_model_universal_v3"
        or encoder_type == "world_model_universal_v3"
        or ("world" in experiment_name.lower() and "v3" in experiment_name.lower())
        or ("world" in experiment_name.lower() and "phase5" in experiment_name.lower())
    )

    # If config explicitly requests V3 but checkpoint lacks V3 heads -> hard error
    if is_world_model_v3_by_name and not has_v3_keys:
        sample_keys = list(state_dict.keys())[:10]
        raise RuntimeError(
            f"Loaded checkpoint '{model_path.name}' is missing World Model V3 heads "
            f"(fc_health / fc_rul / traj_head), but config requests V3.\n"
            f"Experiment: {experiment_name}\n"
            f"Hint: Ensure you are loading 'world_model_v3_best.pt' for V3 runs.\n"
            f"Sample keys: {sample_keys}"
        )

    # Now we can safely flag V3 if both config and heads agree
    is_world_model_v3 = is_world_model_v3_by_name and has_v3_keys
    # Then check for v2 or general world model
    is_world_model = (
        is_world_model_v3
        or model_type == "world_model_universal_v2"
        or encoder_type == "world_model_universal_v2"
        or ("world_phase" in experiment_name.lower() and not is_world_model_v3)
        or ("world_model" in experiment_name.lower() and not is_world_model_v3)
        or config.get("training_mode") == "world_model"
    )
    
    # If we detect world model, set encoder_type accordingly
    if is_world_model:
        if is_world_model_v3:
            encoder_type = "world_model_universal_v3"
            print(f"Detected world-model v3 experiment: {experiment_name}")
            print(f"  model_type={model_type}, encoder_type={encoder_type}")
        else:
            encoder_type = "world_model_universal_v2"
            print(f"Detected world-model v2 experiment: {experiment_name}")
            print(f"  model_type={model_type}, encoder_type={encoder_type}")
    elif encoder_type is None:
        # If encoder_type is still None, try to infer from state_dict
        # Check for world model v3 first (has decoder, fc_health, fc_rul, shared_head, traj_head)
        if ("decoder.weight_ih_l0" in state_dict and 
            "fc_health.weight" in state_dict and 
            "fc_rul.weight" in state_dict and 
            "shared_head.0.weight" in state_dict and
            "traj_head.weight" in state_dict):
            # This looks like a world model v3
            encoder_type = "world_model_universal_v3"
            is_world_model = True
            is_world_model_v3 = True
            print(f"Inferred world_model_universal_v3 from checkpoint state_dict (decoder + fc_health + fc_rul + shared_head + traj_head present)")
        # Check for world model v2 (has decoder and eol_head, but not fc_health/fc_rul)
        elif "decoder.weight_ih_l0" in state_dict and "eol_head.0.weight" in state_dict:
            # This looks like a world model v2
            encoder_type = "world_model_universal_v2"
            is_world_model = True
            print(f"Inferred world_model_universal_v2 from checkpoint state_dict (decoder + eol_head present)")
        # Check for universal_v2: has encoder.cnn_branches
        elif "encoder.cnn_branches.0.0.weight" in state_dict:
            # Check if it's v1 or v2 by looking for seq_encoder (v2 has transformer/lstm inside)
            if "encoder.seq_encoder" in state_dict or any("seq_encoder" in k for k in state_dict.keys()):
                encoder_type = "universal_v2"
            else:
                encoder_type = "universal_v1"
        elif "encoder.lstm.weight_ih_l0" in state_dict:
            encoder_type = "lstm"
        elif "encoder.input_proj.weight" in state_dict:
            encoder_type = "transformer"
        else:
            encoder_type = "lstm"  # Default fallback
        print(f"Inferred encoder_type={encoder_type} from checkpoint state_dict")
    else:
        print(f"Using encoder_type={encoder_type} (from {'metadata' if 'encoder_type' in meta else 'config'})")
        if not is_world_model:
            print(f"Detected RUL/HI experiment: {experiment_name}")
    
    # Use metadata from checkpoint if available
    if input_dim is None and "input_dim" in meta:
        input_dim = meta["input_dim"]
        print(f"Using input_dim={input_dim} from checkpoint metadata")
    if num_conditions is None and "num_conditions" in meta:
        num_conditions = meta["num_conditions"]
        print(f"Using num_conditions={num_conditions} from checkpoint metadata")
    
    # Determine num_conditions: use provided value, then try checkpoint metadata, then infer from checkpoint, then config
    if num_conditions is None:
        # First try metadata (already checked above)
        if "num_conditions" in meta:
            num_conditions = meta["num_conditions"]
            use_condition_embedding = meta.get("use_condition_embedding", False)
            print(f"Using num_conditions={num_conditions} from checkpoint metadata")
        # Then try to infer from checkpoint state dict
        elif "condition_embedding.weight" in state_dict:
            num_conditions = state_dict["condition_embedding.weight"].shape[0]
            use_condition_embedding = True
            print(f"Inferred num_conditions={num_conditions} from checkpoint (condition_embedding)")
        elif "encoder.cond_emb.weight" in state_dict:
            num_conditions = state_dict["encoder.cond_emb.weight"].shape[0]
            use_condition_embedding = True
            print(f"Inferred num_conditions={num_conditions} from checkpoint (encoder.cond_emb)")
        else:
            # Try config
            use_condition_embedding = config.get("use_condition_embedding", False)
            if use_condition_embedding:
                num_conditions = config.get("num_conditions")
                if num_conditions is None:
                    raise ValueError(
                        "num_conditions must be provided or stored in checkpoint/config. "
                        "Please provide num_conditions parameter or ensure it's in summary.json."
                    )
                print(f"Using num_conditions={num_conditions} from config")
            else:
                num_conditions = 1
                print(f"No condition embeddings detected, using num_conditions=1")
    else:
        # Use provided value
        use_condition_embedding = config.get("use_condition_embedding", False)
        # Check if condition embeddings are actually used in checkpoint
        if "condition_embedding.weight" in state_dict or "encoder.cond_emb.weight" in state_dict:
            use_condition_embedding = True
        print(f"Using provided num_conditions={num_conditions}")
    
    # Determine cond_emb_dim
    cond_emb_dim = config.get("cond_emb_dim", 4) if use_condition_embedding else 0
    if use_condition_embedding:
        # Try to infer from checkpoint
        if "condition_embedding.weight" in state_dict:
            cond_emb_dim = state_dict["condition_embedding.weight"].shape[1]
        elif "encoder.cond_emb.weight" in state_dict:
            cond_emb_dim = state_dict["encoder.cond_emb.weight"].shape[1]
    
    # Handle different encoder types
    if encoder_type == "world_model_universal_v3":
        # World Model v3 with Health Index head
        from src.models.world_model import WorldModelUniversalV3
        
        # Determine input_dim: use provided value, then try checkpoint, then config
        if input_dim is None:
            # Try to infer from CNN first layer (encoder.cnn_branches.*)
            if "encoder.cnn_branches.0.0.weight" in state_dict:
                input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from WorldModel v3 UniversalEncoderV2 CNN checkpoint")
            else:
                # Fallback: use num_features / input_dim from summary.json
                input_dim = config.get("num_features") or config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")
        
        d_model = config.get("d_model", 96)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", 384)
        dropout = config.get("dropout", 0.1)
        kernel_sizes = config.get("kernel_sizes", [3, 5, 9])
        seq_encoder_type = config.get("seq_encoder_type", "transformer")
        decoder_num_layers = config.get("decoder_num_layers", 2)
        horizon = config.get("horizon", 40)
        
        # Check for condition embeddings
        if "encoder.cond_emb.weight" in state_dict:
            num_conditions = state_dict["encoder.cond_emb.weight"].shape[0]
            cond_emb_dim = state_dict["encoder.cond_emb.weight"].shape[1]
            print(f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} from WorldModel v3 checkpoint")
        else:
            num_conditions = config.get("num_conditions", None)
            cond_emb_dim = config.get("cond_emb_dim", 4) if num_conditions and num_conditions > 1 else 0
        
        # HI fusion and tail-weighting config (may be absent in older runs)
        wm_cfg = config.get("world_model_config", {})
        use_hi_in_eol = wm_cfg.get("use_hi_in_eol", False)
        use_hi_slope_in_eol = wm_cfg.get("use_hi_slope_in_eol", False)

        model = WorldModelUniversalV3(
            input_size=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=True,
            max_seq_len=300,
            decoder_num_layers=decoder_num_layers,
            horizon=horizon,
            use_hi_in_eol=use_hi_in_eol,
            use_hi_slope_in_eol=use_hi_slope_in_eol,
        )
        
        # Load state dict for world model v3
        try:
            model.load_state_dict(state_dict, strict=True)
            print("  Successfully loaded world model v3 checkpoint (strict=True)")
        except RuntimeError as e:
            print(f"  Warning: Could not load with strict=True: {e}")
            print("  Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("  Loaded world model v3 checkpoint (strict=False)")
        
        model = model.to(device)
        return model, config
    elif encoder_type == "world_model_universal_v2":
        # World Model with UniversalEncoderV2
        from src.models.world_model import WorldModelEncoderDecoderUniversalV2
        
        # Determine input_dim: use provided value, then try checkpoint, then config
        if input_dim is None:
            # Try to infer from CNN first layer
            if "encoder.cnn_branches.0.0.weight" in state_dict:
                input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from WorldModel UniversalEncoderV2 CNN checkpoint")
            else:
                # Try config
                input_dim = config.get("num_features") or config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")
        
        d_model = config.get("d_model", 96)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", 384)
        dropout = config.get("dropout", 0.1)
        kernel_sizes = config.get("kernel_sizes", [3, 5, 9])
        seq_encoder_type = config.get("seq_encoder_type", "transformer")
        decoder_num_layers = config.get("decoder_num_layers", 2)
        
        # Check for condition embeddings
        if "encoder.cond_emb.weight" in state_dict:
            num_conditions = state_dict["encoder.cond_emb.weight"].shape[0]
            cond_emb_dim = state_dict["encoder.cond_emb.weight"].shape[1]
            print(f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} from WorldModel checkpoint")
        else:
            num_conditions = config.get("num_conditions", None)
            cond_emb_dim = config.get("cond_emb_dim", 4) if num_conditions and num_conditions > 1 else 0
        
        model = WorldModelEncoderDecoderUniversalV2(
            input_size=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_size=1,
            num_conditions=num_conditions if num_conditions and num_conditions > 1 else None,
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=True,
            max_seq_len=300,
            decoder_num_layers=decoder_num_layers,
        )
        
        # Load state dict for world model
        try:
            model.load_state_dict(state_dict, strict=True)
            print("  Successfully loaded world model checkpoint (strict=True)")
        except RuntimeError as e:
            print(f"  Warning: Could not load with strict=True: {e}")
            print("  Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("  Loaded world model checkpoint (strict=False)")
        
        model = model.to(device)
        return model, config
    elif encoder_type == "universal_v2":
        # For UniversalEncoderV2, infer from checkpoint
        # First, check if condition embeddings are used by inspecting checkpoint
        if "encoder.cond_emb.weight" in state_dict:
            # Condition embeddings are present in checkpoint
            num_conditions = state_dict["encoder.cond_emb.weight"].shape[0]
            use_cond_fusion = True
            cond_emb_dim = state_dict["encoder.cond_emb.weight"].shape[1]
            print(f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} from UniversalEncoderV2 checkpoint")
        else:
            # No condition embeddings in checkpoint
            use_cond_fusion = config.get("use_condition_embedding", False)
            cond_emb_dim = config.get("cond_emb_dim", 4) if use_cond_fusion else 0
            num_conditions = config.get("num_conditions", None) if use_cond_fusion else None
            if use_cond_fusion:
                print(f"Warning: use_condition_embedding=True in config but no cond_emb in checkpoint, using num_conditions={num_conditions} from config")
        
        # Determine input_dim: use provided value, then try checkpoint, then config
        if input_dim is None:
            # Try to infer from CNN first layer
            if "encoder.cnn_branches.0.0.weight" in state_dict:
                input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from UniversalEncoderV2 CNN checkpoint")
            else:
                # Try config
                input_dim = config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")
        
        d_model = config.get("d_model", 64)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", None)
        dropout = config.get("dropout", 0.1)
        kernel_sizes = config.get("kernel_sizes", [3, 5, 9])
        seq_encoder_type = config.get("seq_encoder_type", "transformer")
        use_layer_norm = config.get("use_layer_norm", True)
        
        encoder = UniversalEncoderV2(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim if use_cond_fusion else 0,
            max_seq_len=300,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
        )
        
        model = RULHIUniversalModelV2(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    elif encoder_type == "universal_v1":
        # For UniversalEncoderV1, infer from checkpoint
        # First, check if condition embeddings are used by inspecting checkpoint
        if "encoder.cond_emb.weight" in state_dict:
            # Condition embeddings are present in checkpoint
            num_conditions = state_dict["encoder.cond_emb.weight"].shape[0]
            use_cond_fusion = True
            cond_emb_dim = state_dict["encoder.cond_emb.weight"].shape[1]
            print(f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} from UniversalEncoderV1 checkpoint")
        else:
            # No condition embeddings in checkpoint
            use_cond_fusion = config.get("use_condition_embedding", False)
            cond_emb_dim = config.get("cond_emb_dim", 4) if use_cond_fusion else 0
            num_conditions = config.get("num_conditions", None) if use_cond_fusion else None
            if use_cond_fusion:
                print(f"Warning: use_condition_embedding=True in config but no cond_emb in checkpoint, using num_conditions={num_conditions} from config")
        
        # Determine input_dim: use provided value, then try checkpoint, then config
        if input_dim is None:
            # Try to infer from CNN first layer
            if "encoder.cnn_branches.0.0.weight" in state_dict:
                input_dim = state_dict["encoder.cnn_branches.0.0.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from UniversalEncoderV1 CNN checkpoint")
            else:
                # Try config
                input_dim = config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")
        
        d_model = config.get("d_model", 48)
        cnn_channels = config.get("cnn_channels", None)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 4)
        dim_feedforward = config.get("dim_feedforward", 256)
        dropout = config.get("dropout", 0.1)
        
        encoder = UniversalEncoderV1(
            input_dim=input_dim,
            d_model=d_model,
            cnn_channels=cnn_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim if use_cond_fusion else 0,
            max_seq_len=300,
        )
        
        model = RULHIUniversalModel(
            encoder=encoder,
            d_model=d_model,
            dropout=dropout,
        )
    elif encoder_type == "universal_v3_attention":
        # Transformer + Attention encoder (UniversalEncoderV3Attention)
        # Infer condition embedding setup from checkpoint or config
        if "cond_emb.weight" in state_dict:
            num_conditions = state_dict["cond_emb.weight"].shape[0]
            use_cond_fusion = True
            cond_emb_dim = state_dict["cond_emb.weight"].shape[1]
            print(
                f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} "
                f"from UniversalEncoderV3Attention checkpoint"
            )
        else:
            use_cond_fusion = config.get("use_condition_embedding", False)
            cond_emb_dim = config.get("cond_emb_dim", 4) if use_cond_fusion else 0
            num_conditions = config.get("num_conditions", None) if use_cond_fusion else None
            if use_cond_fusion:
                print(
                    "Warning: use_condition_embedding=True in config but no cond_emb in checkpoint, "
                    f"using num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} from config"
                )

        # Determine input_dim: prefer CNN conv weights, then input projection, then config
        if input_dim is None:
            if "cnn_branches.0.0.weight" in state_dict:
                input_dim = state_dict["cnn_branches.0.0.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from UniversalEncoderV3Attention CNN checkpoint")
            elif "input_proj.weight" in state_dict:
                input_dim = state_dict["input_proj.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from UniversalEncoderV3Attention input_proj checkpoint")
            else:
                input_dim = config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")

        d_model = config.get("d_model", 64)
        num_layers = config.get("num_layers", 3)
        n_heads = config.get("n_heads", config.get("nhead", 4))
        dim_feedforward = config.get("dim_feedforward", None)
        dropout = config.get("dropout", 0.1)
        use_ms_cnn = config.get("use_ms_cnn", True)
        kernel_sizes = config.get("kernel_sizes", [3, 5, 9])

        model = UniversalEncoderV3Attention(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_ms_cnn=use_ms_cnn,
            kernel_sizes=kernel_sizes,
            num_conditions=num_conditions,
            condition_embedding_dim=cond_emb_dim if use_cond_fusion else 0,
            max_seq_len=config.get("max_seq_len", 300),
        )
    elif encoder_type == "transformer_encoder_v1":
        # Pure Transformer-Encoder EOL+HI model (EOLFullTransformerEncoder)
        # Infer condition embedding setup from checkpoint or config
        if "condition_embedding.weight" in state_dict:
            num_conditions = state_dict["condition_embedding.weight"].shape[0]
            cond_emb_dim = state_dict["condition_embedding.weight"].shape[1]
            print(
                f"Inferred num_conditions={num_conditions} and cond_emb_dim={cond_emb_dim} "
                f"from EOLFullTransformerEncoder checkpoint"
            )
        else:
            num_conditions = config.get("num_conditions", None)
            cond_emb_dim = config.get("cond_emb_dim", 4) if num_conditions and num_conditions > 1 else 0

        # Determine input_dim from input_proj or config
        if input_dim is None:
            if "input_proj.weight" in state_dict:
                input_dim = state_dict["input_proj.weight"].shape[1]
                print(f"Inferred input_dim={input_dim} from EOLFullTransformerEncoder input_proj checkpoint")
            else:
                input_dim = config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")

        d_model = config.get("d_model", 64)
        num_layers = config.get("num_layers", 3)
        n_heads = config.get("n_heads", config.get("nhead", 4))
        dim_feedforward = config.get("dim_feedforward", None)
        dropout = config.get("dropout", 0.1)

        # NEW: continuous condition encoder flags (ms+DT v2 encoder).
        # These are stored flat in summary.json via **encoder_kwargs.
        use_cond_encoder = bool(config.get("use_cond_encoder", False))
        # cond_in_dim may be None in config (runtime-inferred during training) – treat None as 0 here.
        cond_in_dim_cfg = config.get("cond_in_dim", 0)
        cond_in_dim = int(cond_in_dim_cfg) if cond_in_dim_cfg is not None else 0
        cond_encoder_dim = config.get("cond_encoder_dim", None)
        use_cond_recon_head = bool(config.get("use_cond_recon_head", False))
        
        # Damage head parameters (must match training configuration exactly)
        use_damage_head = bool(config.get("use_damage_head", False))
        damage_L_ref = float(config.get("L_ref", 300.0))
        damage_alpha_base = float(config.get("alpha_base", 0.1))
        damage_hidden_dim = int(config.get("damage_hidden_dim", 64))
        # NEW (v3c): optional MLP-based damage head configuration
        damage_use_mlp = bool(config.get("damage_use_mlp", False))
        damage_mlp_hidden_factor = int(config.get("damage_mlp_hidden_factor", 2))
        damage_mlp_num_layers = int(config.get("damage_mlp_num_layers", 2))
        damage_mlp_dropout = float(config.get("damage_mlp_dropout", 0.1))
        # NEW (v3d): delta cumsum parameters
        damage_use_delta_cumsum = bool(config.get("damage_use_delta_cumsum", False))
        damage_delta_alpha = float(config.get("damage_delta_alpha", 1.0))
        # NEW (v3e): temporal smoothing
        damage_use_temporal_conv = bool(config.get("damage_use_temporal_conv", False))
        damage_temporal_conv_kernel_size = int(config.get("damage_temporal_conv_kernel_size", 3))
        damage_temporal_conv_num_layers = int(config.get("damage_temporal_conv_num_layers", 1))

        # v4/v5: calibrated HI head + v5-specific flags
        # Prefer checkpoint structure when summary is missing these flags (backward compatible)
        keys = list(state_dict.keys())
        has_hi_cal_head = any("hi_cal_head.0.weight" in k for k in keys)
        has_cond_norm = any("condition_normalizer.net.0.weight" in k for k in keys)
        fc_rul_key = next((k for k in keys if k.endswith("fc_rul.weight")), None)
        fc_rul_in = state_dict[fc_rul_key].shape[1] if fc_rul_key is not None else d_model

        use_hi_cal_head = bool(config.get("use_hi_cal_head", has_hi_cal_head))
        use_condition_normalizer = bool(config.get("use_condition_normalizer", has_cond_norm))
        cond_hidden_cfg = config.get("condition_normalizer_hidden_dim")
        if cond_hidden_cfg is None and has_cond_norm and "condition_normalizer.net.0.weight" in state_dict:
            # condition_normalizer.net.0 is the first Linear layer: [hidden_dim, cond_dim]
            cond_hidden_cfg = state_dict["condition_normalizer.net.0.weight"].shape[0]
        condition_normalizer_hidden_dim = int(cond_hidden_cfg or 64)
        # Detect if RUL head was trained with HI_cal fusion by comparing input dim
        use_hi_cal_fusion_for_rul = bool(config.get("use_hi_cal_fusion_for_rul", False))
        if fc_rul_key is not None and fc_rul_in != d_model:
            # Checkpoint fc_rul expects a larger input (typically d_model+1), so
            # enable HI_cal fusion flag to match training-time architecture.
            use_hi_cal_fusion_for_rul = True

        model = EOLFullTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_condition_embedding=bool(num_conditions and num_conditions > 1),
            num_conditions=num_conditions if num_conditions and num_conditions > 1 else 1,
            cond_emb_dim=cond_emb_dim,
            max_seq_len=config.get("max_seq_len", 300),
            # Continuous condition encoder (only active for ms_dt_v2-type runs)
            use_cond_encoder=use_cond_encoder,
            cond_in_dim=cond_in_dim,
            cond_encoder_dim=cond_encoder_dim,
            use_cond_recon_head=use_cond_recon_head,
            # Damage head parameters (must match training exactly)
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
            # v4/v5: calibrated HI + condition normaliser + HI_cal fusion
            use_hi_cal_head=use_hi_cal_head,
            use_condition_normalizer=use_condition_normalizer,
            condition_normalizer_hidden_dim=condition_normalizer_hidden_dim,
            use_hi_cal_fusion_for_rul=use_hi_cal_fusion_for_rul,
        )

        # For v5 runs with a ConditionNormalizer, instantiate it before loading
        # the state dict so that its parameters are present in the module tree.
        if use_condition_normalizer and "condition_normalizer.net.0.weight" in state_dict:
            w0 = state_dict["condition_normalizer.net.0.weight"]  # [hidden_dim, cond_dim]
            w4 = state_dict["condition_normalizer.net.4.weight"]  # [sensor_dim, hidden_dim]
            cond_dim_sd = w0.shape[1]
            sensor_dim_sd = w4.shape[0]
            try:
                model.set_condition_normalizer_dims(cond_dim=cond_dim_sd, sensor_dim=sensor_dim_sd)
            except Exception as e:
                print(f"[WARNING] Failed to initialise ConditionNormalizer from checkpoint shapes: {e}")

        # Optional advanced RUL head (phys_v3/phys_v4 experiments).
        # summary.json stores flat keys describing the RUL head configuration.
        rul_head_type = config.get("rul_head_type", "linear")
        if rul_head_type == "improved":
            from src.models.transformer_eol import ImprovedRULHead

            model.rul_head_type = "improved"

            max_rul_cfg = float(config.get("max_rul", 125.0))
            tau_cfg = float(config.get("rul_beta", 45.0))
            model.max_rul = max_rul_cfg
            model.tau = tau_cfg

            hidden_dim = int(config.get("rul_head_hidden_dim", 128) or 128)
            num_hidden_layers = int(config.get("rul_head_num_layers", 3) or 3)
            head_dropout = float(config.get("rul_head_dropout", dropout) or dropout)
            use_skip = bool(config.get("rul_head_use_skip", True))
            use_hi_fusion = bool(config.get("rul_head_use_hi_fusion", True))
            use_piecewise = bool(config.get("use_piecewise_rul_mapping", True))

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
                max_rul=max_rul_cfg,
                tau=tau_cfg,
            )
        elif rul_head_type == "v4":
            from src.models.transformer_eol import RULHeadV4

            model.rul_head_type = "v4"

            max_rul_cfg = float(config.get("max_rul", 125.0))
            tau_cfg = float(config.get("rul_beta", 45.0))
            model.max_rul = max_rul_cfg
            model.tau = tau_cfg

            model.rul_head = RULHeadV4(
                d_model=d_model,
                max_rul=max_rul_cfg,
            )
    else:
        # LSTM or Transformer (EOLFullLSTMWithHealth)
        if encoder_type == "lstm":
            hidden_dim = config.get("hidden_dim", 50)
            num_layers = config.get("num_layers", 2)
            dropout = config.get("dropout", 0.1)
            bidirectional = config.get("bidirectional", False)
            transformer_nhead = 4  # Not used
            transformer_dim_feedforward = 256  # Not used
        else:  # transformer
            hidden_dim = config.get("d_model", 48)
            num_layers = config.get("num_layers", 3)
            dropout = config.get("dropout", 0.1)
            bidirectional = False
            transformer_nhead = config.get("nhead", 4)
            transformer_dim_feedforward = config.get("dim_feedforward", 256)
        
        # Determine input_dim: use provided value, then try checkpoint, then config
        if input_dim is None:
            # Infer input_dim from checkpoint encoder weights
            # For transformer: encoder.input_proj.weight shape is [d_model, encoder_input_dim]
            # encoder_input_dim = input_dim + cond_emb_dim (if condition embeddings used)
            if encoder_type == "transformer" and "encoder.input_proj.weight" in state_dict:
                encoder_input_dim = state_dict["encoder.input_proj.weight"].shape[1]
                if use_condition_embedding:
                    input_dim = encoder_input_dim - cond_emb_dim
                else:
                    input_dim = encoder_input_dim
                print(f"Inferred input_dim={input_dim} from checkpoint (encoder_input_dim={encoder_input_dim}, cond_emb_dim={cond_emb_dim})")
            elif encoder_type == "lstm":
                # For LSTM, check if condition embeddings are used by checking lstm input size
                if "encoder.lstm.weight_ih_l0" in state_dict:
                    lstm_input_dim = state_dict["encoder.lstm.weight_ih_l0"].shape[1]
                    if use_condition_embedding:
                        input_dim = lstm_input_dim - cond_emb_dim
                    else:
                        input_dim = lstm_input_dim
                    print(f"Inferred input_dim={input_dim} from LSTM checkpoint (lstm_input_dim={lstm_input_dim}, cond_emb_dim={cond_emb_dim})")
                else:
                    # Try config
                    input_dim = config.get("input_dim")
                    if input_dim is None:
                        raise ValueError(
                            "input_dim must be provided or stored in checkpoint/config. "
                            "Please provide input_dim parameter or ensure it's in summary.json."
                        )
                    print(f"Using input_dim={input_dim} from config for LSTM")
            else:
                # Try config
                input_dim = config.get("input_dim")
                if input_dim is None:
                    raise ValueError(
                        "input_dim must be provided or stored in checkpoint/config. "
                        "Please provide input_dim parameter or ensure it's in summary.json."
                    )
                print(f"Using input_dim={input_dim} from config")
        else:
            print(f"Using provided input_dim={input_dim}")
        
        # Create model
        model = EOLFullLSTMWithHealth(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            lambda_health=config.get("health_loss_weight", 0.35),
            use_condition_embedding=use_condition_embedding,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim,
            encoder_type=encoder_type,
            transformer_nhead=transformer_nhead,
            transformer_dim_feedforward=transformer_dim_feedforward,
        )
    
    # Load weights (state_dict already extracted above)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, print diagnostic info
        print(f"Error loading state dict: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    return model, config


def rebuild_scaler_from_training_data(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    use_condition_wise_scaling: bool = None,
) -> StandardScaler | Dict[int, StandardScaler]:
    """
    Rebuild the feature scaler exactly as in the training pipeline.
    
    This function mirrors the logic from create_full_dataloaders in src/eol_full_lstm.py.
    
    For condition-wise scaling:
    - Group by ConditionID and fit one StandardScaler per condition.
    For global scaling:
    - Fit a single StandardScaler over all training rows.
    
    Args:
        df_train: Training DataFrame with features and ConditionID column
        feature_cols: List of feature column names to scale
        use_condition_wise_scaling: If True, use condition-wise scaling (dict of scalers).
                                    If None, uses config default (USE_CONDITION_WISE_SCALING).
    
    Returns:
        scaler: StandardScaler (global) or Dict[int, StandardScaler] (condition-wise)
    """
    if use_condition_wise_scaling is None:
        use_condition_wise_scaling = USE_CONDITION_WISE_SCALING
    
    # Extract feature matrix from training data
    X_train = df_train[feature_cols].values  # [N, num_features]
    
    if use_condition_wise_scaling and "ConditionID" in df_train.columns:
        # Condition-wise scaling: separate scaler per condition
        scalers: Dict[int, StandardScaler] = {}
        unique_conds = sorted(df_train["ConditionID"].unique())
        
        for cond_id in unique_conds:
            df_cond = df_train[df_train["ConditionID"] == cond_id]
            if len(df_cond) == 0:
                continue
            
            X_cond = df_cond[feature_cols].values  # [N_cond, num_features]
            
            scaler_cond = StandardScaler()
            scaler_cond.fit(X_cond)
            scalers[int(cond_id)] = scaler_cond
        
        if not scalers:
            # Fallback: if no conditions found, use global scaler
            print("[WARNING] No conditions found for condition-wise scaling, falling back to global scaler")
            scaler = StandardScaler()
            scaler.fit(X_train)
            return scaler
        
        print(f"[INFO] Rebuilt condition-wise scaler with {len(scalers)} conditions: {sorted(scalers.keys())}")
        return scalers
    else:
        # Global scaling: single scaler for all data
        scaler = StandardScaler()
        scaler.fit(X_train)
        print(f"[INFO] Rebuilt global scaler on {len(X_train)} training samples")
        return scaler


def run_inference_for_experiment(
    experiment_dir: Path,
    split: str = "test",
    device: torch.device | str = "cpu",
    return_hi_trajectories: bool = True,
) -> Tuple[List[EngineEOLMetrics], Dict[int, EngineTrajectory]]:
    """
    Load model and run inference on specified split.
    
    Args:
        experiment_dir: Path to experiment directory
        split: "test" or "val"
        device: Device to run inference on
        return_hi_trajectories: If True, return full HI trajectories per engine
    
    Returns:
        eol_metrics: List of EOL metrics per engine
        trajectories: Dict mapping unit_id -> EngineTrajectory (if return_hi_trajectories=True)
    """
    # Load config first to get dataset name
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {experiment_dir}")
    
    with open(summary_path, "r") as f:
        config = json.load(f)
    
    dataset_name = config.get("dataset", "FD004")
    
    print(f"Running inference on {dataset_name} {split} set...")
    
    # Load data FIRST to derive input_dim and num_conditions
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset_name,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )
    
    # Feature engineering (match training; enable residuals ONLY for dedicated
    # residual experiments, and use phys_features for condition vector + twin).
    from src.config import ResidualFeatureConfig
    exp_name_lower = experiment_dir.name.lower()

    # Residuals in the sense of PhysicsFeatureConfig (phase4-style residual
    # experiments) are controlled by experiment name / explicit flag – this
    # must match run_experiments.py.
    use_residuals = bool(
        config.get("use_residuals", False)
        or ("residual" in exp_name_lower)
        or ("resid" in exp_name_lower)
    )

    phys_features_cfg = config.get("phys_features", {})

    residual_cfg = ResidualFeatureConfig(
        enabled=use_residuals,
        mode=phys_features_cfg.get("mode", "per_engine"),
        baseline_len=phys_features_cfg.get("twin_baseline_len", 30),
        include_original=phys_features_cfg.get("include_original", True),
    )
    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=bool(phys_features_cfg.get("use_extended", False)),
        use_residuals=use_residuals,
        use_temporal_on_physics=False,
        residual=residual_cfg,
    )

    # Optional high-level feature configuration (kept in sync with run_experiments.py)
    features_cfg = config.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    use_temporal_features = features_cfg.get("use_multiscale_features", True)

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

    # ------------------------------------------------------------------
    # Reproduce the ms+DT feature pipeline from run_experiments.py:
    #   1) Physics core features
    #   2) Continuous condition vector (Cond_*)
    #   3) Digital Twin + residuals
    #   4) Temporal / multi-scale features
    # ------------------------------------------------------------------

    # Optional physics-informed condition/twin features
    phys_opts = phys_features_cfg
    use_phys_condition_vec = phys_opts.get("use_condition_vector", False)
    use_twin_features = phys_opts.get(
        "use_twin_features",
        phys_opts.get("use_digital_twin_residuals", False),
    )
    twin_baseline_len = phys_opts.get("twin_baseline_len", 30)
    condition_vector_version = phys_opts.get("condition_vector_version", 2)

    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Continuous condition vector
    if use_phys_condition_vec:
        from src.additional_features import build_condition_features

        print("  Using continuous condition vector features (Cond_*) [inference]")
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

    # 3) Digital twin + residuals (HealthyTwinRegressor)
    if use_twin_features:
        from src.additional_features import create_twin_features

        print(f"  Using HealthyTwinRegressor (baseline_len={twin_baseline_len}) [inference]")
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) Temporal / multi-scale features
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
    
    feature_cols = [
        c for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    feature_cols, _ = remove_rul_leakage(feature_cols)
    print(f"Feature columns after engineering: {len(feature_cols)} (residuals={'on' if use_residuals else 'off'})")
    
    # Get sequence parameters from config
    past_len = config.get("sequence_length", 30)
    max_rul = config.get("max_rul", 125)
    
    # Determine which data to use
    if split == "test":
        df_split = df_test
    else:  # val
        # For validation, we need to reconstruct from training data
        # This is more complex - we'd need to know the train/val split
        # For now, we'll use test data and note this limitation
        print("Warning: Validation split inference not fully implemented, using test data")
        df_split = df_test
    
    # Build sequences FIRST to derive input_dim and num_conditions
    X_split, unit_ids_split, cond_ids_split = build_test_sequences_from_df(
        df_split,
        feature_cols=feature_cols,
        past_len=past_len,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
    )
    
    # Create true_rul_dict based on the order of unit_ids_split
    # CRITICAL: y_test_true from load_cmapps_subset is in the same order as
    # the engines appear in the RUL file. The RUL file contains RUL values
    # in the order that engines first appear in the test data file.
    # build_test_sequences_from_df uses df_test[unit_col].unique() which returns
    # units in the order of first appearance, so the order should match.
    # However, to be safe, we use the actual order from unit_ids_split after
    # build_test_sequences_from_df has been called.
    if split == "test":
        # Get unique unit IDs in the same order as they appear in df_test
        # This should match the order used by build_test_sequences_from_df
        unit_ids_test_ordered = df_test["UnitNumber"].unique()
        
        # Verify that lengths match
        if len(unit_ids_test_ordered) != len(y_test_true):
            raise ValueError(
                f"Mismatch: {len(unit_ids_test_ordered)} unique units in test data, "
                f"but {len(y_test_true)} RUL values in RUL file. "
                f"This suggests a data loading issue."
            )
        
        # IMPORTANT: The RUL file contains RUL values in the order that engines
        # first appear in the test data file. df_test["UnitNumber"].unique() returns
        # units in the order of first appearance, so y_test_true[i] should correspond
        # to unit_ids_test_ordered[i]. However, we need to verify this matches
        # the order in unit_ids_split (which comes from build_test_sequences_from_df).
        # Create mapping: unit_id -> true_rul (based on order in y_test_true)
        true_rul_dict = dict(zip(unit_ids_test_ordered, y_test_true))
        
        # Verify that unit_ids_split matches unit_ids_test_ordered
        # (they should be in the same order)
        unit_ids_split_array = unit_ids_split.numpy() if hasattr(unit_ids_split, 'numpy') else np.array(unit_ids_split)
        if len(unit_ids_split_array) != len(unit_ids_test_ordered):
            raise ValueError(
                f"Mismatch: {len(unit_ids_split_array)} units in unit_ids_split, "
                f"but {len(unit_ids_test_ordered)} unique units in test data."
            )
        
        # Check if the order matches (first few should be the same)
        if not np.array_equal(unit_ids_split_array, unit_ids_test_ordered):
            # Order doesn't match - we need to reorder y_test_true to match unit_ids_split
            # Create a mapping from unit_id to its index in unit_ids_test_ordered
            unit_to_index = {uid: idx for idx, uid in enumerate(unit_ids_test_ordered)}
            # Reorder y_test_true to match unit_ids_split order
            y_test_true_reordered = np.array([
                y_test_true[unit_to_index[uid]] for uid in unit_ids_split_array
            ])
            # Update true_rul_dict with reordered values
            true_rul_dict = dict(zip(unit_ids_split_array, y_test_true_reordered))
            print(f"[WARNING] Reordered y_test_true to match unit_ids_split order")
    else:
        # For validation, we don't have true RUL - would need to reconstruct
        # For now, use test data
        unit_ids_test_ordered = df_test["UnitNumber"].unique()
        if len(unit_ids_test_ordered) != len(y_test_true):
            raise ValueError(
                f"Mismatch: {len(unit_ids_test_ordered)} unique units in test data, "
                f"but {len(y_test_true)} RUL values in RUL file."
            )
        true_rul_dict = dict(zip(unit_ids_test_ordered, y_test_true))
    
    # Derive input_dim and num_conditions from actual data
    input_dim = X_split.shape[-1]  # [N, T, F] -> F
    if cond_ids_split is not None and len(cond_ids_split) > 0:
        num_conditions = int(cond_ids_split.max().item()) + 1
    else:
        # Check if ConditionID column exists in dataframe
        if "ConditionID" in df_split.columns:
            num_conditions = int(df_split["ConditionID"].max()) + 1
        else:
            num_conditions = 1
    
    print(f"Derived input_dim={input_dim} and num_conditions={num_conditions} from {split} data")
    
    # Load scaler if available, or rebuild from training data
    # CRITICAL: Apply scaling ONCE, before model loading
    # The model expects once-scaled inputs (as during training)
    # Double-scaling would cause incorrect predictions and wild metric values
    scaler_path = experiment_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print(f"[WARNING] scaler.pkl not found at {scaler_path}")
        print("[INFO] Rebuilding scaler from df_train to match training pipeline...")
        # Rebuild scaler using the same logic as training
        scaler = rebuild_scaler_from_training_data(df_train, feature_cols)
        # Optionally save for future runs
        try:
            import pickle
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"[INFO] Saved rebuilt scaler to {scaler_path}")
        except Exception as e:
            print(f"[WARNING] Could not save rebuilt scaler: {e}")

    # Hard safety check: ensure scaler vs. pipeline features are consistent
    # BEFORE we apply any scaling or load the model. This catches the classic
    # 295 vs. 244 feature mismatch early with a clear error message.
    check_feature_dimensions(
        feature_cols=feature_cols,
        scaler=scaler,
        model=None,
        context="inference",
    )

    # Apply scaling ONCE (before model loading)
    # This ensures the model receives inputs in the same scale as during training
    if scaler is not None:
        # Apply scaling based on scaler type (dict for condition-wise or single scaler)
        if isinstance(scaler, dict):
            # Condition-wise scaling
            X_split_scaled_list = []
            for i in range(len(X_split)):
                cond_id = int(cond_ids_split[i].item()) if cond_ids_split is not None else 0
                x_sample = X_split[i].numpy()  # [past_len, num_features]
                if cond_id in scaler:
                    x_scaled = scaler[cond_id].transform(x_sample)
                else:
                    # Fallback: use first available scaler
                    if scaler:
                        first_scaler = list(scaler.values())[0]
                        x_scaled = first_scaler.transform(x_sample)
                    else:
                        x_scaled = x_sample
                X_split_scaled_list.append(torch.from_numpy(x_scaled))
            X_split = torch.stack(X_split_scaled_list)
        else:
            # Global scaling
            X_split_flat = X_split.numpy().reshape(-1, X_split.shape[-1])
            X_split_scaled_flat = scaler.transform(X_split_flat)
            X_split = torch.from_numpy(X_split_scaled_flat.reshape(X_split.shape))
        print("Applied feature scaling")
    else:
        # This should never happen if scaler was rebuilt, but guard against it
        raise RuntimeError(
            "Scaler could not be loaded or rebuilt; diagnostics cannot safely proceed. "
            "Please ensure scaler.pkl is present or fix scaler rebuilding."
        )
    
    # NOW load model with correct dimensions
    # Note: X_split is already scaled above (once, matching training pipeline)
    model, config = load_model_from_experiment(
        experiment_dir,
        device,
        input_dim=input_dim,
        num_conditions=num_conditions,
    )

    # Second safety check: ensure model.input_dim (if present) matches the
    # feature dimensionality used by the inference pipeline.
    check_feature_dimensions(
        feature_cols=feature_cols,
        scaler=scaler,
        model=model,
        context="inference",
    )
    
    # Run inference
    # X_split is already scaled (once) and is a torch.Tensor on CPU
    # Move to device before inference
    
    # Ensure we respect the model's use_condition_embedding flag
    # Check strictly against the model instance loaded
    use_cond_emb = getattr(model, "use_condition_embedding", False)
    
    # Also check if model.encoder has it (common in WorldModelV3/V2)
    if not use_cond_emb and hasattr(model, "encoder"):
         use_cond_emb = getattr(model.encoder, "use_condition_embedding", False) or \
                        getattr(model.encoder, "use_condition_fusion", False)
    
    # If model uses condition embeddings, we MUST provide them if available
    # If cond_ids_split is None but model needs them, we have a problem (handled in forward)
    if use_cond_emb:
         cond_ids_tensor = cond_ids_split.to(device) if cond_ids_split is not None else None
         if cond_ids_tensor is None:
             print("Warning: Model expects condition embeddings (use_condition_embedding=True) but no cond_ids available in data. Inference may fail.")
    else:
         cond_ids_tensor = None
    
    with torch.no_grad():
        # Move X_split to device (already scaled once above)
        X_split = X_split.to(device)
        
        # Prefer the unified helper for RUL decoding (same as evaluate_on_test_data)
        # This ensures consistent RUL decoding across evaluation and diagnostics
        # Always pass cond_ids_tensor if available (or None) - forward_rul_only will handle it
        rul_pred = forward_rul_only(model, X_split, cond_ids=cond_ids_tensor)
        
        # If we want HI trajectories AND forward_rul_only does not provide HI,
        # we can do an additional model(X_split, cond_ids=...) call, but
        # NEVER use that output as raw RUL in cycles without matching decoding.
        if return_hi_trajectories and hasattr(model, "forward"):
            # Get full model output (RUL, HI, HI_seq) for HI trajectories
            if cond_ids_tensor is not None:
                outputs = model(X_split, cond_ids=cond_ids_tensor)
            else:
                 # Try calling without cond_ids (might fail if model strictly needs them)
                 # But if use_cond_emb is True and cond_ids_tensor is None, we likely failed earlier
                 try:
                     outputs = model(X_split)
                 except TypeError:
                      # If model requires cond_ids but we don't have them
                      # This is a fallback, but likely we should have had them
                      outputs = model(X_split, cond_ids=None) # Let model raise the specific error
            
            # In run_inference_for_experiment, if returning 3 items, it expects:
            # rul_pred, hi_last, hi_seq
            # WorldModelUniversalV3 returns a dict: {"traj", "eol", "hi"}
            # We need to handle both cases
            
            hi_last = None
            hi_seq = None
            
            if isinstance(outputs, dict):
                # V3 output format
                if "hi" in outputs:
                    hi_last = outputs["hi"]  # [B, 1]
                if "traj" in outputs:
                    hi_seq = outputs["traj"] # [B, H, 1]
            elif isinstance(outputs, (tuple, list)):
                if len(outputs) == 3:
                    _, hi_last, hi_seq = outputs
                elif len(outputs) == 2:
                     _, hi_last = outputs
            
            # Post-process hi_seq if present
            if hi_seq is not None and torch.is_tensor(hi_seq):
                 hi_seq = hi_seq.squeeze(-1) if hi_seq.dim() == 3 else hi_seq
        else:
            hi_last = None
            hi_seq = None

        # Optional: HI_cal_v2 sequence for Transformer encoder v4
        hi_cal_seq = None
        if (
            return_hi_trajectories
            and isinstance(model, EOLFullTransformerEncoder)
            and getattr(model, "use_hi_cal_head", False)
        ):
            try:
                enc_seq, _ = model.encode(
                    X_split,
                    cond_ids=cond_ids_tensor,
                    return_seq=True,
                )
                hi_cal_seq = model.predict_hi_cal_seq(enc_seq)  # [B, T]
            except Exception as e:  # pragma: no cover - defensive logging
                print(f"[WARNING] Failed to compute HI_cal_v2 sequence for diagnostics: {e}")
                hi_cal_seq = None
    
    rul_pred = rul_pred.cpu().numpy().flatten()
    
    # Apply RUL capping (same as evaluate_on_test_data)
    # Get max_rul from config (default 125)
    max_rul = config.get("max_rul", 125)
    rul_pred = np.minimum(rul_pred, max_rul)
    rul_pred = np.maximum(rul_pred, 0.0)  # Ensure non-negative
    
    # Collect per-engine metrics
    eol_metrics = []
    trajectories = {}
    
    # Debug: Print first few mappings to verify consistency
    # This helps verify that true_rul and pred_rul are correctly aligned and capped
    if len(unit_ids_split) > 0:
        first_5_unit_ids = unit_ids_split[:5].numpy()
        first_5_true_rul_raw = [true_rul_dict.get(int(uid), 0.0) for uid in first_5_unit_ids]
        first_5_true_rul_capped = [min(tr, max_rul) if max_rul is not None else tr for tr in first_5_true_rul_raw]
        print(f"[DEBUG] First 5 unit_ids from build_test_sequences: {first_5_unit_ids}")
        print(f"[DEBUG] First 5 true_rul values (capped at {max_rul}): {first_5_true_rul_capped}")
        print(f"[DEBUG] First 5 pred_rul values (after scaling + capping): {rul_pred[:5]}")
    
    # Quick sanity check: compare mean true vs mean pred RUL
    # This catches scaling mismatches early
    true_vals = np.array([true_rul_dict.get(int(uid), 0.0) for uid in unit_ids_split.numpy()])
    true_vals_capped = np.minimum(true_vals, max_rul) if max_rul is not None else true_vals
    mean_true = float(true_vals_capped.mean())
    mean_pred = float(rul_pred.mean())
    
    print(f"[DEBUG] mean_true_rul={mean_true:.2f}, mean_pred_rul={mean_pred:.2f}")
    
    if mean_true > 30 and mean_pred < 5:
        raise RuntimeError(
            "RUL predictions are clustered near zero while true RUL is large. "
            "This strongly suggests a scaling mismatch. "
            "Check that feature scaling and RUL decoding match the training pipeline. "
            f"mean_true_rul={mean_true:.2f}, mean_pred_rul={mean_pred:.2f}"
        )
    
    # Build EOL metrics per engine
    # EOL metrics and NASA are computed on per-engine, last-cycle values
    # using the same RUL capping as evaluate_on_test_data
    for i, unit_id in enumerate(unit_ids_split.numpy()):
        unit_id = int(unit_id)
        true_rul_raw = true_rul_dict.get(unit_id, 0.0)
        # Apply RUL capping to true_rul (same as evaluate_on_test_data)
        # This ensures consistency: both true and pred RUL are capped at max_rul
        true_rul = min(true_rul_raw, max_rul) if max_rul is not None else true_rul_raw
        pred_rul = float(rul_pred[i])  # Already capped to [0, max_rul] above
        
        # Error convention: error = pred - true (same as evaluate_on_test_data)
        error = pred_rul - true_rul
        
        # Compute NASA contribution for this engine (EOL only)
        # Use the same function as evaluate_on_test_data
        # This calculates NASA score for a single EOL prediction (one value per engine)
        # Both true_rul and pred_rul are capped, matching evaluate_on_test_data
        nasa_contribution = nasa_phm_score_single(true_rul, pred_rul)
        
        eol_metrics.append(EngineEOLMetrics(
            unit_id=unit_id,
            true_rul=true_rul,
            pred_rul=pred_rul,
            error=error,
            nasa=nasa_contribution,
        ))
        
        # Build trajectory if requested
        if return_hi_trajectories:
            # Get engine data
            df_engine = df_split[df_split["UnitNumber"] == unit_id].sort_values("TimeInCycles")
            cycles = df_engine["TimeInCycles"].values
            
            if len(cycles) == 0:
                continue
            
            # For HI trajectory, we'll use the HI sequence from the last window
            # This gives us HI values for the last `past_len` cycles
            if hi_seq is not None:
                # hi_seq[i] is the HI sequence for this engine's last window [past_len]
                hi_traj_window = hi_seq[i].cpu().numpy() if torch.is_tensor(hi_seq) else hi_seq[i]
                # Map to the last past_len cycles
                if len(cycles) >= past_len:
                    # Use HI for the last past_len cycles
                    hi_traj = np.full(len(cycles), np.nan)
                    
                    # Robust assignment handling length mismatches (e.g. V3 horizon vs past_len)
                    target_len = past_len
                    source_len = len(hi_traj_window)
                    copy_len = min(target_len, source_len)
                    
                    # Assign to the end of the trajectory
                    if copy_len > 0:
                        hi_traj[-copy_len:] = hi_traj_window[:copy_len]
                    
                    # Forward fill for earlier cycles (simplified)
                    for j in range(len(cycles) - past_len - 1, -1, -1):
                        hi_traj[j] = hi_traj[j + 1] if not np.isnan(hi_traj[j + 1]) else 1.0
                else:
                    # Engine shorter than past_len
                    hi_traj = np.full(len(cycles), np.nan)
                    hi_traj[-len(hi_traj_window):] = hi_traj_window[-len(cycles):]
                    # Forward fill
                    for j in range(len(cycles) - len(hi_traj_window) - 1, -1, -1):
                        hi_traj[j] = hi_traj[j + 1] if not np.isnan(hi_traj[j + 1]) else 1.0
            elif hi_last is not None:
                # Only have last HI value - use it for all cycles (simplified)
                hi_val = hi_last[i].cpu().numpy() if torch.is_tensor(hi_last) else hi_last[i]
                hi_traj = np.full(len(cycles), float(hi_val))
            else:
                # No HI available - use placeholder
                hi_traj = np.full(len(cycles), 0.5)

            # HI_cal_v2 trajectory (if available)
            hi_cal_traj: Optional[np.ndarray]
            if hi_cal_seq is not None:
                hi_cal_window = hi_cal_seq[i].cpu().numpy() if torch.is_tensor(hi_cal_seq) else hi_cal_seq[i]
                if len(cycles) >= past_len:
                    hi_cal_traj = np.full(len(cycles), np.nan)
                    target_len = past_len
                    source_len = len(hi_cal_window)
                    copy_len = min(target_len, source_len)
                    if copy_len > 0:
                        hi_cal_traj[-copy_len:] = hi_cal_window[:copy_len]
                    for j in range(len(cycles) - past_len - 1, -1, -1):
                        hi_cal_traj[j] = hi_cal_traj[j + 1] if not np.isnan(hi_cal_traj[j + 1]) else 1.0
                else:
                    hi_cal_traj = np.full(len(cycles), np.nan)
                    hi_cal_traj[-len(hi_cal_window):] = hi_cal_window[-len(cycles):]
                    for j in range(len(cycles) - len(hi_cal_window) - 1, -1, -1):
                        hi_cal_traj[j] = hi_cal_traj[j + 1] if not np.isnan(hi_cal_traj[j + 1]) else 1.0
            else:
                hi_cal_traj = None
            
            # True RUL trajectory (decreasing from max_rul to 0)
            # For test data, we know the true RUL at EOL
            # We'll construct a linear trajectory assuming constant degradation rate
            if len(cycles) > 0:
                # Assume RUL decreases linearly from max_rul to true_rul
                # At cycle 0: RUL = max_rul (or estimated from cycles)
                # At last cycle: RUL = true_rul
                cycle_range = cycles.max() - cycles.min()
                if cycle_range > 0:
                    true_rul_traj = np.maximum(0, max_rul - (max_rul - true_rul) * (cycles - cycles.min()) / cycle_range)
                else:
                    true_rul_traj = np.full(len(cycles), true_rul)
            else:
                true_rul_traj = np.array([true_rul])
            
            # Predicted RUL (constant for now - EOL prediction)
            pred_rul_traj = np.full(len(cycles), pred_rul) if len(cycles) > 0 else np.array([pred_rul])
            
            trajectories[unit_id] = EngineTrajectory(
                unit_id=unit_id,
                cycles=cycles,
                hi=hi_traj,
                true_rul=true_rul_traj,
                pred_rul=pred_rul_traj,
                hi_cal=hi_cal_traj,
            )
    
    return eol_metrics, trajectories


def select_representative_engines(
    eol_metrics: List[EngineEOLMetrics],
    num: int = 5,
) -> List[int]:
    """
    Select representative engines for visualization.
    
    Strategy:
    - Worst NASA (max nasa)
    - Median NASA
    - Best NASA (min nasa)
    - Plus 2 random engines
    
    Args:
        eol_metrics: List of EOL metrics
        num: Number of engines to select
    
    Returns:
        List of unit_ids
    """
    if len(eol_metrics) < num:
        return [m.unit_id for m in eol_metrics]
    
    # Sort by NASA score
    sorted_metrics = sorted(eol_metrics, key=lambda m: m.nasa, reverse=True)
    
    selected = []
    
    # Worst NASA
    if len(sorted_metrics) > 0:
        selected.append(sorted_metrics[0].unit_id)
    
    # Median NASA
    if len(sorted_metrics) > 1:
        median_idx = len(sorted_metrics) // 2
        selected.append(sorted_metrics[median_idx].unit_id)
    
    # Best NASA
    if len(sorted_metrics) > 2:
        selected.append(sorted_metrics[-1].unit_id)
    
    # Random engines (excluding already selected)
    remaining = [m.unit_id for m in sorted_metrics if m.unit_id not in selected]
    if len(remaining) >= (num - len(selected)):
        import random
        random.seed(42)  # For reproducibility
        selected.extend(random.sample(remaining, num - len(selected)))
    else:
        selected.extend(remaining)
    
    return selected[:num]

