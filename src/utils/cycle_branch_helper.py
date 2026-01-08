"""
Cycle Branch Training Helper.

This module provides helper functions for integrating the cycle branch
into the training loop, encapsulating initialization, forward pass,
loss computation, and logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

# Avoid circular import - CycleBranchConfig is used for type hints only
if TYPE_CHECKING:
    from src.world_model_training import CycleBranchConfig

from src.models.physics.nominal_head import NominalHead
from src.models.physics.cycle_layer_mvp import CycleLayerMVP
from src.models.heads.param_head_theta6 import ParamHeadTheta6
from src.losses.cycle_losses import (
    compute_cycle_loss,
    compute_theta_smooth_loss,
    compute_theta_mono_loss,
    CycleBranchLoss,
)
from src.utils.sensor_mapping import resolve_cycle_target_cols



@dataclass
class CycleBranchComponents:
    """Container for cycle branch model components."""
    nominal_head: NominalHead
    param_head: ParamHeadTheta6
    cycle_layer: CycleLayerMVP
    loss_fn: CycleBranchLoss
    target_col_map: Dict[str, str]
    target_indices: List[int]
    ops_indices: List[int]


def initialize_cycle_branch(
    cfg: CycleBranchConfig,
    z_dim: int,
    feature_cols: List[str],
    num_conditions: int,
    device: torch.device,
    scaler_dict: Optional[Dict[int, Any]] = None,
) -> Optional[CycleBranchComponents]:
    """Initialize cycle branch components.
    
    Args:
        cfg: CycleBranchConfig with all settings
        z_dim: Dimension of encoder latent z_t
        feature_cols: List of feature column names
        num_conditions: Number of operating conditions
        device: PyTorch device
        scaler_dict: Dict mapping condition_id -> fitted StandardScaler
            (used for condition-wise normalization in loss)
        
    Returns:
        CycleBranchComponents or None if not enabled
    """
    if not cfg.enable:
        return None
    
    print("\n[CycleBranch] Initializing cycle branch components...")
    
    # 1. Resolve target columns
    try:
        target_col_map = resolve_cycle_target_cols(feature_cols, cfg.targets)
        print(f"  Resolved targets: {target_col_map}")
    except ValueError as e:
        print(f"  ⚠️  Could not resolve cycle targets: {e}")
        print(f"  Disabling cycle branch.")
        return None
    
    # 2. Find target indices in feature_cols
    target_indices = []
    for target_name in cfg.targets:
        col_name = target_col_map[target_name]
        if col_name in feature_cols:
            target_indices.append(feature_cols.index(col_name))
        else:
            print(f"  ⚠️  Target column {col_name} not in feature_cols")
            return None
    print(f"  Target indices: {target_indices}")
    
    # 3. Find operating settings indices
    ops_patterns = ["Setting1", "Setting2", "Setting3", "setting_1", "setting_2", "setting_3"]
    ops_indices = []
    for pat in ops_patterns:
        if pat in feature_cols:
            ops_indices.append(feature_cols.index(pat))
    
    # Try lowercase versions
    if len(ops_indices) < 3:
        for i, col in enumerate(feature_cols):
            if col.lower() in ["setting1", "setting2", "setting3"]:
                if i not in ops_indices:
                    ops_indices.append(i)
    
    if len(ops_indices) < 3:
        print(f"  ⚠️  Could not find 3 operating settings. Found indices: {ops_indices}")
        print(f"  Disabling cycle branch.")
        return None
    ops_indices = ops_indices[:3]  # Take first 3
    print(f"  Operating settings indices: {ops_indices}")
    
    # 4. Initialize NominalHead
    head_type = cfg.nominal_head_type
    if head_type == "table" and num_conditions < 1:
        print(f"  NominalHead: table mode requires num_conditions >= 1, falling back to MLP")
        head_type = "mlp"
    
    nominal_head = NominalHead(
        head_type=head_type,
        num_conditions=num_conditions if head_type == "table" else None,
        ops_dim=3,
        hidden_dim=cfg.nominal_head_hidden,
        eta_bounds=cfg.eta_nom_bounds,
        output_dp_nom=False,  # MVP: dp_nom is always constant
    ).to(device)
    print(f"  NominalHead: type={head_type}, num_conditions={num_conditions}")
    
    # 5. Initialize ParamHeadTheta6
    param_head = ParamHeadTheta6(
        z_dim=z_dim,
        hidden_dim=cfg.param_head_hidden,
        m_bounds_eta=cfg.m_bounds_eta,
        m_bounds_dp=cfg.m_bounds_dp,
        num_layers=cfg.param_head_num_layers,
    ).to(device)
    print(f"  ParamHeadTheta6: z_dim={z_dim}, hidden={cfg.param_head_hidden}")
    
    # 6. Initialize CycleLayerMVP (outputs RAW thermodynamic units)
    cycle_layer = CycleLayerMVP(
        n_targets=len(cfg.targets),
        num_conditions=num_conditions,
        pr_mode=cfg.pr_mode,
        pr_head_hidden=cfg.pr_head_hidden,
        dp_nom_constant=cfg.dp_nom_constant,
    ).to(device)
    print(f"  CycleLayerMVP: n_targets={len(cfg.targets)}, pr_mode={cfg.pr_mode}")
    
    # 7. Extract cycle target scaler stats for condition-wise normalization
    cycle_target_mean = None
    cycle_target_std = None
    
    try:
        from src.utils.cycle_target_scaler import extract_cycle_target_stats
        
        if scaler_dict is not None and len(scaler_dict) > 0:
            cycle_target_mean, cycle_target_std = extract_cycle_target_stats(
                scaler_dict=scaler_dict,
                feature_cols=feature_cols,
                target_indices=target_indices,
                num_conditions=num_conditions,
            )
            cycle_target_mean = cycle_target_mean.to(device)
            cycle_target_std = cycle_target_std.to(device)
            print(f"  Extracted cycle target stats: mean/std shape=({cycle_target_mean.shape})")
            print(f"    Example cond=0 mean: {[f'{cycle_target_mean[0, i].item():.1f}' for i in range(min(4, cycle_target_mean.shape[-1]))]}")
            print(f"    Example cond=0 std:  {[f'{cycle_target_std[0, i].item():.1f}' for i in range(min(4, cycle_target_std.shape[-1]))]}")
    except Exception as e:
        print(f"  ⚠️  Could not extract cycle target stats: {e}")
        print(f"  Will use un-normalized loss (may cause scale mismatch)")
    
    # 8. Initialize loss function with scaler stats
    loss_fn = CycleBranchLoss(
        lambda_cycle=cfg.lambda_cycle,
        lambda_smooth=cfg.lambda_theta_smooth,
        lambda_mono=cfg.lambda_theta_mono,
        cycle_loss_type=cfg.cycle_loss_type,
        cycle_huber_beta=cfg.cycle_huber_beta,
        mono_on_eta=cfg.mono_on_eta_mods,
        mono_on_dp=cfg.mono_on_dp_mod,
        mono_eps=cfg.mono_eps,
        lambda_power_balance=cfg.lambda_power_balance,
        lambda_theta_prior=getattr(cfg, 'lambda_theta_prior', 0.05),  # Default 0.05 for anti-saturation
        target_names=cfg.targets,
        cycle_target_mean=cycle_target_mean,
        cycle_target_std=cycle_target_std,
    )
    print(f"  CycleBranchLoss: λ_cycle={cfg.lambda_cycle}, λ_smooth={cfg.lambda_theta_smooth}, λ_prior={getattr(cfg, 'lambda_theta_prior', 0.05)}")
    
    total_params = sum(
        p.numel() for p in 
        list(nominal_head.parameters()) + 
        list(param_head.parameters()) + 
        list(cycle_layer.parameters())
    )
    print(f"  Total cycle branch parameters: {total_params:,}")
    
    return CycleBranchComponents(
        nominal_head=nominal_head,
        param_head=param_head,
        cycle_layer=cycle_layer,
        loss_fn=loss_fn,
        target_col_map=target_col_map,
        target_indices=target_indices,
        ops_indices=ops_indices,
    )



def cycle_branch_forward(
    components: CycleBranchComponents,
    X_batch: torch.Tensor,
    z_t: torch.Tensor,
    cond_ids: Optional[torch.Tensor],
    cfg: CycleBranchConfig,
    epoch: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute cycle branch forward pass.
    
    Args:
        components: CycleBranchComponents
        X_batch: Input features (B, T, D) or (B, D)
        z_t: Encoder latent (B, T, z_dim) or (B, z_dim)
        cond_ids: Condition IDs (B,)
        cfg: CycleBranchConfig
        epoch: Current epoch (for warmup logic)
        
    Returns:
        cycle_pred: Predicted sensor values
        cycle_target: Target sensor values
        m_t: Degradation modifiers
        eta_nom: Nominal efficiencies
        intermediates: Dict with additional info
    """
    B = X_batch.shape[0]
    is_seq = X_batch.dim() == 3
    
    # Extract operating settings
    if is_seq:
        ops_t = X_batch[:, :, components.ops_indices]  # (B, T, 3)
        cycle_target = X_batch[:, :, components.target_indices]  # (B, T, n_targets)
    else:
        ops_t = X_batch[:, components.ops_indices]  # (B, 3)
        cycle_target = X_batch[:, components.target_indices]  # (B, n_targets)
    
    # Get nominal efficiencies
    if components.nominal_head.head_type == "table":
        eta_nom = components.nominal_head(ops_t=ops_t, cond_ids=cond_ids)
    else:
        eta_nom = components.nominal_head(ops_t=ops_t)
    
    # Get degradation modifiers
    m_t = components.param_head(z_t)
    
    
    # Expand m_t if sequence mode
    if is_seq and m_t.dim() == 2:
        T = ops_t.shape[1]
        m_t = m_t.unsqueeze(1).expand(-1, T, -1)

    # Warmup: freeze m=1 during warmup epochs
    if cfg.deg_warmup_epochs > 0 and epoch < cfg.deg_warmup_epochs:
        m_t = torch.ones_like(m_t)
    
    # Compute cycle predictions
    cycle_pred, layer_inter = components.cycle_layer(
        ops_t=ops_t,
        m_t=m_t,
        eta_nom=eta_nom,
        cond_ids=cond_ids,
        return_intermediates=True,
    )
    
    intermediates = {
        "ops_t": ops_t,
        "eta_nom": eta_nom,
        "m_t": m_t,
    }
    
    # Debug: print scale stats once at epoch 0 (now shows raw AND scaled)
    if epoch == 0 and not hasattr(components, '_target_debug_printed'):
        with torch.no_grad():
            print("\n[CycleBranch] Target Scale Debug (epoch 0):")
            
            # Get scaler stats from loss_fn if available
            has_scaler = (
                hasattr(components.loss_fn, 'cycle_target_mean') and 
                components.loss_fn.cycle_target_mean is not None and
                cond_ids is not None
            )
            
            # Global stats: target (scaled), pred (raw)
            print(f"  target_scaled: mean={cycle_target.mean():.4f}, std={cycle_target.std():.4f}")
            print(f"  pred_raw:      mean={cycle_pred.mean():.2f}, std={cycle_pred.std():.2f}")
            
            # Compute pred_scaled if scaler available
            if has_scaler:
                mean_buf = components.loss_fn.cycle_target_mean  # (C, 4)
                std_buf = components.loss_fn.cycle_target_std    # (C, 4)
                cond_mean = mean_buf[cond_ids]  # (B, 4)
                cond_std = std_buf[cond_ids]    # (B, 4)
                
                if cycle_pred.dim() == 3:  # (B, T, 4)
                    cond_mean = cond_mean.unsqueeze(1)  # (B, 1, 4)
                    cond_std = cond_std.unsqueeze(1)
                
                pred_scaled = (cycle_pred - cond_mean) / (cond_std + 1e-6)
                print(f"  pred_scaled:   mean={pred_scaled.mean():.4f}, std={pred_scaled.std():.4f}")

                # ===== P1 FIX: Additional diagnostics =====
                # Check condition distribution
                unique_conds, cond_counts = torch.unique(cond_ids, return_counts=True)
                print(f"  Condition distribution: {dict(zip(unique_conds.tolist(), cond_counts.tolist()))}")

                # Check scaler stats for this batch's conditions
                batch_mean = mean_buf[cond_ids].mean(dim=0)  # (4,) average across batch
                batch_std = std_buf[cond_ids].mean(dim=0)
                print(f"  Scaler stats (batch avg): mean={batch_mean.tolist()}, std={batch_std.tolist()}")

                # Residual stats
                residual = pred_scaled - cycle_target
                print(f"  Residual (pred-target): mean={residual.mean():.4f}, std={residual.std():.4f}")

                # Range checks
                print(f"  pred_raw range: [{cycle_pred.min():.1f}, {cycle_pred.max():.1f}]")
                print(f"  pred_scaled range: [{pred_scaled.min():.3f}, {pred_scaled.max():.3f}]")
                print(f"  target_scaled range: [{cycle_target.min():.3f}, {cycle_target.max():.3f}]")
                # ===== End P1 fix =====
            else:
                print(f"  pred_scaled:   N/A (no scaler stats available)")

            # Per-sensor stats
            target_names = ["T24", "T30", "P30", "T50"]
            for i, name in enumerate(target_names[:cycle_target.shape[-1]]):
                print(f"  {name}:")
                print(f"    target_scaled: mean={cycle_target[..., i].mean():.4f}, std={cycle_target[..., i].std():.4f}")
                print(f"    pred_raw:      mean={cycle_pred[..., i].mean():.2f}, std={cycle_pred[..., i].std():.2f}")
                if has_scaler:
                    print(f"    pred_scaled:   mean={pred_scaled[..., i].mean():.4f}, std={pred_scaled[..., i].std():.4f}")
        components._target_debug_printed = True
    intermediates.update(layer_inter)
    
    return cycle_pred, cycle_target, m_t, eta_nom, intermediates


def cycle_branch_loss(
    components: CycleBranchComponents,
    cycle_pred: torch.Tensor,
    cycle_target: torch.Tensor,
    m_t: torch.Tensor,
    cfg: CycleBranchConfig,
    epoch: int,
    num_epochs: int,
    mask: Optional[torch.Tensor] = None,
    intermediates: Optional[dict] = None,
    cond_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute cycle branch losses.
    
    Args:
        components: CycleBranchComponents
        cycle_pred: Predicted sensor values (RAW thermodynamic units)
        cycle_target: Target sensor values (RAW from X_batch)
        m_t: Degradation modifiers
        cfg: CycleBranchConfig
        epoch: Current epoch
        num_epochs: Total epochs
        mask: Optional valid mask
        intermediates: Optional dict with cycle layer intermediates
        cond_ids: Condition IDs (B,) for condition-wise normalization
        
    Returns:
        total_loss: Scalar loss
        metrics: Dict with loss components
    """
    # Curriculum: ramp up lambda_cycle over cfg.cycle_ramp_epochs
    if cfg.cycle_ramp_epochs > 0:
        epoch_frac = min(1.0, (epoch + 1) / cfg.cycle_ramp_epochs)
    else:
        epoch_frac = 1.0
    
    # Expand m_t to sequence if needed for smoothness loss
    if m_t.dim() == 2:
        # No sequence dimension, can't compute smoothness
        theta_seq = m_t.unsqueeze(1)  # (B, 1, 6)
    else:
        theta_seq = m_t  # (B, T, 6)
    
    total_loss, metrics = components.loss_fn(
        cycle_pred=cycle_pred,
        cycle_target=cycle_target,
        theta_seq=theta_seq,
        mask=mask,
        epoch_frac=epoch_frac,
        intermediates=intermediates,
        cond_ids=cond_ids,
        epoch=epoch,
    )
    
    return total_loss, metrics


def get_cycle_branch_parameters(
    components: CycleBranchComponents,
) -> List[nn.Parameter]:
    """Get all trainable parameters from cycle branch components."""
    params = []
    params.extend(components.nominal_head.parameters())
    params.extend(components.param_head.parameters())
    params.extend(components.cycle_layer.parameters())
    return list(params)


def log_cycle_branch_metrics(
    epoch: int,
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
) -> None:
    """Append cycle branch metrics to history."""
    for key, value in metrics.items():
        history_key = f"cycle_{key}"
        if history_key not in history:
            history[history_key] = []
        history[history_key].append(value)
