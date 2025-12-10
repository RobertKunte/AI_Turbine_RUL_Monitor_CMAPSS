try:
    import torch  # type: ignore[import]
    from typing import Optional, Tuple, Union  # type: ignore[import]
    import numpy as np  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the custom loss. Please install torch."
    ) from exc

# Import config for damage head check
try:
    from src.config import USE_DAMAGE_HEALTH_HEAD
except ImportError:
    # Fallback if config not available
    USE_DAMAGE_HEALTH_HEAD = False


# ===================================================================
# Refactored Health Index Loss Functions
# ===================================================================

def compute_health_loss(
    pred_hi: torch.Tensor,
    target_hi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MSE loss between predicted and target Health Index.
    
    Args:
        pred_hi: (batch,) or (batch, 1) predicted HI in [0, 1]
        target_hi: (batch,) or (batch, 1) target HI in [0, 1]
        mask: Optional (batch,) mask to apply (1 = include, 0 = exclude)
    
    Returns:
        Scalar MSE loss
    """
    pred_hi = pred_hi.view(-1).float()
    target_hi = target_hi.view(-1).float()
    
    error = (pred_hi - target_hi) ** 2
    
    if mask is not None:
        mask = mask.view(-1).float()
        loss = (error * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = error.mean()
    
    return loss


def compute_eol_health_loss(
    pred_hi: torch.Tensor,
    rul: torch.Tensor,
    eol_threshold: float,
    weight: float,
) -> torch.Tensor:
    """
    Compute EOL-specific health penalty: enforce HI→0 for small RUL.
    
    Args:
        pred_hi: (batch,) or (batch, 1) predicted HI in [0, 1]
        rul: (batch,) or (batch, 1) true RUL values
        eol_threshold: RUL threshold for EOL-tail zone (e.g., 25 cycles)
        weight: Weight multiplier for this penalty
    
    Returns:
        Scalar EOL health loss
    """
    pred_hi = pred_hi.view(-1).float()
    rul = rul.view(-1).float()
    
    # EOL mask: 1 in tail zone (RUL <= threshold), 0 otherwise
    eol_mask = (rul <= eol_threshold).float()
    
    # Health should be close to 0 in the EOL tail
    # Penalty = (pred_hi * eol_mask)^2 (target = 0)
    eol_health_error = (pred_hi * eol_mask) ** 2
    eol_health_loss = eol_health_error.mean() * weight
    
    return eol_health_loss


def health_smoothness_loss(
    health_seq: torch.Tensor,
    rul_seq: Optional[torch.Tensor] = None,
    smooth_weight: float = 0.0,
    plateau_rul_threshold: float = 80.0,
) -> torch.Tensor:
    """
    Phase 2: Compute smoothness loss for Health Index sequences.
    
    Penalizes large changes in HI between consecutive time steps,
    especially in early-life (high RUL) region.
    
    Args:
        health_seq: (batch, seq_len) or (batch, seq_len, 1) predicted HI over sequence
        rul_seq: Optional (batch, seq_len) true RUL for each time step (for masking)
        smooth_weight: Weight multiplier for smoothness loss (0.0 to disable)
        plateau_rul_threshold: RUL threshold above which smoothness is enforced (early-life)
    
    Returns:
        Scalar smoothness loss (0.0 if smooth_weight <= 0)
    """
    if smooth_weight <= 0:
        return torch.tensor(0.0, device=health_seq.device, requires_grad=True)
    
    # Ensure 2D: (batch, seq_len)
    if health_seq.dim() == 3:
        health_seq = health_seq.squeeze(-1)  # (batch, seq_len, 1) -> (batch, seq_len)
    elif health_seq.dim() == 1:
        health_seq = health_seq.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
    
    # Compute differences: HI_t - HI_{t-1}
    # diff: (batch, seq_len-1)
    diff = health_seq[:, 1:] - health_seq[:, :-1]
    
    # Smoothness = squared differences
    smooth = diff.pow(2)  # (batch, seq_len-1)
    
    # Optional: Mask to apply smoothness only in early-life (high RUL) region
    if rul_seq is not None:
        # Ensure 2D: (batch, seq_len)
        if rul_seq.dim() == 3:
            rul_seq = rul_seq.squeeze(-1)
        elif rul_seq.dim() == 1:
            rul_seq = rul_seq.unsqueeze(0)
        
        # Mask: 1 for RUL > threshold (early-life), 0 otherwise
        # Apply mask to rul_seq[:, :-1] (aligned with diff)
        mask = (rul_seq[:, :-1] > plateau_rul_threshold).float()  # (batch, seq_len-1)
        
        # Only penalize smoothness in masked region
        if mask.sum() > 0:
            smooth = smooth * mask
            # Normalize by number of masked elements
            loss = smooth.sum() / (mask.sum() + 1e-8)
        else:
            # No early-life samples, return zero loss
            loss = torch.tensor(0.0, device=health_seq.device, requires_grad=True)
    else:
        # No masking: apply smoothness to all time steps
        loss = smooth.mean()
    
    return smooth_weight * loss


def compute_monotonicity_loss(
    pred_hi: torch.Tensor,
    rul: torch.Tensor,
    beta: float = 30.0,
    weight: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalize HI increases over time in the late-RUL region.
    
    Simple, unsmoothed version that only considers pairs where both timesteps
    are in the late-RUL region (RUL <= beta).
    
    Args:
        pred_hi: (batch, seq_len) predicted HI in [0, 1] over sequence (temporal order: early -> late)
        rul: (batch, seq_len) true RUL for each time step (capped at max_rul)
        beta: RUL threshold for late region (e.g., 30 cycles)
        weight: Weight multiplier for this penalty (applied outside, but kept for API compatibility)
    
    Returns:
        Tuple of (raw_mono_loss, raw_mono_loss)
        - raw_mono_loss: mean of positive diffs in late region pairs (before weighting)
        - Note: weight is applied outside this function
    """
    # Ensure shapes: (batch, seq_len)
    if pred_hi.dim() == 3:
        pred_hi = pred_hi.squeeze(-1)
    if rul.dim() == 3:
        rul = rul.squeeze(-1)
    
    # Rename for clarity
    health_seq = pred_hi
    rul_seq = rul
    
    # mask: only consider late phase, where true RUL <= beta
    late_mask = (rul_seq <= beta)  # [B, T]
    
    # if nothing in late phase, return zeros
    if not late_mask.any():
        zero = health_seq.new_tensor(0.0)
        return zero, zero
    
    # Compute forward differences
    diffs = health_seq[:, 1:] - health_seq[:, :-1]  # [B, T-1]
    
    # build a mask for pairs where both times are late
    late_pairs = late_mask[:, 1:] & late_mask[:, :-1]
    
    # only positive diffs (HI increases) in late region
    viol = torch.relu(diffs) * late_pairs.float()
    
    if viol.numel() == 0 or not late_pairs.any():
        raw_mono = health_seq.new_tensor(0.0)
    else:
        # Mean violation over valid late pairs
        raw_mono = viol.sum() / late_pairs.float().sum()
    
    # loss = raw_mono (the weight is applied outside)
    return raw_mono, raw_mono


def debug_monotonicity_loss(
    health_seq: torch.Tensor,
    rul_seq: torch.Tensor,
    beta: float = 30.0,
    max_print: int = 5,
) -> None:
    """
    Debug helper to inspect monotonicity violations for HI.
    
    This function uses the same logic as compute_monotonicity_loss to verify
    that the monotonicity loss is working correctly.
    
    Args:
        health_seq: [B, T] predicted health index in temporal order
        rul_seq: [B, T] corresponding RUL targets
        beta: RUL threshold for late region
        max_print: Maximum number of example sequences to print
    """
    with torch.no_grad():
        # Ensure shapes: (batch, seq_len)
        if health_seq.dim() == 3:
            health_seq = health_seq.squeeze(-1)
        if rul_seq.dim() == 3:
            rul_seq = rul_seq.squeeze(-1)
        
        # 1) Basic stats
        print(f"[DEBUG] health_seq shape={tuple(health_seq.shape)}, "
              f"rul_seq shape={tuple(rul_seq.shape)}")
        
        # 2) Compute simple forward diffs over time
        diffs = health_seq[:, 1:] - health_seq[:, :-1]  # [B, T-1]
        violations = diffs[diffs > 0]
        num_viol = violations.numel()
        max_viol = violations.max().item() if num_viol > 0 else 0.0
        mean_viol = violations.mean().item() if num_viol > 0 else 0.0
        
        print(f"[DEBUG] num_violations={num_viol}, "
              f"max_violation={max_viol:.6f}, mean_violation={mean_viol:.6f}")
        
        # 3) Run the *same* logic as compute_monotonicity_loss
        raw_mono, mono_loss = compute_monotonicity_loss(
            pred_hi=health_seq,
            rul=rul_seq,
            beta=beta,
            weight=1.0,  # Use weight=1.0 to get raw loss
        )
        print(f"[DEBUG] raw_mono={raw_mono.item():.6f}, "
              f"mono_loss={mono_loss.item():.6f}")
        
        # 4) Optionally print a few example time series where violations occur
        if num_viol > 0:
            # Boolean mask of sequences that have at least one violation
            has_viol = (diffs > 0).any(dim=1)
            idx = torch.nonzero(has_viol, as_tuple=False).flatten()
            idx = idx[:max_print]
            
            for i in idx:
                hi_i = health_seq[i].cpu().numpy()
                rul_i = rul_seq[i].cpu().numpy()
                print(f"[DEBUG] Example seq idx={int(i)}")
                print("  HI :", np.round(hi_i, 4))
                print("  RUL:", np.round(rul_i, 1))
        
        print("=" * 80)

def multitask_rul_health_loss(
    rul_pred: torch.Tensor,
    rul_true: torch.Tensor,
    health_pred: torch.Tensor,
    max_rul: float = 125.0,
    tau: float = 40.0,
    lambda_health: float = 0.3,
    plateau_thresh: float = 80.0,
    hi_eol_thresh: float = 25.0,
    hi_eol_weight: float = 4.0,
    health_seq: Optional[torch.Tensor] = None,
    rul_seq: Optional[torch.Tensor] = None,
    hi_mono_weight: float = 0.0,
    hi_mono_rul_beta: float = 30.0,
    return_components: bool = False,
    rul_traj_weight: float = 1.0,  # NEW: Weight for RUL MSE part
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Multi-task loss that combines:
    - weighted RUL MSE (with higher weight near EOL, i.e. small RUL)
    - Health Index MSE with plateau+tail target: HI=1 for RUL > plateau_thresh, 
      then linear decay to 0 at EOL
    - EOL health penalty: extra weight to enforce HI≈0 at EOL (small RUL)
    - Optional monotonicity regularizer: penalizes increases of HI along sequence
    
    The RUL loss weighting emphasizes small RUL (near EOL), which is more
    relevant for maintenance decisions. Health Index is learned as a monotone
    proxy from RUL during training, but at inference time depends only on
    sensor features.
    
    Args:
        rul_pred: (batch,) or (batch, 1) predicted RUL (clamped to [0, max_rul])
        rul_true: (batch,) or (batch, 1) true RUL (already clamped to [0, max_rul])
        health_pred: (batch,) or (batch, 1) predicted HI in [0,1] at final time step
        max_rul: maximum RUL used for clamping (typically 125)
        tau: scale parameter for the exponential RUL weighting
        lambda_health: trade-off between RUL loss and health loss
        plateau_thresh: RUL threshold above which HI_target=1 (plateau region)
        hi_eol_thresh: RUL threshold for EOL-tail zone (e.g., 20-30 cycles)
        hi_eol_weight: extra weight for EOL health penalty (HI→0 near failure)
        health_seq: Optional (batch, seq_len) or (batch, seq_len, 1) predicted HI 
                   over full sequence (for monotonicity regularizer)
        rul_seq: Optional (batch, seq_len) or (batch, seq_len, 1) true RUL for each 
                time step (for RUL-weighted monotonicity)
        hi_mono_weight: weight of monotonicity penalty (0.0 to disable)
        hi_mono_rul_beta: scale parameter (in cycles) for RUL weighting in monotonicity.
                         Smaller beta -> stronger emphasis on late cycles.
        return_components: if True, return (loss, components_dict) instead of just loss
        rul_traj_weight: multiplier for the RUL MSE loss term
    
    Returns:
        scalar loss tensor, or (loss, components_dict) if return_components=True
    """
    # Ensure shapes: flatten to (batch,)
    rul_pred = rul_pred.view(-1).float()
    rul_true = rul_true.view(-1).float()
    health_pred = health_pred.view(-1).float()
    
    # Clamp RUL predictions to valid range
    rul_pred = torch.clamp(rul_pred, min=0.0, max=max_rul)
    
    # Import RUL loss scale from config
    try:
        from src.config import RUL_LOSS_SCALE
    except ImportError:
        RUL_LOSS_SCALE = 1.0 / 125.0  # Default fallback
    
    # 1) Weighted RUL MSE:
    #    weights = exp(-rul_true / tau)
    #    Higher weight for small RUL (near EOL)
    weights = torch.exp(-rul_true / tau)
    rul_error = (rul_pred - rul_true) ** 2
    rul_loss_unscaled = (weights * rul_error).mean()
    
    # Apply RUL loss scaling to balance with HI losses
    rul_loss = rul_loss_unscaled * RUL_LOSS_SCALE * rul_traj_weight
    
    # 2) Health Index target: plateau at 1 for RUL > plateau_thresh, 
    #    then linear decay from plateau_thresh → 0
    #    HI_target = 1 for RUL > plateau_thresh
    #    HI_target = RUL / plateau_thresh for 0 <= RUL <= plateau_thresh
    HI_target = torch.where(
        rul_true > plateau_thresh,
        torch.ones_like(rul_true),
        (rul_true / plateau_thresh).clamp(0.0, 1.0),
    )
    
    # Use refactored health loss function
    health_loss = compute_health_loss(health_pred, HI_target)
    
    # 3) EOL-specific health penalty: enforce HI→0 for RUL in tail zone
    #    Use refactored EOL health loss function
    eol_health_loss = compute_eol_health_loss(
        pred_hi=health_pred,
        rul=rul_true,
        eol_threshold=hi_eol_thresh,
        weight=hi_eol_weight,
    )
    
    # 4) Optional monotonicity regularizer: penalize increases of HI along sequence
    #    Focus on late-RUL region where monotonicity is most critical
    #    NOTE: If using damage-based health head, HI is already monotonic by construction,
    #          so monotonicity loss is not needed and should be disabled.
    mono_loss_raw = torch.tensor(0.0, device=rul_pred.device)
    mono_loss = torch.tensor(0.0, device=rul_pred.device)
    
    if (not USE_DAMAGE_HEALTH_HEAD) and health_seq is not None and hi_mono_weight > 0.0:
        # Ensure shapes
        if health_seq.dim() == 3:
            health_seq_flat = health_seq.squeeze(-1)
        else:
            health_seq_flat = health_seq
        
        if rul_seq is not None:
            # Use refactored monotonicity loss (focuses on late-RUL region)
            mono_loss_raw, _ = compute_monotonicity_loss(
                pred_hi=health_seq_flat,
                rul=rul_seq.squeeze(-1) if rul_seq.dim() == 3 else rul_seq,
                beta=hi_mono_rul_beta,
                weight=hi_mono_weight,  # kept for API compatibility, but not used
            )
            # Apply weight outside (function returns raw values)
            mono_loss = mono_loss_raw * hi_mono_weight
        else:
            # Fallback: compute without RUL sequence (use dummy RUL = 0 for all steps)
            # This will focus on all time steps
            batch_size, seq_len = health_seq_flat.shape
            dummy_rul = torch.zeros(batch_size, seq_len, device=health_seq_flat.device)
            mono_loss_raw, _ = compute_monotonicity_loss(
                pred_hi=health_seq_flat,
                rul=dummy_rul,
                beta=hi_mono_rul_beta,
                weight=hi_mono_weight,  # kept for API compatibility, but not used
            )
            # Apply weight outside (function returns raw values)
            mono_loss = mono_loss_raw * hi_mono_weight
    
    # 5) Total loss composition
    loss = rul_loss + lambda_health * (health_loss + eol_health_loss) + mono_loss
    
    if return_components:
        components = {
            "rul_loss_unscaled": rul_loss_unscaled.item(),
            "rul_loss": rul_loss.item(),
            "health_loss": health_loss.item(),
            "eol_health_loss": eol_health_loss.item(),
            "mono_loss_raw": mono_loss_raw.item() if isinstance(mono_loss_raw, torch.Tensor) else mono_loss_raw,
            "mono_loss": mono_loss.item() if isinstance(mono_loss, torch.Tensor) else mono_loss,
            "total_loss": loss.item(),
        }
        return loss, components
    
    return loss


def monotonic_health_loss(
    pred_hi: torch.Tensor,
    target_hi: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Combined Health Index sequence loss:
    - Base MSE to target HI
    - Monotonicity penalty (penalize HI increases over time)
    - Smoothness penalty (second-order differences to reduce jitter)

    Assumes physics-informed Health Index semantics:
        HI ≈ 1.0  => healthy (large RUL)
        HI ≈ 0.0  => failed / near EOL

    Args:
        pred_hi: (batch, T) or (batch, T, 1) predicted HI sequence in [0, 1]
        target_hi: (batch, T) or (batch, T, 1) target HI sequence in [0, 1]
        alpha: Weight for monotonicity penalty
        beta: Weight for smoothness penalty

    Returns:
        Scalar loss = mse + alpha * mono_loss + beta * smoothness_loss
    """
    # Ensure 2D: (B, T)
    if pred_hi.dim() == 3:
        pred_hi = pred_hi.squeeze(-1)
    if target_hi.dim() == 3:
        target_hi = target_hi.squeeze(-1)
    if pred_hi.dim() == 1:
        pred_hi = pred_hi.unsqueeze(0)
    if target_hi.dim() == 1:
        target_hi = target_hi.unsqueeze(0)

    # Base MSE between predicted and target HI
    mse = ((pred_hi - target_hi) ** 2).mean()

    # First-order differences over time
    if pred_hi.size(1) > 1:
        diff = pred_hi[:, 1:] - pred_hi[:, :-1]  # (B, T-1)
        # Monotonicity: penalize *increases* in HI (ReLU of positive slopes)
        mono_loss = torch.relu(diff).mean()

        # Second-order differences for smoothness
        if diff.size(1) > 1:
            diff2 = diff[:, 1:] - diff[:, :-1]  # (B, T-2)
            smoothness_loss = (diff2 ** 2).mean()
        else:
            smoothness_loss = pred_hi.new_tensor(0.0)
    else:
        # Sequence too short for temporal penalties
        mono_loss = pred_hi.new_tensor(0.0)
        smoothness_loss = pred_hi.new_tensor(0.0)

    total = mse + alpha * mono_loss + beta * smoothness_loss

    if return_components:
        # Return total plus raw components (without alpha/beta)
        return total, mse, mono_loss, smoothness_loss

    return total


def rul_asymmetric_weighted_loss(pred, target,
                                 over_factor=2.0,
                                 min_rul_weight=1.0,
                                 max_rul_weight=0.3):
    """
    Custom loss for RUL:
    - Overestimation (pred > target) is penalized stronger than underestimation.
    - Low RUL values are weighted higher than large RUL values.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    error = pred - target
    over  = torch.clamp(error, min=0.0)      # overestimation
    under = torch.clamp(-error, min=0.0)     # underestimation
    
    # Asymmetric penalty (like MSE but scaled for overestimation)
    base_loss = over_factor * over**2 + under**2
    
    # RUL-based weights: higher weight for small RUL
    t_norm = target / (target.max() + 1e-6)  # normalize to [0,1]
    # Map to [max_rul_weight, min_rul_weight]
    weights = max_rul_weight + (min_rul_weight - max_rul_weight) * (1.0 - t_norm)
    
    weighted_loss = weights * base_loss
    return weighted_loss.mean()