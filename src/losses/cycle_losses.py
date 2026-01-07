"""
Cycle Branch Loss Functions.

This module provides loss functions for the differentiable cycle branch:
- Cycle reconstruction loss (sensor prediction)
- Theta smoothness loss (temporal smoothness of degradation modifiers)
- Theta monotonicity loss (optional soft penalty for increasing modifiers)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_power_balance_penalty(
    work_balance: dict,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute power balance penalty (HPT->HPC, LPT->Fan+LPC).
    
    Args:
        work_balance: Dict with W_fan, W_lpc, W_hpc, W_hpt, W_lpt
        mask: Optional mask (B, T)
        
    Returns:
        Scalar penalty
    """
    keys = ["W_hpc", "W_hpt", "W_fan", "W_lpc", "W_lpt"]
    for k in keys:
        if k not in work_balance:
            return torch.tensor(0.0)

    W_hpc = work_balance["W_hpc"]
    W_hpt = work_balance["W_hpt"]
    W_fan_lpc = work_balance["W_fan"] + work_balance["W_lpc"]
    W_lpt = work_balance["W_lpt"]
    
    # Relative error penalty (squared)
    eps = 1e-6
    penalty_hpt = ((W_hpt - W_hpc) / (W_hpc.abs() + eps)) ** 2
    penalty_lpt = ((W_lpt - W_fan_lpc) / (W_fan_lpc.abs() + eps)) ** 2
    
    total_penalty = penalty_hpt + penalty_lpt
    
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        total_penalty = total_penalty * mask
        valid_count = mask.sum().clamp(min=1.0)
        return total_penalty.sum() / valid_count
    
    return total_penalty.mean()


def compute_theta_prior_loss(
    m_t: torch.Tensor,
    upper_bound: float = 0.995,
    lower_bound: float = 0.90,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute anti-saturation prior for theta (degradation modifiers).
    
    Penalizes m(t) values that approach the upper bound (prevents m=1 collapse)
    and optionally pushes away from lower bound.
    
    L_prior = mean(ReLU(m_t - upper_bound)) + 0.1 * mean(ReLU(lower_bound - m_t))
    
    Args:
        m_t: Degradation modifiers (B, T, 6) or (B, 6)
        upper_bound: Soft upper limit (default 0.995)
        lower_bound: Soft lower limit (default 0.90)
        mask: Optional mask (B, T) or (B, T, 1)
        
    Returns:
        Scalar prior loss
    """
    # Upper bound penalty: ReLU(m_t - upper_bound)
    upper_penalty = F.relu(m_t - upper_bound)
    
    # Lower bound penalty (weaker): 0.1 * ReLU(lower_bound - m_t)
    lower_penalty = 0.1 * F.relu(lower_bound - m_t)
    
    total_penalty = upper_penalty + lower_penalty
    
    # Average over theta dimensions
    total_penalty = total_penalty.mean(dim=-1)  # (B, T) or (B,)
    
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # (B, T)
        total_penalty = total_penalty * mask
        valid_count = mask.sum().clamp(min=1.0)
        return total_penalty.sum() / valid_count
    
    return total_penalty.mean()


def compute_cycle_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "huber"] = "huber",
    huber_beta: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cycle reconstruction loss.
    
    Args:
        pred: Predicted sensor values (B, T, n_targets) or (B, n_targets)
        target: Target sensor values (same shape as pred)
        loss_type: 'mse' or 'huber'
        huber_beta: Beta parameter for Huber loss (threshold for L1/L2 transition)
        mask: Optional mask (B, T) or (B,) for valid timesteps
        
    Returns:
        Scalar loss value
    """
    if loss_type == "huber":
        loss = F.huber_loss(pred, target, reduction="none", delta=huber_beta)
    else:
        loss = F.mse_loss(pred, target, reduction="none")
    
    # Average over targets dimension
    loss = loss.mean(dim=-1)  # (B, T) or (B,)
    
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # (B, T)
        # Apply mask
        loss = loss * mask
        valid_count = mask.sum().clamp(min=1.0)
        return loss.sum() / valid_count
    
    return loss.mean()


def compute_cycle_loss_per_sensor(
    pred: torch.Tensor,
    target: torch.Tensor,
    sensor_names: list[str],
    loss_type: Literal["mse", "huber"] = "huber",
    huber_beta: float = 0.1,
) -> dict[str, float]:
    """Compute per-sensor cycle loss for diagnostics.
    
    Args:
        pred: (B, T, n_targets)
        target: (B, T, n_targets)
        sensor_names: List of sensor names matching target dimension
        loss_type: 'mse' or 'huber'
        huber_beta: Huber beta
        
    Returns:
        Dict mapping sensor name to loss value
    """
    result = {}
    for i, name in enumerate(sensor_names):
        p = pred[..., i]
        t = target[..., i]
        if loss_type == "huber":
            loss = F.huber_loss(p, t, reduction="mean", delta=huber_beta)
        else:
            loss = F.mse_loss(p, t, reduction="mean")
        result[name] = float(loss.item())
    return result


def compute_theta_smooth_loss(
    theta_seq: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute temporal smoothness loss for degradation modifiers.
    
    L_smooth = mean ||m_t - m_{t-1}||²
    
    Encourages slow-varying degradation trajectories.
    
    Args:
        theta_seq: Degradation modifiers (B, T, 6)
        mask: Optional valid timestep mask (B, T)
        
    Returns:
        Scalar smoothness loss
    """
    if theta_seq.dim() != 3:
        # No temporal dimension, return zero
        return torch.tensor(0.0, device=theta_seq.device)
    
    B, T, D = theta_seq.shape
    if T < 2:
        return torch.tensor(0.0, device=theta_seq.device)
    
    # Compute temporal differences: m_t - m_{t-1}
    diff = theta_seq[:, 1:, :] - theta_seq[:, :-1, :]  # (B, T-1, 6)
    
    # Squared L2 norm per timestep
    diff_sq = (diff ** 2).sum(dim=-1)  # (B, T-1)
    
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        # Mask for valid transitions: both t and t-1 must be valid
        mask_transitions = mask[:, 1:] * mask[:, :-1]  # (B, T-1)
        diff_sq = diff_sq * mask_transitions
        valid_count = mask_transitions.sum().clamp(min=1.0)
        return diff_sq.sum() / valid_count
    
    return diff_sq.mean()


def compute_theta_mono_loss(
    theta_seq: torch.Tensor,
    mono_on_eta: bool = False,
    mono_on_dp: bool = False,
    eps: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute monotonicity loss for degradation modifiers.
    
    Soft penalty for increases in degradation modifiers (should slowly decrease).
    L_mono = mean ReLU((m_t - m_{t-1}) - eps)
    
    Args:
        theta_seq: Degradation modifiers (B, T, 6)
        mono_on_eta: Apply monotonicity to eta modifiers (indices 0-4)
        mono_on_dp: Apply monotonicity to dp modifier (index 5)
        eps: Small tolerance before penalizing increases
        mask: Optional valid timestep mask (B, T)
        
    Returns:
        Scalar monotonicity loss
    """
    if not mono_on_eta and not mono_on_dp:
        return torch.tensor(0.0, device=theta_seq.device)
    
    if theta_seq.dim() != 3:
        return torch.tensor(0.0, device=theta_seq.device)
    
    B, T, D = theta_seq.shape
    if T < 2:
        return torch.tensor(0.0, device=theta_seq.device)
    
    # Build mask for which parameters to apply monotonicity
    param_mask = torch.zeros(6, device=theta_seq.device)
    if mono_on_eta:
        param_mask[:5] = 1.0
    if mono_on_dp:
        param_mask[5] = 1.0
    
    # Compute increases: m_t - m_{t-1}
    diff = theta_seq[:, 1:, :] - theta_seq[:, :-1, :]  # (B, T-1, 6)
    
    # Penalize increases beyond eps
    violations = F.relu(diff - eps)  # (B, T-1, 6)
    
    # Apply parameter mask
    violations = violations * param_mask.unsqueeze(0).unsqueeze(0)  # (B, T-1, 6)
    
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask_transitions = mask[:, 1:] * mask[:, :-1]  # (B, T-1)
        # Unsqueeze to broadcast with violations (B, T-1, 6)
        violations = violations * mask_transitions.unsqueeze(-1)
        valid_count = mask_transitions.sum().clamp(min=1.0)
        return violations.sum() / valid_count
    
    return violations.mean()


def compute_variance_attribution(
    eta_nom: torch.Tensor,
    m_t: torch.Tensor,
    eta_eff: Optional[torch.Tensor] = None,
) -> dict[str, dict[str, float]]:
    """Compute variance attribution between nominal and degradation components.
    
    For each efficiency component, computes:
    - Var(eta_nom): ops-driven variation
    - Var(m): degradation-driven variation
    - var_share_deg = Var(m) / (Var(m) + Var(eta_nom))
    
    Args:
        eta_nom: Nominal efficiencies (B, T, 5) or flattened
        m_t: Degradation modifiers (B, T, 6) — only first 5 used
        eta_eff: Optional effective efficiencies (B, T, 5)
        
    Returns:
        Dict with variance stats per component
    """
    # Flatten to (N, *)
    eta_nom_flat = eta_nom.reshape(-1, eta_nom.shape[-1])  # (N, 5)
    m_flat = m_t.reshape(-1, m_t.shape[-1])[:, :5]  # (N, 5)
    
    comp_names = ["fan", "lpc", "hpc", "hpt", "lpt"]
    result = {}
    
    for i, name in enumerate(comp_names):
        var_nom = float(eta_nom_flat[:, i].var().item())
        var_m = float(m_flat[:, i].var().item())
        
        total_var = var_nom + var_m
        var_share_deg = var_m / total_var if total_var > 1e-10 else 0.0
        
        result[name] = {
            "var_eta_nom": var_nom,
            "var_m": var_m,
            "var_share_deg": var_share_deg,
        }
        
        if eta_eff is not None:
            eta_eff_flat = eta_eff.reshape(-1, eta_eff.shape[-1])
            result[name]["var_eta_eff"] = float(eta_eff_flat[:, i].var().item())
    
    # Overall warning flags
    low_deg_usage = all(v["var_share_deg"] < 0.05 for v in result.values())
    result["_warnings"] = {
        "deg_usage_warning": low_deg_usage,
    }
    
    return result


class CycleBranchLoss(torch.nn.Module):
    """Combined cycle branch loss module.
    
    Combines:
    - L_cycle: Sensor reconstruction loss
    - L_smooth: Temporal smoothness of m(t)
    - L_mono: Optional monotonicity penalty
    
    Args:
        lambda_cycle: Weight for cycle reconstruction loss
        lambda_smooth: Weight for smoothness loss
        lambda_mono: Weight for monotonicity loss
        cycle_loss_type: 'mse' or 'huber'
        cycle_huber_beta: Beta for Huber loss
        mono_on_eta: Apply mono to eta modifiers
        mono_on_dp: Apply mono to dp modifier
        mono_eps: Epsilon for mono penalty
        cycle_target_mean: Optional (num_conditions, 4) mean for condition-wise normalization
        cycle_target_std: Optional (num_conditions, 4) std for condition-wise normalization
    """
    
    def __init__(
        self,
        lambda_cycle: float = 0.1,
        lambda_smooth: float = 0.05,
        lambda_mono: float = 0.0,
        cycle_loss_type: Literal["mse", "huber"] = "huber",
        cycle_huber_beta: float = 0.1,
        mono_on_eta: bool = False,
        mono_on_dp: bool = False,
        mono_eps: float = 1e-4,
        lambda_power_balance: float = 0.0,
        lambda_theta_prior: float = 0.001,
        target_names: Optional[list[str]] = None,
        cycle_target_mean: Optional[torch.Tensor] = None,
        cycle_target_std: Optional[torch.Tensor] = None,
        cycle_target_space: str = "scaled",  # "scaled" or "raw"
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono
        self.lambda_power_balance = lambda_power_balance
        self.lambda_theta_prior = lambda_theta_prior  # Anti-saturation
        self.cycle_loss_type = cycle_loss_type
        self.cycle_huber_beta = cycle_huber_beta
        self.mono_on_eta = mono_on_eta
        self.mono_on_dp = mono_on_dp
        self.mono_eps = mono_eps
        self.target_names = target_names or ["T24", "T30", "P30", "T50"]
        self.cycle_target_space = cycle_target_space
        
        # Condition-wise normalization buffers (NOT learnable)
        if cycle_target_mean is not None:
            self.register_buffer("cycle_target_mean", cycle_target_mean)
        else:
            self.register_buffer("cycle_target_mean", None)
            
        if cycle_target_std is not None:
            self.register_buffer("cycle_target_std", cycle_target_std)
        else:
            self.register_buffer("cycle_target_std", None)
        
        # Debug flag
        self._debug_printed = False
    
    def forward(
        self,
        cycle_pred: torch.Tensor,
        cycle_target: torch.Tensor,
        theta_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        epoch_frac: float = 1.0,
        intermediates: Optional[dict] = None,
        cond_ids: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined cycle branch loss.
        
        Args:
            cycle_pred: Predicted sensors (B, T, n_targets) - RAW thermodynamic units
            cycle_target: Target sensors (B, T, n_targets) - StandardScaled if cycle_target_space='scaled'
            theta_seq: Degradation modifiers (B, T, 6)
            mask: Optional valid mask (B, T)
            epoch_frac: Fraction of total epochs (for curriculum)
            intermediates: Optional dict with work_balance etc.
            cond_ids: Condition IDs (B,) for condition-wise normalization
            epoch: Current epoch for epoch-specific debug
            
        Returns:
            total_loss: Scalar combined loss
            components: Dict with individual loss components
        """
        # =====================================================================
        # Condition-wise normalization (Option A: fixed, no learnable params)
        # =====================================================================
        pred_used = cycle_pred
        target_used = cycle_target
        
        if self.cycle_target_mean is not None and self.cycle_target_std is not None and cond_ids is not None:
            # Get mean/std for each batch element based on cond_id
            mean = self.cycle_target_mean[cond_ids]  # (B, 4)
            std = self.cycle_target_std[cond_ids]    # (B, 4)
            
            # Expand for sequence dimension if needed
            if cycle_pred.dim() == 3:
                mean = mean.unsqueeze(1)  # (B, 1, 4)
                std = std.unsqueeze(1)    # (B, 1, 4)
            
            eps = 1e-6
            
            # Always normalize pred (raw -> scaled)
            pred_used = (cycle_pred - mean) / (std + eps)
            
            # Only normalize target if in "raw" mode
            if self.cycle_target_space == "raw":
                target_used = (cycle_target - mean) / (std + eps)
            else:
                # "scaled" mode: target is already StandardScaled, use as-is
                target_used = cycle_target
            
            # Debug prints at epochs 0 and 2
            should_debug = (epoch == 0 and not self._debug_printed) or (epoch == 2)
            if should_debug:
                with torch.no_grad():
                    tgt_mean = cycle_target.mean().item()
                    tgt_std = cycle_target.std().item()
                    
                    print(f"\n[CycleBranchLoss] Epoch {epoch} Debug (cycle_target_space={self.cycle_target_space}):")
                    print(f"  cycle_pred (raw): mean={cycle_pred.mean():.2f}, std={cycle_pred.std():.2f}")
                    print(f"  cycle_target:     mean={tgt_mean:.4f}, std={tgt_std:.4f}")
                    print(f"  pred_used:        mean={pred_used.mean():.4f}, std={pred_used.std():.4f}")
                    print(f"  target_used:      mean={target_used.mean():.4f}, std={target_used.std():.4f}")
                    
                    # Per-sensor stats
                    print(f"  Per-sensor (scaled space):")
                    for i, name in enumerate(self.target_names[:cycle_pred.shape[-1]]):
                        pred_m = pred_used[..., i].mean().item()
                        pred_s = pred_used[..., i].std().item()
                        tgt_m = target_used[..., i].mean().item()
                        tgt_s = target_used[..., i].std().item()
                        delta = abs(pred_m - tgt_m)
                        print(f"    {name}: pred_used({pred_m:.4f}±{pred_s:.2f}) vs target_used({tgt_m:.4f}±{tgt_s:.2f}) Δmean={delta:.4f}")
                    
                    # Theta stats
                    theta_mean = theta_seq.mean().item()
                    theta_min = theta_seq.min().item()
                    theta_max = theta_seq.max().item()
                    near_upper = (theta_seq > 0.99).float().mean().item() * 100
                    near_lower = (theta_seq < 0.91).float().mean().item() * 100
                    sat_frac = near_upper + near_lower
                    print(f"  Theta: mean={theta_mean:.4f}, range=[{theta_min:.4f}, {theta_max:.4f}], sat_frac={sat_frac:.1f}%")
                    
                    if epoch == 0:
                        self._debug_printed = True
        
        # Cycle reconstruction loss (in normalized space)
        l_cycle = compute_cycle_loss(
            pred_used, target_used,
            loss_type=self.cycle_loss_type,
            huber_beta=self.cycle_huber_beta,
            mask=mask,
        )
        
        # Smoothness loss
        l_smooth = compute_theta_smooth_loss(theta_seq, mask=mask)
        
        # Monotonicity loss
        l_mono = compute_theta_mono_loss(
            theta_seq,
            mono_on_eta=self.mono_on_eta,
            mono_on_dp=self.mono_on_dp,
            eps=self.mono_eps,
            mask=mask,
        )
        
        # Power balance penalty
        l_power = torch.tensor(0.0, device=cycle_pred.device)
        if self.lambda_power_balance > 0 and intermediates is not None:
             if "work_balance" in intermediates:
                 l_power = compute_power_balance_penalty(intermediates["work_balance"], mask)
        
        # Per-sensor loss breakdown (for logging)
        per_sensor_metrics = compute_cycle_loss_per_sensor(
             cycle_pred, cycle_target, self.target_names,
             loss_type=self.cycle_loss_type,
             huber_beta=self.cycle_huber_beta
        )
        
        # Apply curriculum to cycle loss weight (ramp up over epochs)
        lambda_cycle_curr = self.lambda_cycle * epoch_frac
        
        # Theta prior (anti-saturation)
        l_theta_prior = compute_theta_prior_loss(theta_seq, mask=mask)
        
        # Combine
        total = (
            lambda_cycle_curr * l_cycle +
            self.lambda_smooth * l_smooth +
            self.lambda_mono * l_mono +
            self.lambda_power_balance * l_power +
            self.lambda_theta_prior * l_theta_prior
        )
        
        components = {
            "cycle_loss": float(l_cycle.item()),
            "theta_smooth_loss": float(l_smooth.item()),
            "theta_mono_loss": float(l_mono.item()),
            "power_balance_loss": float(l_power.item()),
            "theta_prior_loss": float(l_theta_prior.item()),
            "lambda_cycle_effective": lambda_cycle_curr,
            "total_cycle_branch_loss": float(total.item()),
        }
        # Add per-sensor metrics
        for name, val in per_sensor_metrics.items():
            components[f"cycle_loss_{name}"] = val
        
        return total, components
