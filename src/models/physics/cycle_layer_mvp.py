"""
CycleLayerMVP: Differentiable 0D Thermodynamic Cycle Model (Revised).

This module implements an algebraic (no implicit solver) cycle model that
predicts sensor outputs from operating conditions and degradation parameters.

Mode 1 Factorized: η_eff = η_nom(ops) × m(t)
where m(t) are slow-varying degradation modifiers from ParamHeadTheta6.

CMAPSS Sensor Mapping:
- T24 = Total temperature at LPC outlet (T_lpc_out)
- T30 = Total temperature at HPC outlet (T_hpc_out)
- P30 = Total pressure at HPC outlet (P_hpc_out)
- T50 = Total temperature at LPT outlet (T_lpt_out)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CycleLayerMVP(nn.Module):
    """Algebraic 0D cycle model for CMAPSS turbofan sensor prediction (Revised).
    
    Predicts cycle-relevant sensors (T24, T30, P30, T50) from:
    - Operating settings (ops_t): assumed normalized [0,1]
    - Degradation modifiers m(t) from ParamHeadTheta6
    - Nominal efficiencies η_nom from NominalHead
    
    Uses corrected thermodynamic relations:
    - Proper isentropic compression with efficiency
    - Proper isentropic expansion with efficiency (turbines)
    - T4 (combustor exit) depends on TRA (throttle)
    - Work balance as optional soft penalty (not hard constraint)
    - Soft lower/upper bounds (no hard clamps that kill gradients)
    
    Args:
        n_targets: Number of sensor targets to predict (default 4: T24, T30, P30, T50)
        num_conditions: Number of operating conditions for per-condition PR constants
        pr_mode: 'per_cond' (per-condition constants) or 'head' (learnable MLP)
        pr_head_hidden: Hidden dim for PR head if pr_mode='head'
        dp_nom_constant: Constant value for combustor pressure drop ratio (0-1)
        gamma_c: Specific heat ratio for compressor (default 1.4)
        gamma_t: Specific heat ratio for turbine (default 1.33)
        cp_c: Specific heat capacity for cold air (J/kg·K, default 1005.0)
        cp_t: Specific heat capacity for hot gas (J/kg·K, default 1150.0)
        t2_ambient: Ambient/inlet temperature baseline (°R, default 518.67)
        p2_ambient: Ambient/inlet pressure baseline (PSIA, default 14.7)
    """
    
    def __init__(
        self,
        n_targets: int = 4,
        num_conditions: Optional[int] = None,
        pr_mode: Literal["per_cond", "head"] = "per_cond",
        pr_head_hidden: int = 16,
        dp_nom_constant: float = 0.95,
        gamma_c: float = 1.4,
        gamma_t: float = 1.33,
        cp_c: float = 1005.0,
        cp_t: float = 1150.0,
        t2_ambient: float = 518.67,
        p2_ambient: float = 14.7,
    ):
        super().__init__()
        self.n_targets = n_targets
        self.pr_mode = pr_mode
        self.dp_nom_constant = dp_nom_constant
        
        # Thermodynamic constants as buffers
        self.register_buffer("gamma_c", torch.tensor(gamma_c))
        self.register_buffer("gamma_t", torch.tensor(gamma_t))
        self.register_buffer("cp_c", torch.tensor(cp_c))
        self.register_buffer("cp_t", torch.tensor(cp_t))
        self.register_buffer("t2_ambient", torch.tensor(t2_ambient))
        self.register_buffer("p2_ambient", torch.tensor(p2_ambient))
        
        # Exponents for isentropic relations
        # Compressor: (gamma_c - 1) / gamma_c
        # Turbine: (gamma_t - 1) / gamma_t
        exp_c = (gamma_c - 1) / gamma_c
        exp_t = (gamma_t - 1) / gamma_t
        self.register_buffer("exp_c", torch.tensor(exp_c))
        self.register_buffer("exp_t", torch.tensor(exp_t))
        
        # Pressure ratios: per-condition constants or learnable head
        if pr_mode == "per_cond":
            if num_conditions is None or num_conditions < 1:
                raise ValueError("num_conditions required for pr_mode='per_cond'")
            self.num_conditions = num_conditions
            # Initialize with reasonable defaults for turbofan
            # PR_fan ~ 1.5, PR_lpc ~ 2.0, PR_hpc ~ 10.0
            pr_init = torch.tensor([
                [1.5, 2.0, 10.0] for _ in range(num_conditions)
            ], dtype=torch.float32)
            self.pr_table = nn.Parameter(pr_init)
            self.pr_head_raw = None
            # Bounds for soft clamping per_cond mode
            self.register_buffer("pr_max", torch.tensor([3.0, 5.0, 25.0]))
        else:
            self.num_conditions = None
            self.pr_table = None
            # Small MLP: ops -> PRs (raw, no activation - bounds applied via sigmoid)
            self.pr_head_raw = nn.Sequential(
                nn.Linear(3, pr_head_hidden),
                nn.SiLU(),
                nn.Linear(pr_head_hidden, 3),
            )
            # Bounded PR range for head mode
            self.register_buffer("pr_min", torch.tensor([1.1, 1.1, 5.0]))
            self.register_buffer("pr_max", torch.tensor([2.0, 3.0, 20.0]))
    
    def _safe_lower_bound(
        self,
        x: torch.Tensor,
        min_val: float,
        softness: float = 0.05,
    ) -> torch.Tensor:
        """Soft lower bound using softplus (gradient-friendly).
        
        Returns values >= min_val with smooth transition.
        """
        return min_val + F.softplus(x - min_val, beta=1.0 / softness)
    
    def _safe_upper_bound(
        self,
        x: torch.Tensor,
        max_val: torch.Tensor,
        softness: float = 0.1,
    ) -> torch.Tensor:
        """Soft upper bound using softplus (gradient-friendly).
        
        Returns values <= max_val with smooth transition.
        """
        return max_val - F.softplus(max_val - x, beta=1.0 / softness)
    
    def _compress(
        self,
        T_in: torch.Tensor,
        pr: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressor outlet temperature.
        
        Uses isentropic reference with efficiency:
        T_out = T_in * (1 + (PR^exp - 1) / eta)
        where exp = (gamma_c - 1) / gamma_c
        
        Args:
            T_in: Inlet temperature
            pr: Pressure ratio (>= 1)
            eta: Isentropic efficiency (0-1)
            
        Returns:
            T_out: Outlet temperature
        """
        # Soft bounds for numerical stability
        eta_safe = self._safe_lower_bound(eta, 0.5, softness=0.02)
        pr_safe = self._safe_lower_bound(pr, 1.01, softness=0.02)
        
        # Isentropic temperature ratio
        T_ratio_ideal = pr_safe ** self.exp_c
        
        # Actual temperature ratio with efficiency
        T_ratio_actual = 1.0 + (T_ratio_ideal - 1.0) / eta_safe
        
        return T_in * T_ratio_actual
    
    def _expand(
        self,
        T_in: torch.Tensor,
        expansion_ratio: torch.Tensor,
        eta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbine outlet temperature.
        
        Uses isentropic reference with efficiency:
        Step 1: T_out_s = T_in * (1/ER)^((gamma_t - 1)/gamma_t)  [isentropic]
        Step 2: T_out = T_in - eta_t * (T_in - T_out_s)          [actual with efficiency]
        
        Args:
            T_in: Inlet temperature (hot gas)
            expansion_ratio: ER = P_in / P_out (>= 1)
            eta_t: Turbine isentropic efficiency (0-1)
            
        Returns:
            T_out: Outlet temperature
        """
        # Soft lower bound for ER (must be >= 1)
        er_safe = self._safe_lower_bound(expansion_ratio, 1.01, softness=0.02)
        eta_safe = self._safe_lower_bound(eta_t, 0.5, softness=0.02)
        
        # Isentropic outlet temperature
        # T_out_s / T_in = (1/ER)^((gamma_t - 1)/gamma_t)
        T_out_s = T_in * (1.0 / er_safe) ** self.exp_t
        
        # Actual temperature with efficiency
        # eta_t = (T_in - T_out) / (T_in - T_out_s)
        # => T_out = T_in - eta_t * (T_in - T_out_s)
        T_out = T_in - eta_safe * (T_in - T_out_s)
        
        return T_out
    
    def _get_pressure_ratios(
        self,
        ops_flat: torch.Tensor,
        cond_ids: Optional[torch.Tensor],
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Get pressure ratios [PR_fan, PR_lpc, PR_hpc]."""
        if self.pr_mode == "per_cond":
            if cond_ids is None:
                raise ValueError("cond_ids required for pr_mode='per_cond'")
            # Expand cond_ids if needed
            if cond_ids.dim() == 1:
                # (B,) -> (B*T,)
                cond_ids_flat = cond_ids.unsqueeze(1).expand(B, T).reshape(B * T)
            else:
                cond_ids_flat = cond_ids.reshape(B * T)
            cond_ids_clamped = cond_ids_flat.clamp(0, self.num_conditions - 1)
            prs = self.pr_table[cond_ids_clamped]  # (B*T, 3)
            # Soft lower bound
            prs = self._safe_lower_bound(prs, 1.01, softness=0.05)
            # Soft upper bound to prevent blow-ups
            prs = self._safe_upper_bound(prs, self.pr_max, softness=0.2)
        else:
            # MLP head with bounded output via sigmoid
            raw = self.pr_head_raw(ops_flat)  # (B*T, 3)
            # Map to bounded range: pr_min + sigmoid(raw) * (pr_max - pr_min)
            prs = self.pr_min + torch.sigmoid(raw) * (self.pr_max - self.pr_min)
        
        return prs
    
    def forward(
        self,
        ops_t: torch.Tensor,
        m_t: torch.Tensor,
        eta_nom: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass: predict cycle sensors.
        
        Args:
            ops_t: Operating settings (B, T, 3) or (B, 3), assumed normalized [0,1]
                   [TRA, Altitude, Mach]
            m_t: Degradation modifiers (B, T, 6) or (B, 6)
                 [m_fan, m_lpc, m_hpc, m_hpt, m_lpt, m_dp_comb]
            eta_nom: Nominal efficiencies (B, T, 5) or (B, 5)
                     [η_fan, η_lpc, η_hpc, η_hpt, η_lpt]
            cond_ids: Condition IDs (B,) or (B, T) for per-cond PR lookup
            return_intermediates: If True, also return dict of intermediate values
            
        Returns:
            cycle_pred: (B, T, n_targets) predicted sensor values [T24, T30, P30, T50]
            intermediates: (optional) dict with stations, eta_eff, work balance, etc.
        """
        # Handle shape: support both (B, *) and (B, T, *)
        is_seq = ops_t.dim() == 3
        if not is_seq:
            ops_t = ops_t.unsqueeze(1)      # (B, 1, 3)
            m_t = m_t.unsqueeze(1)          # (B, 1, 6)
            eta_nom = eta_nom.unsqueeze(1)  # (B, 1, 5)
        
        B, T, _ = ops_t.shape
        
        # Flatten for computation
        ops_flat = ops_t.reshape(B * T, 3)
        m_flat = m_t.reshape(B * T, 6)
        eta_nom_flat = eta_nom.reshape(B * T, 5)
        
        # =====================================================================
        # Mode 1: Compute effective efficiencies η_eff = η_nom × m(t)
        # =====================================================================
        eta_eff_fan = eta_nom_flat[:, 0] * m_flat[:, 0]
        eta_eff_lpc = eta_nom_flat[:, 1] * m_flat[:, 1]
        eta_eff_hpc = eta_nom_flat[:, 2] * m_flat[:, 2]
        eta_eff_hpt = eta_nom_flat[:, 3] * m_flat[:, 3]
        eta_eff_lpt = eta_nom_flat[:, 4] * m_flat[:, 4]
        
        # Combustor pressure drop: dp_eff = dp_nom_constant * m_dp
        m_dp = m_flat[:, 5]
        dp_eff = self.dp_nom_constant * m_dp
        
        # =====================================================================
        # Get pressure ratios (fan, lpc, hpc)
        # =====================================================================
        prs = self._get_pressure_ratios(ops_flat, cond_ids, B, T)
        pr_fan = prs[:, 0]
        pr_lpc = prs[:, 1]
        pr_hpc = prs[:, 2]
        
        # =====================================================================
        # Inlet conditions (Station 2)
        # =====================================================================
        T2 = self.t2_ambient.expand(B * T)
        P2 = self.p2_ambient.expand(B * T)
        
        # Apply ram effect based on Mach number
        # ops_t assumed normalized [0,1], Mach is ops[2]
        # Scale Mach to physical range ~0-0.9
        mach = ops_flat[:, 2].clamp(0.0, 1.0) * 0.9
        
        # Ram temperature rise: T_stag/T_static ≈ 1 + 0.2*M^2 (gamma=1.4)
        T2 = T2 * (1.0 + 0.2 * mach ** 2)
        
        # Ram pressure rise: P_stag/P_static ≈ (1 + 0.2*M^2)^3.5
        P2 = P2 * (1.0 + 0.2 * mach ** 2) ** 3.5
        
        # Ensure P2 > 0 for stability
        P2 = self._safe_lower_bound(P2, 1.0, softness=0.1)
        
        # =====================================================================
        # Compressor Train: Fan -> LPC -> HPC
        # =====================================================================
        # Fan outlet
        T_fan_out = self._compress(T2, pr_fan, eta_eff_fan)
        P_fan_out = P2 * pr_fan
        
        # LPC outlet (T_lpc_out -> CMAPSS T24)
        T_lpc_out = self._compress(T_fan_out, pr_lpc, eta_eff_lpc)
        P_lpc_out = P_fan_out * pr_lpc
        
        # HPC outlet (T_hpc_out, P_hpc_out -> CMAPSS T30, P30)
        T_hpc_out = self._compress(T_lpc_out, pr_hpc, eta_eff_hpc)
        P_hpc_out = P_lpc_out * pr_hpc
        
        # =====================================================================
        # Combustor (Station 4)
        # =====================================================================
        # Pressure drop: P4 = P_hpc_out * dp_eff
        P4 = P_hpc_out * dp_eff
        
        # T4 depends on TRA (throttle resolver angle = load)
        # TRA is ops[:, 0], assumed normalized [0, 1]
        tra = ops_flat[:, 0].clamp(0.0, 1.0)
        
        # T4/T_hpc_out ratio as load proxy:
        # Idle (~TRA=0): ratio ~1.8
        # Max (~TRA=1): ratio ~2.8-3.0
        t4_ratio = 1.8 + 1.0 * tra
        T4 = T_hpc_out * t4_ratio
        
        # =====================================================================
        # Turbine pressure ratios (from total expansion)
        # =====================================================================
        # Total expansion ratio across turbines
        PR_turb_total = P4 / P2
        PR_turb_total = self._safe_lower_bound(PR_turb_total, 1.2, softness=0.1)
        
        # Split between HPT and LPT (log-space additive heuristic)
        # HPT handles ~40% (smaller delta-P, higher work per stage)
        # LPT handles ~60%
        PR_hpt = PR_turb_total ** 0.4
        PR_lpt = PR_turb_total ** 0.6
        
        # Station pressures
        P45 = P4 / PR_hpt  # HPT outlet
        P50 = P45 / PR_lpt  # LPT outlet
        
        # =====================================================================
        # Turbine Train: HPT -> LPT
        # =====================================================================
        # HPT outlet
        T_hpt_out = self._expand(T4, PR_hpt, eta_eff_hpt)
        
        # LPT outlet (T_lpt_out -> CMAPSS T50)
        T_lpt_out = self._expand(T_hpt_out, PR_lpt, eta_eff_lpt)
        
        # =====================================================================
        # Work balance computation (for debugging/soft penalty)
        # =====================================================================
        # Compressor work (temperature rise × cp_c)
        W_fan = self.cp_c * (T_fan_out - T2)
        W_lpc = self.cp_c * (T_lpc_out - T_fan_out)
        W_hpc = self.cp_c * (T_hpc_out - T_lpc_out)
        
        # Turbine work (temperature drop × cp_t)
        W_hpt = self.cp_t * (T4 - T_hpt_out)
        W_lpt = self.cp_t * (T_hpt_out - T_lpt_out)
        
        # =====================================================================
        # Map internal stations to CMAPSS outputs
        # =====================================================================
        # T24 = T_lpc_out (LPC outlet, NOT fan outlet!)
        # T30 = T_hpc_out
        # P30 = P_hpc_out
        # T50 = T_lpt_out
        T24_pred = T_lpc_out
        T30_pred = T_hpc_out
        P30_pred = P_hpc_out
        T50_pred = T_lpt_out
        
        # =====================================================================
        # Stack predictions (NO hard clamps - use soft bounds if needed externally)
        # =====================================================================
        cycle_pred = torch.stack([T24_pred, T30_pred, P30_pred, T50_pred], dim=-1)
        
        # Reshape back to (B, T, 4)
        cycle_pred = cycle_pred.reshape(B, T, self.n_targets)
        
        if not is_seq:
            cycle_pred = cycle_pred.squeeze(1)  # (B, n_targets)
        
        if return_intermediates:
            # Reshape all intermediates for return
            def _reshape(x):
                if is_seq:
                    return x.reshape(B, T) if x.dim() == 1 else x.reshape(B, T, -1)
                return x
            
            # CR-2: Return reshaped eta_nom for consistency
            eta_nom_out = eta_nom_flat.reshape(B, T, 5) if is_seq else eta_nom_flat.reshape(B, 5)
            
            eta_eff_dict = {
                "fan": _reshape(eta_eff_fan),
                "lpc": _reshape(eta_eff_lpc),
                "hpc": _reshape(eta_eff_hpc),
                "hpt": _reshape(eta_eff_hpt),
                "lpt": _reshape(eta_eff_lpt),
            }
            
            T_stations = {
                "T2": _reshape(T2),
                "T_fan_out": _reshape(T_fan_out),
                "T_lpc_out": _reshape(T_lpc_out),
                "T_hpc_out": _reshape(T_hpc_out),
                "T4": _reshape(T4),
                "T_hpt_out": _reshape(T_hpt_out),
                "T_lpt_out": _reshape(T_lpt_out),
            }
            
            P_stations = {
                "P2": _reshape(P2),
                "P_fan_out": _reshape(P_fan_out),
                "P_lpc_out": _reshape(P_lpc_out),
                "P_hpc_out": _reshape(P_hpc_out),
                "P4": _reshape(P4),
                "P45": _reshape(P45),
                "P50": _reshape(P50),
            }
            
            work_balance = {
                "W_fan": _reshape(W_fan),
                "W_lpc": _reshape(W_lpc),
                "W_hpc": _reshape(W_hpc),
                "W_hpt": _reshape(W_hpt),
                "W_lpt": _reshape(W_lpt),
            }
            
            intermediates = {
                "eta_nom": eta_nom_out,
                "eta_eff": eta_eff_dict,
                "dp_eff": _reshape(dp_eff),
                "prs": prs.reshape(B, T, 3) if is_seq else prs,
                "T_stations": T_stations,
                "P_stations": P_stations,
                "work_balance": work_balance,
            }
            
            return cycle_pred, intermediates
        
        return cycle_pred
    
    def compute_power_balance_penalty(
        self,
        work_balance: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute soft power balance penalty (optional regularizer).
        
        Penalizes deviation from:
        - W_HPT ≈ W_HPC (HPT drives HPC)
        - W_LPT ≈ W_Fan + W_LPC (LPT drives Fan and LPC)
        
        Args:
            work_balance: Dict with W_fan, W_lpc, W_hpc, W_hpt, W_lpt
            
        Returns:
            Scalar penalty term (mean over batch)
        """
        W_hpc = work_balance["W_hpc"]
        W_hpt = work_balance["W_hpt"]
        W_fan_lpc = work_balance["W_fan"] + work_balance["W_lpc"]
        W_lpt = work_balance["W_lpt"]
        
        # Relative error penalty (squared)
        eps = 1e-6
        penalty_hpt = ((W_hpt - W_hpc) / (W_hpc.abs() + eps)) ** 2
        penalty_lpt = ((W_lpt - W_fan_lpc) / (W_fan_lpc.abs() + eps)) ** 2
        
        return (penalty_hpt + penalty_lpt).mean()


def test_cycle_layer_mvp_smoke():
    """Smoke test: basic forward + gradient flow."""
    model = CycleLayerMVP(num_conditions=6, pr_mode="per_cond")
    
    B, T = 4, 50
    ops = torch.rand(B, T, 3)  # assumed normalized [0,1]
    m_t = torch.ones(B, T, 6) * 0.95
    eta_nom = torch.ones(B, T, 5) * 0.88
    cond_ids = torch.zeros(B, dtype=torch.long)
    
    # Forward with intermediates
    pred, inter = model(ops, m_t, eta_nom, cond_ids, return_intermediates=True)
    
    # Shape check (CR-4: fixed message)
    assert pred.shape == (B, T, 4), f"Expected shape {(B, T, 4)}, got {pred.shape}"
    
    # Finite check
    assert torch.isfinite(pred).all(), "Non-finite values in predictions"
    
    # Gradient flow
    loss = pred.mean()
    loss.backward()
    
    # Check gradients exist on learnable params
    assert model.pr_table.grad is not None, "No gradient on pr_table"
    assert (model.pr_table.grad != 0).any(), "Zero gradients on pr_table"
    
    # Sanity check output ranges (not hard assert, just logging)
    print(f"[Smoke Test] pred shape: {pred.shape}")
    print(f"  T24 range: {pred[..., 0].min():.1f} - {pred[..., 0].max():.1f}")
    print(f"  T30 range: {pred[..., 1].min():.1f} - {pred[..., 1].max():.1f}")
    print(f"  P30 range: {pred[..., 2].min():.1f} - {pred[..., 2].max():.1f}")
    print(f"  T50 range: {pred[..., 3].min():.1f} - {pred[..., 3].max():.1f}")
    
    # Check intermediates contain reshaped eta_nom (CR-2)
    assert inter["eta_nom"].shape == (B, T, 5), f"eta_nom wrong shape: {inter['eta_nom'].shape}"
    
    # Log work balance
    wb = inter["work_balance"]
    penalty = model.compute_power_balance_penalty(wb)
    print(f"  Power balance penalty: {penalty.item():.4f}")
    
    print("[Smoke Test] PASSED ✓")


def test_cycle_layer_head_mode():
    """Test pr_mode='head' with bounded PRs."""
    model = CycleLayerMVP(pr_mode="head", pr_head_hidden=16)
    
    B, T = 2, 10
    ops = torch.rand(B, T, 3)
    m_t = torch.ones(B, T, 6) * 0.95
    eta_nom = torch.ones(B, T, 5) * 0.88
    
    pred, inter = model(ops, m_t, eta_nom, cond_ids=None, return_intermediates=True)
    
    # Check PRs are bounded
    prs = inter["prs"]
    assert (prs >= 1.0).all(), f"PRs below 1.0: {prs.min()}"
    assert (prs <= 25.0).all(), f"PRs above 25.0: {prs.max()}"
    
    # Gradient flow
    loss = pred.mean()
    loss.backward()
    
    # Check head has gradients
    for name, p in model.pr_head_raw.named_parameters():
        assert p.grad is not None, f"No gradient on pr_head_raw.{name}"
    
    print("[Head Mode Test] PASSED ✓")


if __name__ == "__main__":
    test_cycle_layer_mvp_smoke()
    test_cycle_layer_head_mode()
