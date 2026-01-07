"""
Cycle Branch Integration Test Script.

Run this to test the B-Track Cycle Branch components on Colab or locally.

Usage:
    # Colab cell:
    !python scripts/test_cycle_branch.py
    
    # Local:
    python scripts/test_cycle_branch.py
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np


def test_all_components():
    """Test all cycle branch components end-to-end."""
    print("=" * 60)
    print("CYCLE BRANCH INTEGRATION TEST")
    print("=" * 60)
    
    # =========================================================================
    # 1. Test CycleLayerMVP
    # =========================================================================
    print("\n[1] Testing CycleLayerMVP...")
    from src.models.physics.cycle_layer_mvp import CycleLayerMVP
    
    model = CycleLayerMVP(num_conditions=6, pr_mode="per_cond")
    
    B, T = 4, 50
    ops = torch.rand(B, T, 3)
    m_t = torch.ones(B, T, 6) * 0.95
    eta_nom = torch.ones(B, T, 5) * 0.88
    cond_ids = torch.zeros(B, dtype=torch.long)
    
    pred, inter = model(ops, m_t, eta_nom, cond_ids, return_intermediates=True)
    
    assert pred.shape == (B, T, 4), f"Shape mismatch: {pred.shape}"
    assert torch.isfinite(pred).all(), "Non-finite predictions"
    
    print(f"  ✓ Output shape: {pred.shape}")
    print(f"  ✓ T24: {pred[..., 0].min():.1f} - {pred[..., 0].max():.1f} °R")
    print(f"  ✓ T30: {pred[..., 1].min():.1f} - {pred[..., 1].max():.1f} °R")
    print(f"  ✓ P30: {pred[..., 2].min():.1f} - {pred[..., 2].max():.1f} psia")
    print(f"  ✓ T50: {pred[..., 3].min():.1f} - {pred[..., 3].max():.1f} °R")
    
    # Gradient flow
    loss = pred.mean()
    loss.backward()
    assert model.pr_table.grad is not None, "No gradient on pr_table"
    print(f"  ✓ Gradients flow correctly")
    
    # =========================================================================
    # 2. Test NominalHead
    # =========================================================================
    print("\n[2] Testing NominalHead...")
    from src.models.physics.nominal_head import NominalHead
    
    head = NominalHead(head_type="table", num_conditions=6)
    ops = torch.rand(4, 3)
    cond_ids = torch.zeros(4, dtype=torch.long)
    eta_nom = head(ops_t=ops, cond_ids=cond_ids)
    
    assert eta_nom.shape == (4, 5), f"Shape mismatch: {eta_nom.shape}"
    print(f"  ✓ Output shape: {eta_nom.shape}")
    print(f"  ✓ η_nom range: {eta_nom.min():.3f} - {eta_nom.max():.3f}")
    
    # =========================================================================
    # 3. Test ParamHeadTheta6
    # =========================================================================
    print("\n[3] Testing ParamHeadTheta6...")
    from src.models.heads.param_head_theta6 import ParamHeadTheta6
    
    head = ParamHeadTheta6(z_dim=128)
    z_t = torch.randn(4, 50, 128)
    m_t = head(z_t)
    
    assert m_t.shape == (4, 50, 6), f"Shape mismatch: {m_t.shape}"
    print(f"  ✓ Output shape: {m_t.shape}")
    print(f"  ✓ m_t range: {m_t.min():.3f} - {m_t.max():.3f}")
    
    # =========================================================================
    # 4. Test Loss Functions
    # =========================================================================
    print("\n[4] Testing CycleBranchLoss...")
    from src.losses.cycle_losses import CycleBranchLoss
    
    loss_fn = CycleBranchLoss(lambda_cycle=0.1, lambda_smooth=0.05)
    
    cycle_pred = torch.randn(4, 50, 4)
    cycle_target = torch.randn(4, 50, 4)
    theta_seq = torch.rand(4, 50, 6) * 0.15 + 0.85
    
    total_loss, metrics = loss_fn(cycle_pred, cycle_target, theta_seq, epoch_frac=1.0)
    
    assert torch.isfinite(total_loss), "Non-finite loss"
    print(f"  ✓ Total loss: {total_loss.item():.4f}")
    print(f"  ✓ Metrics: {list(metrics.keys())}")
    
    # =========================================================================
    # 5. Test Sensor Mapping
    # =========================================================================
    print("\n[5] Testing sensor_mapping...")
    from src.utils.sensor_mapping import resolve_cycle_target_cols
    
    feature_cols = ["Setting1", "Setting2", "Setting3", "Sensor1", "Sensor2",
                    "Sensor9", "Sensor11", "Sensor13", "Sensor17"]
    target_names = ["T24", "T30", "P30", "T50"]
    
    mapping = resolve_cycle_target_cols(feature_cols, target_names)
    print(f"  ✓ Mapping: {mapping}")
    
    # =========================================================================
    # 6. Test Power Balance Penalty
    # =========================================================================
    print("\n[6] Testing power balance penalty...")
    model = CycleLayerMVP(num_conditions=6, pr_mode="per_cond")
    
    pred, inter = model(
        torch.rand(4, 50, 3),
        torch.ones(4, 50, 6) * 0.95,
        torch.ones(4, 50, 5) * 0.88,
        torch.zeros(4, dtype=torch.long),
        return_intermediates=True,
    )
    
    wb = inter["work_balance"]
    penalty = model.compute_power_balance_penalty(wb)
    print(f"  ✓ Power balance penalty: {penalty.item():.4f}")
    
    # =========================================================================
    # 7. Test head mode (bounded PRs)
    # =========================================================================
    print("\n[7] Testing pr_mode='head' (bounded PRs)...")
    model = CycleLayerMVP(pr_mode="head", pr_head_hidden=16)
    
    pred, inter = model(
        torch.rand(2, 10, 3),
        torch.ones(2, 10, 6) * 0.95,
        torch.ones(2, 10, 5) * 0.88,
        cond_ids=None,
        return_intermediates=True,
    )
    
    prs = inter["prs"]
    print(f"  ✓ PRs bounded: [{prs.min():.2f}, {prs.max():.2f}]")
    assert (prs >= 1.0).all(), "PRs below 1.0"
    assert (prs <= 25.0).all(), "PRs above 25.0"
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nCycle branch is ready for training integration.")
    print("\nTo enable in training config:")
    print("""
from src.world_model_training import WorldModelTrainingConfig, CycleBranchConfig

config = WorldModelTrainingConfig(
    cycle_branch=CycleBranchConfig(
        enable=True,
        targets=["T24", "T30", "P30", "T50"],
        lambda_cycle=0.1,
        lambda_theta_smooth=0.05,
    )
)
""")


if __name__ == "__main__":
    test_all_components()
