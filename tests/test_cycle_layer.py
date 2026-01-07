"""
Unit Tests for Cycle Branch Components.

Tests for:
- Sensor mapping resolver
- CycleLayerMVP forward pass and gradients
- NominalHead table and MLP modes
- ParamHeadTheta6 bounds enforcement
- Cycle loss functions
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


class TestSensorMapping(unittest.TestCase):
    """Tests for sensor mapping resolver."""
    
    def test_get_sensor_index(self):
        """Test semantic name to index mapping."""
        from src.utils.sensor_mapping import get_sensor_index
        
        # Standard mappings per CMAPSS spec
        self.assertEqual(get_sensor_index("T24"), 2)
        self.assertEqual(get_sensor_index("T30"), 3)
        self.assertEqual(get_sensor_index("P30"), 7)
        self.assertEqual(get_sensor_index("T50"), 4)
        self.assertEqual(get_sensor_index("Nf"), 8)
        self.assertEqual(get_sensor_index("Nc"), 9)
    
    def test_resolve_cycle_target_cols_sensor_format(self):
        """Test resolver with Sensor1, Sensor2, ... format."""
        from src.utils.sensor_mapping import resolve_cycle_target_cols
        
        feature_cols = [f"Sensor{i}" for i in range(1, 22)] + ["Setting1", "Setting2"]
        
        result = resolve_cycle_target_cols(feature_cols, ["T24", "T30", "P30", "T50"])
        
        self.assertEqual(result["T24"], "Sensor2")
        self.assertEqual(result["T30"], "Sensor3")
        self.assertEqual(result["P30"], "Sensor7")
        self.assertEqual(result["T50"], "Sensor4")
    
    def test_resolve_raises_on_missing(self):
        """Test that resolver raises when target cannot be found."""
        from src.utils.sensor_mapping import resolve_cycle_target_cols
        
        # Only sensors 1-5 available
        feature_cols = [f"Sensor{i}" for i in range(1, 6)]
        
        with self.assertRaises(ValueError) as ctx:
            resolve_cycle_target_cols(feature_cols, ["T24", "P30"])  # P30 = Sensor7
        
        self.assertIn("P30", str(ctx.exception))


class TestNominalHead(unittest.TestCase):
    """Tests for NominalHead."""
    
    def test_table_mode_shape(self):
        """Test table mode output shape."""
        from src.models.physics.nominal_head import NominalHead
        
        head = NominalHead(head_type="table", num_conditions=6)
        
        # Test with batch of condition IDs
        cond_ids = torch.tensor([0, 2, 5, 1])
        ops = torch.randn(4, 10, 3)  # (B, T, 3)
        
        eta_nom = head(ops_t=ops, cond_ids=cond_ids)
        
        self.assertEqual(eta_nom.shape, (4, 10, 5))
    
    def test_mlp_mode_shape(self):
        """Test MLP mode output shape."""
        from src.models.physics.nominal_head import NominalHead
        
        head = NominalHead(head_type="mlp", num_conditions=None)
        
        ops = torch.randn(4, 10, 3)
        eta_nom = head(ops_t=ops)
        
        self.assertEqual(eta_nom.shape, (4, 10, 5))
    
    def test_bounds_enforcement(self):
        """Test that outputs are within bounds."""
        from src.models.physics.nominal_head import NominalHead
        
        head = NominalHead(
            head_type="mlp",
            eta_bounds=(0.80, 0.99),
        )
        
        # Random inputs
        for _ in range(10):
            ops = torch.randn(8, 20, 3) * 10  # Extreme values
            eta_nom = head(ops_t=ops)
            
            self.assertTrue((eta_nom >= 0.80).all())
            self.assertTrue((eta_nom <= 0.99).all())


class TestParamHeadTheta6(unittest.TestCase):
    """Tests for ParamHeadTheta6."""
    
    def test_output_shape(self):
        """Test output shape for sequence input."""
        from src.models.heads.param_head_theta6 import ParamHeadTheta6
        
        head = ParamHeadTheta6(z_dim=64)
        
        z = torch.randn(4, 10, 64)  # (B, T, z_dim)
        m_t = head(z)
        
        self.assertEqual(m_t.shape, (4, 10, 6))
    
    def test_bounds_enforcement(self):
        """Test that all modifiers are within bounds."""
        from src.models.heads.param_head_theta6 import ParamHeadTheta6
        
        head = ParamHeadTheta6(
            z_dim=64,
            m_bounds_eta=(0.85, 1.00),
            m_bounds_dp=(0.90, 1.00),
        )
        
        for _ in range(10):
            z = torch.randn(8, 20, 64) * 5  # Various magnitudes
            m_t = head(z)
            
            # Check eta modifier bounds (first 5)
            self.assertTrue((m_t[..., :5] >= 0.85).all())
            self.assertTrue((m_t[..., :5] <= 1.00).all())
            
            # Check dp modifier bounds (6th)
            self.assertTrue((m_t[..., 5] >= 0.90).all())
            self.assertTrue((m_t[..., 5] <= 1.00).all())
    
    def test_gradient_flow(self):
        """Test that gradients flow back through the head."""
        from src.models.heads.param_head_theta6 import ParamHeadTheta6
        
        head = ParamHeadTheta6(z_dim=64)
        
        z = torch.randn(4, 10, 64, requires_grad=True)
        m_t = head(z)
        
        loss = m_t.mean()
        loss.backward()
        
        self.assertIsNotNone(z.grad)
        self.assertTrue((z.grad != 0).any())


class TestCycleLayerMVP(unittest.TestCase):
    """Tests for CycleLayerMVP."""
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        from src.models.physics.cycle_layer_mvp import CycleLayerMVP
        
        layer = CycleLayerMVP(
            n_targets=4,
            num_conditions=6,
            pr_mode="per_cond",
        )
        
        B, T = 4, 10
        ops = torch.randn(B, T, 3)
        m_t = torch.ones(B, T, 6) * 0.95
        eta_nom = torch.ones(B, T, 5) * 0.90
        cond_ids = torch.randint(0, 6, (B,))
        
        pred = layer(ops, m_t, eta_nom, cond_ids)
        
        self.assertEqual(pred.shape, (B, T, 4))
    
    def test_output_reasonable_range(self):
        """Test that outputs are in physically reasonable ranges.
        
        Note: Ranges updated for revised CycleLayerMVP with proper turbine
        expansion and TRA-dependent T4. The model now produces realistic
        gas-path temperatures without hard clamps.
        """
        from src.models.physics.cycle_layer_mvp import CycleLayerMVP
        
        layer = CycleLayerMVP(num_conditions=6)
        
        ops = torch.tensor([[[0.5, 0.3, 0.6]]])  # TRA, Alt, Mach (normalized)
        m_t = torch.ones(1, 1, 6) * 0.95
        eta_nom = torch.ones(1, 1, 5) * 0.88
        cond_ids = torch.tensor([0])
        
        pred = layer(ops, m_t, eta_nom, cond_ids)
        
        # T24, T30, P30, T50
        T24, T30, P30, T50 = pred[0, 0].tolist()
        
        # Physically plausible ranges for turbofan gas path (°R and psia)
        # T24 (LPC outlet): ~600-1000°R
        # T30 (HPC outlet): ~1200-2000°R (high due to compression)
        # P30 (HPC outlet): ~50-700 psia
        # T50 (LPT outlet): ~1200-2500°R (depends on T4 and expansion)
        self.assertTrue(500 <= T24 <= 1000, f"T24={T24} out of range")
        self.assertTrue(1000 <= T30 <= 2500, f"T30={T30} out of range")
        self.assertTrue(50 <= P30 <= 1000, f"P30={P30} out of range")
        self.assertTrue(1000 <= T50 <= 3000, f"T50={T50} out of range")
    
    def test_gradient_flow(self):
        """Test that gradients flow through the cycle layer."""
        from src.models.physics.cycle_layer_mvp import CycleLayerMVP
        
        layer = CycleLayerMVP(num_conditions=6)
        
        # Create leaf tensors that require gradients
        ops = torch.randn(4, 10, 3, requires_grad=True)
        m_t_raw = torch.randn(4, 10, 6, requires_grad=True)
        eta_nom_raw = torch.randn(4, 10, 5, requires_grad=True)
        
        # Transform to proper ranges (non-leaf tensors)
        m_t = m_t_raw.sigmoid() * 0.15 + 0.85
        eta_nom = eta_nom_raw.sigmoid() * 0.19 + 0.80
        cond_ids = torch.randint(0, 6, (4,))
        
        pred = layer(ops, m_t, eta_nom, cond_ids)
        loss = pred.mean()
        loss.backward()
        
        # Check that gradients flow to leaf tensors
        self.assertIsNotNone(ops.grad)
        self.assertIsNotNone(m_t_raw.grad)


class TestCycleLosses(unittest.TestCase):
    """Tests for cycle loss functions."""
    
    def test_cycle_loss_shape(self):
        """Test cycle reconstruction loss."""
        from src.losses.cycle_losses import compute_cycle_loss
        
        pred = torch.randn(4, 10, 4)
        target = torch.randn(4, 10, 4)
        
        loss = compute_cycle_loss(pred, target, loss_type="huber")
        
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)
    
    def test_smooth_loss_sequence(self):
        """Test smoothness loss on sequences."""
        from src.losses.cycle_losses import compute_theta_smooth_loss
        
        # Smooth sequence (constant)
        theta_smooth = torch.ones(4, 20, 6) * 0.95
        loss_smooth = compute_theta_smooth_loss(theta_smooth)
        
        # Noisy sequence
        theta_noisy = torch.randn(4, 20, 6) * 0.1 + 0.90
        loss_noisy = compute_theta_smooth_loss(theta_noisy)
        
        # Smooth should have lower loss
        self.assertLess(loss_smooth.item(), loss_noisy.item())
    
    def test_mono_loss_respects_flags(self):
        """Test that mono loss respects eta/dp flags."""
        from src.losses.cycle_losses import compute_theta_mono_loss
        
        # Increasing sequence (violates monotonicity)
        theta = torch.linspace(0.85, 1.0, 20).unsqueeze(0).unsqueeze(-1).expand(4, 20, 6)
        
        # With both flags OFF, should be zero
        loss_off = compute_theta_mono_loss(theta, mono_on_eta=False, mono_on_dp=False)
        self.assertEqual(loss_off.item(), 0.0)
        
        # With eta flag ON
        loss_eta = compute_theta_mono_loss(theta, mono_on_eta=True, mono_on_dp=False)
        self.assertGreater(loss_eta.item(), 0.0)


class TestVarianceAttribution(unittest.TestCase):
    """Tests for variance attribution analysis."""
    
    def test_variance_attribution(self):
        """Test variance attribution computation."""
        from src.losses.cycle_losses import compute_variance_attribution
        
        # eta_nom varies a lot (ops-driven)
        eta_nom = torch.randn(100, 5) * 0.05 + 0.88
        
        # m varies less (degradation)
        m_t = torch.randn(100, 6) * 0.01 + 0.95
        
        result = compute_variance_attribution(eta_nom, m_t)
        
        self.assertIn("hpc", result)
        self.assertIn("var_share_deg", result["hpc"])
        
        # Since eta_nom has more variance, var_share_deg should be small
        for comp in ["fan", "lpc", "hpc", "hpt", "lpt"]:
            self.assertLess(result[comp]["var_share_deg"], 0.5)


if __name__ == "__main__":
    unittest.main()
