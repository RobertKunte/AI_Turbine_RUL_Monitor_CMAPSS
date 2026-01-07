"""
Integration tests for Cycle Branch B2.2 integration.

Tests:
1. Model instantiation with cycle_branch.enable=True
2. Forward pass + cycle loss + backward (gradient flow)
3. Risk metrics functions return finite floats
4. Sensor resolver works for SensorX columns
"""

from __future__ import annotations

import pytest
import numpy as np


class TestRiskMetrics:
    """Test risk_metrics.py functions."""
    
    def test_compute_risk_metrics_last_basic(self):
        """Test basic risk metrics computation."""
        from src.analysis.risk_metrics import compute_risk_metrics_last
        
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([15, 25, 28, 38, 55])  # Mix of over/under
        
        metrics = compute_risk_metrics_last(y_true, y_pred)
        
        # Check all keys exist
        expected_keys = [
            "mean_err", "std_err", "over_rate_10", "over_rate_20",
            "under_rate_10", "p95_over", "max_over", "mean_over_pos"
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
            assert isinstance(metrics[key], float), f"Key {key} is not float"
            assert np.isfinite(metrics[key]) or key in ["p99_over", "p95_abs_err"], f"Key {key} is not finite"
    
    def test_compute_risk_metrics_overestimation(self):
        """Test metrics correctly identify overestimation."""
        from src.analysis.risk_metrics import compute_risk_metrics_last
        
        y_true = np.array([10, 10, 10, 10, 10])
        y_pred = np.array([25, 35, 12, 15, 22])  # All overestimate
        
        metrics = compute_risk_metrics_last(y_true, y_pred)
        
        # Check overestimation detection
        assert metrics["over_rate_10"] == 0.6, "Should detect 3/5 over by 10"
        assert metrics["over_rate_20"] == 0.4, "Should detect 2/5 over by 20"
        assert metrics["max_over"] == 25, "Max over should be 25"
    
    def test_compute_risk_metrics_empty_input(self):
        """Test handling of empty inputs."""
        from src.analysis.risk_metrics import compute_risk_metrics_last
        
        metrics = compute_risk_metrics_last([], [])
        
        # Should return NaN values, not crash
        assert np.isnan(metrics["mean_err"])
    
    def test_compute_risk_metrics_torch_tensor(self):
        """Test handling of torch tensors."""
        pytest.importorskip("torch")
        import torch
        from src.analysis.risk_metrics import compute_risk_metrics_last
        
        y_true = torch.tensor([10.0, 20.0, 30.0])
        y_pred = torch.tensor([15.0, 22.0, 28.0])
        
        metrics = compute_risk_metrics_last(y_true, y_pred)
        
        assert isinstance(metrics["mean_err"], float)
        assert np.isfinite(metrics["mean_err"])


class TestCycleBranchIntegration:
    """Test Cycle Branch integration with B2.2 model."""
    
    @pytest.fixture
    def cycle_config(self):
        """Create a test CycleBranchConfig."""
        from src.world_model_training import CycleBranchConfig
        
        return CycleBranchConfig(
            enable=True,
            targets=["T24", "T30", "P30", "T50"],
            lambda_cycle=0.1,
            lambda_theta_smooth=0.05,
            lambda_theta_mono=0.0,
            cycle_ramp_epochs=5,
        )
    
    def test_cycle_branch_components_instantiation(self, cycle_config):
        """Test that cycle branch components can be instantiated."""
        pytest.importorskip("torch")
        import torch
        
        from src.utils.cycle_branch_helper import initialize_cycle_branch
        
        # Mock feature columns (must include required sensors)
        feature_cols = [
            "op1", "op2", "op3",  # Operating settings
            "Sensor1", "Sensor2", "Sensor3", "Sensor4",  # T24=Sensor2, T30=Sensor3, T50=Sensor4
            "Sensor5", "Sensor6", "Sensor7",  # P30=Sensor7
            "Sensor8", "Sensor9", "Sensor10",
        ]
        
        components = initialize_cycle_branch(
            config=cycle_config,
            feature_cols=feature_cols,
            d_model=64,
            num_conditions=7,
            device=torch.device("cpu"),
        )
        
        assert components is not None
        assert components.nominal_head is not None
        assert components.param_head is not None
        assert components.cycle_layer is not None
        assert components.loss_fn is not None
    
    def test_cycle_branch_forward_pass(self, cycle_config):
        """Test forward pass through cycle branch."""
        pytest.importorskip("torch")
        import torch
        
        from src.utils.cycle_branch_helper import (
            initialize_cycle_branch,
            cycle_branch_forward,
        )
        
        device = torch.device("cpu")
        B, T, F = 4, 30, 12  # Batch, Time, Features
        
        feature_cols = [
            "op1", "op2", "op3",
            "Sensor1", "Sensor2", "Sensor3", "Sensor4",
            "Sensor5", "Sensor6", "Sensor7",
            "Sensor8", "Sensor9",
        ]
        
        components = initialize_cycle_branch(
            config=cycle_config,
            feature_cols=feature_cols,
            d_model=64,
            num_conditions=7,
            device=device,
        )
        
        # Create dummy inputs
        X_batch = torch.randn(B, T, F)
        z_t = torch.randn(B, 64)  # Latent from encoder
        cond_ids = torch.zeros(B, dtype=torch.long)
        
        # Forward pass
        cycle_pred, cycle_target, m_t, eta_nom, intermediates = cycle_branch_forward(
            components=components,
            X_batch=X_batch,
            z_t=z_t,
            cond_ids=cond_ids,
            cfg=cycle_config,
            epoch=0,
        )
        
        # Check outputs
        assert cycle_pred.shape == (B, T, 4), f"Expected (B, T, 4), got {cycle_pred.shape}"
        assert cycle_target.shape == (B, T, 4)
        assert m_t.shape == (B, T, 6), f"m_t should be (B, T, 6), got {m_t.shape}"
        assert eta_nom.shape == (B, T, 5)
        assert torch.isfinite(cycle_pred).all(), "cycle_pred contains NaN/Inf"
    
    def test_cycle_branch_backward(self, cycle_config):
        """Test that gradients flow through cycle branch."""
        pytest.importorskip("torch")
        import torch
        
        from src.utils.cycle_branch_helper import (
            initialize_cycle_branch,
            cycle_branch_forward,
            cycle_branch_loss,
        )
        
        device = torch.device("cpu")
        B, T, F = 4, 30, 12
        
        feature_cols = [
            "op1", "op2", "op3",
            "Sensor1", "Sensor2", "Sensor3", "Sensor4",
            "Sensor5", "Sensor6", "Sensor7",
            "Sensor8", "Sensor9",
        ]
        
        components = initialize_cycle_branch(
            config=cycle_config,
            feature_cols=feature_cols,
            d_model=64,
            num_conditions=7,
            device=device,
        )
        
        X_batch = torch.randn(B, T, F)
        z_t = torch.randn(B, 64, requires_grad=True)
        cond_ids = torch.zeros(B, dtype=torch.long)
        
        # Forward
        cycle_pred, cycle_target, m_t, eta_nom, intermediates = cycle_branch_forward(
            components=components,
            X_batch=X_batch,
            z_t=z_t,
            cond_ids=cond_ids,
            cfg=cycle_config,
            epoch=5,
        )
        
        # Compute loss
        loss, metrics = cycle_branch_loss(
            components=components,
            cycle_pred=cycle_pred,
            cycle_target=cycle_target,
            theta_seq=m_t,
            cfg=cycle_config,
            epoch=5,
            num_epochs=10,
            intermediates=intermediates,
        )
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        assert z_t.grad is not None, "z_t should have gradients"
        assert torch.isfinite(z_t.grad).all(), "z_t gradients contain NaN/Inf"
        
        # Check param_head gradients
        for name, param in components.param_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"param_head.{name} should have gradients"


class TestSensorResolver:
    """Test sensor mapping/resolver functionality."""
    
    def test_resolve_cycle_target_cols(self):
        """Test sensor column resolution."""
        from src.utils.sensor_mapping import resolve_cycle_target_cols
        
        feature_cols = [
            "op1", "op2", "op3",
            "Sensor1", "Sensor2", "Sensor3", "Sensor4",
            "Sensor5", "Sensor6", "Sensor7",
        ]
        
        targets = ["T24", "T30", "P30", "T50"]
        
        resolved = resolve_cycle_target_cols(feature_cols, targets)
        
        assert "T24" in resolved
        assert "T30" in resolved
        assert "P30" in resolved
        assert "T50" in resolved
    
    def test_resolve_sensor_indices(self):
        """Test that sensor indices are correctly resolved."""
        from src.utils.sensor_mapping import resolve_sensor_indices
        
        feature_cols = [
            "op1", "op2", "op3",
            "Sensor1", "Sensor2", "Sensor3", "Sensor4",
        ]
        
        resolved_cols = {
            "T24": "Sensor2",
            "T30": "Sensor3",
        }
        
        indices = resolve_sensor_indices(feature_cols, resolved_cols)
        
        assert 4 in indices, "Sensor2 should map to index 4"
        assert 5 in indices, "Sensor3 should map to index 5"


class TestCycleDiagnostics:
    """Test cycle diagnostics functions."""
    
    def test_engine_groups_by_error(self):
        """Test engine grouping by RUL error."""
        from src.analysis.cycle_diagnostics import get_engine_groups_by_error
        
        engine_ids = list(range(1, 51))  # 50 engines
        err_last = {e: (e - 25) * 2 for e in engine_ids}  # Errors from -48 to +48
        
        groups = get_engine_groups_by_error(engine_ids, err_last, n_per_group=10)
        
        assert "worst20_over" in groups
        assert "worst20_under" in groups
        assert "mid20" in groups
        
        # Worst20_over should have highest positive errors
        assert all(err_last[e] > 0 for e in groups["worst20_over"])
        
        # Worst20_under should have most negative errors
        assert all(err_last[e] < 0 for e in groups["worst20_under"])
    
    def test_safe_pearson_constant(self):
        """Test correlation handles constant arrays."""
        from src.analysis.cycle_diagnostics import _safe_pearson
        
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        
        corr = _safe_pearson(x, y)
        assert np.isnan(corr), "Correlation with constant array should be NaN"
    
    def test_safe_pearson_valid(self):
        """Test correlation with valid data."""
        from src.analysis.cycle_diagnostics import _safe_pearson
        
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect positive correlation
        
        corr = _safe_pearson(x, y)
        assert abs(corr - 1.0) < 0.01, f"Expected ~1.0, got {corr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
