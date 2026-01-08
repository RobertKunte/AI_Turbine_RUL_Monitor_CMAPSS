"""
Unit tests for CycleBranchLoss audit fixes.

Tests:
1. Scaled mode with normalization config works correctly
2. Scaled mode without config raises ValueError (fail-fast)
3. Power balance penalty returns device-correct zero
4. Per-sensor metrics use masked loss correctly
"""

import pytest
import torch

from src.losses.cycle_losses import (
    CycleBranchLoss,
    compute_power_balance_penalty,
    compute_cycle_loss_per_sensor,
)


class TestCycleBranchLossScaledMode:
    """Tests for CycleBranchLoss scaled mode normalization."""
    
    def test_scaled_mode_with_config_runs_correctly(self):
        """Scaled mode with mean/std/cond_ids should normalize pred and run."""
        # Setup
        B, T, n_targets = 4, 10, 4
        num_conditions = 3
        
        # Create dummy scaler stats
        cycle_target_mean = torch.randn(num_conditions, n_targets)
        cycle_target_std = torch.abs(torch.randn(num_conditions, n_targets)) + 0.1
        
        loss_fn = CycleBranchLoss(
            lambda_cycle=1.0,
            cycle_target_space="scaled",
            cycle_target_mean=cycle_target_mean,
            cycle_target_std=cycle_target_std,
        )
        
        # Inputs
        cycle_pred = torch.randn(B, T, n_targets) * 100 + 500  # Raw-ish values
        cycle_target = torch.randn(B, T, n_targets)  # Already scaled ~N(0,1)
        theta_seq = torch.ones(B, T, 6) * 0.92
        cond_ids = torch.randint(0, num_conditions, (B,))
        
        # Forward pass should work
        total_loss, components = loss_fn(
            cycle_pred=cycle_pred,
            cycle_target=cycle_target,
            theta_seq=theta_seq,
            cond_ids=cond_ids,
            epoch=0,
        )
        
        # Assertions
        assert torch.isfinite(total_loss), "Loss should be finite"
        assert "cycle_loss" in components
        assert "cycle_loss_T24" in components  # Per-sensor metrics exist
        
    def test_scaled_mode_missing_config_raises_valueerror(self):
        """Scaled mode without mean/std should raise ValueError (fail-fast)."""
        loss_fn = CycleBranchLoss(
            lambda_cycle=1.0,
            cycle_target_space="scaled",
            # No cycle_target_mean/std provided
        )
        
        B, T, n_targets = 4, 10, 4
        cycle_pred = torch.randn(B, T, n_targets)
        cycle_target = torch.randn(B, T, n_targets)
        theta_seq = torch.ones(B, T, 6) * 0.92
        cond_ids = torch.randint(0, 3, (B,))
        
        with pytest.raises(ValueError) as excinfo:
            loss_fn(
                cycle_pred=cycle_pred,
                cycle_target=cycle_target,
                theta_seq=theta_seq,
                cond_ids=cond_ids,
                epoch=0,
            )
        
        assert "cycle_target_space='scaled'" in str(excinfo.value)
        assert "normalization config missing" in str(excinfo.value)
        
    def test_scaled_mode_missing_cond_ids_raises_valueerror(self):
        """Scaled mode with mean/std but no cond_ids should raise ValueError."""
        num_conditions = 3
        n_targets = 4
        
        loss_fn = CycleBranchLoss(
            lambda_cycle=1.0,
            cycle_target_space="scaled",
            cycle_target_mean=torch.randn(num_conditions, n_targets),
            cycle_target_std=torch.ones(num_conditions, n_targets),
        )
        
        B, T = 4, 10
        cycle_pred = torch.randn(B, T, n_targets)
        cycle_target = torch.randn(B, T, n_targets)
        theta_seq = torch.ones(B, T, 6) * 0.92
        
        with pytest.raises(ValueError) as excinfo:
            loss_fn(
                cycle_pred=cycle_pred,
                cycle_target=cycle_target,
                theta_seq=theta_seq,
                cond_ids=None,  # Missing cond_ids
                epoch=0,
            )
        
        assert "cond_ids" in str(excinfo.value)


class TestLambdaThetaPriorConfig:
    """Tests for lambda_theta_prior config plumbing."""
    
    def test_lambda_theta_prior_default_is_0_01(self):
        """Default lambda_theta_prior should be 0.01."""
        loss_fn = CycleBranchLoss()
        assert loss_fn.lambda_theta_prior == 0.01, (
            f"Expected lambda_theta_prior=0.01, got {loss_fn.lambda_theta_prior}"
        )
    
    def test_lambda_theta_prior_custom_value_used(self):
        """Custom lambda_theta_prior should be respected."""
        loss_fn = CycleBranchLoss(lambda_theta_prior=0.05)
        assert loss_fn.lambda_theta_prior == 0.05
        
        loss_fn2 = CycleBranchLoss(lambda_theta_prior=0.001)
        assert loss_fn2.lambda_theta_prior == 0.001


class TestPowerBalancePenaltyDevice:
    """Tests for compute_power_balance_penalty device handling."""
    
    def test_missing_keys_returns_zero_on_correct_device_cpu(self):
        """Missing keys should return 0 on CPU device."""
        work_balance = {
            "W_hpc": torch.tensor([1.0, 2.0]),  # CPU
            # Missing other keys
        }
        result = compute_power_balance_penalty(work_balance)
        
        assert result.device == torch.device("cpu")
        assert result.item() == 0.0
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_missing_keys_returns_zero_on_correct_device_cuda(self):
        """Missing keys should return 0 on CUDA device when input is on CUDA."""
        work_balance = {
            "W_hpc": torch.tensor([1.0, 2.0], device="cuda"),
            # Missing other keys
        }
        result = compute_power_balance_penalty(work_balance)
        
        assert result.device.type == "cuda"
        assert result.item() == 0.0

    def test_empty_dict_returns_zero_on_cpu(self):
        """Empty dict should return 0 on CPU (fallback)."""
        result = compute_power_balance_penalty({})
        
        assert result.device == torch.device("cpu")
        assert result.item() == 0.0


class TestPerSensorMetricsMask:
    """Tests for compute_cycle_loss_per_sensor with mask support."""
    
    def test_per_sensor_with_mask_excludes_masked_values(self):
        """Masked values should be excluded from per-sensor loss."""
        B, T, n_targets = 2, 5, 2
        
        pred = torch.zeros(B, T, n_targets)
        target = torch.ones(B, T, n_targets)
        
        # Mask: only first 3 timesteps valid per sample
        mask = torch.zeros(B, T)
        mask[:, :3] = 1.0
        
        sensor_names = ["T24", "T30"]
        
        result = compute_cycle_loss_per_sensor(
            pred, target, sensor_names,
            loss_type="mse",
            mask=mask,
        )
        
        # Loss should be computed only on valid timesteps
        # pred=0, target=1 -> squared error = 1.0
        # MSE with valid mask should be 1.0
        assert abs(result["T24"] - 1.0) < 0.01
        assert abs(result["T30"] - 1.0) < 0.01
        
    def test_per_sensor_without_mask_uses_all_values(self):
        """Without mask, all values should be included."""
        B, T, n_targets = 2, 5, 2
        
        pred = torch.zeros(B, T, n_targets)
        target = torch.ones(B, T, n_targets)
        
        sensor_names = ["T24", "T30"]
        
        result = compute_cycle_loss_per_sensor(
            pred, target, sensor_names,
            loss_type="mse",
            mask=None,
        )
        
        # All timesteps: pred=0, target=1 -> MSE = 1.0
        assert abs(result["T24"] - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
