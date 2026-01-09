"""
Unit tests for cycle_target_scaler.py

Tests for:
- extract_cycle_target_stats with legacy StandardScaler format
- extract_cycle_target_stats with new dict format {'features': StandardScaler, ...}
- Index mapping from global to feature space
- Fail-fast on identity stats
- Fail-fast when target is in ops_indices
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from sklearn.preprocessing import StandardScaler

from src.utils.cycle_target_scaler import extract_cycle_target_stats


class TestExtractCycleTargetStatsLegacyFormat:
    """Tests for legacy StandardScaler format."""
    
    def test_legacy_format_extracts_correct_stats(self):
        """Legacy format: scaler_dict[cond] = StandardScaler."""
        scaler = StandardScaler()
        scaler.mean_ = np.array([100, 1528, 1682, 452, 1456, 200, 300])
        scaler.scale_ = np.array([10, 42.5, 58.2, 87.3, 95.1, 20, 30])
        
        scaler_dict = {0: scaler}
        feature_cols = ['f0', 'T24', 'T30', 'P30', 'T50', 'f5', 'f6']
        target_indices = [1, 2, 3, 4]  # T24, T30, P30, T50
        
        mean, std = extract_cycle_target_stats(
            scaler_dict, feature_cols, target_indices, num_conditions=1
        )
        
        assert mean.shape == (1, 4)
        assert std.shape == (1, 4)
        assert torch.allclose(mean[0], torch.tensor([1528, 1682, 452, 1456], dtype=torch.float32), atol=1)
        assert torch.allclose(std[0], torch.tensor([42.5, 58.2, 87.3, 95.1], dtype=torch.float32), atol=0.1)
    
    def test_legacy_format_multiple_conditions(self):
        """Legacy format with multiple conditions."""
        scaler0 = StandardScaler()
        scaler0.mean_ = np.array([1500, 1600, 400, 1400])
        scaler0.scale_ = np.array([40, 50, 80, 90])
        
        scaler1 = StandardScaler()
        scaler1.mean_ = np.array([1550, 1650, 450, 1450])
        scaler1.scale_ = np.array([45, 55, 85, 95])
        
        scaler_dict = {0: scaler0, 1: scaler1}
        feature_cols = ['T24', 'T30', 'P30', 'T50']
        target_indices = [0, 1, 2, 3]
        
        mean, std = extract_cycle_target_stats(
            scaler_dict, feature_cols, target_indices, num_conditions=2
        )
        
        assert mean.shape == (2, 4)
        assert torch.allclose(mean[0], torch.tensor([1500, 1600, 400, 1400], dtype=torch.float32))
        assert torch.allclose(mean[1], torch.tensor([1550, 1650, 450, 1450], dtype=torch.float32))


class TestExtractCycleTargetStatsDictFormat:
    """Tests for new dict format with ops/features separation."""
    
    def test_dict_format_extracts_correct_stats(self):
        """Dict format: scaler_dict[cond] = {'features': StandardScaler, ...}."""
        scaler = StandardScaler()
        # non_ops_indices = [0, 3, 4, 5, 6] -> 5 features in scaler
        # ops_indices = [1, 2] -> ops not in scaler
        # Feature scaler order: [f0, P30, T50, f5, f6] but we want [T24, T30, P30, T50]
        # Wait, need to think through this more carefully...
        
        # Global feature order: [f0, Setting1, Setting2, T24, T30, P30, T50]
        # ops_indices = [1, 2] (Setting1, Setting2)
        # non_ops_indices = [0, 3, 4, 5, 6] (f0, T24, T30, P30, T50)
        # So in the features scaler, order is: index 0->f0, 1->T24, 2->T30, 3->P30, 4->T50
        
        scaler.mean_ = np.array([100, 1528, 1682, 452, 1456])  # 5 non-ops features
        scaler.scale_ = np.array([10, 42.5, 58.2, 87.3, 95.1])
        
        scaler_dict = {
            0: {
                'features': scaler,
                'ops': MagicMock(),  # Not used for cycle targets
                'non_ops_indices': [0, 3, 4, 5, 6],  # Global indices
                'ops_indices': [1, 2],
            }
        }
        
        feature_cols = ['f0', 'Setting1', 'Setting2', 'T24', 'T30', 'P30', 'T50']
        target_indices = [3, 4, 5, 6]  # T24, T30, P30, T50 (global indices)
        
        mean, std = extract_cycle_target_stats(
            scaler_dict, feature_cols, target_indices, num_conditions=1
        )
        
        assert mean.shape == (1, 4)
        # target_indices [3,4,5,6] map to non_ops positions [1,2,3,4]
        assert torch.allclose(mean[0], torch.tensor([1528, 1682, 452, 1456], dtype=torch.float32), atol=1)
        assert torch.allclose(std[0], torch.tensor([42.5, 58.2, 87.3, 95.1], dtype=torch.float32), atol=0.1)
    
    def test_dict_format_raises_when_target_is_ops(self):
        """Should raise ValueError if target_index is in ops_indices."""
        scaler = StandardScaler()
        scaler.mean_ = np.array([100, 1528])
        scaler.scale_ = np.array([10, 42])
        
        scaler_dict = {
            0: {
                'features': scaler,
                'ops': MagicMock(),
                'non_ops_indices': [0, 3],
                'ops_indices': [1, 2],  # Setting1, Setting2
            }
        }
        
        feature_cols = ['f0', 'Setting1', 'Setting2', 'T24']
        target_indices = [1, 3]  # Setting1 (ops!) and T24
        
        with pytest.raises(ValueError, match="is in ops_indices"):
            extract_cycle_target_stats(
                scaler_dict, feature_cols, target_indices, num_conditions=1
            )
    
    def test_dict_format_raises_when_missing_features_key(self):
        """Should raise ValueError if 'features' key is missing."""
        scaler_dict = {
            0: {
                'ops': MagicMock(),
                'non_ops_indices': [0, 1, 2],
            }
        }
        
        with pytest.raises(ValueError, match="missing 'features' key"):
            extract_cycle_target_stats(
                scaler_dict, ['f0', 'f1', 'f2'], [0, 1], num_conditions=1
            )
    
    def test_dict_format_raises_when_missing_non_ops_indices(self):
        """Should raise ValueError if 'non_ops_indices' key is missing."""
        scaler = StandardScaler()
        scaler.mean_ = np.array([100, 200])
        scaler.scale_ = np.array([10, 20])
        
        scaler_dict = {
            0: {
                'features': scaler,
                # Missing 'non_ops_indices'
            }
        }
        
        with pytest.raises(ValueError, match="missing 'non_ops_indices' key"):
            extract_cycle_target_stats(
                scaler_dict, ['f0', 'f1'], [0, 1], num_conditions=1
            )


class TestExtractCycleTargetStatsValidation:
    """Tests for C1 validation checks."""
    
    def test_raises_on_identity_stats(self):
        """Should raise ValueError if stats are identity (mean=0, std=1)."""
        scaler = StandardScaler()
        scaler.mean_ = np.array([0.0, 0.0, 0.0, 0.0])
        scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0])
        
        scaler_dict = {0: scaler}
        
        with pytest.raises(ValueError, match="IDENTITY"):
            extract_cycle_target_stats(
                scaler_dict, ['T24', 'T30', 'P30', 'T50'], [0, 1, 2, 3], num_conditions=1
            )
    
    def test_raises_on_degenerate_std(self):
        """Should raise ValueError if any std is near zero."""
        scaler = StandardScaler()
        scaler.mean_ = np.array([1500, 1600, 400, 1400])
        scaler.scale_ = np.array([40, 0.0, 80, 90])  # Second feature has zero std
        
        scaler_dict = {0: scaler}
        
        with pytest.raises(ValueError, match="Degenerate"):
            extract_cycle_target_stats(
                scaler_dict, ['T24', 'T30', 'P30', 'T50'], [0, 1, 2, 3], num_conditions=1
            )
    
    def test_raises_on_missing_scaler(self):
        """Should raise ValueError if scaler for condition is missing."""
        scaler_dict = {}  # Empty dict
        
        with pytest.raises(ValueError, match="No scaler for cond_id"):
            extract_cycle_target_stats(
                scaler_dict, ['T24', 'T30', 'P30', 'T50'], [0, 1, 2, 3], num_conditions=1
            )


class TestIndexMapping:
    """Tests for global -> feature space index mapping."""
    
    def test_correct_mapping_with_gaps(self):
        """Test mapping when non_ops_indices have gaps (ops in between)."""
        scaler = StandardScaler()
        # Global: [f0, S1, f2, S3, f4, f5]
        # ops_indices = [1, 3]
        # non_ops_indices = [0, 2, 4, 5]
        # Scaler has 4 features in order: f0, f2, f4, f5
        scaler.mean_ = np.array([100, 200, 300, 400])
        scaler.scale_ = np.array([10, 20, 30, 40])
        
        scaler_dict = {
            0: {
                'features': scaler,
                'ops': MagicMock(),
                'non_ops_indices': [0, 2, 4, 5],
                'ops_indices': [1, 3],
            }
        }
        
        feature_cols = ['f0', 'S1', 'f2', 'S3', 'f4', 'f5']
        target_indices = [2, 5]  # f2 and f5 (global indices)
        
        mean, std = extract_cycle_target_stats(
            scaler_dict, feature_cols, target_indices, num_conditions=1
        )
        
        # f2 is at position 1 in scaler (non_ops_indices.index(2) = 1)
        # f5 is at position 3 in scaler (non_ops_indices.index(5) = 3)
        assert torch.allclose(mean[0], torch.tensor([200, 400], dtype=torch.float32))
        assert torch.allclose(std[0], torch.tensor([20, 40], dtype=torch.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
