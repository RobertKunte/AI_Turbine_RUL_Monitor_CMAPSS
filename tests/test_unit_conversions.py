"""
Unit tests for thermodynamics unit conversions and denormalization.

Tests the critical denorm_by_condition() workflow and conversion functions
to ensure physically correct SI unit handling.
"""

import pytest
import numpy as np
from src.utils.unit_conversions import (
    rankine_to_kelvin,
    kelvin_to_rankine,
    psia_to_pascal,
    pascal_to_psia,
    denorm_by_condition,
    ThermoConfig,
    validate_thermo_values_SI,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Temperature Conversion Tests
# ============================================================================

def test_rankine_to_kelvin_scalar():
    """Test Rankine to Kelvin conversion with scalar values."""
    # 0°C = 273.15 K = 491.67 °R
    assert abs(rankine_to_kelvin(491.67) - 273.15) < 0.01

    # 100°C = 373.15 K = 671.67 °R
    assert abs(rankine_to_kelvin(671.67) - 373.15) < 0.01

    # Typical turbine temperature: 1500 °R ≈ 833.33 K
    assert abs(rankine_to_kelvin(1500.0) - 833.33) < 0.01


def test_rankine_to_kelvin_array():
    """Test Rankine to Kelvin conversion with numpy arrays."""
    T_R = np.array([500.0, 600.0, 700.0, 800.0])
    T_K = rankine_to_kelvin(T_R)

    # Check shape preserved
    assert T_K.shape == T_R.shape

    # Check values
    expected_K = T_R * (5.0 / 9.0)
    np.testing.assert_allclose(T_K, expected_K, rtol=1e-6)


def test_kelvin_to_rankine_roundtrip():
    """Test Kelvin ↔ Rankine roundtrip conversion."""
    T_R_original = np.array([500, 600, 700, 800, 900, 1000])
    T_K = rankine_to_kelvin(T_R_original)
    T_R_back = kelvin_to_rankine(T_K)

    np.testing.assert_allclose(T_R_back, T_R_original, rtol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_rankine_to_kelvin_torch():
    """Test Rankine to Kelvin conversion with torch tensors."""
    T_R = torch.tensor([500.0, 600.0, 700.0])
    T_K = rankine_to_kelvin(T_R)

    # Check type preserved
    assert isinstance(T_K, torch.Tensor)

    # Check values
    expected_K = T_R * (5.0 / 9.0)
    torch.testing.assert_close(T_K, expected_K, rtol=1e-6, atol=1e-6)


# ============================================================================
# Pressure Conversion Tests
# ============================================================================

def test_psia_to_pascal_scalar():
    """Test PSIA to Pascal conversion with scalar values."""
    # 1 PSIA = 6894.757 Pa
    assert abs(psia_to_pascal(1.0) - 6894.757) < 0.1

    # 14.7 PSIA ≈ 1 atm ≈ 101325 Pa
    assert abs(psia_to_pascal(14.7) - 101352.93) < 1.0

    # Typical turbine pressure: 500 PSIA
    assert abs(psia_to_pascal(500.0) - 3447378.65) < 1.0


def test_psia_to_pascal_array():
    """Test PSIA to Pascal conversion with numpy arrays."""
    P_psia = np.array([10.0, 20.0, 30.0, 40.0])
    P_pa = psia_to_pascal(P_psia)

    # Check shape preserved
    assert P_pa.shape == P_psia.shape

    # Check values
    expected_pa = P_psia * 6894.757293168
    np.testing.assert_allclose(P_pa, expected_pa, rtol=1e-6)


def test_pascal_to_psia_roundtrip():
    """Test Pascal ↔ PSIA roundtrip conversion."""
    P_psia_original = np.array([10, 20, 30, 40, 50, 100])
    P_pa = psia_to_pascal(P_psia_original)
    P_psia_back = pascal_to_psia(P_pa)

    np.testing.assert_allclose(P_psia_back, P_psia_original, rtol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_psia_to_pascal_torch():
    """Test PSIA to Pascal conversion with torch tensors."""
    P_psia = torch.tensor([10.0, 20.0, 30.0])
    P_pa = psia_to_pascal(P_psia)

    # Check type preserved
    assert isinstance(P_pa, torch.Tensor)

    # Check values
    expected_pa = P_psia * 6894.757293168
    torch.testing.assert_close(P_pa, expected_pa, rtol=1e-6, atol=1e-6)


# ============================================================================
# Denormalization Tests (CRITICAL)
# ============================================================================

def test_denorm_by_condition_2d():
    """Test denormalization with 2D arrays (B, D)."""
    # Synthetic scaled data
    x_scaled = np.array([
        [0.5, -0.5, 1.0, -1.0],  # Sample 0, condition 0
        [0.0, 0.0, 0.5, -0.5],   # Sample 1, condition 0
        [1.0, 1.0, -1.0, -1.0],  # Sample 2, condition 1
    ])

    # Condition-wise statistics
    mean_by_cond = np.array([
        [1500.0, 1600.0, 500.0, 1400.0],  # Condition 0 stats
        [1520.0, 1620.0, 520.0, 1420.0],  # Condition 1 stats
    ])

    std_by_cond = np.array([
        [100.0, 120.0, 50.0, 80.0],   # Condition 0 std
        [110.0, 130.0, 55.0, 85.0],   # Condition 1 std
    ])

    cond_ids = np.array([0, 0, 1])

    # Denormalize
    x_raw = denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids)

    # Check shape preserved
    assert x_raw.shape == x_scaled.shape

    # Check sample 0 (condition 0)
    expected_0 = x_scaled[0] * std_by_cond[0] + mean_by_cond[0]
    np.testing.assert_allclose(x_raw[0], expected_0, rtol=1e-5)

    # Check sample 2 (condition 1)
    expected_2 = x_scaled[2] * std_by_cond[1] + mean_by_cond[1]
    np.testing.assert_allclose(x_raw[2], expected_2, rtol=1e-5)


def test_denorm_by_condition_3d():
    """Test denormalization with 3D arrays (B, T, D)."""
    batch_size = 4
    seq_len = 10
    n_features = 4

    # Synthetic scaled data
    x_scaled = np.random.randn(batch_size, seq_len, n_features)

    # Condition-wise statistics
    mean_by_cond = np.array([
        [1500.0, 1600.0, 500.0, 1400.0],
        [1520.0, 1620.0, 520.0, 1420.0],
    ])

    std_by_cond = np.array([
        [100.0, 120.0, 50.0, 80.0],
        [110.0, 130.0, 55.0, 85.0],
    ])

    cond_ids = np.array([0, 0, 1, 1])

    # Denormalize
    x_raw = denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids)

    # Check shape preserved
    assert x_raw.shape == x_scaled.shape

    # Manually verify sample 0, timestep 0
    expected_00 = x_scaled[0, 0] * std_by_cond[0] + mean_by_cond[0]
    np.testing.assert_allclose(x_raw[0, 0], expected_00, rtol=1e-5)


def test_denorm_by_condition_validates_cond_ids():
    """Test that denorm_by_condition validates condition ID range."""
    x_scaled = np.array([[0.5, -0.5]])
    mean_by_cond = np.array([[1500.0, 1600.0]])
    std_by_cond = np.array([[100.0, 120.0]])

    # Invalid condition ID (out of range)
    cond_ids = np.array([5])  # Only condition 0 exists

    with pytest.raises(ValueError, match="cond_ids out of range"):
        denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_denorm_by_condition_torch():
    """Test denormalization with torch tensors."""
    x_scaled = torch.randn(4, 10, 4)
    mean_by_cond = torch.tensor([
        [1500.0, 1600.0, 500.0, 1400.0],
        [1520.0, 1620.0, 520.0, 1420.0],
    ])
    std_by_cond = torch.tensor([
        [100.0, 120.0, 50.0, 80.0],
        [110.0, 130.0, 55.0, 85.0],
    ])
    cond_ids = torch.tensor([0, 0, 1, 1])

    # Denormalize
    x_raw = denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids)

    # Check type preserved
    assert isinstance(x_raw, torch.Tensor)
    assert x_raw.shape == x_scaled.shape
    assert x_raw.device == x_scaled.device
    assert x_raw.dtype == x_scaled.dtype


def test_denorm_then_convert_workflow():
    """Test the CRITICAL workflow: denormalize → convert to SI."""
    # Simulate StandardScaled cycle predictions
    cycle_pred_scaled = np.array([
        [0.0, 0.5, -0.5, 1.0],  # Sample 0 (mean-centered)
    ])

    # Scaler stats (condition 0)
    mean_by_cond = np.array([[1500.0, 1600.0, 500.0, 1400.0]])  # °R, °R, PSIA, °R
    std_by_cond = np.array([[100.0, 120.0, 50.0, 80.0]])
    cond_ids = np.array([0])

    # Step 1: Denormalize to raw imperial units
    cycle_pred_raw = denorm_by_condition(
        cycle_pred_scaled, mean_by_cond, std_by_cond, cond_ids
    )

    # Expected raw values
    expected_raw = np.array([[1500.0, 1660.0, 475.0, 1480.0]])
    np.testing.assert_allclose(cycle_pred_raw, expected_raw, rtol=1e-5)

    # Step 2: Convert to SI
    # Temperatures: indices 0, 1, 3
    temps_K = rankine_to_kelvin(cycle_pred_raw[:, [0, 1, 3]])
    expected_temps_K = expected_raw[:, [0, 1, 3]] * (5.0 / 9.0)
    np.testing.assert_allclose(temps_K, expected_temps_K, rtol=1e-5)

    # Pressure: index 2
    pressure_Pa = psia_to_pascal(cycle_pred_raw[:, 2])
    expected_pressure_Pa = expected_raw[:, 2] * 6894.757293168
    np.testing.assert_allclose(pressure_Pa, expected_pressure_Pa, rtol=1e-5)

    # Verify SI values are physically plausible
    assert np.all((temps_K > 200) & (temps_K < 2000))  # Reasonable turbine range
    assert np.all((pressure_Pa > 1e4) & (pressure_Pa < 1e8))  # 10 kPa to 100 MPa


# ============================================================================
# ThermoConfig Tests
# ============================================================================

def test_thermo_config_defaults():
    """Test ThermoConfig default values."""
    config = ThermoConfig()

    # Check sensor units
    assert config.sensor_units["T24"] == "RANKINE"
    assert config.sensor_units["P30"] == "PSIA"

    # Check bounds
    assert config.temp_bounds_K == (200.0, 2000.0)
    assert config.pressure_bounds_Pa == (1e4, 1e8)

    # Check thresholds
    assert config.warn_if_viol_frac == 0.01
    assert config.fail_if_viol_frac == 0.05
    assert config.strict_mode is False


def test_thermo_config_custom():
    """Test ThermoConfig with custom values."""
    config = ThermoConfig(
        temp_bounds_K=(300.0, 1800.0),
        pressure_bounds_Pa=(5e4, 5e7),
        strict_mode=True,
    )

    assert config.temp_bounds_K == (300.0, 1800.0)
    assert config.pressure_bounds_Pa == (5e4, 5e7)
    assert config.strict_mode is True


# ============================================================================
# Validation Tests
# ============================================================================

def test_validate_thermo_values_all_valid():
    """Test validation with all values within bounds."""
    temps_K = np.array([300, 400, 500, 600, 700, 800])
    pressures_Pa = np.array([1e5, 2e5, 3e5, 4e5, 5e5, 6e5])

    config = ThermoConfig(
        temp_bounds_K=(200, 2000),
        pressure_bounds_Pa=(1e4, 1e8),
    )

    result = validate_thermo_values_SI(temps_K, pressures_Pa, config)

    # No violations
    assert result["violations"]["temp"] == 0
    assert result["violations"]["pressure"] == 0
    assert result["violation_fractions"]["temp"] == 0.0
    assert result["violation_fractions"]["pressure"] == 0.0
    assert result["should_fail"] is False
    assert len(result["warnings"]) == 0


def test_validate_thermo_values_temp_violations():
    """Test validation with temperature violations."""
    temps_K = np.array([100, 300, 400, 500, 3000])  # 2/5 violations (100K, 3000K)
    pressures_Pa = np.array([1e5, 2e5, 3e5, 4e5, 5e5])

    config = ThermoConfig(
        temp_bounds_K=(200, 2000),
        pressure_bounds_Pa=(1e4, 1e8),
        warn_if_viol_frac=0.01,
    )

    result = validate_thermo_values_SI(temps_K, pressures_Pa, config)

    # Check violations
    assert result["violations"]["temp"] == 2
    assert result["violation_fractions"]["temp"] == 0.4  # 2/5
    assert result["should_fail"] is True  # > 5% threshold
    assert len(result["warnings"]) > 0


def test_validate_thermo_values_pressure_violations():
    """Test validation with pressure violations."""
    temps_K = np.array([300, 400, 500, 600])
    pressures_Pa = np.array([1e3, 1e5, 2e5, 1e9])  # 2/4 violations

    config = ThermoConfig(
        temp_bounds_K=(200, 2000),
        pressure_bounds_Pa=(1e4, 1e8),
    )

    result = validate_thermo_values_SI(temps_K, pressures_Pa, config)

    # Check violations
    assert result["violations"]["pressure"] == 2
    assert result["violation_fractions"]["pressure"] == 0.5  # 2/4
    assert result["should_fail"] is True


def test_validate_thermo_values_strict_mode_raises():
    """Test that strict_mode raises on violations."""
    temps_K = np.array([100, 300, 400, 3000])  # 2/4 violations
    pressures_Pa = np.array([1e5, 2e5, 3e5, 4e5])

    config = ThermoConfig(
        temp_bounds_K=(200, 2000),
        strict_mode=True,
        fail_if_viol_frac=0.05,
    )

    with pytest.raises(ValueError, match="violate physical bounds"):
        validate_thermo_values_SI(temps_K, pressures_Pa, config)


# ============================================================================
# Integration Test: Full Pipeline
# ============================================================================

def test_full_pipeline_cmapss_like_data():
    """Integration test with CMAPSS-like synthetic data."""
    # Simulate a batch of cycle predictions (StandardScaled)
    batch_size = 32
    cycle_pred_scaled = np.random.randn(batch_size, 4)  # 4 sensors: T24, T30, P30, T50

    # Scaler stats for 7 conditions (CMAPSS FD004)
    num_conditions = 7
    mean_by_cond = np.random.uniform(1400, 1600, (num_conditions, 4))
    mean_by_cond[:, 2] = np.random.uniform(400, 600, num_conditions)  # P30 in PSIA

    std_by_cond = np.random.uniform(80, 120, (num_conditions, 4))
    std_by_cond[:, 2] = np.random.uniform(40, 60, num_conditions)  # P30 std

    cond_ids = np.random.randint(0, num_conditions, batch_size)

    # Step 1: Denormalize
    cycle_pred_raw = denorm_by_condition(
        cycle_pred_scaled, mean_by_cond, std_by_cond, cond_ids
    )

    # Verify raw values are in reasonable ranges
    assert np.all(cycle_pred_raw[:, [0, 1, 3]] > 1000)  # Temps in °R
    assert np.all(cycle_pred_raw[:, [0, 1, 3]] < 2000)
    assert np.all(cycle_pred_raw[:, 2] > 200)  # Pressure in PSIA
    assert np.all(cycle_pred_raw[:, 2] < 800)

    # Step 2: Convert to SI
    temps_K = rankine_to_kelvin(cycle_pred_raw[:, [0, 1, 3]].flatten())
    pressures_Pa = psia_to_pascal(cycle_pred_raw[:, 2])

    # Step 3: Validate
    config = ThermoConfig()
    result = validate_thermo_values_SI(temps_K, pressures_Pa, config)

    # Should have no violations for reasonable synthetic data
    assert result["violation_fractions"]["temp"] < 0.01
    assert result["violation_fractions"]["pressure"] < 0.01
    assert result["should_fail"] is False


# ============================================================================
# Anti-pattern Test (should catch bugs)
# ============================================================================

def test_antipattern_convert_scaled_directly():
    """Test that converting scaled values directly produces nonsense."""
    # Scaled values (mean ≈ 0)
    cycle_pred_scaled = np.array([[0.0, 0.5, -0.5, 1.0]])

    # ❌ WRONG: Convert scaled values directly to SI
    temps_K_wrong = rankine_to_kelvin(cycle_pred_scaled[:, [0, 1, 3]])

    # This produces nonsense: temps near 0K!
    assert np.any(temps_K_wrong < 100)  # Physically impossible

    # ✅ CORRECT: Denormalize first
    mean_by_cond = np.array([[1500.0, 1600.0, 500.0, 1400.0]])
    std_by_cond = np.array([[100.0, 120.0, 50.0, 80.0]])
    cond_ids = np.array([0])

    cycle_pred_raw = denorm_by_condition(
        cycle_pred_scaled, mean_by_cond, std_by_cond, cond_ids
    )
    temps_K_correct = rankine_to_kelvin(cycle_pred_raw[:, [0, 1, 3]])

    # Now we get physically plausible values
    assert np.all(temps_K_correct > 200)
    assert np.all(temps_K_correct < 2000)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
