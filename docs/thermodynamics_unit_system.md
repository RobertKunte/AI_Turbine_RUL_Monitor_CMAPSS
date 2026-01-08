# Thermodynamics Unit System

## Overview

The C-MAPSS turbine RUL monitor uses a hybrid unit system optimized for both machine learning and physics-based modeling:

- **Data layer**: Imperial units (°R, PSIA) as provided by NASA C-MAPSS dataset
- **ML layer**: StandardScaler Z-normalization for encoder features
- **Physics layer**: Raw imperial units (°R, PSIA) for CycleLayerMVP thermodynamic calculations
- **Validation layer**: SI units (K, Pa) for physical bounds checking

**Critical Requirement**: Always denormalize StandardScaled values before converting to SI units!

---

## Unit Mappings

### Temperature Sensors

| Sensor | Description | Raw Unit | Typical Range (°R) | SI Range (K) |
|--------|-------------|----------|-------------------|--------------|
| T2 (Sensor1) | Fan inlet temperature | °R | 518-550 | 288-306 |
| T24 (Sensor2) | LPC outlet temperature | °R | 640-680 | 356-378 |
| T30 (Sensor3) | HPC outlet temperature | °R | 1570-1620 | 872-900 |
| T50 (Sensor4) | LPT outlet temperature | °R | 1370-1420 | 761-789 |

**Conversion Formula**: `K = °R × 5/9`

**Example**:
```python
from src.utils.unit_conversions import rankine_to_kelvin

T_R = 1500.0  # °R
T_K = rankine_to_kelvin(T_R)  # 833.33 K
```

### Pressure Sensors

| Sensor | Description | Raw Unit | Typical Range (PSIA) | SI Range (Pa) |
|--------|-------------|----------|---------------------|---------------|
| P2 (Sensor5) | Fan inlet pressure | PSIA | 14.6-14.7 | 100,680-101,352 |
| P15 (Sensor6) | Bypass duct pressure | PSIA | 21-23 | 144,789-158,579 |
| P30 (Sensor7) | HPC outlet pressure | PSIA | 550-555 | 3,792,116-3,826,610 |
| Ps30 (Sensor8) | HPC outlet static pressure | PSIA | 47-48 | 324,053-330,948 |

**Conversion Formula**: `Pa = PSIA × 6894.757293168`

**Example**:
```python
from src.utils.unit_conversions import psia_to_pascal

P_psia = 500.0  # PSIA
P_pa = psia_to_pascal(P_psia)  # 3,447,379 Pa (≈3.45 MPa)
```

### Operating Settings

| Setting | Description | Raw Unit | Physical Range | Normalized Range |
|---------|-------------|----------|----------------|------------------|
| Setting1 (TRA) | Throttle resolver angle | degrees | 0-100 | [0, 1] |
| Setting2 (Alt) | Altitude | feet | 0-42,000 | [0, 1] |
| Setting3 (Mach) | Mach number | dimensionless | 0-0.9 | [0, 1] |

**Normalization**: Operating settings use `MinMaxScaler(0, 1)` with physical ranges, NOT StandardScaler!

**Why this matters**: CycleLayerMVP uses ops_t values directly in thermodynamic calculations. Mach numbers must be in [0, 1] range for physically meaningful compressor/turbine modeling.

---

## Data Flow Pipeline

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Raw C-MAPSS Data (°R, PSIA, degrees, feet)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Separate Scaling Paths                                       │
│    ├─ Operating settings → MinMaxScaler [0,1]                  │
│    ├─ ML features → StandardScaler (Z-score, mean≈0, std≈1)   │
│    └─ Cycle targets → Condition-wise StandardScaler            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Model Forward Pass                                           │
│    ├─ Encoder(ML features Z-normalized) → z_t embedding        │
│    └─ CycleLayer(ops [0,1], z_t) → cycle_pred (raw °R, PSIA)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Loss Computation                                             │
│    └─ Normalize cycle_pred to match scaled targets             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Thermo Sanity Check (Validation)                            │
│    ├─ Denormalize cycle_pred to raw (°R, PSIA)                │
│    ├─ Convert to SI (K, Pa)                                    │
│    ├─ Validate physical bounds                                 │
│    └─ Save diagnostics to JSON                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Workflow: Denormalize → Convert to SI

**❌ WRONG - DO NOT DO THIS:**
```python
# WRONG: Converting StandardScaled values directly to SI
cycle_pred_scaled  # Mean≈0, Std≈1
cycle_pred_K = rankine_to_kelvin(cycle_pred_scaled)  # ❌ NONSENSE!
# This converts ≈0 to ≈0K which is physically impossible!
```

**✅ CORRECT - ALWAYS USE THIS:**
```python
from src.utils.unit_conversions import denorm_by_condition, rankine_to_kelvin

# Step 1: Denormalize to raw imperial units
cycle_pred_scaled  # (B, T, 4) StandardScaler output, mean≈0, std≈1
cycle_pred_raw = denorm_by_condition(
    cycle_pred_scaled,
    mean_by_cond,  # (7, 4) condition-wise means
    std_by_cond,   # (7, 4) condition-wise stds
    cond_ids       # (B,) condition IDs per sample
)  # Now in raw CMAPSS units: °R, PSIA

# Step 2: Convert to SI for validation
cycle_pred_K = rankine_to_kelvin(cycle_pred_raw[..., temp_indices])  # K
cycle_pred_Pa = psia_to_pascal(cycle_pred_raw[..., pressure_indices])  # Pa

# Step 3: Validate
from src.utils.unit_conversions import validate_thermo_values_SI, ThermoConfig
config = ThermoConfig()
result = validate_thermo_values_SI(cycle_pred_K, cycle_pred_Pa, config)
```

---

## Physical Bounds (Validation)

### SI Unit Bounds

| Quantity | SI Unit | Lower Bound | Upper Bound | Rationale |
|----------|---------|-------------|-------------|-----------|
| Temperature | K | 200 | 2000 | Ambient to max combustion temp |
| Pressure | Pa | 10,000 (10 kPa) | 100,000,000 (100 MPa) | Ambient to max combustion pressure |
| Compressor Efficiency | - | 0.70 | 0.98 | Typical turbomachinery range |
| Turbine Efficiency | - | 0.70 | 0.98 | Typical turbomachinery range |
| Theta (degradation) | - | 0.83 | 1.00 | Health degradation multipliers |

### Derived Checks

- **Pressure ratio**: `PR = P_out / P_in > 1.0` (compressor stages must increase pressure)
- **Temperature rise**: `ΔT > 0` across compressor stages
- **Theta saturation**: `<5%` of values near bounds (0.83 or 1.00)
- **Operating settings**: `ops_t ∈ [0, 1]` for Mach/altitude/TRA

---

## Configuration

### ThermoConfig

Use `ThermoConfig` to configure unit mappings and validation thresholds:

```python
from src.utils.unit_conversions import ThermoConfig

config = ThermoConfig(
    # Sensor unit mappings
    sensor_units={
        "T24": "RANKINE",
        "T30": "RANKINE",
        "P30": "PSIA",
        "T50": "RANKINE",
        "T2": "RANKINE",
        "P2": "PSIA",
        "P15": "PSIA",
    },

    # Physical bounds (SI units)
    temp_bounds_K=(200.0, 2000.0),
    pressure_bounds_Pa=(1e4, 1e8),
    efficiency_bounds=(0.70, 0.98),
    theta_bounds=(0.83, 1.00),

    # Fail-fast thresholds
    warn_if_viol_frac=0.01,  # Warn if >1% violations
    fail_if_viol_frac=0.05,  # Fail if >5% violations
    strict_mode=False,        # Set True to raise on failures

    # Operating settings physical ranges
    TRA_range=(0.0, 100.0),           # degrees
    altitude_range_ft=(0.0, 42000.0), # feet
    mach_range=(0.0, 0.9),            # dimensionless
)
```

### Fail-Fast Thresholds

- **`warn_if_viol_frac=0.01`**: Print warning if >1% of values violate bounds
- **`fail_if_viol_frac=0.05`**: Flag as failed if >5% violate
- **`strict_mode=False`**: When True, raises `ValueError` on failures (stops training)

**Recommendation**: Keep `strict_mode=False` during training to log issues without crashing.

---

## Diagnostics

### Thermo Sanity Reports

Thermo sanity checks run at epochs **[0, 2, 5, 10, 20, 50, 100]** and at test time, generating comprehensive JSON reports:

```
results/<dataset>/<experiment>/diagnostics/thermo_sanity/
├── thermo_sanity_epoch000.json  # First epoch (critical for unit verification)
├── thermo_sanity_epoch002.json
├── thermo_sanity_epoch005.json
├── thermo_sanity_epoch010.json
├── ...
├── thermo_sanity_epoch999.json  # Test phase
└── thermo_sanity_latest.json    # Latest report
```

### Report Structure

```json
{
  "metadata": {
    "epoch": 0,
    "step": 0,
    "timestamp": "2026-01-08T12:34:56",
    "num_samples": 32
  },
  "theta_health": {
    "global": {
      "saturation_frac": [0.02, 0.01, 0.03, 0.02, 0.01, 0.00],
      "delta_l1_mean": 0.045
    },
    "saturation_frac_total": 0.015
  },
  "cycle_prediction_health": {
    "per_sensor": {
      "T24": {
        "pred_raw_mean": 1520.5,
        "pred_raw_std": 45.2,
        "pred_raw_min": 1420.0,
        "pred_raw_max": 1650.0,
        "pred_SI_K_mean": 844.7,
        "pred_SI_K_std": 25.1,
        "pred_SI_K_min": 788.9,
        "pred_SI_K_max": 916.7,
        "pred_violations_frac": 0.0,
        "target_violations_frac": 0.0
      },
      "P30": {
        "pred_raw_mean": 520.3,
        "pred_SI_Pa_mean": 3586890.0,
        "pred_violations_frac": 0.0
      }
    },
    "overall_violation_frac": 0.0
  },
  "warnings": [],
  "fail_fast_flags": {
    "theta_saturated": false,
    "physics_violation": false
  }
}
```

### Console Output

During training, compact summaries are printed:

```
[THERMO SANITY] Epoch 0 Summary:
  Theta saturation: 1.50%
  Overall physical violations: 0.00%
  Warnings: 0
```

---

## Usage Examples

### Example 1: Validate Cycle Predictions

```python
from src.utils.unit_conversions import (
    denorm_by_condition,
    rankine_to_kelvin,
    psia_to_pascal,
    validate_thermo_values_SI,
    ThermoConfig,
)

# Get cycle predictions from model (scaled)
cycle_pred_scaled = model_output["cycle_pred"]  # (B, T, 4)

# Get scaler stats
mean_by_cond = scaler_dict["mean"]  # (7, 4)
std_by_cond = scaler_dict["std"]    # (7, 4)
cond_ids = batch["cond_ids"]        # (B,)

# Step 1: Denormalize
cycle_pred_raw = denorm_by_condition(
    cycle_pred_scaled, mean_by_cond, std_by_cond, cond_ids
)

# Step 2: Convert to SI
sensor_names = ["T24", "T30", "P30", "T50"]
temp_indices = [0, 1, 3]  # T24, T30, T50
pressure_indices = [2]     # P30

temps_K = rankine_to_kelvin(cycle_pred_raw[..., temp_indices])
pressures_Pa = psia_to_pascal(cycle_pred_raw[..., pressure_indices])

# Step 3: Validate
config = ThermoConfig()
result = validate_thermo_values_SI(temps_K.flatten(), pressures_Pa.flatten(), config)

# Check results
if result["should_fail"]:
    print(f"⚠️ Physical violations detected: {result['warnings']}")
```

### Example 2: Generate Thermo Sanity Report

```python
from src.analysis.thermo_sanity import (
    compute_thermo_sanity_report,
    save_thermo_sanity_report,
    print_thermo_sanity_summary,
)
from src.utils.unit_conversions import ThermoConfig

# Configure thermo validation
config = ThermoConfig(strict_mode=False)

# Compute report
report = compute_thermo_sanity_report(
    cycle_pred=cycle_pred.cpu().numpy(),      # Raw imperial units
    cycle_target=cycle_target.cpu().numpy(),
    m_t=m_t.cpu().numpy(),
    eta_nom=eta_nom.cpu().numpy(),
    cond_ids=cond_ids.cpu().numpy(),
    scaler_stats=(mean_by_cond, std_by_cond),
    sensor_names=["T24", "T30", "P30", "T50"],
    epoch=epoch,
    step=global_step,
    config=config,
)

# Save to JSON
save_thermo_sanity_report(report, results_dir, epoch)

# Print summary
print_thermo_sanity_summary(report)
```

### Example 3: Unit Tests

```python
import pytest
from src.utils.unit_conversions import rankine_to_kelvin, psia_to_pascal

def test_temperature_conversion():
    """Test Rankine to Kelvin conversion."""
    # 0°C = 273.15 K = 491.67 °R
    assert abs(rankine_to_kelvin(491.67) - 273.15) < 0.01

    # Typical turbine: 1500 °R ≈ 833.33 K
    assert abs(rankine_to_kelvin(1500.0) - 833.33) < 0.01

def test_pressure_conversion():
    """Test PSIA to Pascal conversion."""
    # 1 atm ≈ 14.7 PSIA ≈ 101,353 Pa
    assert abs(psia_to_pascal(14.7) - 101352.93) < 1.0

# Run tests
pytest.main([__file__, "-v"])
```

---

## Troubleshooting

### Issue 1: "pred_violations_frac > 50% - values out of bounds"

**Cause**: Predictions were likely not denormalized before SI conversion.

**Fix**: Ensure you call `denorm_by_condition()` before `rankine_to_kelvin()` or `psia_to_pascal()`.

```python
# ✅ Correct
cycle_pred_raw = denorm_by_condition(cycle_pred_scaled, mean, std, cond_ids)
temps_K = rankine_to_kelvin(cycle_pred_raw)

# ❌ Wrong
temps_K = rankine_to_kelvin(cycle_pred_scaled)  # Will produce nonsense!
```

### Issue 2: "ops_t values outside [0,1] range"

**Cause**: Operating settings were StandardScaled instead of MinMaxScaled.

**Fix**: Check that operating settings use `MinMaxScaler(0, 1)` in [world_model_training_v3.py](../src/world_model_training_v3.py:436-463).

```python
# ✅ Correct
scaler_ops = MinMaxScaler(feature_range=(0, 1))
scaler_ops.fit(X_train_ops)

# ❌ Wrong
scaler = StandardScaler()
scaler.fit(X_train_all)  # Don't mix ops with other features!
```

### Issue 3: "Theta saturation >10%"

**Cause**: Theta values (degradation multipliers) are saturating at bounds (0.83 or 1.00).

**Possible causes**:
- Learning rate too high
- Insufficient regularization
- Physics priors too strong

**Fix**: Check theta health in diagnostics and adjust hyperparameters if needed.

---

## API Reference

### Unit Conversion Functions

#### `rankine_to_kelvin(T_R)`
Convert Rankine to Kelvin.
- **Args**: `T_R` (float, np.ndarray, or torch.Tensor) - Temperature in °R
- **Returns**: Temperature in K
- **Formula**: `K = °R × 5/9`

#### `kelvin_to_rankine(T_K)`
Convert Kelvin to Rankine.
- **Args**: `T_K` (float, np.ndarray, or torch.Tensor) - Temperature in K
- **Returns**: Temperature in °R
- **Formula**: `°R = K × 9/5`

#### `psia_to_pascal(P_psia)`
Convert PSIA to Pascal.
- **Args**: `P_psia` (float, np.ndarray, or torch.Tensor) - Pressure in PSIA
- **Returns**: Pressure in Pa
- **Formula**: `Pa = PSIA × 6894.757293168`

#### `pascal_to_psia(P_pa)`
Convert Pascal to PSIA.
- **Args**: `P_pa` (float, np.ndarray, or torch.Tensor) - Pressure in Pa
- **Returns**: Pressure in PSIA
- **Formula**: `PSIA = Pa / 6894.757293168`

### Denormalization Function

#### `denorm_by_condition(x_scaled, mean_by_cond, std_by_cond, cond_ids, eps=1e-6)`
**CRITICAL**: Denormalize StandardScaled values using condition-specific statistics.

- **Args**:
  - `x_scaled` (np.ndarray or torch.Tensor): Scaled values, shape (B, T, D) or (B, D)
  - `mean_by_cond` (np.ndarray or torch.Tensor): Mean per condition, shape (C, D)
  - `std_by_cond` (np.ndarray or torch.Tensor): Std per condition, shape (C, D)
  - `cond_ids` (np.ndarray or torch.Tensor): Condition IDs, shape (B,)
  - `eps` (float): Numerical stability constant
- **Returns**: Denormalized values in original units (e.g., °R, PSIA)
- **Formula**: `x_raw = x_scaled × std[cond_id] + mean[cond_id]`

**Must be called BEFORE converting to SI units!**

### Validation Function

#### `validate_thermo_values_SI(temps_K, pressures_Pa, config, context="thermo_validation")`
Validate thermodynamic values against physical bounds.

- **Args**:
  - `temps_K` (np.ndarray): Temperatures in Kelvin
  - `pressures_Pa` (np.ndarray): Pressures in Pascal
  - `config` (ThermoConfig): Validation configuration
  - `context` (str): Description for error messages
- **Returns**: Dict with violations, warnings, and should_fail flag

---

## References

- [NASA C-MAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [CycleLayerMVP Implementation](../src/models/physics/cycle_layer_mvp.py)
- [Unit Conversion Implementation](../src/utils/unit_conversions.py)
- [Thermo Sanity Implementation](../src/analysis/thermo_sanity.py)
- [Training Integration](../src/world_model_training_v3.py)

---

## Changelog

- **2026-01-08**: Initial documentation created
  - Added comprehensive unit mapping tables
  - Documented critical denormalize-first pattern
  - Added troubleshooting guide
  - Included API reference
