# FD004 Phase-1 Health-Head Results Summary

This document summarizes the results from the FD004 Phase-1 hyperparameter sweep for the Multi-Task LSTM with Health Index head.

## Results Table

| Dataset | Experiment                               | λ (health_loss) | mono_late | mono_global | Test RMSE | Test MAE | Test Bias | Test R² | Test NASA Mean |
|---------|------------------------------------------|----------------|-----------|-------------|-----------|----------|-----------|---------|----------------|
| FD004   | fd004_phase1_l0.35_ml0.030_mg0.0030 (RMSE) | 0.35            | 0.03      | 0.003       | 24.69     | 19.41    | -5.82    | 0.67    | 43.98          |
| FD004   | fd004_phase1_l0.30_ml0.030_mg0.0050 (NASA) | 0.30            | 0.03      | 0.005       | 25.58     | 20.85    | -7.66    | 0.65    | 19.29          |

## Configuration Details

### RMSE-Optimized Model
- **Experiment Name**: `fd004_phase1_l0.35_ml0.030_mg0.0030`
- **Health Loss Weight (λ)**: 0.35
- **Late Monotonicity Weight**: 0.03
- **Global Monotonicity Weight**: 0.003
- **RUL Beta (τ)**: 45.0
- **Condition Calibration Weight**: 0.0 (disabled)

**Test Metrics:**
- RMSE: 24.69 cycles
- MAE: 19.41 cycles
- Bias: -5.82 cycles
- R²: 0.67
- NASA Mean: 43.98

### NASA-Optimized Model
- **Experiment Name**: `fd004_phase1_l0.30_ml0.030_mg0.0050`
- **Health Loss Weight (λ)**: 0.30
- **Late Monotonicity Weight**: 0.03
- **Global Monotonicity Weight**: 0.005
- **RUL Beta (τ)**: 45.0
- **Condition Calibration Weight**: 0.0 (disabled)

**Test Metrics:**
- RMSE: 25.58 cycles
- MAE: 20.85 cycles
- Bias: -7.66 cycles
- R²: 0.65
- NASA Mean: 19.29

## Key Observations

1. **RMSE-Optimized Model**: Achieves better pointwise accuracy (lower RMSE/MAE) but higher NASA score, indicating more late predictions.

2. **NASA-Optimized Model**: Achieves significantly lower NASA score (19.29 vs 43.98), indicating better safety performance with fewer late predictions, at the cost of slightly higher RMSE.

3. **Monotonicity Weights**: Both models use the same late monotonicity weight (0.03), but the NASA-optimized model uses a higher global monotonicity weight (0.005 vs 0.003), which helps enforce better overall trajectory behavior.

4. **Health Loss Weight**: The RMSE-optimized model uses a higher λ (0.35), giving more weight to the health loss component, while the NASA-optimized model uses λ=0.30.

## Usage

To train these models:
```bash
python -m experiments.exp_fd004_phase1 --config config/fd004_phase1_rmse.yaml
python -m experiments.exp_fd004_phase1 --config config/fd004_phase1_nasa.yaml
```

To generate Health Index plots:
```bash
python scripts/plot_fd004_phase1_health_index.py --config config/fd004_phase1_rmse.yaml
python scripts/plot_fd004_phase1_health_index.py --config config/fd004_phase1_nasa.yaml
```

