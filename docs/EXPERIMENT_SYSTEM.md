# Experiment System Documentation

This document describes the systematic experiment framework for comparing LSTM vs Transformer encoders across all FD datasets.

## Overview

The experiment system consists of:

1. **Encoder Abstraction** (`src/encoders.py`): Clean interfaces for LSTM and Transformer encoders
2. **Experiment Configs** (`src/experiment_configs.py`): Centralized experiment definitions
3. **Experiment Runner** (`run_experiments.py`): Script to run experiments systematically
4. **Results Aggregation** (`scripts/aggregate_experiment_results.py`): Collect all results into CSV

## Experiment Groups

### Group A: LSTM Baselines (Phase-2 style)
- **Purpose**: Establish LSTM baselines across all datasets
- **Experiments**: `fd001_phase2_lstm_baseline`, `fd002_phase2_lstm_baseline`, `fd003_phase2_lstm_baseline`, `fd004_phase2_lstm_baseline`
- **Configuration**:
  - Encoder: LSTM (hidden_dim=50, num_layers=2, dropout=0.1)
  - Loss: rul_beta=45, health_loss_weight=0.35, mono_late_weight=0.03, mono_global_weight=0.003
  - Condition embeddings: Enabled for FD002/FD004 (7 conditions), disabled for FD001/FD003

### Group B: Transformer Cross-Dataset Baseline
- **Purpose**: Apply successful FD004 Transformer config to all datasets
- **Experiments**: `fd001_phase2_transformer_baseline`, `fd002_phase2_transformer_baseline`, `fd003_phase2_transformer_baseline`, `fd004_phase2_transformer_baseline`
- **Configuration**:
  - Encoder: Transformer (d_model=48, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1)
  - Loss: Same as Group A
  - Condition embeddings: Enabled for FD002/FD004, disabled for FD001/FD003

### Group C: FD004 Transformer Hyperparameter Sweep
- **Purpose**: Fine-tune Transformer on FD004 to reduce NASA score
- **Experiments**:
  - `fd004_phase2_transformer_hi_strong`: Stronger HI/monotonicity (health_loss_weight=0.50, mono_late_weight=0.08, mono_global_weight=0.005)
  - `fd004_phase2_transformer_hi_condcalib`: Condition calibration enabled (hi_condition_calib_weight=0.03)
  - `fd004_phase2_transformer_small_regularized`: Smaller encoder with stronger regularization (d_model=32, dropout=0.2)

## Usage

### Running Experiments

```bash
# Run a specific group
python run_experiments.py --group A
python run_experiments.py --group B
python run_experiments.py --group C

# Run all experiments
python run_experiments.py --group all

# Run specific experiments by name
python run_experiments.py --experiments fd001_phase2_lstm_baseline fd004_phase2_transformer_baseline

# Run all experiments for a dataset
python run_experiments.py --dataset FD004
```

### Aggregating Results

After running experiments, aggregate all results into a CSV:

```bash
python scripts/aggregate_experiment_results.py
python scripts/aggregate_experiment_results.py --output results/my_summary.csv
```

## Results Structure

Each experiment saves results to:
```
results/
  <dataset>/
    <experiment_name>/
      summary.json          # Full experiment results
      eol_full_lstm_best_<name>.pt  # Best model checkpoint
      training_curves_<name>.png    # Training plots
```

The `summary.json` contains:
- Experiment metadata (name, dataset, phase, encoder_type)
- All hyperparameters (encoder, loss, optimizer, training)
- Validation metrics (RMSE, MAE, Bias, R², NASA Mean)
- Test metrics (RMSE, MAE, Bias, R², NASA Mean)

## Expected Results

### FD004 Transformer Baseline (Reference)
- Test RMSE: ~22.0 cycles
- Test NASA Mean: ~10.8
- Test Bias: ~-5.2 cycles

### Comparison Targets
- **LSTM vs Transformer**: Compare across all datasets
- **FD004 Improvement**: Target NASA Mean < 10 (from baseline ~10.8)
- **Cross-Dataset Consistency**: Verify Transformer works well on FD001-FD003

## Adding New Experiments

To add a new experiment:

1. Add a function in `src/experiment_configs.py`:
```python
def get_my_new_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="my_experiment",
        dataset="FD004",
        encoder_type="transformer",
        encoder_kwargs={...},
        loss_params={...},
        optimizer_params={...},
        training_params={...},
        phase_2_params={...},
    )
```

2. Add it to a group or call it directly:
```python
python run_experiments.py --experiments my_experiment
```

## Notes

- All experiments use the same feature engineering pipeline
- Condition embeddings are automatically enabled/disabled based on dataset
- Results are saved incrementally (each experiment saves its own summary.json)
- The aggregation script can be run at any time to update the CSV

