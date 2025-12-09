# Phase-1 Cross-Dataset Baseline

This document describes the Phase-1 Health-Head cross-dataset baseline experiment, which applies the best hyperparameters from the FD004 Phase-1 sweep to all datasets (FD001, FD002, FD003, FD004).

## Configuration

The base configuration is defined in:
- `config/health_head_phase1/base_phase1_health_head.yaml`

**Hyperparameters (from FD004 Phase-1 RMSE-optimized model):**
- `rul_beta`: 45.0
- `health_loss_weight`: 0.35
- `mono_late_weight`: 0.03
- `mono_global_weight`: 0.003
- `hi_condition_calib_weight`: 0.0

## Running the Experiment

To train Phase-1 Health-Head models for all datasets:

```bash
python -m experiments.exp_phase1_health_head_all_datasets
```

This will:
1. Load the base configuration
2. Train a model for each dataset (FD001, FD002, FD003, FD004)
3. Evaluate on validation and test sets
4. Save results to `results/health_index/phase1_cross_dataset_baseline.csv`

## Results

After running the experiment, results are stored in:
- **CSV**: `results/health_index/phase1_cross_dataset_baseline.csv`
- **Models**: `results/health_index/{fdxxx}/phase1/{fdxxx}_phase1_baseline/`

## Generating Summary Table

To generate a Markdown summary table from the CSV:

```bash
python scripts/generate_phase1_summary_table.py
```

This creates:
- `docs/phase1_cross_dataset_summary.md`

## Results Table

| Dataset | Test RMSE | Test MAE | Test Bias | Test R² | Test NASA Mean |
|---------|-----------|----------|-----------|---------|----------------|
| FD001   | ...       | ...      | ...       | ...     | ...            |
| FD002   | ...       | ...      | ...       | ...     | ...            |
| FD003   | ...       | ...      | ...       | ...     | ...            |
| FD004   | ...       | ...      | ...       | ...     | ...            |

*Note: Run the experiment to populate the table with actual values.*

