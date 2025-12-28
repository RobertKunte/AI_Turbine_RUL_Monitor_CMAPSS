# FD00X Baseline Port Summary

**Last updated**: 2025-12-28  
**Baseline reference**: `fd004_wm_v1_p0_softcap_k3_hm_pad`  
**Port status**: Configs created, runs pending execution

---

## Overview

This document summarizes the cross-dataset baseline port of the FD004 World Model baseline (`fd004_wm_v1_p0_softcap_k3_hm_pad`) to FD001, FD002, and FD003. All configs use identical semantics:

- **Horizon padding**: `use_padded_horizon_targets=True` (near-EOL windows included)
- **Horizon masking**: `use_horizon_mask=True` (padded timesteps excluded from loss)
- **Soft cap weighting**: `soft_cap_enable=True` (distance-based soft weighting)
- **Evaluation**: `LAST_AVAILABLE_PER_UNIT` (truncated-aware, right-censored)
- **Training**: 10 epochs, batch_size=256, seed=42, engine_train_ratio=0.8
- **Horizon**: H=30, past_len=30, max_rul=125

---

## Results Table

| Dataset | Run Name | Features Count | Num Conditions | n_units (LAST) | RMSE_LAST | MAE_LAST | Bias_LAST | R²_LAST | NASA_LAST_MEAN | Notes |
|---------|----------|----------------|----------------|----------------|-----------|----------|-----------|---------|----------------|-------|
| FD001 | `fd001_wm_v1_p0_softcap_k3_hm_pad` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Single-condition dataset (simpler than FD004) |
| FD002 | `fd002_wm_v1_p0_softcap_k3_hm_pad` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Multiple operating conditions (similar to FD004) |
| FD003 | `fd003_wm_v1_p0_softcap_k3_hm_pad` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Single-condition dataset (simpler than FD004) |
| FD004 | `fd004_wm_v1_p0_softcap_k3_hm_pad` | 659 | 7 | TBD | ~17 | ~12 | ~−4 | ~0.84 | ~7 | Reference baseline (7 ConditionIDs) |

**Status**: Configs created, runs pending execution. Metrics will be populated after runs complete.

---

## Run Commands

```bash
# FD001
python -u run_experiments.py --experiments fd001_wm_v1_p0_softcap_k3_hm_pad --device cuda

# FD002
python -u run_experiments.py --experiments fd002_wm_v1_p0_softcap_k3_hm_pad --device cuda

# FD003
python -u run_experiments.py --experiments fd003_wm_v1_p0_softcap_k3_hm_pad --device cuda
```

---

## Expected Artifacts (Per Run)

Each run should produce in `results/<dataset_lower>/<run_name>/`:

- `summary.json` (metrics summary)
- `metrics_test.json` (test set metrics)
- `metrics_val.json` (validation set metrics)
- `training_history.json` (training curves data)
- `training_curves.png` (training/validation loss plots)
- `feature_cols.json` (feature column list)
- `feature_pipeline_config.json` (feature pipeline config snapshot)
- `condition_metrics.json` (per-condition breakdown)
- `error_hist.png` (error distribution histogram)
- `true_vs_pred.png` (scatter plot: true vs predicted RUL)
- `hi_rul_10_degraded.png` (HI/RUL trajectories for worst 10 engines)
- `dynamics_kpis.json` (HI/RUL dynamics KPIs)

---

## Dataset-Specific Notes

### FD001
- **Single operating condition**: Simpler than FD004 (no condition-wise scaling needed)
- **Expected**: Lower feature count than FD004 (no condition vectors)
- **Note**: Condition handling should work with num_conditions=1

### FD002
- **Multiple operating conditions**: Similar complexity to FD004
- **Expected**: Feature count and condition handling similar to FD004
- **Note**: May have different number of ConditionIDs than FD004 (verify from data)

### FD003
- **Single operating condition**: Simpler than FD004 (no condition-wise scaling needed)
- **Expected**: Lower feature count than FD004 (no condition vectors)
- **Note**: Condition handling should work with num_conditions=1

---

## Semantic Correctness Checks

After each run, verify:

- [ ] Horizon mask active: `pad_frac > 0.0` in logs
- [ ] Near-EOL windows included: `y_eol min ≤ 5` in logs
- [ ] Feature dimension check passes: No mismatch errors
- [ ] Scaler consistency: Training scaler matches inference/diagnostics scaler
- [ ] Evaluation semantics: `LAST_AVAILABLE_PER_UNIT` used (not optimistic assumptions)

---

## Quality Notes

If any run shows suspicious behavior (e.g., floor effects, constant predictions), document here:

- **FD001**: TBD (after run)
- **FD002**: TBD (after run)
- **FD003**: TBD (after run)

---

**Document version**: 1.0  
**Maintainer**: Implementer + Experimenter

