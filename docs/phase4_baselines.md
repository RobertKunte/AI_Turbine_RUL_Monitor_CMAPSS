# Phase‑4 Residual / Digital‑Twin Baselines (FD001–FD004)

This document summarizes the **Phase‑4 residual / digital‑twin EOL models** that serve as
frozen baselines for future experiments (e.g. WorldModelV3, Transformer+Attention).

All baselines:

- use **physics‑informed features** (sensor + settings + physics proxies),
- add **multi‑scale temporal features** (rolling means, trends),
- enable **residual / digital‑twin features** (measured – “healthy” baseline),
- use **condition‑wise feature scaling** for multi‑condition datasets (FD002/FD004),
- train a **joint RUL + HI model** with monotonicity regularization.

## 1. Configs and Checkpoints

The canonical YAML configs live under:

- `config/phase4/*.yaml`

Each file defines:

- `experiment_id`: experiment name (matches `results/<fd>/...` subdirectory),
- `fd_id`: NASA C‑MAPSS subset (`FD001`–`FD004`),
- `model_class` / `encoder_type`: Python model definition in `src/models/`,
- `model`: encoder / sequence hyper‑parameters,
- `loss`: RUL + HI loss hyper‑parameters,
- `data`: feature engineering + scaling flags,
- `training`: optimizer and schedule.

The corresponding checkpoints and metrics are stored in:

- `results/<fd>/<experiment_id>/`

with at least:

- `summary.json` – global metrics and config snapshot,
- `condition_metrics.json` – per‑ConditionID metrics (FD002/FD004),
- `world_model_best.pt` / `best_model.pt` – best model weights,
- `error_hist.png`, `true_vs_pred.png`, `hi_rul_10_degraded.png` – diagnostics plots (if generated).

### 1.1 Baseline Table

> NOTE: RMSE / NASA values below are placeholders – please replace them with the values
> from `summary.json` once the final Phase‑4 runs are completed.

| FD    | Experiment ID                                | Config Path                                              | Checkpoint Path                                                           | RMSE (test) | NASA Mean (test) |
|-------|----------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------------|-------------|------------------|
| FD001 | `fd001_phase4_universal_v1_residual`         | `config/phase4/fd001_phase4_universal_v1_residual.yaml` | `results/fd001/fd001_phase4_universal_v1_residual/best_model.pt`         | _tbd_       | _tbd_            |
| FD002 | `fd002_phase4_universal_v2_ms_cnn_d96_residual` | `config/phase4/fd002_phase4_universal_v2_ms_cnn_residual.yaml` | `results/fd002/fd002_phase4_universal_v2_ms_cnn_d96_residual/best_model.pt` | _tbd_       | _tbd_            |
| FD003 | `fd003_phase4_universal_v1_residual`         | `config/phase4/fd003_phase4_universal_v1_residual.yaml` | `results/fd003/fd003_phase4_universal_v1_residual/best_model.pt`         | _tbd_       | _tbd_            |
| FD004 | `fd004_phase4_universal_v2_ms_cnn_d96_residual` | `config/phase4/fd004_phase4_universal_v2_ms_cnn_residual.yaml` | `results/fd004/fd004_phase4_universal_v2_ms_cnn_d96_residual/best_model.pt` | _tbd_       | _tbd_            |

These four experiments are considered **frozen baselines** for Phase‑4. New models
(World Models, Transformers, etc.) should report their results relative to these.

## 2. Intended Usage

### 2.1 Training / Re‑training

While training is currently orchestrated via `run_experiments.py` and the helpers in
`src/experiment_configs.py`, the Phase‑4 YAMLs provide a **single source of truth** for:

- sequence lengths (`past_len`, `max_rul`),
- encoder architecture (`d_model`, `num_layers`, `kernel_sizes`, …),
- loss weights (`health_loss_weight`, `mono_late_weight`, `mono_global_weight`),
- feature flags (physics, temporal, residual, condition embeddings),
- training setup (batch size, LR, epochs, patience).

In the future, the training script can be refactored to read these YAMLs directly and
instantiate the correct model and pipeline.

### 2.2 Inference & Diagnostics

The Phase‑4 baselines should be evaluated using the **standard diagnostics pipeline**:

- Build features and sequences with the same options as in the YAML (`data` section),
- Run per‑engine EOL inference (one RUL prediction per test engine),
- Compute metrics (RMSE, MAE, Bias, R², NASA PHM08),
- Generate diagnostics plots:
  - Error histogram (`error_hist.png`),
  - True vs. Predicted RUL scatter (`true_vs_pred.png`),
  - HI + RUL trajectories for degraded engines (`hi_rul_10_degraded.png`).

WorldModel V3 and future Transformer+Attention models should plug into the same
diagnostics script so that **metrics and plots are directly comparable**.

## 3. Notes & Caveats

- Exact hyper‑parameters (e.g. `d_model` vs. `dim_feedforward`) are taken from the
  Python configs in `src/experiment_configs.py`. If there is a discrepancy between YAML
  and code, the code is currently authoritative; the YAMLs should be updated to match.

- RUL labels are **clamped to `max_rul`** during training and evaluation for comparability
  with NASA PHM08 benchmarks.

- Health Index (HI) targets follow a **piecewise linear physics model**:
  - HI ≈ 1.0 for large RUL (plateau region),
  - HI ≈ RUL / `hi_plateau_threshold` in the linear decay region,
  - HI ≈ 0.0 near EOL (additional EOL penalty).

- Condition‑wise scaling is crucial for FD002/FD004; for FD001/FD003 a single global
  scaler is sufficient.


