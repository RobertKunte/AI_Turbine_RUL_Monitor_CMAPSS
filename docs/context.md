### AI-Based Turbine RUL Monitor – Physics-Informed World Models (NASA C-MAPSS FD001–FD004)

**Project**: AI-Based Turbine RUL Monitor – Physics-Informed World Models  
**Domain**: NASA C-MAPSS (FD001–FD004), turbofan Remaining Useful Life (RUL) monitoring  
**Core idea**: Combine physics-informed feature engineering (residual/twin, HI_phys) with sequence models (Transformer encoders, trajectory decoders, world models) to achieve strong EOL accuracy *and* realistic degradation dynamics.

### Objectives
- **Primary**: High-quality RUL predictions at the **last observed cycle** (test is right-censored) and stable behavior across operating conditions (especially FD002/FD004).
- **Trajectory quality**: RUL trajectories and learned health indices should be monotone/smooth where appropriate and reflect degradation dynamics (avoid “flat then cliff”).
- **Metrics** (report consistently):
  - **RMSE**, **MAE**, **Bias** (pred−true), **R²**
  - **NASA PHM08 score** (mean + sum if available)
- **PHM best practices**:
  - Avoid leakage (no per-engine “future-aware” calibration; train-only fits).
  - Reproducible feature pipelines, fixed scalers, deterministic splits.
  - Diagnostics must explicitly consider censoring/truncation in test.

### Current known issues / risks
- **“Worst engines” / late degradation**:
  - A subset of engines show large positive last-cycle error (“late detection”).
  - Need targeted diagnosis: condition mix, censoring severity, residual sensor signatures.
- **Right-censored FD004 test**:
  - “EOL” for test means **last observed cycle**, not failure.
  - Worst-20 may correlate strongly with truncation (`true_rul_last`) and/or `num_cycles`.
- **Condition-wise scaling (FD002/FD004)**:
  - Must keep scaler logic identical between training/inference/diagnostics.
  - Misalignment causes feature-dimension or distribution drift and invalid conclusions.
- **Residual/twin features**:
  - Digital-twin residuals (`Resid_*`, `Twin_*`) must be enabled consistently (config-driven).
  - Distinguish “phase4 residual features” vs “HealthyTwin residuals”.
- **Multi-scale temporal features**:
  - High feature count increases mismatch risk; requires strict feature registry and safety checks.
- **Decoder blindness risk**:
  - Trajectory decoders/world models can overfit EOL while producing unrealistic mid-life dynamics.
  - Require explicit dynamics validation (slopes, monotonicity, smoothness, per-engine plots).

### Conventions

#### Run naming
Use short, descriptive names with dataset + model + features + version:
- **Example**: `fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm`
- **Pattern** (recommended):  
  `fd{dataset_id}_{model}_{features}_{target}_{version}_{notes}`

#### Metrics table fields (standard)
- **run_name**, **dataset**, **split** (val/test)
- **rmse**, **mae**, **bias**, **r2**
- **nasa_mean**, **nasa_sum** (if available)
- **num_engines**
- **notes** (important caveats: censoring, scaling, ablations)

#### Artifact paths
- **Primary**: `results/<dataset_lower>/<run_name>/...`
  - `summary.json`
  - `scaler.pkl`
  - `*_best*.pt`
  - `diagnostics_*.png`
  - `eol_metrics.json` (if written)
- Optional later: `artifacts/` for consolidated reports and exports.

#### Definition of “done” (for a change)
- **Code/Docs changes**: documented scope + rationale (ADR if decision-level).
- **Reproducibility**: same pipeline in train/inference; feature-dimension safety check passes.
- **Evaluation**:
  - Metrics computed on the relevant split(s) and saved to `summary.json` (or report).
  - Diagnostics plots produced/updated (esp. worst engines + censoring plots for FD004).
- **Acceptance**: explicitly stated acceptance criteria met (see `docs/roadmap.md`).

### Current focus (next 2 weeks)
- Improve **Worst-20 FD004 diagnosis** and censoring/truncation understanding (plots + stats).
- Stabilize **feature/scaler reproducibility** across training and diagnostics.
- Add a minimal **ablation framework** (features + heads) with comparable reporting.
- Validate **trajectory realism** (decoder/world model) beyond EOL metrics.
- Establish **documentation governance** (ADR + prompts + workflow) and use it routinely.


