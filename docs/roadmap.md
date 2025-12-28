# Roadmap: FD004 World Model → Agent-Ready Latent Sequence Forecasting

**Last updated**: 2025-12-28  
**Primary dataset**: FD004 (NASA C-MAPSS)  
**Baseline run**: `fd004_wm_v1_p0_softcap_k3_hm_pad`

---

## Vision & Product Thesis

**Vision**: Demonstrator for World Models → transferable to Mechanical Engineer Agents.

**Product thesis**: This project is not an end product, but a demonstrator for:
- **Scenario-capable world models**: Models that can answer "what-if" queries (condition changes, maintenance actions)
- **Mechanical Engineer Agents**: Foundation for future agent-based systems that reason about degradation dynamics

This project develops a semantically correct, risk-aware RUL World Model that can:
1. Predict realistic degradation trajectories (not just EOL point estimates)
2. Support scenario rollouts ("what-if" queries: condition changes, maintenance actions)
3. Provide conservative, deployment-ready outputs (quantile-based safety margins)
4. Transfer to agent-based systems via latent sequence forecasting: `p_θ(z_{t+1:t+H} | z_{≤t}, c)`

**Central target operator**: `p_θ(z_{t+1:t+H} | z_{≤t}, c)` where:
- `z_t` is a global latent state capturing coherent degradation
- `c` is a condition vector enabling condition-conditioned rollouts
- `H` is the forecast horizon (multi-step autoregressive)

**Intermediate milestones**:
- **M1**: Paper-ready baseline (semantically correct, reproducible, documented)
- **M2**: Risk-aware interface (conservative outputs, overshoot control, calibration)
- **M3**: Scenario rollouts (deterministic → distributional latent forecasting)
- **M4**: Agent-query API (condition-conditioned rollouts, action-conditioned dynamics)

**Target end-state**: Scenario-capable latent sequence model where:
- Global latent `z_t` captures coherent degradation state
- Autoregressive Transformer Decoder enables multi-step rollouts
- Condition vectors `c` enable "what-if" queries (operating condition changes)
- Distributional outputs (CVAE-style) enable uncertainty-aware rollouts

---

## Current Baseline (Status)

**Baseline run**: `fd004_wm_v1_p0_softcap_k3_hm_pad`

**What it fixes**:
- **Prediction floor removed**: Near-EOL windows (RUL < 30) now included in training via padding
- **Horizon masking**: Padded timesteps excluded from RUL loss (no synthetic target learning)
- **Padded horizon targets**: `use_padded_horizon_targets=True` enables `require_full_horizon=False` in Stage-1
- **Soft cap weighting**: Replaces binary masking to prevent vanishing gradients on capped timesteps

**Key gates achieved**:
- ✅ `schema_version=2` feature pipeline (deterministic, no heuristics)
- ✅ Feature-dimension safety checks (fail-fast on mismatches)
- ✅ Reproducible artifacts (scaler.pkl, summary.json, config snapshots)
- ✅ Run registry + sync (SQLite-based, Colab-compatible)
- ✅ Truncated-aware evaluation (`LAST_AVAILABLE_PER_UNIT` semantics)

**Baseline configuration**:
- Model: UniversalEncoderV2 + LSTM Decoder (H=30)
- Features: 659 (multiscale [5,10,30,60,120] + digital-twin residuals + condition vectors)
- Training: 10 epochs (extended variant: `_e50` with 50 epochs)
- Flags: `use_horizon_mask=True`, `use_padded_horizon_targets=True`, `soft_cap_enable=True`
- Operating conditions: 7 ConditionIDs derived from (S1_r, S2_r, S3_r) discretization

**Baseline performance** (test set, `LAST_AVAILABLE_PER_UNIT`):
- RMSE_LAST ≈ 17 cycles
- MAE_LAST ≈ 12 cycles
- Bias_LAST ≈ −4 cycles (slight underestimation, conservative)
- R²_LAST ≈ 0.84
- NASA PHM08 (mean) ≈ 7

**Documentation**: See `docs/fd004_world_model_baseline.md` for full technical details.

---

## Sprint Tracks (Parallel)

Four parallel tracks addressing different risk vectors. Each track has 2–4 sprint items with explicit artifacts and acceptance criteria.

---

### Track A: Ablation & Overfitting Control (Feature-Bloat)

**Goal**: Systematically quantify feature contributions and isolate overfitting risk from 659-feature space.

#### Sprint A1: Feature Group Ablations

**Goal**: Isolate contribution of each feature group (raw, multiscale, residual, twin, condition) to EOL metrics and trajectory realism.

**Artifacts**:
- [ ] Ablation matrix table (Markdown): `docs/fd004_feature_ablation_matrix.md`
- [ ] Metrics comparison CSV: `results/fd004/ablation_feature_groups_metrics.csv`
- [ ] Plots per ablation:
  - [ ] `true_vs_pred_last.png` (EOL scatter)
  - [ ] `error_vs_true_rul_bins.png` (error by RUL bin, emphasize near-EOL)
  - [ ] `worst20_trajectories.png` (trajectory plots for worst engines)

**Ablation variants** (each as separate experiment):
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_raw_only` (raw sensors only, ~21 features)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_ms_only` (multiscale only, no residuals/twin/cond)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_no_ms` (residuals+twin+cond, no multiscale)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_no_residual` (no digital-twin residuals)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_no_cond` (no condition vectors)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_ms_cond_only` (multiscale + cond, no residuals)

**Comparability rules** (MUST be identical across all ablations):
- Same train/val split (seed=42, engine-based 80/20)
- Same `max_rul=125` (capping)
- Same evaluation protocol (`LAST_AVAILABLE_PER_UNIT`)
- Same horizon H=30, same padding/masking flags
- Same training epochs (10) and batch size (256)

**Acceptance criteria**:
- [ ] All ablations complete without feature-dimension errors
- [ ] Metrics table shows clear ranking (which groups drive RMSE vs NASA vs trajectory quality)
- [ ] Feature count per ablation documented in table
- [ ] At least one ablation shows >5% RMSE degradation vs full baseline (confirms overfitting risk)
- [ ] Overfitting risk documented: 249 training engines vs 659 features (high-dimensional regime)
- [ ] Note: Multi-seed validation optional for now (future work if needed)

**Owner**: Planner (matrix design), Implementer (config variants), Experimenter (runs), Reviewer (comparability)

---

#### Sprint A2: Soft-Cap & Horizon-Mask Isolated Effects

**Goal**: Quantify isolated impact of soft-cap weighting and horizon masking (sanity checks).

**Artifacts**:
- [ ] Comparison table: `docs/fd004_softcap_horizon_mask_isolated.md`
- [ ] Metrics CSV: `results/fd004/ablation_softcap_hm_metrics.csv`
- [ ] Plots: `error_vs_true_rul_bins_softcap_off.png`, `error_vs_true_rul_bins_hm_off.png`

**Variants**:
- `fd004_wm_v1_p0_softcap_k3_hm_pad_softcap_off` (soft_cap_enable=False, binary masking on)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_hm_off` (use_horizon_mask=False, but keep padding)

**Comparability rules**: Same as A1 (split, seeds, max_rul, evaluation protocol).

**Acceptance criteria**:
- [ ] Soft-cap OFF shows measurable degradation (confirms soft-cap helps)
- [ ] Horizon-mask OFF shows floor at ~24–30 cycles (confirms mask prevents padding artifacts)
- [ ] Metrics table clearly shows which fix drives which improvement

**Owner**: Implementer (config flags), Experimenter (runs), Reviewer (interpretation)

---

#### Sprint A3: Feature Count vs Performance Curve

**Goal**: Plot performance (RMSE, NASA) vs feature count to identify diminishing returns threshold.

**Artifacts**:
- [ ] Plot: `feature_count_vs_rmse_nasa.png` (scatter with trend line)
- [ ] Table: Feature count per ablation variant (from A1)

**Acceptance criteria**:
- [ ] Clear inflection point identified (e.g., ">400 features shows <2% RMSE improvement")
- [ ] Recommendation: target feature count for future experiments

**Owner**: Experimenter (analysis), Reviewer (interpretation)

---

### Track B: Architecture (Decoder Variants)

**Goal**: Explore decoder architectures while maintaining LSTM baseline as CONTROL. Target: scenario-capable latent sequence forecasting.

#### Sprint B1: LSTM Decoder Baseline (CONTROL)

**Goal**: Establish LSTM decoder as stable CONTROL for architecture comparisons.

**Artifacts**:
- [ ] Baseline run: `fd004_wm_v1_p0_softcap_k3_hm_pad` (already exists)
- [ ] Extended training variant: `fd004_wm_v1_p0_softcap_k3_hm_pad_e50` (50 epochs)

**Acceptance criteria**:
- [ ] CONTROL metrics documented: RMSE_LAST, MAE_LAST, Bias_LAST, R²_LAST, NASA
- [ ] Training stability confirmed (val loss decreases, no divergence)
- [ ] CONTROL used as reference for all Track B comparisons

**Owner**: Experimenter (run), Reviewer (stability check)

---

#### Sprint B2: Transformer Decoder Baseline (Deterministic)

**Goal**: Replace LSTM decoder with autoregressive Transformer decoder. Keep deterministic (point predictions only).

**Artifacts**:
- [ ] Experiment config: `fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar`
- [ ] Metrics comparison: `docs/fd004_decoder_lstm_vs_tf.md`
- [ ] Plots: `trajectory_comparison_lstm_vs_tf.png` (same engines, both decoders)

**Architecture**:
- Encoder: UniversalEncoderV2 (unchanged)
- Decoder: Autoregressive Transformer Decoder
  - Causal masking (no future leakage)
  - Cross-attention to encoder outputs
  - Self-attention over past decoder states
- Output: Deterministic (mean prediction only)

**Benefit thresholds** (must meet ALL to proceed):
- RMSE_LAST improvement ≥ 3% vs CONTROL
- NASA mean improvement ≥ 5% vs CONTROL
- Trajectory quality: No degradation in worst-case engines (qualitative check)
- Risk/Overshoot: Overshoot rate and unsafe fraction must NOT be worse than CONTROL (industrial safety lens)

**Comparability rules**: Same as Track A (split, seeds, max_rul, evaluation, features).

**Acceptance criteria**:
- [ ] Transformer decoder trains stably (no NaN, val loss decreases)
- [ ] Metrics table shows clear comparison vs CONTROL
- [ ] If thresholds NOT met: document why (e.g., "Transformer overfits", "LSTM sufficient")
- [ ] Decision: Continue with Transformer OR keep LSTM as baseline

**Owner**: Planner (architecture spec), Implementer (decoder code), Experimenter (run), Reviewer (comparison)

---

#### Sprint B3: Latent Sequence Forecasting (Distributional)

**Goal**: Extend Transformer decoder to distributional outputs (CVAE-style) for scenario rollouts.

**Artifacts**:
- [ ] ADR: `docs/decisions/ADR-0011-latent-sequence-forecasting.md`
- [ ] Experiment config: `fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_latent`
- [ ] Rollout demo: `scenario_rollout_demo.ipynb` (condition-conditioned examples)

**Architecture**:
- Global latent `z_t` (per timestep, sampled from `q_φ(z_t | x_{≤t}, c)`)
- Autoregressive decoder: `p_θ(z_{t+1:t+H} | z_{≤t}, c)`
- Output: Distributional (mean + variance, or quantiles)

**Scenario capabilities**:
- Condition-conditioned: "What if operating condition changes at t=50?"
- Multi-step rollouts: Predict H=30 steps autoregressively
- Uncertainty propagation: Variance grows with horizon

**Acceptance criteria**:
- [ ] Latent model trains stably (KL divergence regularized, no collapse)
- [ ] Rollout demo shows coherent trajectories (no mode collapse, realistic variance)
- [ ] Metrics: RMSE/NASA comparable to deterministic baseline (±5% tolerance)
- [ ] Documentation: Clear API for scenario queries

**Owner**: Planner (ADR + spec), Implementer (latent code), Experimenter (rollout demo), Reviewer (coherence)

---

### Track C: Semantics & Evaluation Realism

**Goal**: Ensure evaluation matches deployment reality. Emphasize worst-case analysis and per-condition breakdowns.

#### Sprint C1: Worst-Case Engine Trajectory Reporting

**Goal**: Systematically report worst engines with emphasis on near-EOL region (true_rul_last ≤ 20).

**Artifacts**:
- [ ] Plot: `worst20_trajectories_overall.png` (RUL trajectories, all worst 20 engines by absolute error)
- [ ] Plot: `worst20_trajectories_near_eol.png` (RUL trajectories, zoom on last 30 cycles, subset with true_rul_last ≤ 20)
- [ ] Table: `worst20_engines_overall.csv` (engine_id, true_rul_last, pred_rul_last, abs_error, condition, truncation_severity)
- [ ] Table: `worst20_engines_near_eol.csv` (subset: true_rul_last ≤ 20, same columns)
- [ ] Analysis: `docs/fd004_worst20_analysis.md` (correlations: condition? truncation? residual patterns?)

**Metrics per worst engine**:
- RMSE_LAST, MAE_LAST, Bias_LAST
- NASA score (individual)
- Condition ID, truncation severity (num_cycles / max_cycles)

**Acceptance criteria**:
- [ ] Worst-20 overall clearly identified (sorted by absolute error, not just RMSE)
- [ ] Worst-20 near-EOL subset identified (true_rul_last ≤ 20, sorted by absolute error)
- [ ] Near-EOL emphasis: At least 10/20 worst overall engines have true_rul_last ≤ 20
- [ ] Analysis identifies primary driver: (A) truncation, (B) condition, (C) residual sensors, or mixed

**Owner**: Implementer (diagnostics script), Experimenter (run analysis), Reviewer (interpretation)

---

#### Sprint C2: Per-Condition Metrics + Error vs RUL Bins

**Goal**: Break down metrics by operating condition and by true RUL bins (emphasize near-EOL).

**Artifacts**:
- [ ] Table: `per_condition_metrics.csv` (condition_id, RMSE, MAE, Bias, NASA, num_engines)
- [ ] Plot: `error_vs_true_rul_bins.png` (boxplot: error distribution per RUL bin [0-10, 10-20, 20-30, ..., 100-125])
- [ ] Plot: `error_vs_true_rul_bins_near_eol.png` (zoom on [0-30] bins)
- [ ] Table: `error_by_rul_bin.csv` (bin, mean_error, std_error, p95_error, num_samples)

**RUL bins** (emphasize near-EOL):
- [0-10], [10-20], [20-30], [30-50], [50-75], [75-100], [100-125]

**Acceptance criteria**:
- [ ] Per-condition table shows clear condition-dependent performance (FD004 has 7 ConditionIDs derived from (S1_r, S2_r, S3_r) discretization)
- [ ] Error vs RUL bins shows degradation pattern (error increases near EOL? constant? decreases?)
- [ ] Near-EOL bins ([0-10], [10-20], [20-30]) have sufficient sample count (n ≥ 10 per bin)

**Owner**: Implementer (metrics script), Experimenter (analysis), Reviewer (interpretation)

---

#### Sprint C3: "ALL Windows" vs "LAST" Breakdown

**Goal**: Report both LAST_AVAILABLE_PER_UNIT (primary) and ALL windows (secondary) for transparency.

**Artifacts**:
- [ ] Metrics table: `eol_metrics_last_vs_all.csv` (split, metric, LAST_value, ALL_value, diff)
- [ ] Plot: `error_distribution_last_vs_all.png` (histogram comparison)

**Semantics**:
- **LAST**: One prediction per test engine (last observed cycle only) — PRIMARY
- **ALL**: All sliding windows in test set (includes mid-life windows) — SECONDARY

**Acceptance criteria**:
- [ ] LAST metrics remain PRIMARY (reported first, used for comparisons)
- [ ] ALL metrics shown for transparency (may be optimistic due to mid-life windows)
- [ ] Clear documentation: "LAST matches deployment (right-censored), ALL includes mid-life"

**Owner**: Implementer (metrics script), Reviewer (semantics check)

---

### Track D: Risk/Conservatism & Deployment Checks

**Goal**: Quantify overshoot risk and define safe output policy. Prepare for deployment readiness.

#### Sprint D1: Conservatism / Risk Metrics

**Goal**: Quantify dangerous overestimation (overshoot) near EOL and define "unsafe" fraction.

**Artifacts**:
- [ ] Metrics: `risk_metrics_near_eol.json` (overshoot_rate, p95_overshoot, unsafe_fraction)
- [ ] Plot: `overshoot_hist_near_eol.png` (histogram of (pred - true) for true_rul_last ≤ 20)
- [ ] Table: `risk_by_rul_bin.csv` (bin, overshoot_rate, p95_overshoot, unsafe_fraction)

**Risk definitions** (industrial safety lens):
- **Overshoot**: `pred_rul_last > true_rul_last` (overestimation, dangerous)
- **Overshoot rate**: Fraction of engines with overshoot (for true_rul_last ≤ 20)
- **p95 overshoot**: 95th percentile of (pred - true) for true_rul_last ≤ 20
- **Unsafe fraction**: Fraction with `pred_rul_last > true_rul_last + 10` (dangerous overestimation)

**RUL bins** (fine-grained near-EOL):
- [0-5], [5-10], [10-20], [20-30], [30-50], [50-75], [75-100], [100-125]

**Acceptance criteria**:
- [ ] Overshoot rate < 0.5 for true_rul_last ≤ 20 (at least 50% conservative)
- [ ] p95 overshoot < 20 cycles (95% of errors below 20 cycles overestimation)
- [ ] Unsafe fraction < 0.1 (less than 10% dangerously optimistic)
- [ ] Risk metrics computed per fine-grained bin ([0-5], [5-10], [10-20], [20-30])
- [ ] Industrial safety lens: Conservative bias preferred over optimistic bias near EOL

**Owner**: Implementer (risk metrics), Experimenter (analysis), Reviewer (thresholds)

---

#### Sprint D2: Safe Output Policy Definition

**Goal**: Define recommended "safe" output policy (e.g., quantile or μ − q_τ) as interface contract.

**Artifacts**:
- [ ] ADR: `docs/decisions/ADR-0012-safe-output-policy.md`
- [ ] Interface spec: `docs/safe_output_policy_interface.md`
- [ ] Demo: `safe_output_demo.ipynb` (compare μ vs safe_RUL)

**Policy options** (evaluate and choose):
- **Option A**: `safe_RUL = μ - q_τ` (mean minus upper quantile of error)
- **Option B**: `safe_RUL = q_10` (10th percentile, conservative)
- **Option C**: `safe_RUL = μ - k·σ` (mean minus k standard deviations)

**Evaluation**:
- Compare overshoot rates: μ vs safe_RUL
- Compare RMSE degradation: safe_RUL vs μ (acceptable trade-off?)
- Compare NASA scores: safe_RUL vs μ (should improve)

**Acceptance criteria**:
- [ ] Policy chosen and documented (ADR)
- [ ] Interface spec defines input/output contract
- [ ] Demo shows clear improvement in overshoot rate (safe_RUL more conservative)
- [ ] RMSE degradation acceptable (<10% increase for safe_RUL vs μ)

**Owner**: Planner (ADR), Implementer (interface), Experimenter (demo), Reviewer (policy)

---

#### Sprint D3: Calibration Checks (Coverage, Reliability)

**Goal**: Assess prediction calibration (coverage, reliability) without overcommitting to full uncertainty quantification.

**Artifacts**:
- [ ] Plot: `calibration_reliability_curve.png` (predicted vs observed quantiles)
- [ ] Metrics: `calibration_metrics.json` (coverage_80, coverage_95, ECE)

**Scope** (lightweight, not full UQ):
- If quantile head exists: Check coverage (80% interval contains 80% of true values?)
- If distributional outputs exist: Check reliability (predicted variance correlates with actual error?)

**Acceptance criteria**:
- [ ] Calibration metrics computed (if applicable)
- [ ] Documentation: "Calibration status: [well-calibrated / underconfident / overconfident]"
- [ ] No overcommitment: If UQ not implemented, document as "future work"

**Owner**: Implementer (calibration script), Reviewer (interpretation)

---

## Definition of Done (DoD) for "Paper-Ready Baseline"

A baseline is considered "paper-ready" when it meets ALL of the following criteria:

### Minimal Artifacts

- [ ] **Metrics JSON**: `results/fd004/<run>/summary.json` with:
  - RMSE_LAST, MAE_LAST, Bias_LAST, R²_LAST (test set)
  - NASA PHM08 (mean + sum, test set)
  - Per-condition breakdown (if applicable)
  - Risk metrics (overshoot_rate, p95_overshoot, unsafe_fraction)

- [ ] **Plots**:
  - [ ] `true_vs_pred_last.png` (scatter: true vs predicted RUL at last cycle)
  - [ ] `error_hist_last.png` (histogram of errors)
  - [ ] `training_curves.png` (train/val loss over epochs)
  - [ ] `worst20_trajectories.png` (RUL trajectories for worst 20 engines)
  - [ ] `error_vs_true_rul_bins.png` (error distribution by RUL bin)

- [ ] **Per-condition table**: `per_condition_metrics.csv` (condition_id, metrics, num_engines)

- [ ] **Risk report**: `risk_report_near_eol.md` (overshoot analysis, unsafe fraction)

- [ ] **Baseline documentation**: `docs/fd004_world_model_baseline.md` (technical reference)

### Reproducibility Gates

- [ ] Feature pipeline: `schema_version=2`, deterministic, no heuristics
- [ ] Feature-dimension check: Passes (659 == 659, or documented variant)
- [ ] Scaler consistency: Training scaler matches inference/diagnostics scaler
- [ ] Run registry: Entry created with config_json + metrics_json

### Evaluation Semantics

- [ ] **Primary metric**: `LAST_AVAILABLE_PER_UNIT` (truncated-aware, right-censored)
- [ ] **Secondary metric**: `ALL windows` (for transparency, clearly marked as secondary)
- [ ] **Documentation**: Explicit statement: "Test EOL = last observed cycle (right-censored)"

### Acceptance Thresholds

- [ ] **Semantic correctness**: Horizon mask active (pad_frac > 0), near-EOL windows included (y_eol min ≤ 5)
- [ ] **Performance**: RMSE_LAST < 50 cycles, Bias_LAST < +30 cycles, R²_LAST > 0.3
- [ ] **Risk**: Overshoot rate < 0.5 for true_rul_last ≤ 20, unsafe fraction < 0.1

### Baseline Status Note

**Important**: The baseline (`fd004_wm_v1_p0_softcap_k3_hm_pad`) is considered a **paper-ready starting point**, not a final result. It establishes semantic correctness and reproducibility, enabling:
- Valid comparisons for future experiments (architecture, features, loss functions)
- Foundation for research/publication (reproducible, correct semantics)
- Deployment consideration (training matches real-world distribution)

Future work (Tracks A–D) will optimize performance, reduce overfitting, improve risk metrics, and enable scenario rollouts.

---

## Run Naming Convention

**Pattern**: `fd004_wm_v1_p0_softcap_k3_hm_pad{_suffix}`

**Base**: `fd004_wm_v1_p0_softcap_k3_hm_pad` (baseline with horizon masking + padding)

**Suffixes** (append to base):
- `_e50`: Extended training (50 epochs)
- `_ablation_{group}`: Feature ablation (e.g., `_ablation_raw_only`, `_ablation_no_ms`)
- `_softcap_off`: Soft cap disabled (binary masking)
- `_hm_off`: Horizon mask disabled
- `_dec_tf_ar`: Transformer decoder (autoregressive)
- `_dec_tf_latent`: Transformer decoder (latent/distributional)

**Examples**:
- `fd004_wm_v1_p0_softcap_k3_hm_pad_e50` (extended baseline)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_ablation_raw_only` (raw sensors only)
- `fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar` (Transformer decoder)

**Rules**:
- Base name identifies core configuration (p0 fixes + horizon masking)
- Suffixes identify variants (training length, features, architecture)
- Keep names concise but unambiguous

---

## Notes & Governance

### Scope & Focus

- **FD004 focus**: This roadmap assumes FD004 as primary dataset. Items are reusable for FD001–FD003 but not explicitly scoped.
- **Parallel tracks**: Tracks A–D can proceed in parallel. No strict dependencies except CONTROL baseline (B1) should complete before B2/B3.
- **Gates**: Each sprint item has explicit acceptance criteria. Do not proceed to next sprint if gates not met.
- **Comparability**: All ablation/architecture experiments MUST use identical splits, seeds, max_rul, evaluation protocol (see Track A comparability rules).

### Experiment Naming

- Follow the **Run Naming Convention** (see section above)
- Base name: `fd004_wm_v1_p0_softcap_k3_hm_pad`
- Suffixes identify variants (training length, features, architecture)
- Keep names concise but unambiguous

### Documentation Requirements

- Each sprint must produce documented artifacts (tables, plots, analysis)
- ADRs required for decision-level changes (architecture, policy, evaluation protocol)
- Baseline documentation: `docs/fd004_world_model_baseline.md` (technical reference)

### Quality Gates

- Feature-dimension safety checks must pass (fail-fast on mismatches)
- Scaler consistency: Training scaler matches inference/diagnostics scaler
- Run registry: All runs must be logged with config_json + metrics_json
- Evaluation semantics: `LAST_AVAILABLE_PER_UNIT` is PRIMARY, `ALL windows` is SECONDARY

---

**Document version**: 2.1  
**Last updated**: 2025-12-28  
**Maintainer**: Technical Program Manager + ML Lead
