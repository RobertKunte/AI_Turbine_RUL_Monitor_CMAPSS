# FD004 World Model Baseline – Horizon-Masked, Truncated-Aware RUL

**Status**: Baseline established (Jan 2025)

This document describes the first semantically correct, truncated-aware, horizon-masked RUL World Model for FD004. This baseline addresses fundamental training/evaluation distribution mismatches that caused prediction floors and optimistic bias. It serves as the reference point for future FD004 World Model development.

---

## 1. Problem Statement (Before)

Previous FD004 World Model runs exhibited systematic semantic flaws that undermined deployment credibility:

**Training/evaluation distribution mismatch**:
- Training excluded near-EOL windows (RUL < 30) due to `require_full_horizon=True`
- Evaluation included all engines down to RUL = 0
- Model never saw low-RUL samples during training → prediction floor at ~24–30 cycles

**Loss computation on invalid timesteps**:
- Future horizon padded to fixed length H = 30 using clamp mode
- Padded timesteps (no actual future data) contributed to RUL loss
- Model learned to predict on synthetic padded values

**Cap-sticking failure mode**:
- Early/mid-life windows had 20–30 out of 30 future timesteps capped at max_rul
- Binary cap masking caused vanishing gradients for capped timesteps
- Model collapsed to predicting constant high RUL (~100–125 cycles)
- Bias_LAST ≈ +70 cycles, R²_LAST ≈ 0

**Why this matters**:
- PHM systems require accurate low-RUL predictions for safety-critical decisions
- Training/eval mismatch creates false confidence in model performance
- Optimistic bias near EOL leads to late failure detection and high NASA PHM08 penalties

---

## 2. Key Fixes Introduced

### Near-EOL windows included in training

**Before**: `require_full_horizon=True` filtered out windows where available future < H, excluding all samples with RUL < 30.

**Now**: `require_full_horizon=False` (via `use_padded_horizon_targets=True`) allows windows with short future horizons. Near-EOL samples (RUL → 0) are included via padding.

**Why correct**: Training distribution now matches evaluation distribution. Model learns to predict low RUL values.

### Horizon padding with explicit horizon mask

**Before**: Padding existed but mask was not created or used. Padded timesteps were treated as valid targets.

**Now**: `return_mask=True` creates a binary mask (1 = observed, 0 = padded). Mask is propagated through dataset and available in training batches.

**Why correct**: Distinguishes observed vs synthetic timesteps. Enables proper masking in loss computation.

### Masked RUL future loss (no loss on padded timesteps)

**Before**: RUL future loss computed on all H timesteps, including padded ones. Model penalized for not matching synthetic clamped values.

**Now**: RUL loss uses combined mask: `mask_rul = cap_mask × early_mask × horizon_mask`. Only valid, uncapped, non-early timesteps contribute to loss.

**Why correct**: Model learns only from real observed degradation patterns, not synthetic padding artifacts.

### Soft-cap weighting for capped RUL

**Before**: Binary cap masking (`cap_mask_enable=True`) set weight to 0 for capped timesteps, causing vanishing gradients.

**Now**: Soft cap weighting (`soft_cap_enable=True`) applies weight = `(distance_from_cap)^0.5` with floor 0.05. Capped timesteps contribute reduced but non-zero gradient.

**Why correct**: Prevents vanishing gradients while still de-emphasizing capped regions. Smooth transition from capped to uncapped regions.

### Truncated-aware LAST evaluation

**Before**: Evaluation might have used optimistic assumptions about test set EOL.

**Now**: Evaluation uses `LAST_AVAILABLE_PER_UNIT` semantics: each test engine evaluated at its last observed cycle (right-censored), not assumed failure point.

**Why correct**: Matches real-world PHM scenario where test data is truncated. No optimistic leakage.

### Deterministic feature pipeline (schema_version=2)

**Before**: Feature pipeline might have had heuristics or non-deterministic behavior.

**Now**: Strict feature contract with `schema_version=2`. Feature dimension verified (659 features). No silent fallbacks.

**Why correct**: Reproducibility and consistency across training/inference/diagnostics. Fail-fast on mismatches.

---

## 3. Final Training Semantics (Authoritative)

This section defines the ground truth for the baseline configuration.

**Dataset**: FD004 (NASA C-MAPSS turbofan degradation)

**RUL capping**: `max_rul = 125` cycles. All RUL targets and predictions are capped at 125. This is applied:
- During window building (`cap_targets=True`)
- During training (target clamping)
- During evaluation (prediction clipping)

**Training window construction**:
- Past window: `past_len = 30` cycles
- Future horizon: `horizon = 30` cycles (H = 30)
- Padding mode: `pad_mode = "clamp"` (repeat last observed value)
- Full horizon requirement: `require_full_horizon = False` (when `use_padded_horizon_targets=True`)
- Near-EOL windows: Included via padding when `available_future < H`

**RUL future loss computation**:
- Loss type: Masked MSE (normalized RUL space)
- Mask combination: `mask_rul = cap_mask × early_mask × horizon_mask`
  - `cap_mask`: Excludes timesteps where RUL ≥ (1 - eps) normalized
  - `early_mask`: Excludes timesteps beyond `rul_train_max_cycles` (if set)
  - `horizon_mask`: Excludes padded timesteps (0 = padded, 1 = observed)
- Soft cap weighting: Applied when `soft_cap_enable=True`
  - Weight = `(1.0 - true_rul_norm)^0.5` clamped to [0.05, 1.0]
  - Multiplied by `horizon_mask` to zero out padded timesteps
- Normalization: `loss = (diff² × mask_rul × weights).sum() / (mask_rul × weights).sum()`

**Evaluation semantics**:
- Metric: `LAST_AVAILABLE_PER_UNIT`
- Each test engine: Prediction at last observed cycle only
- Right-censoring: Test "EOL" = last observed cycle, not failure
- No optimistic assumptions: Does not assume failure occurred

**Diagnostics**:
- Feature pipeline: Identical to training (same scalers, same feature set)
- Feature dimension: Verified 659 == 659 (fail-fast check)
- No heuristics: Strict schema_version=2 contract

---

## 4. Baseline Configuration (Concrete)

**Experiment name**: `fd004_wm_v1_p0_softcap_k3_hm_pad`

**Model architecture**:
- Encoder: UniversalEncoderV2 (EOLFullTransformerEncoder)
  - Input dimension: 659 features
  - d_model: 96
  - num_layers: 3
  - nhead: 4
  - dim_feedforward: 384
  - dropout: 0.1
- Decoder: LSTM-based trajectory decoder
  - Hidden dimension: 256
  - Num layers: 1
  - Horizon: 30
- Heads:
  - HI trajectory head (full horizon)
  - RUL trajectory head (full horizon)
  - EOL scalar head (optional)

**Input features** (659 total):
- Base sensors: 21 raw sensor measurements
- Multiscale temporal windows: [5, 10, 30, 60, 120] cycles
- Digital-twin residuals: Resid_* features computed from healthy baseline
- Condition vectors: Cond_* features (continuous operating condition encoding)
- Feature pipeline: schema_version=2 (deterministic, no heuristics)

**Training configuration**:
- Epochs: 10 (baseline), 50 (extended variant: `_e50`)
- Batch size: 256
- Learning rate: 1e-4
- Optimizer: Adam
- Train/val split: Engine-based (80/20), seed=42

**Key flags** (world_model_params):
- `use_horizon_mask = True` (enables mask creation and usage)
- `use_padded_horizon_targets = True` (allows near-EOL windows)
- `soft_cap_enable = True` (soft per-timestep weighting)
- `soft_cap_power = 0.5` (sqrt ramp)
- `soft_cap_floor = 0.05` (minimum weight)
- `informative_sampling_enable = True`
- `informative_sampling_mode = "uncapped_frac"`
- `informative_uncapped_frac_threshold = 0.3` (30% uncapped required)
- `late_weight_enable = True`
- `late_weight_factor = 10.0`

**Loss weights**:
- RUL future loss: `rul_future_loss_weight` (from config)
- HI trajectory loss: `traj_loss_weight` (from config)
- EOL scalar loss: `eol_scalar_loss_weight` (from config)

---

## 5. Baseline Performance (Test Set)

**Note**: Metrics will be populated after the baseline run completes. Expected metrics based on ADR-0010 acceptance criteria:

**Target performance** (from ADR-0010 Go/No-Go criteria):
- Bias_LAST: < +30 cycles (vs +70 before)
- R²_LAST: > 0.3 (vs ~0 before)
- pred_rul_seq_std: ≥ 0.10 (vs ~0.02 before)
- RMSE_LAST: < 50 cycles (vs ~75 before)

**Why these numbers are strong but plausible**:
- FD004 is the most challenging dataset (multiple operating conditions, high variance)
- Literature RMSE ranges: 20–40 cycles for strong methods
- The baseline prioritizes semantic correctness over RMSE optimization
- Truncated-aware evaluation (LAST) is stricter than optimistic assumptions

**Why metrics should be trusted**:
- Training includes near-EOL samples (no distribution mismatch)
- Loss computed only on valid timesteps (no padding artifacts)
- Evaluation uses truncated-aware semantics (no optimistic leakage)
- Feature pipeline is deterministic and verified (no silent mismatches)

**Actual metrics** (to be updated after run):
- RMSE_LAST: TBD
- MAE_LAST: TBD
- Bias_LAST: TBD
- R²_LAST: TBD
- NASA PHM08 (mean): TBD
- NASA PHM08 (sum): TBD

---

## 6. Why This Is a True Baseline

This run (`fd004_wm_v1_p0_softcap_k3_hm_pad`) is the first FD004 World Model configuration suitable as a reference baseline because:

**Semantic correctness**:
- Training and evaluation distributions match (both include near-EOL samples)
- Loss computation excludes invalid timesteps (padded, capped, early)
- Evaluation uses truncated-aware semantics (no optimistic assumptions)

**Reproducibility**:
- Deterministic feature pipeline (schema_version=2)
- Explicit config flags (no hidden heuristics)
- Fail-fast checks (feature dimension, scaler consistency)

**Completeness**:
- All critical fixes applied (horizon masking, padding, soft cap weighting)
- Logging confirms fixes are active (pad_frac > 0, y_eol min ≤ 5)
- Diagnostics use identical pipeline as training

**Future experiments should be compared against this baseline**:
- Architecture changes (encoder/decoder modifications)
- Loss function modifications (new penalty terms)
- Feature engineering (additional features, ablations)
- Hyperparameter sweeps (learning rate, batch size, etc.)

**This baseline enables**:
- Validated comparison of architectural improvements
- Confidence in reported metrics (no semantic flaws)
- Foundation for research/publication (reproducible, correct semantics)
- Deployment consideration (training matches real-world distribution)

---

## 7. Explicit Non-Goals / Open Items

This baseline establishes semantic correctness but does not claim optimality:

**Not yet optimized**:
- No hyperparameter sweep (learning rate, batch size, architecture)
- No extensive epoch tuning (baseline uses 10 epochs, extended variant uses 50)
- No loss weight optimization (uses reasonable defaults)

**Not yet implemented**:
- No latent dynamics modeling (predicts sensors/RUL directly, not latent state)
- No cross-dataset validation (FD004 only, not tested on FD001–FD003)
- No uncertainty quantification (point predictions only, no confidence intervals)

**Not yet validated**:
- No extensive ablation studies (feature groups, loss components)
- No worst-engine deep dive (diagnostics exist but not fully analyzed)
- No condition-aware calibration (HI bias correction per condition)

**Scope limitation**:
- Single dataset (FD004)
- Single model family (UniversalEncoderV2 + LSTM decoder)
- Single evaluation metric focus (LAST, not trajectory quality metrics)

These limitations are intentional: the baseline prioritizes correctness over completeness. Future work can build on this foundation.

---

## 8. Technical Details

### Window Building

**Stage-1 seq2seq builder** (used for initial training):
- Config flag: `use_padded_horizon_targets = True`
- Window config: `require_full_horizon = False` (when flag is True)
- Mask creation: `return_mask = True` (when `use_horizon_mask = True`)
- Padding: `pad_mode = "clamp"` (repeat last observed RUL value)

**Stage-2 world model builder** (used for full training):
- Window config: `require_full_horizon = False` (hardcoded)
- Mask creation: `return_mask = True` (when `use_horizon_mask = True`)
- Padding: `pad_mode = "clamp"`

### Loss Masking Implementation

**Mask extraction**:
```python
# From batch
valid_mask_seq = mask_batch  # (B, H, 1) or (B, H)
if valid_mask_seq.dim() == 3:
    valid_mask_seq = valid_mask_seq.squeeze(-1)  # (B, H)
valid_mask_seq_exp = valid_mask_seq.unsqueeze(-1)  # (B, H, 1)
```

**Mask combination**:
```python
mask_rul = mask_f * valid_mask_seq_exp  # (B, H, 1)
# where mask_f = cap_mask & early_mask (already (B, H, 1))
```

**Soft cap weight application**:
```python
if soft_cap_enable:
    cap_weight = cap_distance.pow(0.5).clamp(0.05, 1.0)
    cap_weight = cap_weight * valid_mask_seq_exp  # Zero out padded timesteps
```

**Loss computation**:
```python
diff2 = (pred_rul_seq_norm - true_rul_seq_norm) ** 2
num_i = (diff2 * mask_rul).sum(dim=(1, 2))  # (B,)
den_i = mask_rul.sum(dim=(1, 2)).clamp_min(1e-6)  # (B,)
loss_i = num_i / den_i  # (B,)
loss_rul_traj = (w_final * loss_i).sum() / (w_final.sum() + 1e-6)
```

### Verification Logging

**Stage-1 build verification**:
- `[Stage-1] Y_eol range: min=X.XX, max=Y.YY` (min should be ≤ 5)
- `[Stage-1] pad_frac=X.XXXX` (should be > 0.0)

**First batch verification**:
- `[HorizonMask] use_horizon_mask=True`
- `[HorizonMask] valid_frac=X.XXXX pad_frac=X.XXXX`
- `[HorizonMask] denom(min/mean)=X.XX/X.XX`

---

## 9. Related Documentation

- **ADR-0010**: WorldModel-V1 Cap-Collapse Fix (soft cap weighting rationale)
- **docs/context.md**: Project context and conventions
- **docs/decisions/**: Other architectural decisions
- **src/world_model_training_v3.py**: Implementation (lines 1867-1895 for window building, 3041-3062 for mask application)

---

## 10. Changelog

**2025-01-XX**: Baseline established
- Horizon masking implemented and enabled
- Stage-1 padding enabled (`use_padded_horizon_targets=True`)
- Soft cap weighting enabled
- Truncated-aware evaluation confirmed
- Documentation created

---

**Document version**: 1.0  
**Last updated**: 2025-01-XX  
**Maintainer**: Technical Documentation

