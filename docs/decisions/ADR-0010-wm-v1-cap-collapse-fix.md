### ADR-0010: WorldModel-V1 Cap-Collapse Fix (Soft Cap Weighting + Stricter Informative Sampling)

#### Status
Proposed

#### Context

**Problem observed**:
WorldModelV1 on FD004 exhibits a severe "high-RUL collapse" / "cap-sticking" failure mode:
- Predictions cluster near the RUL cap (~100-125 cycles, normalized ~0.9-1.0)
- Bias_LAST ≈ +70 cycles (massive overestimation)
- R²_LAST ≈ 0 (near-constant/degenerate predictions)
- pred_rul_seq_std ≈ 0.02 (almost no variance in output)

**Root cause analysis** (from Reviewer deep-dive):
1. **Capped targets dominate training data**: With `max_rul=125` and `pad_mode="clamp"`, early/mid-life windows have future RUL sequences that are entirely capped at 1.0 (normalized). Debug shows `frac_all_cap_future_batch ≈ 0.18-0.24`.

2. **Informative sampling is too weak**: The `future_min_lt_cap` mode only checks if ANY timestep is below cap. Windows with 29/30 capped timesteps are still marked "informative".

3. **Cap reweighting applied at sample-level, not timestep-level**: `cap_reweight_weight=0.05` only affects samples where the ENTIRE horizon is capped. Windows with 90% capped timesteps get full weight.

4. **Binary cap masking causes vanishing gradients**: When `cap_mask_enable=True`, fully-capped windows contribute zero loss → zero gradient → decoder learns nothing from early-life windows.

**Why encoder-only works but WM doesn't**:
- Encoder sees local windows with real RUL variation
- Encoder is not forced to "explain" a 30-step capped future
- World Model must predict 30 future timesteps, 20-30 of which are often capped
- Loss minimum lies at `pred_rul ≈ 1.0` (always predict "healthy")

**Evidence** (from runs `fd004_wm_v1_infwin_wiringcheck_k0`, `fd004_wm_v1_infwin_capweight_k1`):
- `true_rul_cap_frac_batch ≈ 0.24`
- `frac_all_cap_future_batch ≈ 0.18`
- `pred_rul_min_mean_max: [0.89, 0.94, 0.99]` (clustered at cap)
- `Bias_LAST ≈ +70 cycles`

#### Decision

We implement a **P0 fix** consisting of two complementary changes:

**P0.1: Soft Cap Weighting (replace binary masking)**

Instead of binary masking (capped=0, uncapped=1), apply a soft weight that scales with distance from cap:

```
cap_distance = 1.0 - true_rul_seq_norm  # 0=capped, 1=RUL=0
cap_weight = cap_distance^power         # power=0.5 (sqrt) by default
cap_weight = clamp(cap_weight, floor, 1.0)  # floor=0.05, never fully zero
```

**Benefits**:
- Capped timesteps still contribute gradient (no vanishing)
- Uncapped late-life timesteps get full weight
- Smooth transition, no discontinuities

**Config knobs**:
- `soft_cap_enable: bool = True` (default ON)
- `soft_cap_power: float = 0.5` (sqrt gives gradual ramp)
- `soft_cap_floor: float = 0.05` (minimum weight for capped timesteps)

**P0.2: Stricter Informative Sampling**

Require at least X% of future timesteps to be uncapped (not just any single one):

```
uncapped_frac = (y_tr_norm < (1.0 - eps)).mean(axis=1)  # Per-sample
is_informative = uncapped_frac >= threshold            # threshold=0.3 (30%)
```

**Modes**:
- `"uncapped_frac"` (NEW default): Require ≥30% uncapped future timesteps
- `"mean_below_threshold"`: Require mean(future_rul) < 0.9
- `"future_min_lt_cap"` (legacy): Any single uncapped timestep
- `"future_has_zero"`: Must include RUL=0

**Config knobs**:
- `informative_sampling_mode: str = "uncapped_frac"`
- `informative_uncapped_frac_threshold: float = 0.3`
- `informative_mean_threshold: float = 0.9`
- `keep_prob_noninformative: float = 0.05` (reduce from 0.1)

**Scope**:
- **In-scope**: `src/world_model_training_v3.py` (loss computation, sampling), `src/world_model_training.py` (WorldModelConfig), `src/experiment_configs.py` (new experiment)
- **Out-of-scope**: Model architecture changes, Tobit/survival losses (deferred to P1/P2)

#### Alternatives considered

1. **Tobit / Survival-Analysis Loss**
   - Pros: Theoretically correct for censored data
   - Cons: High implementation complexity, harder to debug, interacts with existing HI/damage losses
   - Decision: Defer to P1 after validating P0 works

2. **Delta-RUL Prediction (architectural)**
   - Pros: Prevents plateau by construction (predict negative deltas)
   - Cons: Already partially implemented in `latent_hi_rul_dynamic_delta_v2`; needs validation first
   - Decision: Defer to P1, enable existing implementation

3. **Stratified Sampling by RUL Range**
   - Pros: Forces balanced representation (33% late/mid/early)
   - Cons: Requires custom sampler, changes epoch definition
   - Decision: Defer; P0.2 (stricter informative) is simpler and may suffice

4. **Keep current approach with higher cap_reweight_weight**
   - Pros: Minimal code change
   - Cons: Still sample-level, doesn't address timestep-level issue
   - Decision: Rejected; doesn't address root cause

#### Consequences

**Positive**:
- Capped timesteps contribute gradient (no vanishing gradient zones)
- Training focuses on uncapped late-life dynamics
- Expected: Bias_LAST reduction from +70 to <+30 cycles
- Expected: pred_rul_seq_std increase from ~0.02 to ≥0.15
- Minimal code change, backwards compatible (new flags default to safe values)

**Negative**:
- Adds 5 new config knobs (manageable)
- May reduce effective training set size (stricter sampling)
- Requires validation run to confirm effectiveness

**Risks**:
- **Leakage**: None introduced (sampling uses train data only)
- **Feature/scaler mismatch**: None (no changes to feature pipeline)
- **Censoring misinterpretation**: N/A for this fix (training-only change)
- **Metric pitfalls**: Must track split metrics (capped vs uncapped LAST) to verify improvement

#### Validation plan (tests / metrics / plots)

**Tests**:
- Verify `soft_cap_enable=True` produces non-zero gradients for all batches
- Verify `informative_sampling_mode="uncapped_frac"` reduces training set appropriately
- Wiring debug: log `cap_weight_mean`, `cap_weight_std`, `frac_floor_weight`

**Metrics** (comparison: baseline vs P0 fix):

| Metric | Baseline | Target (Go) |
|--------|----------|-------------|
| Bias_LAST | ~+70 cycles | < +30 cycles |
| R²_LAST | ~0 | > 0.3 |
| pred_rul_seq_std | ~0.02 | ≥ 0.15 |
| RMSE_LAST | ~75 | < 50 |

**Plots/Diagnostics**:
- `pred_rul_seq_norm` distribution (should show variance, not spike at 1.0)
- Per-engine trajectory plots (late-life should show declining RUL)
- Split metrics: Bias on capped vs uncapped engines separately

**Gates (Go/No-Go)**:
- **Go** if: Bias_LAST < +30 AND pred_rul_seq_std ≥ 0.10 AND trajectories show late-life decline
- **No-Go** if: RMSE regresses by >20% OR gradients still vanish OR no improvement in Bias

#### Links

- Runs: `results/fd004/fd004_wm_v1_infwin_wiringcheck_k0/`, `results/fd004/fd004_wm_v1_infwin_capweight_k1/`
- Related ADRs:
  - `docs/decisions/ADR-0001-project-governance.md`
  - `docs/decisions/ADR-0005-censoring-aware-training.md`
- Docs: `docs/context.md`, `docs/roadmap.md`
- Code:
  - `src/world_model_training_v3.py` (loss computation, lines ~2700-2970)
  - `src/world_model_training.py` (WorldModelConfig dataclass)
  - `src/experiment_configs.py` (experiment definitions)

