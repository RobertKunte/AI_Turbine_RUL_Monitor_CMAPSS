# Reviewer Assessment: FD004 EOL Windows + Horizon Mask Prompt

**Reviewer**: Technical Critique  
**Date**: 2025-01-XX  
**Prompt Under Review**: Update `fd004_wm_v1_p0_softcap_k3` to include EOL windows with horizon masking  
**Related**: Horizon mask implementation (already completed), ADR-0010

---

## Executive Summary

**Status**: ✅ **GO** with clarifications

The proposed change is **sound and necessary**: enabling horizon masking and ensuring near-EOL samples are included in training will address the "floor around 24–30" issue. The implementation is already partially in place (`require_full_horizon=False` is hardcoded), so this is primarily a config update.

**Key finding**: The code already has `require_full_horizon=False` hardcoded (line 1871), so the main change is enabling `use_horizon_mask=True` in the experiment config. The prompt correctly identifies this.

---

## 1) Critical Issues (Must-Fix)

### Issue 1.1: Missing verification that `return_mask=True` is already set

**Why it matters**:  
The prompt states "the dataset builder returns `mask` for the future horizon" but doesn't verify that `return_mask=True` is already set in the code. If it's not set, the mask won't be created regardless of `use_horizon_mask`.

**Minimal fix required**:  
Add to Step B:

> "Verify that `build_sliding_windows` is called with `return_mask=True` (should already be set at line 1895 in `world_model_training_v3.py`). If not, this must be added."

**Location**: Step B of the prompt.

---

### Issue 1.2: Ambiguity about Stage-1 vs Stage-2 window builders

**Why it matters**:  
The prompt mentions "world_model_universal_v3 stage-1" but the actual function is `train_world_model_universal_v3` which has Stage-2 training. There are TWO window builders:
- Stage-1 (lines 234-240): Uses `use_padded_horizon_targets` flag
- Stage-2 (lines 1867-1895): Hardcodes `require_full_horizon=False`

The prompt should clarify which one is being modified.

**Minimal fix required**:  
Clarify in Step B:

> "Identify BOTH window builders:
> - Stage-1 (lines 234-240): controlled by `use_padded_horizon_targets` flag
> - Stage-2 (lines 1867-1895): hardcoded `require_full_horizon=False` (already correct)
> 
> The Stage-2 builder is the one used for `fd004_wm_v1_p0_softcap_k3`, so verify it already has `require_full_horizon=False` (it does)."

---

### Issue 1.3: Missing check for `use_horizon_mask` propagation

**Why it matters**:  
The prompt assumes setting `use_horizon_mask=True` in the config will automatically propagate to the training function. Need to verify the config key name matches what the code expects.

**Minimal fix required**:  
Add to Step C:

> "Verify the config key name: the code reads `use_horizon_mask` from `world_model_config` (line 2182). Ensure the experiment config sets `wmp["use_horizon_mask"] = True` (not a different key name)."

---

## 2) Important Improvements (Should-Fix)

### Issue 2.1: y_eol min expectation clarification

**Why it matters**:  
The prompt states "y_eol min should be 0" but with `eol_target_mode="future0"` (line 1879), the scalar target is `future_rul[0]`, which may not be exactly 0 even for near-EOL windows (it's the RUL at t+1, not t). The prompt should clarify the expected range.

**Recommendation**:  
Clarify in acceptance criteria:

> "Training seq2seq build log shows y_eol min <= 5 (ideally 0-2) and/or pad_frac is printed and > 0.0. Note: with `eol_target_mode="future0"`, y_eol min may be ~1-2 even for EOL windows since it's RUL at t+1."

**Impact**: Medium (prevents false negatives in validation)

---

### Issue 2.2: Missing verification of mask propagation to Stage-2 dataset

**Why it matters**:  
The prompt mentions "Make sure any Stage-2 dataset creation also includes the mask" but doesn't specify how to verify this. The implementation already handles this, but the prompt should include a verification step.

**Recommendation**:  
Add to Step F:

> "Verify mask propagation: After dataset creation (around line 2184-2196), check that `horizon_mask_tr` is not None when `use_horizon_mask=True`. Add a print statement if needed: `print(f'[Dataset] horizon_mask_tr is None: {horizon_mask_tr is None}')`"

**Impact**: Low (verification only)

---

### Issue 2.3: Acceptance criteria for "no hard floor" is vague

**Why it matters**:  
"Test LAST pred range includes values < 20" is a post-training check, not something that can be verified during implementation. This should be moved to a separate validation section or clarified as a post-run check.

**Recommendation**:  
Split acceptance criteria:

> "**Implementation acceptance** (can verify during code changes):
> - Config sets `use_horizon_mask=True`
> - Training logs show `use_horizon_mask=True` and `pad_frac > 0`
> - Feature dimension check passes (659 == 659)
> 
> **Post-training validation** (requires running experiment):
> - Test LAST pred range includes values < 20 (no hard floor at ~24–30)
> - Training y_eol min <= 5"

**Impact**: Low (clarity)

---

## 3) Nice-to-Haves

### Suggestion 3.1: Add explicit check for mask shape consistency

**Why**:  
Verify that the mask shape matches expectations (should be `(N, H, 1)` or `(N, H)`).

**Location**: Step E (sanity print)

---

### Suggestion 3.2: Document expected pad_frac range

**Why**:  
Help implementer understand what "good" padding fraction looks like. For FD004 with H=30, near-EOL windows will have varying amounts of padding.

**Location**: Step E or acceptance criteria

---

## 4) Patch Suggestions (Prompt Improvements)

### Patch 4.1: Clarify which window builder is used

**Current text**:  
> "B) Identify the windowing / seq2seq builder parameters used by world_model_universal_v3 stage-1"

**Suggested revision**:  
> "B) Identify the windowing / seq2seq builder parameters used by `train_world_model_universal_v3`:
> - Stage-2 window builder (lines 1867-1895): This is the one used for `fd004_wm_v1_p0_softcap_k3`
> - Verify it already has `require_full_horizon=False` (it does, line 1871)
> - Verify it calls `build_sliding_windows` with `return_mask=True` (it does, line 1895)"

---

### Patch 4.2: Add explicit config key verification

**Current text**:  
> "C) Change the config for this experiment (ONLY this experiment) to:
>    - require_full_horizon: False"

**Suggested revision**:  
> "C) Change the config for this experiment (ONLY this experiment) to:
>    - `wmp["use_horizon_mask"] = True` (this enables mask creation and usage)
>    - Note: `require_full_horizon=False` is already hardcoded in the window builder (line 1871), so no config change needed for that"

---

### Patch 4.3: Clarify y_eol min expectation

**Current text**:  
> "During training seq2seq build: print pad_frac (not NA) and y_eol min should be 0"

**Suggested revision**:  
> "During training seq2seq build: print pad_frac (not NA) and y_eol min should be <= 5 (ideally 0-2). Note: with `eol_target_mode="future0"`, y_eol is RUL at t+1, so even EOL windows may show y_eol ~1-2."

---

### Patch 4.4: Add verification step for mask propagation

**Add after Step E**:  
> "**Step E.5**: Verify mask is propagated to dataset:
> ```python
> # After dataset creation (around line 2184), add:
> if use_horizon_mask:
>     assert horizon_mask_tr is not None, 'horizon_mask_tr is None despite use_horizon_mask=True'
>     print(f'[Dataset] horizon_mask_tr shape: {horizon_mask_tr.shape}')
> ```"

---

### Patch 4.5: Split acceptance criteria

**Current text**:  
> "Acceptance Criteria (must meet all)
> - Training seq2seq build log shows y_eol min <= 5 (ideally 0) and/or pad_frac is printed and > 0.0.
> - The first-batch log shows use_horizon_mask=True and pad_frac > 0.
> - Diagnostics: `Verified feature dimension: 659 == 659` and no feature mismatch.
> - Test LAST pred range includes values < 20 (no hard floor at ~24–30).
> - No silent fallback to heuristic feature pipeline; must use schema_version=2 config."

**Suggested revision**:  
> "**Acceptance Criteria - Implementation** (can verify during code changes):
> - Config sets `wmp["use_horizon_mask"] = True` in experiment config
> - Training seq2seq build log shows pad_frac is printed and > 0.0
> - First-batch log shows `use_horizon_mask=True` and `pad_frac > 0`
> - Diagnostics: `Verified feature dimension: 659 == 659` and no feature mismatch
> - No silent fallback to heuristic feature pipeline; must use schema_version=2 config
> 
> **Acceptance Criteria - Post-Training Validation** (requires running experiment):
> - Training seq2seq build log shows y_eol min <= 5 (ideally 0-2)
> - Test LAST pred range includes values < 20 (no hard floor at ~24–30)"

---

## 5) Go / No-Go Decision

### ✅ **GO** (with clarifications)

**Rationale**:  
- ✅ **No data leakage**: Including near-EOL windows doesn't introduce leakage (they're still training-time samples)
- ✅ **No feature/scaler changes**: Pure config update, no pipeline modifications
- ✅ **Backwards compatible**: Only affects this specific experiment
- ✅ **Addresses real issue**: Floor at 24–30 is a known problem from train/eval distribution mismatch
- ✅ **Implementation already exists**: Horizon mask code is already in place

**Mandatory conditions before implementation**:
1. ✅ Verify `return_mask=True` is set in window builder (it is, line 1895)
2. ✅ Clarify which window builder is being modified (Stage-2, already has `require_full_horizon=False`)
3. ✅ Verify config key name matches code expectation (`use_horizon_mask`)

**Recommended before merging**:
- Address Issue 2.1 (y_eol min expectation)
- Address Issue 2.2 (mask propagation verification)
- Split acceptance criteria into implementation vs post-training

**Gates satisfied**:
- ✅ **Reproducibility gate**: No changes to feature pipeline or scalers
- ✅ **Leakage gate**: No future information leakage (near-EOL windows are still training samples)
- ✅ **FD004 censoring gate**: N/A (training-only change, doesn't affect eval labels)
- ✅ **Diagnostics gate**: Prompt includes verification steps

---

## Additional Notes

### Current State Analysis

**Already implemented**:
- ✅ `require_full_horizon=False` hardcoded (line 1871)
- ✅ `return_mask=True` in window builder call (line 1895)
- ✅ Horizon mask extraction and application in loss computation (lines 3041-3062)
- ✅ Mask propagation to dataset (lines 2174-2196)

**Missing**:
- ❌ `use_horizon_mask=True` in experiment config (needs to be added)

**Conclusion**: The implementation is 95% complete. This is primarily a config update with verification logging.

### Risk Assessment

- **Low risk**: Config-only change, implementation already exists
- **High value**: Addresses known "floor" issue
- **Mitigation**: Comprehensive logging to verify fix is active

### Testing Strategy

After implementation, verify:
1. Config file has `use_horizon_mask=True`
2. Training logs show mask is created and used
3. Padding fraction > 0 in logs
4. y_eol min drops below 30 (ideally <= 5)
5. Post-training: test predictions go below 20

---

**Reviewer Signature**: Technical Critique  
**Next Steps**: Implement with clarifications, then validate on FD004 baseline run.

