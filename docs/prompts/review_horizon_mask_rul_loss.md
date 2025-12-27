# Reviewer Assessment: Horizon Mask Application to RUL Future Loss

**Reviewer**: Technical Critique  
**Date**: 2025-01-XX  
**Prompt Under Review**: Advisor agent prompt for applying horizon mask to RUL future trajectory loss  
**Related ADR**: ADR-0010 (WM-V1 Cap-Collapse Fix)  
**Context**: `docs/context.md`, `docs/decisions/ADR-0010-wm-v1-cap-collapse-fix.md`

---

## Executive Summary

**Status**: ✅ **GO** with minor clarifications

The proposed change is **sound and necessary**: padded future timesteps (when available horizon < H) should not contribute to RUL loss, just as they are already excluded from HI losses. The implementation plan is minimal, backwards-compatible, and addresses a real gap in the current codebase.

**Key finding**: The horizon mask (`valid_mask_seq`) is already created and available in batches when `use_horizon_mask=True`, but it's only applied to HI trajectory losses, not RUL future loss. This is an inconsistency that should be fixed.

---

## 1) Critical Issues (Must-Fix)

### Issue 1.1: Missing explicit check for `use_horizon_mask` enablement

**Why it matters**:  
The prompt assumes `valid_mask_seq` is always available, but it's only created when `use_horizon_mask=True` (line 215 in `world_model_training_v3.py`). If this flag is `False`, `mask_batch` will be `None`, and the code will fail or silently skip masking.

**Minimal fix required**:  
Add explicit handling in the prompt:

```python
# Extract horizon mask if enabled
if use_horizon_mask and mask_batch is not None:
    valid_mask_seq = mask_batch.squeeze(-1)  # (B, H)
else:
    valid_mask_seq = torch.ones_like(true_rul_seq_norm[:, :, 0])  # (B, H) - all valid
```

**Location**: Step 2 of the prompt, before combining masks.

---

### Issue 1.2: Mask shape broadcasting ambiguity

**Why it matters**:  
The prompt states `mask_rul = mask_f * valid_mask_seq` but doesn't specify shape alignment. `mask_f` is `(B, H, 1)` (from line 2989), while `valid_mask_seq` is `(B, H)` after squeezing. The multiplication will broadcast, but this should be explicit.

**Minimal fix required**:  
Clarify in Step 2:

```python
# Ensure shapes align: mask_f is (B, H, 1), valid_mask_seq is (B, H)
valid_mask_seq_expanded = valid_mask_seq.unsqueeze(-1) if valid_mask_seq.dim() == 2 else valid_mask_seq  # (B, H, 1)
mask_rul = mask_f * valid_mask_seq_expanded  # (B, H, 1)
```

---

### Issue 1.3: Missing validation that padded samples exist

**Why it matters**:  
The prompt's requirement #5 asks to "confirm that batches include some samples with valid_future_len < H", but this check should happen **before** implementing the change to verify the fix is needed. If no samples have padding, the change is unnecessary (though harmless).

**Minimal fix required**:  
Add a pre-implementation diagnostic step:

```python
# Pre-check: verify padding exists in training data
if horizon_mask_train_split is not None:
    pad_frac = 1.0 - horizon_mask_train_split.mean().item()
    print(f"[Pre-check] Training data padding fraction: {pad_frac:.4f}")
    if pad_frac < 1e-4:
        print("[WARNING] No padding detected. Horizon masking may be unnecessary.")
```

**Location**: Add as Step 0.5 (after locating code paths, before Step 1).

---

## 2) Important Improvements (Should-Fix)

### Issue 2.1: Soft cap weighting interaction with horizon mask

**Why it matters**:  
The prompt doesn't specify how soft cap weighting (from ADR-0010) interacts with horizon masking. If `soft_cap_enable=True`, the code uses `soft_cap_weight` instead of `mask_f`. The horizon mask should still be applied to zero out padded timesteps.

**Recommendation**:  
Clarify in Step 3 that horizon masking applies **regardless** of soft cap mode:

```python
# Apply horizon mask to soft cap weights if enabled
if soft_cap_enable and cap_weight is not None:
    cap_weight = cap_weight * valid_mask_seq_expanded  # Zero out padded timesteps
    # Then use cap_weight for loss computation
```

**Impact**: Medium (ensures consistency)

---

### Issue 2.2: Late-weighting and informative sampling interaction

**Why it matters**:  
Late-weighting (`late_weight_factor`) and informative sampling are applied at the **sample level**, not timestep level. The prompt doesn't clarify whether these should be computed **before** or **after** horizon masking. If a sample has 5 valid timesteps out of 30, should late-weighting use the min of those 5, or the original 30?

**Recommendation**:  
Clarify that late-weighting should use **only valid timesteps**:

```python
# Compute late-weighting mask using only valid timesteps
if late_weight_enable:
    fut_min_valid = (true_rul_seq_norm * valid_mask_seq_expanded).min(dim=1).values  # (B,)
    # Use fut_min_valid instead of fut_min for late-weighting
```

**Impact**: Medium (affects training dynamics)

---

### Issue 2.3: Missing check for Stage-1 vs Stage-2 paths

**Why it matters**:  
The code has multiple loss computation paths (Stage-1 freeze vs Stage-2 unfreeze). The prompt should verify that **both paths** are updated, or explicitly state which path is being modified.

**Recommendation**:  
Add to Step 0:

```python
# Verify which training stage path is active:
# - Stage-1 (freeze_encoder_epochs > 0): lines ~680-900
# - Stage-2 (after unfreeze): lines ~950-1100
# Update BOTH paths if both are used, or document which path is active.
```

**Impact**: Low (code organization)

---

## 3) Nice-to-Haves

### Suggestion 3.1: Add unit test for mask combination

**Why**:  
Mask combination logic (`mask_f * valid_mask_seq`) is critical and should be tested. A simple test can verify:
- Shapes broadcast correctly
- Padded timesteps are zeroed
- Cap/early masks still work

**Location**: `tests/test_world_model_masks.py` (new file)

---

### Suggestion 3.2: Log mask statistics per epoch

**Why**:  
The prompt asks for first-epoch logs, but tracking mask statistics across epochs helps diagnose training dynamics (e.g., if padding fraction changes as informative sampling filters samples).

**Location**: Add to epoch summary logging (around line 2500+)

---

### Suggestion 3.3: Add visualization of masked vs unmasked loss contributions

**Why**:  
A diagnostic plot showing loss contribution per timestep (masked vs unmasked) would help validate the fix is working as intended.

**Location**: Optional diagnostic script

---

## 4) Patch Suggestions (Prompt Improvements)

### Patch 4.1: Clarify mask availability precondition

**Current text**:  
> "1) Locate where the window builder / dataloader provides valid_mask_seq (horizon mask). Ensure it is available in the batch used for RUL loss."

**Suggested revision**:  
> "1) Locate where the window builder / dataloader provides valid_mask_seq (horizon mask). **Verify that `use_horizon_mask=True` is set in the config** (line 215). If not, the mask will be `None` and this change is a no-op. Ensure the mask is available in the batch used for RUL loss (it's already included in `TensorDataset` when `use_horizon_mask=True`, lines 403-405)."

---

### Patch 4.2: Add explicit shape handling

**Current text**:  
> "mask_rul = mask_f * valid_mask_seq (broadcast to match pred/true shape)."

**Suggested revision**:  
> "mask_rul = mask_f * valid_mask_seq_expanded where `valid_mask_seq_expanded = valid_mask_seq.unsqueeze(-1)` if `valid_mask_seq.dim() == 2` else `valid_mask_seq`. This ensures `(B, H, 1)` shape matching `mask_f` and `pred_rul_seq_norm`."

---

### Patch 4.3: Clarify soft cap weighting interaction

**Current text**:  
> "Keep existing weighting (soft-cap reweight, informative, late-weight) but apply them only on valid timesteps..."

**Suggested revision**:  
> "Keep existing weighting (soft-cap reweight, informative, late-weight) but apply them only on valid timesteps. **Important**: If `soft_cap_enable=True`, multiply `cap_weight` by `valid_mask_seq_expanded` before using it in loss computation. Late-weighting should compute `fut_min` using only valid timesteps: `fut_min_valid = (true_rul_seq_norm * valid_mask_seq_expanded).min(dim=1).values`."

---

### Patch 4.4: Add pre-implementation diagnostic

**Add new step after Step 0**:  
> "**Step 0.5 (Pre-check)**: Before implementing, verify that padding exists in training data:
> ```python
> if horizon_mask_train_split is not None:
>     pad_frac = 1.0 - horizon_mask_train_split.mean().item()
>     print(f'[Pre-check] Training data padding fraction: {pad_frac:.4f}')
>     assert pad_frac > 1e-4, 'No padding detected. Horizon masking may be unnecessary.'
> ```
> If padding fraction is negligible (<0.1%), document why this change is still needed."

---

## 5) Go / No-Go Decision

### ✅ **GO** (with clarifications)

**Rationale**:  
- ✅ **No data leakage**: Masking uses only training-time information (window builder knows available horizon)
- ✅ **No feature/scaler changes**: Pure loss computation modification
- ✅ **Backwards compatible**: Only affects behavior when `use_horizon_mask=True` (opt-in)
- ✅ **Addresses real gap**: HI losses use masking, RUL losses should too
- ✅ **Minimal scope**: Single file change, well-localized

**Mandatory conditions before implementation**:
1. ✅ Add explicit `use_horizon_mask` check (Issue 1.1)
2. ✅ Clarify mask shape broadcasting (Issue 1.2)
3. ✅ Add pre-implementation padding check (Issue 1.3)

**Recommended before merging**:
- Address Issue 2.1 (soft cap interaction)
- Address Issue 2.2 (late-weighting interaction)
- Verify both Stage-1 and Stage-2 paths if both are used

**Gates satisfied**:
- ✅ **Reproducibility gate**: No changes to feature pipeline or scalers
- ✅ **Leakage gate**: Masking uses only training-time information
- ✅ **FD004 censoring gate**: N/A (training-only change, doesn't affect eval labels)
- ✅ **Diagnostics gate**: Prompt includes debug logging requirements

---

## Additional Notes

### Relationship to ADR-0010
This change is **complementary** to ADR-0010 (soft cap weighting). ADR-0010 addresses **capped** timesteps (RUL at max), while this change addresses **padded** timesteps (no future data available). Both should be applied together for correct loss computation.

### Testing Strategy
After implementation, verify:
1. Loss values decrease (fewer timesteps contributing)
2. Training still converges (no vanishing gradients from over-masking)
3. EOL metrics improve or remain stable (not regress)
4. Debug logs show non-zero padding fraction

### Risk Assessment
- **Low risk**: Well-localized change, opt-in via config flag
- **Medium impact**: Should improve training dynamics by excluding invalid timesteps
- **Mitigation**: Backwards compatible defaults, comprehensive logging

---

**Reviewer Signature**: Technical Critique  
**Next Steps**: Implement with clarifications, then validate on FD004 baseline run.

