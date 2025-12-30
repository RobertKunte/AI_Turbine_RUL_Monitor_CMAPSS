### ADR-0011: Encoder freeze policy & checkpoint gating for tf_cross world models (FD004)

#### Status
Accepted

#### Context
- **Problem**: In FD004 world_model_universal_v3 with tf_cross decoder, we observed a failure mode where:
  - With encoder frozen for multiple epochs (default 5), the decoder initially predicts near-constant very small RUL values (~0-5 cycles).
  - Validation "last-window" RMSE (`val_rmse_last`) can look artificially good early (epoch 1-2) because:
    - Val set contains many near-EOL windows where true RUL is already low (0-20 cycles).
    - Decoder's constant small predictions (~5 cycles) happen to match these low true values, yielding low RMSE.
    - This is a "fake-good" metric that doesn't generalize to test set.
  - Later, when unfreezing the encoder (epoch 5+) or fully unfreezing, predictions become plausible across [0..max_rul] and RMSE improves, but early checkpoint selection already locked in the wrong model (epoch 1).
  - Test set evaluation then shows severe bias (predictions too low) and poor RMSE (~70 vs expected ~15-20).

- **Evidence**: 
  - Run `wm_v3_fd004_b2_tf_cross_qr_b21_traj` showed best checkpoint at epoch 1 with `val_rmse_last=15.33`, but test RMSE was ~70 with strong negative bias.
  - Diagnostics showed prediction range was very small (min~0, max~5) in early epochs.
  - After unfreeze, prediction range expanded to [0..125] but checkpoint was already selected.

- **Constraints**: 
  - Must preserve transfer learning benefits of encoder freeze.
  - Must maintain backwards compatibility with existing configs.
  - Must work with existing checkpoint selection logic (`best_metric=val_rmse_last` for FD004).

#### Decision
1. **Default freeze duration**: Reduce `freeze_encoder_epochs` from 5 to 2 for tf_cross decoder experiments.
   - Rationale: 2 epochs is sufficient for decoder to stabilize while minimizing fake-good window.
   - Full unfreeze recommended for FD004 tf_cross (with `encoder_lr_mult=0.1` to prevent encoder from overfitting).

2. **Checkpoint gating policy**: Implement minimum epoch gating so "best" checkpoint cannot be selected before:
   - `best_ckpt_min_epoch`: Minimum epoch before any checkpoint selection (default: `freeze_encoder_epochs + 1`).
   - `best_ckpt_min_epoch_after_unfreeze`: Minimum epochs after unfreeze before checkpoint selection (default: 2).
   - Combined: `allow_best_update = (epoch >= best_ckpt_min_epoch) AND (epochs_since_unfreeze >= best_ckpt_min_epoch_after_unfreeze)`.

3. **Prediction sanity gate** (optional, default enabled):
   - Require validation EOL prediction std >= `best_ckpt_require_pred_std_min` (default: 1.0 cycles).
   - Reject checkpoints where predictions are too constant (degenerate solution).
   - Only applies when checkpoint would otherwise be allowed (not during freeze phase).

4. **Unfreeze strategy**: Support `unfreeze_encoder_layers="all"` (or `-1`) for full encoder unfreeze.
   - Recommended for FD004 tf_cross: `unfreeze_encoder_layers="all"` with `encoder_lr_mult=0.1`.

#### Alternatives considered
- **Option A: Keep 5 epoch freeze + raise total epochs**
  - Pros: More encoder stability, preserves transfer benefits longer.
  - Cons: Fake-good window extends to epoch 5, requires longer training, still risk of early checkpoint.
  - **Not chosen**: Doesn't solve the root cause (fake-good metrics during freeze).

- **Option B: No freeze at all**
  - Pros: No fake-good window, simpler logic.
  - Cons: Loses transfer learning benefits, encoder may overfit quickly.
  - **Not chosen**: Transfer learning is valuable; we want to preserve it.

- **Option C: Short freeze (1-2 epochs) + gating** ✅ **CHOSEN**
  - Pros: Preserves transfer benefits, minimizes fake-good window, gating prevents early selection.
  - Cons: Slightly more complex logic, requires careful tuning of gating thresholds.
  - **Chosen**: Best balance of benefits and risks.

- **Option D: Full unfreeze immediately + LR warmup**
  - Pros: No fake-good window, encoder adapts from start.
  - Cons: May lose transfer benefits, requires careful LR scheduling.
  - **Not chosen**: Short freeze + gating is safer and more predictable.

#### Decision drivers
- **Avoid selecting degenerate early checkpoint**: Primary goal is to prevent epoch 1-2 checkpoints with fake-good metrics.
- **Stabilize tf_cross training**: Preserve transfer learning benefits while allowing encoder adaptation.
- **Maintain evaluation integrity**: Ensure checkpoint selection reflects true model quality, not artifacts of freeze phase.
- **Backwards compatibility**: Existing configs should continue to work (with sensible defaults).

#### Implementation details
**New config parameters** (in `world_model_config`):
- `freeze_encoder_epochs`: int (default: 2 for tf_cross, 5 for others)
- `unfreeze_encoder_layers`: int | "all" | -1 (default: 1, recommended "all" for FD004 tf_cross)
- `encoder_lr_mult`: float (default: 0.1, existing)
- `best_ckpt_min_epoch`: Optional[int] (default: `freeze_encoder_epochs + 1` if freeze active, else 0)
- `best_ckpt_min_epoch_after_unfreeze`: int (default: 2)
- `best_ckpt_require_pred_std_min`: float (default: 1.0, set to None to disable)

**Checkpoint gating logic**:
```python
# Compute min_epoch
if best_ckpt_min_epoch is not None:
    min_epoch = best_ckpt_min_epoch
elif freeze_encoder or freeze_encoder_epochs > 0:
    min_epoch = freeze_encoder_epochs + 1
else:
    min_epoch = 0

# Compute epochs since unfreeze
epochs_since_unfreeze = max(0, epoch - freeze_encoder_epochs) if freeze_encoder_epochs > 0 else epoch

# Gate checkpoint selection
checkpoint_allowed = (epoch >= min_epoch) and (epochs_since_unfreeze >= best_ckpt_min_epoch_after_unfreeze)

# Sanity gate (if enabled)
if checkpoint_allowed and best_ckpt_require_pred_std_min is not None:
    pred_std = np.std(val_eol_predictions)
    if pred_std < best_ckpt_require_pred_std_min:
        checkpoint_allowed = False
```

**Logging**:
- Per epoch: `[checkpoint] epoch=X encoder_frozen=True checkpoint_allowed=False min_epoch=3 reason_blocked=...`
- Include: `epoch`, `encoder_frozen`, `checkpoint_allowed`, `min_epoch`, `epochs_since_unfreeze`, `reason_blocked`, `pred_std` (if computed)

**Experiment config updates**:
- tf_cross experiments: `freeze_encoder_epochs=2`
- FD004 tf_cross: `unfreeze_encoder_layers="all"`, `encoder_lr_mult=0.1`
- Default gating params applied automatically

#### Consequences
**Positive**:
- Prevents selection of degenerate early checkpoints with fake-good metrics.
- More reliable checkpoint selection that reflects true model quality.
- Preserves transfer learning benefits with shorter freeze period.
- Better generalization to test set (no severe bias from constant predictions).

**Negative**:
- Slightly slower earliest convergence (checkpoint selection delayed by 2-3 epochs).
- More complex checkpoint logic (but well-logged and configurable).
- Requires careful tuning of gating thresholds (but defaults are sensible).

**Risks**:
- **Leakage**: None (gating is based on epoch count, not data).
- **Evaluation pitfalls**: Reduced risk of fake-good metrics, but still need to monitor prediction distributions.
- **Config complexity**: More parameters to tune, but defaults should work for most cases.

#### Validation plan
**Tests**:
- Unit test: Given fake training history with epochs 0..N, frozen for first K, ensure:
  - `allow_best_update=False` before `(min_epoch AND min_after_unfreeze)`
  - `allow_best_update=True` after both conditions met
- Integration test: Run 1-epoch dry-run with freeze=2, verify checkpoint not saved in epoch 1.

**Metrics**:
- Monitor `val_rmse_last` across epochs (should improve after unfreeze).
- Track prediction range/std per epoch (should expand after unfreeze).
- Compare test RMSE between old (epoch 1 checkpoint) vs new (epoch 7+ checkpoint) policies.

**Plots/Diagnostics**:
- Checkpoint selection timeline plot: show when checkpoints were allowed vs selected.
- Prediction distribution plots: show range/std evolution across epochs.
- Compare early vs late checkpoint predictions on test set.

**Gates**:
- **Go**: Checkpoint selection only after `min_epoch` AND `min_after_unfreeze` epochs.
- **No-Go**: Reject checkpoint if `pred_std < threshold` (sanity gate).

#### Links
- Related ADRs: 
  - ADR-0010: wm-v1-cap-collapse-fix (similar issue with early checkpoints)
- Implementation: `src/world_model_training_v3.py` (checkpoint gating logic)
- Config: `src/experiment_configs.py` (tf_cross experiment defaults)
- Runs: `results/fd004/wm_v3_fd004_b2_tf_cross_qr_b21_traj/...` (example of failure mode)

