# Cursor Implementation Prompt: ADR-0010 WM-V1 Cap-Collapse Fix

## Summary of proposed plan

- **Why now**: WorldModelV1 on FD004 shows severe cap-sticking collapse (Bias_LAST ~+70, R²_LAST ~0) due to capped targets dominating training.
- **P0.1**: Replace binary cap masking with soft per-timestep weighting (distance from cap → weight)
- **P0.2**: Implement stricter informative sampling requiring ≥30% uncapped future timesteps
- **Expected impact**: Bias_LAST reduction to <+30, pred_rul variance increase, visible late-life trajectory decline
- **Backwards compatible**: New flags default to safe values; existing runs unaffected

---

## Files to edit

| File | Status | Purpose |
|------|--------|---------|
| `@src/world_model_training.py` | existing | Add new config fields to `WorldModelTrainingConfig` |
| `@src/world_model_training_v3.py` | existing | Implement soft cap weighting + stricter sampling logic |
| `@src/experiment_configs.py` | existing | Add new experiment config `get_fd004_wm_v1_p0_softcap_k3_config` |

---

## Cursor Implementation Prompt (single block)

```text
You are working in the repo AI_Turbine_RUL_Monitor_CMAPSS.

SCOPE GUARD
- Only edit these files:
  - @src/world_model_training.py
  - @src/world_model_training_v3.py
  - @src/experiment_configs.py

GOAL
Implement ADR-0010 P0 fixes to mitigate FD004 WorldModelV1 cap-collapse:
- P0.1: Soft per-timestep cap weighting for the RUL future loss (no zero-gradient zones)
- P0.2: Stricter informative sampling: require >= 30% uncapped timesteps in the future horizon

STEP 0 (MANDATORY BEFORE EDITS)
- Locate the exact code path that:
  (a) builds training windows/targets for WM-V1,
  (b) performs "informative sampling" filtering,
  (c) computes the RUL future loss that is actually added to total_loss.
- Confirm variable names and tensor shapes used in the active path.
- If multiple loss paths exist, modify ONLY the active WM-V1 RUL-future-loss path.

STEP 1: Config fields (src/world_model_training.py)
Add after the existing cap_mask_* fields (around line 239):

```python
# --------------------------------------------------
# WM-V1: Soft cap weighting (ADR-0010, replaces binary masking)
# --------------------------------------------------
# If True, use soft per-timestep weights based on distance from cap.
# Replaces binary cap_mask when enabled for RUL future loss ONLY.
soft_cap_enable: bool = False  # Default OFF for backwards compatibility
soft_cap_power: float = 0.5  # Weight = distance^power (0.5=sqrt gives gradual ramp)
soft_cap_floor: float = 0.05  # Minimum weight for capped timesteps (never zero)

# Threshold for "uncapped_frac" informative sampling mode
informative_uncapped_frac_threshold: float = 0.3
```

Note: The informative_sampling_mode field already exists. Just ensure "uncapped_frac" is handled
in the validation/runtime code (no Literal type change needed if using getattr).

STEP 2: Stricter informative sampling (src/world_model_training_v3.py)
Find the informative sampling block (around line 1748-1754).

Add handling for mode "uncapped_frac":

```python
# ----------------------------------------------------------
# P0.2: Stricter informative sampling (ADR-0010)
# ----------------------------------------------------------
mode = str(getattr(world_model_config, "informative_sampling_mode", "future_min_lt_cap") or "future_min_lt_cap")
eps = float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6)

if mode == "future_has_zero":
    is_inf = future_min <= float(eps)
elif mode == "uncapped_frac":
    # NEW: Require at least X% of future timesteps to be uncapped
    uncapped_frac_threshold = float(getattr(world_model_config, "informative_uncapped_frac_threshold", 0.3) or 0.3)
    uncapped_mask = y_tr_norm < (1.0 - float(eps))  # (N_tr, H)
    uncapped_frac = uncapped_mask.mean(axis=1)  # (N_tr,) fraction uncapped per sample
    is_inf = uncapped_frac >= uncapped_frac_threshold
else:
    # Legacy: "future_min_lt_cap" - informative if ANY timestep is below cap
    is_inf = future_min < (1.0 - float(eps))

inf_frac = float(is_inf.mean()) if is_inf.size > 0 else 0.0
```

STEP 3: Soft cap weighting for RUL future loss ONLY (src/world_model_training_v3.py)
Find the RUL future loss computation section (around line 2730-2770).

After computing `true_rul_seq_norm`, add soft cap weighting:

```python
# ----------------------------------------------------------
# P0.1: Soft cap weighting (ADR-0010)
# Apply to RUL future loss ONLY, not to HI or other losses.
# ----------------------------------------------------------
soft_cap_enable = bool(getattr(world_model_config, "soft_cap_enable", False))
soft_cap_power = float(getattr(world_model_config, "soft_cap_power", 0.5) or 0.5)
soft_cap_floor = float(getattr(world_model_config, "soft_cap_floor", 0.05) or 0.05)

if soft_cap_enable:
    # Distance from cap: 0 = fully capped, 1 = RUL=0
    cap_distance = (1.0 - true_rul_seq_norm).clamp(0.0, 1.0)  # (B, H, 1)
    
    # Soft weight: power gives gradual ramp from cap to low RUL
    cap_weight = cap_distance.pow(soft_cap_power)
    cap_weight = cap_weight.clamp(soft_cap_floor, 1.0)  # Never fully zero
else:
    cap_weight = None  # Fall back to binary masking
```

Modify the RUL loss computation to use soft weights when enabled:

```python
if soft_cap_enable and cap_weight is not None:
    # Soft-weighted MSE (replaces binary masking for RUL future loss)
    diff2_weighted = diff2 * cap_weight
    num_i = diff2_weighted.sum(dim=(1, 2))
    den_i = cap_weight.sum(dim=(1, 2)).clamp_min(1e-6)
    loss_i = num_i / den_i  # (B,)
else:
    # Legacy binary masking path
    num_i = (diff2 * mask_f).sum(dim=(1, 2))
    den_i = mask_f.sum(dim=(1, 2)).clamp_min(1e-6)
    loss_i = num_i / den_i  # (B,)
```

STEP 4: Wiring/debug stats (src/world_model_training_v3.py)
In the wiring_debug block for RUL loss (around line 2847), add:

```python
if soft_cap_enable and cap_weight is not None:
    wiring_debug[k]["rul"]["soft_cap_stats"] = {
        "soft_cap_enable": True,
        "soft_cap_power": float(soft_cap_power),
        "soft_cap_floor": float(soft_cap_floor),
        "cap_weight_min": float(cap_weight.detach().min().cpu()),
        "cap_weight_mean": float(cap_weight.detach().mean().cpu()),
        "cap_weight_max": float(cap_weight.detach().max().cpu()),
        "cap_weight_std": float(cap_weight.detach().std().cpu()),
        "frac_floor_weight": float((cap_weight <= soft_cap_floor + 1e-6).float().mean().cpu()),
    }

# Also add uncapped_frac batch stats
try:
    yb_norm = true_rul_seq_norm[:, :, 0].detach()  # (B, H)
    eps_dbg = float(getattr(world_model_config, "informative_eps_norm", 1e-6) or 1e-6)
    uncapped_mask_dbg = (yb_norm < (1.0 - eps_dbg)).float()
    uncapped_frac_per_sample = uncapped_mask_dbg.mean(dim=1)  # (B,)
    frac_capped_timesteps = float((yb_norm >= (1.0 - eps_dbg)).float().mean().cpu())
    wiring_debug[k]["rul"]["uncapped_frac_stats"] = {
        "uncapped_frac_batch_mean": float(uncapped_frac_per_sample.mean().cpu()),
        "uncapped_frac_batch_min": float(uncapped_frac_per_sample.min().cpu()),
        "uncapped_frac_batch_max": float(uncapped_frac_per_sample.max().cpu()),
        "frac_capped_timesteps": frac_capped_timesteps,
    }
except Exception:
    pass
```

STEP 5: Experiment config (src/experiment_configs.py)
After get_fd004_wm_v1_infwin_capmask_k2_config (around line 1171), add:

```python
def get_fd004_wm_v1_p0_softcap_k3_config() -> ExperimentConfig:
    """
    P0 cap-collapse fix (ADR-0010):
      - soft_cap_enable=True (P0.1: soft per-timestep weighting for RUL future loss)
      - informative_sampling_mode="uncapped_frac" with threshold=0.3 (P0.2: stricter sampling)
      - keeps late weighting from previous experiments
      - disables binary cap_mask (replaced by soft weighting)
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_infwin_capmask_k2_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3"
    cfg.setdefault("training_params", {})
    cfg["training_params"]["num_epochs"] = 30  # More epochs to see effect
    
    wmp = cfg.setdefault("world_model_params", {})
    
    # P0.1: Soft cap weighting (replaces binary masking for RUL future loss)
    wmp["soft_cap_enable"] = True
    wmp["soft_cap_power"] = 0.5  # sqrt
    wmp["soft_cap_floor"] = 0.05
    wmp["cap_mask_enable"] = False  # Disable binary masking (replaced by soft)
    
    # P0.2: Stricter informative sampling
    wmp["informative_sampling_enable"] = True
    wmp["informative_sampling_mode"] = "uncapped_frac"
    wmp["informative_uncapped_frac_threshold"] = 0.3
    wmp["keep_prob_noninformative"] = 0.05  # Reduce from 0.1
    
    # Keep late weighting
    wmp["late_weight_enable"] = True
    wmp["late_weight_factor"] = 10.0
    
    # Logging
    wmp["debug_wiring_enable"] = True
    wmp["debug_wiring_epochs"] = 1
    wmp["log_informative_stats"] = True
    
    return cfg
```

Also add to the get_experiment_by_name function (around line 3946+):

```python
if experiment_name == "fd004_wm_v1_p0_softcap_k3":
    return get_fd004_wm_v1_p0_softcap_k3_config()
```

ACCEPTANCE CRITERIA (P0)

- **Reproducibility**:
  - New config fields have safe defaults (soft_cap_enable=False)
  - Existing experiments unaffected (backwards compatible)
  - Feature dimension unchanged

- **Evaluation** (comparison: baseline k1 vs P0 fix):
  | Metric | Baseline (k1) | Target (Go) |
  |--------|---------------|-------------|
  | Bias_LAST | ~+70 cycles | < +30 cycles |
  | pred_rul_seq_std | ~0.02 | ≥ 0.08-0.12 |
  | RMSE_LAST | ~75 | ≥15-20% improvement vs k1 |

- **Diagnostics**:
  - wiring_debug.json includes soft_cap_stats with non-trivial distribution (not all near floor)
  - wiring_debug.json includes uncapped_frac_stats
  - Log output shows informative_frac for new mode
  - pred_rul_seq_norm distribution shows variance (not spike at 1.0)

- **FD004**: treat "EOL" as last observed cycle (right-censored) in all labels/titles.

- **Gates (Go/No-Go)**:
  - **Go** if: Bias_LAST < +30 AND pred_rul_seq_std ≥ 0.08 AND cap_weight shows distribution
  - **No-Go** if: RMSE regresses by >20% OR gradients still vanish OR no improvement in Bias

OUTPUT ARTIFACTS
- `results/fd004/fd004_wm_v1_p0_softcap_k3/summary.json`
- `results/fd004/fd004_wm_v1_p0_softcap_k3/wiring_debug.json` (with soft_cap_stats + uncapped_frac_stats)
- `results/fd004/fd004_wm_v1_p0_softcap_k3/transformer_world_model_v1_best_fd004_wm_v1_p0_softcap_k3.pt`

RUN COMMAND
```bash
python -u run_experiments.py --experiments fd004_wm_v1_p0_softcap_k3 --device cuda
```
```

---

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Leakage** | Low | High | No train/test boundary changes; sampling uses train data only |
| **Feature/scaler mismatch** | None | High | No changes to feature pipeline |
| **Censoring misinterpretation** | Low | Medium | Training-only change; eval labels unchanged |
| **Metric pitfalls (Bias)** | Medium | Medium | Track split metrics (capped vs uncapped LAST separately) |
| **Vanishing gradients persist** | Medium | High | `soft_cap_floor=0.05` ensures non-zero gradient; monitor cap_weight_mean |
| **Over-aggressive sampling** | Medium | Medium | If too few samples after filtering, reduce threshold or increase keep_prob |
| **Runtime/compute** | Low | Low | Soft weighting is O(1) additional ops per batch |

---

## Related links

- ADR: `docs/decisions/ADR-0010-wm-v1-cap-collapse-fix.md`
- Baseline runs: `results/fd004/fd004_wm_v1_infwin_wiringcheck_k0/`, `results/fd004/fd004_wm_v1_infwin_capweight_k1/`
- Context: `docs/context.md`
- Roadmap: `docs/roadmap.md`

