### FD004 τ=0.95 Residual-Risk Sweep (Dec 2025)

This note documents the **τ=0.95 residual-risk sweep** and the advisor’s analysis.

**Important (FD004 censoring):** “EOL” on TEST means **last observed cycle (right-censored)**, not true failure.

#### Scope
- Dataset: **FD004**
- Model family: `transformer_encoder_v1` with damage HI + multiview censoring + residual-risk head
- Sweep target: `risk_tau = 0.95`
- Runs: vary `lambda_residual_risk` (aka “w” in the name), keep `risk_low_weight=20`, `low_rul_threshold=20`.

---

### 1) Training-time TEST metrics (authoritative)
These are the metrics printed directly after training under:
**“Evaluating on test set” → “[evaluate_on_test_data] Test Metrics”**
(not the later diagnostics values).

| Run | Test RMSE (cycles) | NASA mean | Test MAE | Test Bias | Test R² |
|---|---:|---:|---:|---:|---:|
| `fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w1_low20` | 20.4248 | 10.9585 | 16.1200 | -3.8271 | 0.7742 |
| `fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w3_low20` | 21.5984 | 9.0727 | 17.9426 | -7.2572 | 0.7475 |
| `fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w5_low20` | 23.2912 | 12.1415 | 19.1378 | -7.5370 | 0.7064 |
| `fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm_multiview_residual_risk_tau95_w8_low20` | 24.7743 | 10.7685 | 20.9318 | -12.8980 | 0.6678 |

**Quick read:**
- Best **RMSE**: **`w1_low20`**
- Best **NASA mean**: **`w3_low20`** (at slightly higher RMSE)

---

### 2) Note on “training vs diagnostics” metric mismatches
We observed cases where diagnostics produced different (sometimes worse) test metrics than the training script, and those values were then written into `summary.json`.

Root cause (high likelihood):
- Diagnostics reconstructs the model from checkpoint and must re-attach runtime attributes that are **not persisted in the state_dict** (e.g. `cond_feature_indices`, `cond_in_dim`, `sensor_feature_indices_for_norm`, condition normalizer dims).
- If missing, the forward path can differ and produce different predictions.

Mitigation implemented (Dec 2025):
- `summary.json:test_metrics` is treated as **training-time authoritative** and must not be overwritten by diagnostics.
- Diagnostics writes to a dedicated block: `diagnostics_test_metrics` (and `test_metrics_diagnostics` alias).
- Diagnostics now re-derives and attaches the runtime feature-index attributes more robustly (even when `cond_in_dim` is missing/0).

---

### 3) Advisor analysis (focus run: `w1_low20`)
This section summarizes the plot-based analysis for `w1_low20`:

#### 3.1 True vs Pred (μ)
**Observed:**
- Strong saturation around **pred ≈ 90–95 cycles**
- High true RUL (>80): systematic **underestimation**, nearly horizontal band
- Low true RUL (<20): comparatively good behavior, **few dangerous overshoots**
- Mid range (30–70): high variance

**Interpretation:**
- μ behaves like a **late-EOL detector** rather than a well-calibrated regressor across the full lifetime.
- The model separates “healthy vs failing” but lacks a smooth “slow degradation” regime.

#### 3.2 True vs Pred (safe = μ − risk)
**Observed:**
- safe is mostly below μ (expected).
- Same saturation structure persists; safe looks like a mostly additive offset.
- For high true RUL, safe can be **30–40 cycles too low** (utility loss).
- For low true RUL, safe is very conservative (safety is good).

**Interpretation:**
- The risk head “works formally”, but it does not correct structural shape issues in μ.
- Risk is currently additive rather than strongly state-dependent.

#### 3.3 Error histogram (μ error = pred − true)
**Observed:**
- Mean error ≈ **-12 cycles**, std ≈ **21 cycles** (left-skewed).
- Bias suggests early-life pessimism / too-early degradation assumption.

**Engineering take:**
- Negative bias is often acceptable for safety.
- But at this magnitude it implies operational inefficiency (early maintenance).

#### 3.4 HI damage trajectories (10 degraded engines)
**Observed:**
- Monotone HI-damage (physically plausible).
- Often long plateau near 1.0 followed by abrupt drop; HI “activates” late.
- Predicted RUL follows HI mainly late (almost constant before).

**Interpretation:**
- Current latent state is close to “distance-to-failure proxy”.
- Main opportunity is to represent **slow degradation earlier** and more continuously.

---

### 4) Concrete next steps (model-near)
1) **Activate HI learning earlier**
   - add / increase an early-life regularizer (e.g. gentle HI slope prior) so HI is not binary.
2) **Make latent state ≠ EOL proxy**
   - shape the state to encode thermodynamic + degradative regime, not only late failure.
3) **Make risk state-dependent**
   - risk conditioned on HI and/or dHI/dt and/or residual energy, not only a global additive correction.

This is aligned with:
- `docs/decisions/ADR-0009-mu-vs-safe-product-contract.md` (product interface)
- `docs/roadmap.md` “avoid blind decoder” / dynamics validation (next phase)


