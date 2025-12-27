# HI / RUL Dynamics Improvement Playbook

This playbook is consulted **only when tasks involve HI or RUL trajectory behavior**
(e.g. flat HI, late collapse, RUL saturation, weak mid-life trend).

EOL / NASA performance is always the top priority.
No step may regress late-life performance.

---

## Target behavior (precise)

### Health Index (HI)
- Slight early decrease, then progressive degradation
- Monotone (or near-monotone) and smooth
- No “flat until the end” behavior

### RUL (right-censored / capped)
- High in early life but **not saturated at the cap**
- Clear downward trend in mid-life
- Conservative near EOL

FD004 note:
- EOL is always treated as **last observed cycle (right-censored)**.
- Text, plots, and diagnostics must reflect censoring explicitly.

---

## Stage 0 — Baseline diagnostics (mandatory first step)

Before changing losses or architecture, add diagnostics next to RMSE / NASA.

### Required KPIs (per-engine + aggregated)
- **HI Plateau Ratio**  
  Fraction of cycles with `HI > 0.98`

- **HI Onset Cycle**  
  First cycle where `HI < 0.95` (threshold configurable)

- **HI Curvature**  
  Mean absolute second difference:  
  `E[ |HI[t+1] - 2·HI[t] + HI[t-1]| ]`

- **RUL Slope Error**  
  Difference between fitted RUL slope (early/mid window) and expected trend under censoring

- **Saturation Rate**  
  Fraction of predictions in `[RUL_cap - δ, RUL_cap]`

### Success criteria
- Plateau ratio ↓
- Onset earlier
- Saturation ↓
- **NASA / EOL not worse**

---

## Stage 1 — Low-risk training changes (preferred)

Apply before any architectural changes.

### A) Curriculum / schedule (recommended)
Three-phase training:

1. **Phase A — Dynamics warmup**
   - No EOL loss
   - Emphasize HI / RUL trajectory shape losses

2. **Phase B — Joint**
   - Gradually ramp EOL loss (linear or cosine)
   - Keep dynamics losses active

3. **Phase C — EOL focus**
   - High EOL / NASA weight
   - Keep dynamics losses as stabilizers

### B) Small shape regularizers (start with tiny weights)
- **Early-slope regularizer**  
  Penalize near-zero slope in early HI

- **Smoothness / curvature penalty**  
  Prevent abrupt late collapse

### C) HI–RUL coupling (hinge-style)
- When HI degrades, predicted RUL must trend down
- Reduces “RUL saturated while HI falls” failure mode

---

## Stage 2 — Medium-risk architecture tweak (optional)

Apply **only if Stage 1 is insufficient**.

Examples:
- **Gated RUL head** consuming `(HI, ΔHI)`
- Explicit saturation penalty near `RUL_cap`
- Auxiliary uncertainty head (quantiles or variance)

If adding new heads or outputs:
- Define a **stable forward output contract**
- Prevent mean shift via loss weighting or detach strategy
- Add **structural constraints** where needed
  (e.g. quantile non-crossing)

---

## Non-goals / Anti-patterns
- Direct cycle index as a feature
- Hard post-hoc smoothing without physical meaning
- Optimizing dynamics at the expense of EOL / NASA
- FD004 plots or text that imply uncensored EOL

---

## Exit criteria
Stop iterating on dynamics when:
- Plateau ratio and saturation are reduced
- Mid-life trends are visible
- Late-life / NASA metrics are unchanged or better
