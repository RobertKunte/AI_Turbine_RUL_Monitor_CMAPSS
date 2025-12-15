### ADR-0008: Residual Quantile Risk Head for Safe RUL (FD004)

#### Status
Proposed

#### Context
- FD004 **test is right-censored**: evaluation labels correspond to the **last observed cycle**, not true failure.
- The project uses a `max_rul` plateau (e.g., 125), which creates a large “flat” target region at early life.
- A previous attempt to predict **RUL quantiles directly** (q10/q50/q90 on RUL) degraded mean-point performance (RMSE/NASA) and sometimes collapsed into conservative/unstable behavior.
  - In the right-censored + max_rul regime, direct quantile learning can over-allocate capacity to the plateau region and can interact badly with auxiliary heads (HI/damage), especially if quantile losses are weighted too strongly.
- The product pain point is **dangerous overestimation** (optimistic tail), e.g. `pred≈40` when `true≈15`, particularly for **low true RUL** engines.

Constraints:
- We must preserve the strong mean predictor (μ) behavior (RMSE/NASA).
- Any safety layer must be **opt-in**, fully reproducible, and not break older runs.
- Avoid leakage and keep FD004 wording: “EOL” = **last observed cycle**.

Note on numbering:
- `ADR-0004` is already taken (`ADR-0004-quantile-head.md`). This ADR uses the next free number to avoid collisions.

#### Decision
We add an **auxiliary residual-risk head** that predicts an upper quantile of **overestimation residual** (overshoot), and derive a conservative prediction **only in inference**:

- **Mean head (primary)**: \(\mu_{\text{RUL}}\), trained with the existing MSE-based objective (unchanged).
- **Risk head (aux)**: predicts \(q_{\tau}(\text{overshoot})\) where:
  - \(\text{err} = \mu_{\text{RUL}} - y\)
  - \(\text{overshoot} = \max(0, \text{err})\)
  - default \(\tau = 0.90\)

Training loss for the risk head:
- Use **pinball loss** on overshoot with **stop-grad through μ** to protect mean performance:
  - \(\text{overshoot} = \max(0, \mu_{\text{RUL}}^{\text{stopgrad}} - y)\)
  - \(L_{\text{risk}} = \text{pinball}(q_{\tau}^{\text{pred}}, \text{overshoot}, \tau)\)
- Total loss:
  - \(L = L_{\text{mean}} + \lambda_{\text{risk}} L_{\text{risk}} + \text{(existing aux losses)}\)

Inference-only safe prediction:
- \(\text{safe\_RUL} = \text{clamp}(\mu_{\text{RUL}} - \max(0, q_{\tau}^{\text{pred}}), 0, \text{max\_rul})\)

Configuration knobs (default OFF):
- `use_residual_risk_head: bool = False`
- `risk_tau: float = 0.90`
- `lambda_risk: float = 0.10`
- Diagnostics-only:
  - `low_rul_threshold: float = 20`
  - `overshoot_threshold: float = 20`

#### Alternatives considered
- Direct RUL quantiles: degraded μ metrics in our setting; sensitive to max_rul plateau and censoring.
- Gaussian σ/NLL: requires distributional assumptions; earlier runs showed potential mean shift unless carefully detached.
- Post-hoc clipping: reduces overshoot but is not data-driven and can be overly blunt.

#### Consequences
- Positive:
  - Tail-risk control without changing the main μ loss.
  - Targeted improvement for low-RUL engines (product-critical).
  - Easy to interpret: “subtract predicted overshoot quantile.”
- Negative:
  - Adds another head/output and inference/diagnostics plumbing.
  - Coverage may be imperfect; needs explicit coverage diagnostics.

#### Validation plan (tests / metrics / plots)
- **Metrics** (FD004 last observed cycle): RMSE/MAE/Bias/R² + NASA for μ; report also for safe_RUL.
- **Safety metrics** (focus on true<=20):
  - overshoot_rate: mean((μ-true) > overshoot_threshold)
  - overshoot_p95, overshoot_max
  - same metrics for safe_RUL
- **Coverage sanity**:
  - fraction where overshoot <= risk_q (overall + low bucket) should be ≈ τ.
- **Plots**:
  - `true_vs_pred_mu.png`, `true_vs_pred_safe.png`
  - `overshoot_hist_mu_vs_safe.png`
  - `risk_quantile_coverage.png`
- **Gates**:
  - When `use_residual_risk_head=False`, behavior identical to baseline.
  - With risk head on, overshoot metrics for true<=20 improve vs μ without catastrophic RMSE/NASA degradation.

#### Links
- `docs/context.md`
- `docs/decisions/ADR-0001-project-governance.md`
- `docs/decisions/ADR-0004-quantile-head.md`
- `docs/decisions/ADR-0007-quantile-risk-penalty-and-bias-calibration.md`
- `docs/roadmap.md`


