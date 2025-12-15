### ADR-0007: Quantile Head + Upper-Tail Risk Penalty + Optional Bias Calibration (FD004)

#### Status
Proposed

#### Context
- FD004 test is **right-censored**: “EOL” means **last observed cycle**. In deployment, optimistic outliers are risky.
- We observe cases where predictions are dangerously optimistic (e.g., pred≈40 when true≈15) and systematic bias (mean error negative / too optimistic).
- The repo already supports optional **quantile head** (P10/P50/P90) and pinball loss, but we want:
  - explicit **upper-tail risk penalty** to reduce optimistic outliers,
  - optional **q50 bias calibration loss** to reduce systematic bias,
  - diagnostics that make the tail risk visible.

Constraints:
- Must not break existing multitask setup (RUL + HI + damage).
- Must remain backward compatible when quantile features are disabled.
- Avoid leakage: right-censored FD004 test must be interpreted as last observed cycle.

#### Decision
We add three opt-in components for FD004 usability:

1) **Quantile head stays first-class**
- Keep quantiles configurable (default [0.1, 0.5, 0.9]).
- Use q10 as conservative “safe” RUL in diagnostics and optional reporting.

2) **Upper-tail risk penalty**
- Define `q_upper` as max quantile (typically q90).
- Penalize optimistic violations: `relu(q_upper - y_true - risk_margin)` with weight `lambda_risk`.

3) **Optional q50 bias calibration loss**
- Compute batch bias: `mean(q50 - y_true)`.
- Penalize absolute bias either per-batch (“batch_abs”) or via EMA (“ema”) with `bias_ema_beta`.

Defaults are set to be conservative and backward compatible:
- risk/bias losses are disabled unless weights > 0 or quantile loss is enabled.

#### Alternatives considered
- Pure MSE on point prediction: does not explicitly control upper-tail optimism.
- Gaussian sigma/NLL: assumes distribution form; earlier experiments showed potential mean-shift issues.
- Post-hoc calibration only: helps bias but does not train away tail risk.

#### Consequences
- Positive:
  - Reduced optimistic tail risk (fewer q90>true violations).
  - Better usability via “safe” RUL (q10) and explicit risk diagnostics.
- Negative:
  - Additional knobs (risk/bias weights, EMA beta).
  - Risk of over-conservatism if weights are too high.

#### Validation plan (tests / metrics / plots)
- **Metrics**: RMSE/MAE/Bias/R² + NASA on last observed cycle.
- **Diagnostics**:
  - Scatter: true vs mu (if available), true vs q50, true vs safe (q10)
  - Risk histogram: q_upper - true (counts >0 and >10)
  - CSV: top optimistic cases by (q_upper - true)
- **Gates**:
  - When quantile head disabled, behaviour matches baseline.
  - No tuple-unpacking or strict-load regressions.

#### Links
- `docs/context.md`
- `docs/decisions/ADR-0004-quantile-head.md`
- `docs/roadmap.md`


