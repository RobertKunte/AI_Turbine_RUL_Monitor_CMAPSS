### ADR-0009: μ vs safe_RUL as Product Contract (Residual Risk Head)

#### Status
Accepted

#### Context
For FD004 TEST, “EOL” refers to the **last observed cycle (right-censored)**. In PHM usage, avoiding unsafe overestimation near EOL is often more important than small RMSE improvements.

We train a point predictor \( \mu \) for RUL and a residual-risk head that predicts an overshoot quantile \(q_{\tau}\) (in cycles). The goal is a conservative decision value that can be consumed by downstream engineering logic (alerts, maintenance policies) without forcing the mean predictor to become pessimistic.

#### Decision
We define a stable product interface:

- **Best estimate**: \( \mu \) (“point prediction”)
- **Decision value**: \( safe = \mathrm{clip}(\mu - q_{\tau}, 0, max\_rul) \)

Training:
- Optimize \(\mu\) for accuracy (RMSE/MAE/Bias).
- Train risk head to calibrate overshoot quantiles, with **detach-through-μ** in the risk loss to reduce mean shift.

Evaluation (always report both):
- **Calibration / safety KPIs** (especially for low-RUL region):
  - `coverage_low` (target ≈ `risk_tau`)
  - `overshoot_safe_rate_pos_low` = P(safe > true | true <= low_thr)
  - `overshoot_safe_low.rate_gt_thr` for a defined overshoot threshold
- **Cost of safety**:
  - `margin_safe = true - safe` (median, p90, p95) for low and all
- **μ quality monitoring**:
  - `mu_metrics_low` and `mu_metrics_all` (rmse/mae/bias)

#### Acceptance gates (default, tune per application)
For FD004 (right-censored), low-RUL region defined by `low_rul_threshold`:
- Calibration: `coverage_low >= risk_tau - 0.02`
- Safety: `overshoot_safe_rate_pos_low` should be near **0–1%** for `risk_tau=0.99` (or stricter if “any over-optimism is dangerous”)
- Utility: `margin_safe_low.p90` must not become excessively large (avoid “safe looks good because risk is huge”)

#### Consequences
- Downstream agents/policies consume `safe` as the conservative decision value.
- We can tune conservatism by adjusting `risk_tau` without retraining μ’s objective.
- Reporting includes both safety and utility KPIs to prevent degenerate solutions.


