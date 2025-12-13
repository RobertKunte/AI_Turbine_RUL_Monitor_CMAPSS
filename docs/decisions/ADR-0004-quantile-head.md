### ADR-0004: Quantile Head for RUL Uncertainty (FD004 last observed cycle / right-censored)

#### Status
Proposed

#### Context
We added a Gaussian uncertainty head (μ/σ) for Encoder v5 and trained with a Gaussian NLL term.
In practice, we observed that enabling the σ-head + NLL can shift the mean prediction μ in an
undesired direction (e.g., more optimistic bias for higher predicted RUL), likely because the
NLL term backpropagates through the shared representation and mean head.

FD004 test is **right-censored**: “EOL” refers to the **last observed cycle**, not failure.
Our uncertainty diagnostics focus on last-observed-cycle predictions and their reliability.

We want:
- A conservative “safe” prediction that does not require a Normality assumption.
- An uncertainty interval that can be asymmetric (skewed errors are common).
- Robust evaluation/diagnostics under censoring-aware labeling.

#### Decision
Adopt a **quantile head** for the encoder RUL prediction at the last observed cycle:
- Predict quantiles \([P10, P50, P90]\) by default.
- Use **P50** as the point prediction for existing metrics/plots.
- Use **P10** as the conservative prediction (“RUL_safe”).
- Use \([P10, P90]\) as the uncertainty band.
- Train using **pinball (quantile) loss** plus a small **non-crossing penalty**
  to enforce \(P10 \le P50 \le P90\).

The Gaussian σ-head/NLL remains supported as a separate run variant but is not the default
uncertainty representation for “safe” reporting.

#### Alternatives considered
- **Gaussian μ/σ (NLL)** only: simple but assumes symmetric/Normal residuals and can shift μ.
- **μ/σ with detached μ for NLL**: reduces mean shift risk, still assumes symmetric uncertainty.
- **Post-hoc calibration of σ**: adds complexity and still depends on parametric form.

#### Consequences
- **Positive**:
  - No Normality assumption; supports asymmetric uncertainty.
  - “Safe” prediction is directly available as P10.
  - Clear reliability metrics: conservative_rate(P10 ≤ true) and coverage(P10–P90).
- **Negative**:
  - Requires additional head and loss weighting/tuning.
  - Quantile crossing can occur without explicit penalties.

#### Validation plan (tests / metrics / plots)
- **Reproducibility gate**:
  - Model reconstruction from `results/.../summary.json` works.
  - `load_model_from_experiment(..., strict=True)` succeeds (loader infers quantile head from checkpoint keys).
- **Evaluation gate**:
  - Report RMSE/MAE/Bias on P50 (point estimate).
  - Report conservative_rate(P10 ≤ true) (nominally ~0.9 for uncensored targets).
  - Report coverage(true ∈ [P10,P90]) (nominally ~0.8).
- **Diagnostics gate**:
  - `fd004_worst20_diagnosis_v1.py` produces quantile band plots and bucket coverage plots.
- **FD004 censoring gate**:
  - All plots/titles/labels say “last observed cycle (right-censored)”.

#### Links
- `docs/context.md`
- `docs/roadmap.md`
- `src/analysis/fd004_worst20_diagnosis_v1.py`


