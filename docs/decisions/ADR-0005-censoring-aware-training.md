### ADR-0005: Censoring-Aware Training (Dynamic Truncation + Ranking Loss + RUL Buckets)

#### Status
Proposed

#### Context
- FD004 test is **right-censored**: “EOL” means **last observed cycle**, not failure (`docs/context.md`).
- Current training primarily uses **fixed-length sliding windows** with per-window RUL targets. This can encourage:
  - learning a **max_rul plateau** (e.g., predicting ~125 for many early windows),
  - weak monotonic consistency across different “observation horizons” of the same engine,
  - poorer calibration when observation length is short (simulated early censoring).
- We want a training strategy that explicitly teaches the model to be robust to **different truncation severities** (observation lengths) and to maintain consistent ordering of RUL as more cycles are observed.

Constraints / governance:
- **No leakage**: any sampling or supervision must use train-only information for train; diagnostics must not “peek” across splits.
- **Reproducibility**: training/inference pipelines must reconstruct from `summary.json` and pass feature-dimension safety checks (`ADR-0001`).
- **Multiprocessing safety**: truncation sampling must be deterministic across DataLoader workers.

#### Decision
We implement **Censoring-Aware Training** as an **opt-in training mode** with three components, gated by config flags:

1) **Dynamic truncation sampling (on-the-fly)**
- Training dataset exposes `num_truncations_per_engine` samples per engine per epoch (default 5).
- Each sample corresponds to a deterministic “cut point” (right-censoring view) within that engine’s available life.
- Cut distribution:
  - with probability `p_full=0.25`: `cut = L` (last observed),
  - else: `cut = int(r * L)` where `r ~ Uniform(0.4, 1.0)`,
  - enforce `cut >= past_len + 1` (in cycle space) so a valid window exists.
- Deterministic randomness:
  - dataset index maps to `(engine_id, truncation_slot_id)`,
  - seed is derived from `(epoch, engine_id, truncation_slot_id)` with a stable hash,
  - ensures consistent behaviour across multiprocessing workers.

2) **Pairwise ranking loss (stabilize monotonic ordering across truncations)**
- For samples from the same engine, enforce earlier cutpoints have larger RUL than later cutpoints.
- Hinge loss: `ReLU(margin - (mu(t1) - mu(t2)))`, weight `lambda_rank` (default 0.1).
- Applies **only** to the point estimate (`mu` / P50), not to sigma/quantiles.

3) **RUL bucket head (auxiliary classification)**
- Add an auxiliary head predicting logits over RUL buckets (e.g., 0–25, 25–50, …, 125+).
- Cross-entropy loss weight `lambda_bucket` (default 0.1).
- Intended to reduce collapse to the max_rul plateau by providing a coarse-grained learning signal.

Scope:
- In-scope: training-time truncation sampling, ranking loss, bucket head, and a basic truncation-colored scatter diagnostic.
- Out-of-scope (for this ADR): survival/hazard models, explicit censoring likelihood, full time-to-event modelling.

#### Alternatives considered
- **Keep full sliding-window training only**: simplest, but does not explicitly teach robustness to truncation severity and can preserve the max_rul plateau.
- **Gaussian NLL / sigma head only**: addresses uncertainty but does not enforce ordering consistency and can shift the mean if weighted poorly.
- **Survival analysis / hazard modelling**: principled for censoring, but higher complexity and larger refactor; deferred.

#### Consequences
- Positive:
  - Better robustness to observation-length variation; improved “true vs pred” scatter away from flat plateau.
  - Stronger monotonic consistency across truncations for the same engine.
  - Coarse bucket supervision can stabilize early-life predictions.
- Negative:
  - Adds dataset complexity and more hyperparameters (num_truncations, p_full, r-range, rank/bucket weights).
  - Requires careful output-contract handling (new bucket head output) to avoid tuple-unpacking regressions.
- Risks:
  - **Leakage**: must ensure truncation diagnostics computed on train/val only when true per-cycle RUL is required.
  - **Feature/scaler mismatch**: dataset wrappers must not change feature dimensionality.
  - **Censoring misinterpretation**: all labels/plots must say “last observed cycle (right-censored)” for FD004 test.

#### Validation plan (tests / metrics / plots)
- **Tests**:
  - Unit test (or lightweight assertion) that truncation dataset length equals `num_engines * num_truncations_per_engine`.
  - Determinism check: same `(epoch, engine, slot)` yields same cut.
- **Metrics** (val/test): RMSE, MAE, Bias, R², NASA mean/sum.
- **Plots/Diagnostics**:
  - `true_vs_pred_by_truncation.png`: scatter colored by truncation ratio bucket (0.4–0.6, 0.6–0.8, 0.8–1.0).
- **Gates**:
  - No feature-dimension mismatch errors.
  - Censoring-aware run improves at least one of: max_rul plateau reduction, scatter shape, worst-20 truncation buckets; while keeping RMSE/NASA within an acceptable delta.

#### Links
- `docs/context.md`
- `docs/decisions/ADR-0001-project-governance.md`
- `docs/roadmap.md`


