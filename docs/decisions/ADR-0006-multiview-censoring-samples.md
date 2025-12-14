### ADR-0006: Multiview Censoring Samples (“One engine, many observation horizons”)

#### Status
Proposed

#### Context
- FD004 test is **right-censored**: “EOL” means **last observed cycle**, not failure (`docs/context.md`).
- Our baseline training uses **full sliding windows** across each engine lifetime. This provides strong supervision coverage but does not explicitly teach robustness to *observation horizon* (how far into the life we have observed the engine).
- We previously tried a censoring-aware mode that effectively **replaced** the sliding-window stream with a small number of per-engine truncation samples; this reduced compute but caused severe underfitting when K was small.

We want a better compromise:
- Keep **all regular sliding windows** (unchanged).
- Add an **auxiliary stream** of “censored views” to simulate “engine pulled from the field at cycle cut”.

#### Decision
We introduce **Multiview Censoring Samples** as an optional training-data augmentation:

- **Regular stream**: original full sliding-window samples (unchanged).
- **Aux stream**: for each train engine, sample:
  - \(K\) truncation cutpoints per epoch (default \(K=5\)),
  - and for each cutpoint sample \(M\) windows uniformly from within the truncated view (default \(M=8\)).

Auxiliary samples carry metadata:
- `unit_id`, `t_end`, `is_censored_view`, `cut_ratio`, `cut_idx`
so future losses/diagnostics can condition on censoring severity and build same-engine pairs.

Mixing:
- Use a controlled mixing ratio, default **70% regular / 30% aux**.
- Keep per-epoch compute comparable to baseline by sampling a fixed number of total samples (e.g., ≈ `len_regular`) rather than iterating over the full concatenated space every epoch.

Worker-safe determinism:
- Use **stable hashing** (no Python randomized `hash`) to seed per-sample RNG.
- Avoid relying on fragile “worker sync hacks”; sampling must remain deterministic under `DataLoader(num_workers>0)`.

#### Alternatives considered
- **Replace** regular stream with K truncations/engine: too few samples unless K is huge; harms RMSE.
- **Pre-materialize** all censored samples on disk: too big and inflexible; violates “no huge datasets”.
- **Survival/hazard models**: more principled but higher complexity; deferred.

#### Consequences
- Positive:
  - Model learns “one engine, many observation horizons”, matching deployment realism (right-censoring).
  - Preserves baseline supervision coverage (regular sliding windows remain intact).
  - Enables future ranking / bucket / calibration losses conditioned on censoring metadata.
- Negative:
  - More complex dataset and sampling logic.
  - Risk of correlated samples (same engine, overlapping windows).

Risks & mitigations:
- **Correlated samples / bias**: keep `aux_sample_ratio` bounded; sample windows uniformly within truncation; optionally cap per-engine aux per epoch.
- **Leakage**: build aux views from train split only; do not use per-cycle truths on FD004 test (right-censored).
- **DataLoader RNG**: stable hashing for per-sample RNG; sampler controls epoch-level mix.
- **Compute**: do not iterate over full `len_regular + K*M*engines`; sample a fixed number per epoch.

#### Validation plan (tests / metrics / plots)
- **Tests**:
  - Dataset length sanity: `len_combined = len_regular + num_engines*K*M` (even if sampler doesn’t iterate all).
  - Determinism: same (run_seed, unit_id, trunc_slot, window_slot, epoch) → same cut/window.
- **Metrics**: RMSE/MAE/Bias/R² + NASA (val/test).
- **Plots/Diagnostics**:
  - `cut_ratio_hist_train.png` (distribution of aux cut_ratio in an epoch)
  - `true_vs_pred_by_cut_ratio.png` (scatter colored by cut_ratio buckets)
- **Gates**:
  - When `use_multiview_censoring=False`, behavior matches baseline.
  - No feature/scaler mismatch regressions; model loader still strict-loads.

#### Links
- `docs/context.md`
- `docs/decisions/ADR-0001-project-governance.md`
- `docs/roadmap.md`


