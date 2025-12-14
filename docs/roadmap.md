### Roadmap / Backlog

This file is the living backlog. Each item includes **Goal**, **Artifacts**, **Acceptance criteria**, and **Owner role**.

---

## Now (0–2 weeks)

### 1) Worst-20 engines diagnosis (FD004) + truncation checks
- **Goal**: Determine whether “worst engines” are driven by censoring severity, condition mix, or specific residual sensor patterns; reduce late-detection failure modes.
- **Artifacts**:
  - `results/fd004/<run>/diagnostics_truncation.png`
  - `results/fd004/<run>/diagnostics_worst_vs_rest_stats.png`
  - `results/fd004/<run>/diagnostics_worst20_residual_deltas.png`
  - Short textual summary in console/log (bucket stats + correlations)
- **Acceptance criteria**:
  - Diagnostics include: histograms, scatter plots, correlation prints, bucket table.
  - Clear statement: “primary driver is (A) truncation, (B) condition, (C) residual sensors, or mixed.”
  - No feature/scaler mismatches (feature-dimension safety check passes).
- **Owner role**: **Planner** (definition), **Implementer** (script), **Reviewer** (leakage + censoring realism), **Experimenter** (run + archive artifacts)

### 2) Reproducibility gates for training/inference/diagnostics
- **Goal**: Prevent silent mismatches in feature pipelines, scalers, and model reconstruction.
- **Artifacts**:
  - Documented “gates” in `docs/WORKFLOW.md`
  - ADR(s) if new rules are introduced (see `docs/decisions/`)
- **Acceptance criteria**:
  - Any run can be reconstructed from `results/.../summary.json` without manual toggles.
  - Feature-dimension check fails fast with a clear message if drift is detected.
- **Owner role**: **Implementer**, **Reviewer**

### 3) Documentation system adoption
- **Goal**: Make planning and key decisions versioned and reviewable in Cursor.
- **Artifacts**:
  - `docs/context.md`, `docs/roadmap.md`
  - `docs/decisions/ADR-0001-project-governance.md`
  - `docs/prompts/planner_prompt.md`, `docs/prompts/reviewer_prompt.md`
  - `docs/WORKFLOW.md`
- **Acceptance criteria**:
  - At least 1 new ADR created for a real decision.
  - Planner + Reviewer prompts used on at least 1 task end-to-end.
- **Owner role**: **Planner**, **Reviewer**

### 4) Colab run automation + artifact sync (registry-first)
- **Goal**: Run experiments on Colab with minimal manual steps, reproducible environment info, and automatic artifact synchronization to Drive per run_id.
- **Artifacts**:
  - `notebooks/templates/colab_runner.py`
  - `src/tools/sync_artifacts.py`
  - `artifacts/run_registry.sqlite` (updated via Colab)
  - Drive artifacts: `/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS/artifacts/runs/<run_id>/`
- **Acceptance criteria**:
  - Single RUN config block in Colab (RUN_NAME, DEVICE)
  - After run: `python -m src.tools.run_registry --show latest` works in Colab
  - After run: `python -m src.tools.sync_artifacts --push --latest` copies only newer and never deletes
  - Clear summary printed (copied files count + bytes)
- **Owner role**: **Planner**, **Implementer**, **Experimenter**, **Reviewer**

### 5) Censoring-aware training (dynamic truncation + ranking loss + RUL buckets) — **In Progress**
- **Goal**: Make training robust to right-censoring severity by sampling multiple truncations per engine per epoch, and stabilize point predictions via ranking loss + bucket head.
- **Artifacts**:
  - `docs/decisions/ADR-0005-censoring-aware-training.md`
  - New run(s): `results/fd004/<run>/summary.json` with censoring-aware params recorded
  - Plot: `results/fd004/<run>/true_vs_pred_by_truncation.png`
- **Acceptance criteria**:
  - Truncation sampling is deterministic across DataLoader workers (same seed → same cut).
  - No tuple-unpacking regressions: older runs still load/infer/diagnose.
  - Metrics reported: RMSE/MAE/Bias/R² + NASA (val/test).
  - Diagnostics plot generated and uses FD004 wording (“last observed cycle (right-censored)”).
- **Owner role**: **Planner** (ADR + plan), **Implementer** (dataset + loss + head), **Reviewer** (leakage + censoring realism), **Experimenter** (runs + archive)

---

## Next (2–6 weeks)

### 4) Feature ablation framework (FD004-focused, reusable across datasets)
- **Goal**: Systematically quantify which feature blocks drive EOL metrics and which drive trajectory realism.
- **Artifacts**:
  - A documented ablation matrix (Markdown report under `docs/` or `results/...`)
  - Plots comparing ablations (error vs life fraction; worst-20 deltas)
- **Acceptance criteria**:
  - Ablations run with identical splits and comparable logging.
  - Report includes metrics table + 2–3 key plots per ablation category.
- **Owner role**: **Planner** (matrix), **Experimenter** (runs), **Reviewer** (leakage + comparability)

### 5) Decoder/world-model dynamics validation (“avoid blind decoder”)
- **Goal**: Ensure models are not “EOL-good but dynamics-bad”; validate slopes/monotonicity/smoothness mid-life.
- **Artifacts**:
  - A standard diagnostics page/report for dynamics validation
  - Per-engine trajectory plots (best/median/worst) and life-fraction error plots
- **Acceptance criteria**:
  - At least one dedicated dynamics metric/plot gate is required for acceptance.
  - Failure mode examples captured (worst engines with late detection).
- **Owner role**: **Planner**, **Reviewer**, **Experimenter**

### 6) Run registry & experiment logging (lightweight now; DB later)
- **Goal**: Make runs discoverable and comparable; standardize “Run Card” metadata.
- **Artifacts**:
  - **Phase 1 (SQLite)**:
    - `src/tools/run_registry.py` (entry point + CLI)
    - DB file: `artifacts/run_registry.sqlite` (default; configurable)
    - Auto logging from `run_experiments.py` (start/success/failure)
  - ADR: `docs/decisions/ADR-0002-run-registry-sqlite.md`
- **Acceptance criteria**:
  - Every run auto-logs: `run_id`, `experiment_name`, `dataset`, timestamps, `status`
  - On success: `config_json` + `metrics_json` stored and consistent with `results/.../summary.json`
  - On failure: error message recorded and `status=failed`
  - CLI can list latest runs and filter by dataset/experiment_name/status
- **Owner role**: **Planner**, **Implementer**

---

## Later (6+ weeks)

### 7) Database-backed experiment tracking + dashboard
- **Goal**: Queryable experiment history, automated comparisons, artifact indexing.
- **Artifacts**:
  - DB schema and migration plan (ADR)
  - Simple dashboard or export scripts
- **Acceptance criteria**:
  - Reproducible ingestion from `results/...` into registry.
- **Owner role**: **Planner**, **Implementer**

### 8) PHM realism improvements for censoring-aware evaluation
- **Goal**: Evaluation methods that reflect right-censoring and deployment realism.
- **Artifacts**:
  - ADR(s) for evaluation protocol
  - Updated diagnostics scripts and reporting templates
- **Acceptance criteria**:
  - Explicit statement of what “true” means on test (last observed cycle vs failure).
  - Added metrics/plots that are censoring-aware (placeholders OK initially).
- **Owner role**: **Reviewer**, **Planner**


