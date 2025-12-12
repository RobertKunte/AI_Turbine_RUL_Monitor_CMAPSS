### ADR-0002: Phase 1 Run Registry (SQLite, auto-logged from run_experiments)

#### Status
Accepted

#### Context
We currently rely on manual logging of experiment outcomes. This causes:
- inconsistent run metadata (missing config flags, missing artifacts),
- harder comparisons across runs,
- slow debugging of common failure modes (feature/scaler mismatches, FD004 censoring confounds),
- weak governance (no standard “run card” for every experiment).

We want a Phase 1 run registry that is:
- **automatic** (no manual entry),
- **low-dependency** (stdlib preferred),
- **queryable** (list/filter/sort),
- **aligned with existing artifacts** (`results/<dataset>/<run>/summary.json`),
- **safe** (never store secrets).

#### Decision
Implement a SQLite-based registry that logs every `run_experiments.py` execution:

- **DB**: a single SQLite file (default: `artifacts/run_registry.sqlite`)
- **Entry point**: `src/tools/run_registry.py`
- **Integration**: `run_experiments.py` calls:
  - `start_run(...)` at run start,
  - `finish_run(...)` on success,
  - `fail_run(...)` on exception.
- **Storage**:
  - `config_json`: sanitized config dict as JSON
  - `metrics_json`: best available metrics dict as JSON
  - `results_dir`, `summary_path`, plus status and timestamps
  - optional artifact paths (plots, checkpoints) via `log_artifact(...)`

#### Alternatives considered
- **Markdown/CSV logging**: simple but fragile, hard to query reliably, easy to diverge.
- **Full tracking stack (MLflow/W&B)**: powerful but adds dependency/ops/policy overhead for Phase 1.
- **Database later only**: rejected; current velocity needs immediate stabilization.

#### Consequences
- **Positive**:
  - Every run becomes discoverable and comparable.
  - Governance gates become enforceable (“every run must have a run card entry”).
- **Negative**:
  - Minor wiring complexity in the experiment runner.
  - DB file management (path, backup, ignore in git).
- **Risks**:
  - Accidentally storing secrets in config → mitigate by sanitization + documentation.
  - Partial logs on crash → mitigate with `status=running` default and updates in transactions.

#### Validation plan (tests / metrics / plots)
- **Tests**:
  - Minimal self-test: create DB, start_run → finish_run, query back.
  - Failure path: start_run → fail_run logs error fields.
- **Metrics**:
  - On success, `metrics_json` must match what is written to `summary.json` (where available).
- **Plots/Diagnostics**:
  - Not required for the registry itself, but artifact paths should be recordable.
- **Gates**:
  - Every run produces exactly one registry record with status: `success` or `failed`.
  - `run_experiments.py` must not crash if the registry is disabled/unavailable.

#### Links
- `docs/context.md`
- `docs/roadmap.md`
- `docs/WORKFLOW.md`
- `docs/decisions/ADR-template.md`


