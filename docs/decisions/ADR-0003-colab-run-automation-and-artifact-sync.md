### ADR-0003: Colab Run Automation + Artifact Sync (Registry-first)

#### Status
Accepted

#### Context
We use Google Colab primarily for GPU execution. Today this often leads to:
- manual copy/paste notebook cells,
- ad-hoc copying of `results/` between Drive and `/content`,
- runs that are hard to reproduce (missing git SHA, unclear environment),
- no consistent mapping from a run to its artifacts and registry entry.

We already introduced a SQLite **Run Registry** (Phase 1). We want Colab execution to be:
- minimal manual steps,
- reproducible (git SHA, config snapshot),
- registry-visible directly from Colab,
- synchronized artifacts (push/pull) without deleting anything.

#### Decision
We standardize Colab runs via:

1) **Single-file Colab runner template**:
- `notebooks/templates/colab_runner.py`
- Mount Drive → clone repo into `/content` → print environment info → run one experiment → show registry latest → sync artifacts.

2) **Per-run artifact root**:
- Local: `artifacts/runs/<run_id>/`
- Drive: `/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS/artifacts/runs/<run_id>/`
- Artifact sync is per-run, not “copy whole results tree”.

3) **Artifact sync CLI**:
- `src/tools/sync_artifacts.py`
- Copy-only-newer behavior; never deletes destination files.
- Run resolution order: `--run_id` → `--run_name` (via registry) → `--latest` (via registry).

4) **Registry visibility from Colab**:
- `python -m src.tools.run_registry --show latest`
- Optional export: `--export latest --format json`

#### Alternatives considered
- Keep notebook-based ad-hoc execution: rejected (non-reproducible, error-prone).
- Sync the full `results/` folder: rejected (slow, noisy, hard to scope).
- Use external tracking tooling now: deferred (Phase 1 aims at minimal dependencies).

#### Consequences
- **Positive**:
  - One consistent Colab workflow, less manual work.
  - Clear mapping: (run_id ↔ registry ↔ artifacts).
  - Ready for future “tool-server + workers” architecture.
- **Negative**:
  - Additional “artifacts/” folder management; needs gitignore (handled externally).

#### Validation plan (tests / metrics / plots)
- **Tests**:
  - Local smoke test: create a fake run entry and validate `sync_artifacts --push/--pull` copies only newer.
- **Metrics**:
  - Registry must show the latest run and record its `artifact_root`.
- **Plots/Diagnostics**:
  - Not required by this ADR, but artifacts should include the run’s plots via `results/...` and/or explicit artifact logging later.
- **Gates**:
  - Colab runner executes with only one config section (RUN_NAME, DEVICE).
  - After run: registry show works; artifacts push works and creates Drive folder for the run_id.

#### Links
- `docs/decisions/ADR-0002-run-registry-sqlite.md`
- `docs/WORKFLOW.md`
- `src/tools/run_registry.py`
- `src/tools/sync_artifacts.py`
- `notebooks/templates/colab_runner.py`


