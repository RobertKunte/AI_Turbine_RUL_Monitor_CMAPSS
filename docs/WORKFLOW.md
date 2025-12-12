### Workflow (Planning → ADR → Implementation → Review → Gates → Close)

This is the standard workflow for doing planning and model/provider discussions directly in Cursor with versioned context.

---

## 1) Start a task
- Read:
  - `@docs/context.md`
  - `@docs/roadmap.md`
  - Relevant ADRs in `@docs/decisions/`
- If the task changes assumptions, update `@docs/context.md` first (keep it short).

## 2) Create or update an ADR (if decision-level)
Create a new ADR when the change affects any of:
- feature pipeline/scaling logic (FD002/FD004 condition-wise scaling, twin residuals, multiscale),
- evaluation protocol (right-censoring, “EOL” definition),
- architecture (encoder/decoder/world model interfaces),
- reproducibility rules (artifact schema, run reconstruction).

Use: `@docs/decisions/ADR-template.md`  
Name: `ADR-XXXX-<short-slug>.md`

## 3) Planning (Planner model)
Use `@docs/prompts/planner_prompt.md` with ChatGPT 5.2.
Planner output must include:
- plan summary
- files to edit
- proposed edits (diff or delimited blocks)
- a single “Cursor Implementation Prompt” block
- risks & mitigations

## 4) Review (Reviewer model)
Use `@docs/prompts/reviewer_prompt.md` with Gemini 3 Pro.
Reviewer must:
- catch leakage risks
- enforce censoring realism (FD004 test)
- validate scaling/feature-dimension compatibility
- require diagnostics and gates
- produce Go/No-Go and patch suggestions

## 5) Implement (in Cursor)
- Paste the Planner’s “Cursor Implementation Prompt” into Cursor.
- Keep edits within scope.
- Prefer small PR-sized changes; avoid mixing unrelated refactors.

## 6) Run gates (must be explicit)
Choose gates based on scope:
- **Reproducibility**:
  - reconstruct model/pipeline from `results/.../summary.json`
  - feature-dimension check passes (scaler vs pipeline)
- **Evaluation**:
  - RMSE/MAE/Bias/R² + NASA (TBD allowed during exploration but label it)
- **Diagnostics**:
  - worst engines plots (worst/median/best)
  - truncation/censoring diagnostics for FD004 test
  - trajectory realism plots where relevant (life-fraction error)

## 7) Log run results (lightweight for now)
Phase 1 Run Registry (SQLite):
- Runs are auto-logged when you execute `run_experiments.py`.
- Default DB path: `artifacts/run_registry.sqlite` (override with `RUN_REGISTRY_DB=...`)
- Disable registry if needed: `RUN_REGISTRY_DISABLE=1`
- Quick inspection:
  - `python -m src.tools.run_registry --list 20`
  - `python -m src.tools.run_registry --dataset FD004 --status success --list 20`

Always also ensure:
- `results/<dataset>/<run>/summary.json` contains metrics and key flags
- plots are saved with stable filenames (e.g., `diagnostics_*.png`)
- update `docs/roadmap.md` if priorities changed

## 8) Close the task
- Update `@docs/roadmap.md`:
  - mark item done / move to next section
  - add follow-up tasks discovered during diagnostics
- Update `@docs/context.md` only if it affects the “single source of truth”.
- Add links to ADR(s) and relevant run artifacts.

---

## Running Experiments on Google Colab
Colab is used primarily for **GPU execution**. The source of truth is:
- the **Run Registry** (`artifacts/run_registry.sqlite`), and
- per-run artifacts under `artifacts/runs/<run_id>/` (synced to Drive).

### Steps
1) Open the template: `notebooks/templates/colab_runner.py`
2) Set:
   - `RUN_NAMES = ["<experiment_name_1>", "<experiment_name_2>", ...]`
   - `DEVICE = "cuda"`
3) Run the script in Colab.
4) Verify registry entry (in Colab output):
   - `python -m src.tools.run_registry --show latest`
5) Verify artifacts synced to Drive:
   - `python -m src.tools.sync_artifacts --push --latest --what both`
   - Drive paths:
     - Artifacts: `/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS/artifacts/runs/<run_id>/`
     - Results:   `/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS/results/<dataset>/<run_name>/`

### Git-friendly status snapshots (recommended)
To keep a lightweight history in Git (without pushing `results/`):
- Export registry snapshot to the repo:
  - `python -m src.tools.export_status --limit 50`
  - Files are written to `docs/status/` (safe to commit).
- Commit/push from your local machine, or enable the optional Colab auto-commit (push requires auth).

### Run Cards (recommended)
Generate one Markdown “Run Card” per `run_id` (easy to review in Git):
- `python -m src.tools.export_run_cards --limit 50`
- Output:
  - `docs/status/run_cards/INDEX.md`
  - `docs/status/run_cards/<run_id>.md`



