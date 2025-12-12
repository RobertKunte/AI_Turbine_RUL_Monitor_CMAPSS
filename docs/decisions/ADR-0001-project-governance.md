### ADR-0001: Project Governance (Planning, Decisions, and Acceptance Gates)

#### Status
Accepted

#### Context
This repository evolves quickly across:
- multiple datasets (FD001–FD004),
- feature pipelines (multi-scale, digital-twin residuals, condition vectors),
- modeling families (encoders, decoders, world models),
- evaluation constraints (FD004 test is right-censored; “EOL” means last observed cycle).

Without governance, we risk:
- silent feature/scaler mismatches between training and diagnostics,
- data leakage (train/test contamination, per-engine future-aware transforms),
- optimizing EOL metrics while producing unrealistic degradation dynamics,
- losing institutional memory on why certain decisions were made.

#### Decision
We adopt a lightweight, versioned documentation system:

- **ADR-first for key decisions**:
  - Any decision that affects data, feature pipelines, scaling, metrics, architecture, evaluation protocol, or reproducibility rules must be captured as an ADR under `docs/decisions/`.
  - ADRs use `docs/decisions/ADR-template.md`.

- **Roadmap as source of priorities**:
  - Work priorities and acceptance criteria live in `docs/roadmap.md`.
  - Planning updates must align with the roadmap (or update it).

- **Single source of truth context**:
  - The current project snapshot and conventions live in `docs/context.md`.
  - Keep it short and stable; link to ADRs for details.

#### Model usage policy (Planner / Reviewer roles)
- **ChatGPT 5.2** is used as **Planner/Architect**:
  - defines approach, risks, acceptance criteria, and proposes ADR/prompt updates.
- **Gemini 3 Pro** is used as **Reviewer/Critic**:
  - stress-tests for leakage, censoring realism, metric pitfalls, and missing diagnostics.
- Optional: **Claude** (or other) can be used for refactors or documentation polishing, but decisions remain in ADRs.

#### Acceptance checks (“gates”) for changes
Any non-trivial change must define and pass gates appropriate to scope:

- **Reproducibility gate**:
  - Model + pipeline reconstructable from `results/<dataset>/<run>/summary.json`
  - Feature-dimension safety checks pass (fail-fast if mismatch)

- **Evaluation gate**:
  - Metrics reported: RMSE/MAE/Bias/R² + NASA (placeholders allowed during early exploration, but must be explicit)
  - For FD004 test: labels and interpretation must state **right-censored / last observed cycle**

- **Diagnostics gate** (when analysis impacts behavior):
  - Worst engines plots (worst/median/best)
  - Truncation/censoring diagnostics (true_rul_last, num_cycles vs error)
  - Life-fraction / trajectory error plots where relevant

#### Escalation rule
When uncertain or conflicting evidence exists:
- Do **not** guess.
- Require at least one of:
  - a focused ablation (feature block / head / loss toggle), or
  - a diagnostic plot/table that isolates the hypothesis (e.g., censoring bucket analysis).

#### Alternatives considered
- “No ADRs, rely on code comments”: rejected (doesn’t capture decision trade-offs).
- “Full experiment tracking DB now”: deferred (see roadmap; later phase).

#### Consequences
- **Positive**:
  - Consistent decision history, fewer regressions, clearer collaboration.
  - Faster iteration with explicit gates and reviewer checks.
- **Negative**:
  - Slight overhead to write ADRs and keep docs current.
  - Requires discipline to avoid bypassing the workflow.

#### Validation plan (tests / metrics / plots)
- **Tests**: N/A for doc-only governance; apply gates to subsequent code changes.
- **Metrics**: N/A for doc-only governance.
- **Plots**: N/A for doc-only governance.
- **Gates**: This ADR is considered “validated” once the workflow is used on at least 1 real task and results in one additional ADR.

#### Links
- `docs/context.md`
- `docs/roadmap.md`
- `docs/WORKFLOW.md`
- `docs/prompts/planner_prompt.md`
- `docs/prompts/reviewer_prompt.md`


