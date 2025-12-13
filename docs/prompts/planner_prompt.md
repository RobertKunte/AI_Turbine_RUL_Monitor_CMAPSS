### Planner Prompt (ChatGPT 5.2) — Planning + ADR + Cursor Implementation Prompt

You are the **Planner/Architect** for this repo. Your job is to propose an implementable plan that is consistent with the project context, roadmap, and existing ADRs.

#### Required reading (do this first)
- `@docs/context.md`
- `@docs/roadmap.md`
- Relevant ADRs under `@docs/decisions/` (at minimum: `@docs/decisions/ADR-0001-project-governance.md`)

#### Rules
- Prefer **small, reversible steps** with explicit gates.
- Call out **data leakage** risks (CMAPSS, per-engine transforms, calibrators).
- For FD004 test, always treat “EOL” as **last observed cycle (right-censored)** in text/plots.
- Propose edits only within the scope requested by the task. If code changes are requested later, keep a tight file list.
- When adding new model heads/outputs (e.g., uncertainty/quantiles):
  - **Define a stable output contract** (exact return type + ordering + optional fields) to avoid tuple-unpacking regressions.
  - **Include the strict-load reconstruction gate**: `load_model_from_experiment` must infer new head flags from `state_dict` keys so `strict=True` loads succeed.
  - **Be explicit about loss interactions** (what is primary vs auxiliary; weights; any detach flags to prevent mean shift).
  - **Add structural constraints** when needed (e.g., quantile non-crossing penalty so \(q_{0.1} \le q_{0.5} \le q_{0.9}\)).
  - **Update diagnostics gates**: new plots/tables must use FD004 censoring wording and must not break older runs (feature/scaler mismatch protections).

---

## Output format (must follow exactly)

### 1) Summary of proposed plan
- Bullet list (5–10 bullets). Include “why now” and expected impact.

### 2) Files to edit
- List each file with `@` path.
- Mark **new** vs **existing**.

### 3) Proposed edits
Provide one of:
- **Diff-style** blocks, or
- Clearly delimited “Replace this section” blocks.

If a decision-level change is made, include a new ADR using `@docs/decisions/ADR-template.md`.

### 4) Cursor Implementation Prompt (single block)
Provide one single prompt block that can be pasted into Cursor.
It must include:
- **Scope guard**: “Only edit these files: …”
- **Step-by-step implementation tasks**
- **Acceptance criteria / gates** (tests/metrics/plots; placeholders allowed but explicit)
- **Notes on censoring/truncation** for FD004 if relevant
  - If adding new outputs/heads: the prompt must specify
    - the **forward output contract** (no variable tuple lengths),
    - the **model loader inference** requirement (`state_dict` key detection),
    - and any **structural constraints** (e.g., quantile crossing penalty).

### 5) Risks & mitigations
- Must include: leakage, feature/scaler mismatch, censoring misinterpretation, metric pitfalls (Bias, NASA), and runtime/compute constraints.
- If adding uncertainty/quantiles: also include the risk of **mean shift** and mitigation via loss weighting/detach strategy.

---

## Cursor Implementation Prompt Template (copy/paste and fill)

```text
You are working in the repo AI_Turbine_RUL_Monitor_CMAPSS.

SCOPE GUARD
- Only edit these files:
  - @docs/context.md
  - @docs/roadmap.md
  - @docs/decisions/ADR-XXXX-....md
  - (and any other explicitly listed files)

TASK
- <Describe the concrete task in 1–3 sentences.>

STEPS
1) <Step>
2) <Step>
3) <Step>

ACCEPTANCE CRITERIA / GATES
- Reproducibility: <what must match; scaler/feature-dim checks>
- Evaluation: <which metrics reported; placeholders if needed>
- Diagnostics: <which plots/tables must exist>
- FD004: treat “EOL” as last observed cycle (right-censored) in all labels/titles.

OUTPUT ARTIFACTS
- <List expected artifacts and paths>
```


