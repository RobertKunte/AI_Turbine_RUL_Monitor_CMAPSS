### ADR Template

#### Title
ADR-XXXX: <Short decision title>

#### Status
Proposed | Accepted | Deprecated

#### Context
- What problem are we solving?
- What constraints apply (CMAPSS, leakage, censoring, condition-wise scaling)?
- What evidence do we already have (runs, diagnostics, failure modes)?

#### Decision
- What we decided.
- Scope: what is in / out.
- Defaults and configuration knobs (if any).

#### Alternatives considered
- Option A: <summary>, pros/cons
- Option B: <summary>, pros/cons
- Why not chosen.

#### Consequences
- Positive: what gets better
- Negative: what gets worse / added complexity
- Risks: leakage, feature mismatch, evaluation pitfalls, censoring misinterpretation

#### Validation plan (tests / metrics / plots)
- **Tests**: what should be added or run
- **Metrics**: which metrics must be reported (RMSE/MAE/Bias/R² + NASA)
- **Plots/Diagnostics**: which plots must be produced (worst engines, truncation, life-fraction error)
- **Gates**: explicit “Go/No-Go” conditions

#### Links
- Runs: `results/<dataset>/<run>/...`
- Related ADRs: `docs/decisions/ADR-XXXX-....md`
- Docs: `docs/context.md`, `docs/roadmap.md`, `docs/WORKFLOW.md`


