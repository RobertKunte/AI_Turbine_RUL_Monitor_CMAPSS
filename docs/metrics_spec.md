# Metrics Spec: LAST vs ALL (C-MAPSS, truncated-aware)

This project reports **two metric sets** for every run/split:

## A) LAST (literature-style, truncated-aware)

**Definition**: for each engine/unit, take exactly **one** sample: the **last available** timepoint in that (possibly truncated) trajectory.

- **Important**: On C-MAPSS **test** data, trajectories are **truncated** (the engine does not reach failure in the file).
  Therefore, “last available” is **not** the true end-of-life (EOL).  
  We avoid the label “EOL” and use **LAST** / **LAST_AVAILABLE_PER_UNIT**.

**Keys (binding)**:
- `rmse_last`, `mae_last`, `bias_last`, `r2_last`
- `nasa_last_sum`, `nasa_last_mean`
- `n_units`
- `last_definition`: `"LAST_AVAILABLE_PER_UNIT (truncated-aware)"`

Implementation: `src.metrics.compute_last_per_unit_metrics(...)`

## B) ALL (deployment diagnostics)

**Definition**: compute metrics over **all available samples** (all sliding windows / all available timepoints).

Interpretation: this is the **deployment-style** “how good are we across the full operating range?” diagnostic.
NASA score is allowed here as an **asymmetric cost** but must be labeled explicitly as **ALL**.

**Keys (binding)**:
- `rmse_all`, `mae_all`, `bias_all`, `r2_all`
- `nasa_all_sum`, `nasa_all_mean`
- `n_samples_all`
- optionally `n_units` (if `unit_ids` are provided)

Implementation: `src.metrics.compute_all_samples_metrics(...)`

## Clipping

When `clip=(0, max_rul)` is provided:
- `y_true` and `y_pred` are clipped to the interval.
- Metrics include `max_rul_used`.

NASA scoring always reuses the centralized implementation in `src.metrics`:
- `nasa_phm_score(...)`
- `compute_eol_errors_and_nasa(...)`

## Test-time truth for ALL on truncated C-MAPSS

C-MAPSS provides `y_test_true` = remaining RUL at the **last observed cycle**.
For any earlier observed cycle \(t\) in the same test trajectory:

\[
RUL(t) = RUL_{last\_obs} + (t_{last} - t)
\]

This enables **ALL timepoints/windows** evaluation on test trajectories without leaking future information.

