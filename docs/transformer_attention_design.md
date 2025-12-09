# Transformer + Attention Encoder – Design Sketch

This document sketches the design for a **Transformer + Attention encoder** that will
build on top of the existing Phase‑4 residual / digital‑twin baselines and WorldModel V3.
It is meant as a **living design note**, not a final specification.

## 1. Motivation

- Improve performance on **multi‑condition datasets** (FD002, FD004), where operating
  conditions strongly affect sensor distributions.
- Provide **better interpretability** via attention:
  - which time steps matter most (temporal attention),
  - which sensors / physics features drive the RUL and HI predictions,
  - how different operating conditions influence the degradation.
- Reuse the **stable data + feature + diagnostics pipeline** from Phase‑4 and WorldModel V3:
  - same residual / physics features,
  - same EOL evaluation protocol (one RUL prediction per engine),
  - same NASA PHM08 metric and diagnostics plots.

## 2. Inputs and Data Pipeline

The planned Transformer encoder should consume **exactly the same type of input** as the
current WorldModel V3:

- Past window length: `past_len = 30` cycles (configurable),
- Features: ~464 columns for Phase‑4 residual setups:
  - raw sensors and settings,
  - physics‑based proxies (e.g. HPC efficiency, EGT drift, fan/HPC ratio),
  - multi‑scale temporal features (rolling mean, trend, deltas),
  - residual / digital‑twin features (measured – healthy baseline),
  - optional condition embeddings (FD002/FD004).

The **data + sequence building APIs** should be shared:

- `build_fd_dataset(fd_id, ...)` – load, feature engineering, scaling,
- `build_eol_sequences_from_df(...)` – build EOL sequences for training and inference.

## 3. Planned Architecture (UniversalEncoderV3 – tentative)

High‑level structure:

1. **Multi‑scale CNN Front‑End (optional)**:
   - 1D convolutions over time with kernel sizes e.g. `[3, 5, 9]`,
   - used to extract local patterns and reduce noise before attention,
   - similar to the existing UniversalEncoderV2 front‑end.

2. **Transformer Encoder over Time**:
   - Input: sequence of length `past_len`, embedding dimension `d_model`,
   - Positional encoding over time,
   - Multi‑Head Self‑Attention + feed‑forward blocks,
   - Condition embeddings:
     - add or concatenate an embedding for `ConditionID` (FD002/FD004),
     - allow the model to differentiate operating conditions.

3. **Optional Feature / Sensor Attention**:
   - Additional attention block over **feature dimension** (channels),
   - Goal: interpretability (which sensors / physics features matter most),
   - Could be implemented as:
     - per‑time‑step feature attention,
     - or a separate pooling block over features.

4. **Heads**:
   - **RUL Head** (EOL):
     - MLP on a pooled encoder representation (e.g. attention pooling or CLS token),
     - outputs scalar RUL at evaluation time (one value per engine).
   - **HI Head** (scalar HI):
     - predicts Health Index in [0, 1] at the evaluation point,
     - trained with physics‑informed HI targets and monotonicity loss.
   - **Trajectory Head** (optional):
     - predicts a short HI or RUL trajectory into the future,
     - can be implemented via a small decoder or direct projection from encoder states.

## 4. Losses and Physics Constraints

We want to keep the **physics‑informed HI behavior** already introduced for WorldModel V3:

- HI ≈ 1.0 when RUL is large (healthy plateau),
- HI declines monotonically (in expectation) as RUL decreases,
- HI ≈ 0.0 near EOL.

Loss components:

- **RUL Loss**:
  - MSE or asymmetric weighted loss on RUL at EOL,
  - optional **tail weighting** for small RUL (e.g. `< 40 cycles`), similar to the
    `eol_tail_weight` used in WorldModel V3.

- **HI Loss (scalar)**:
  - MSE between predicted HI and physics‑informed HI target at evaluation point,
  - HI target computed from RUL via piecewise linear rule (plateau + decay).

- **HI Sequence Loss (if trajectory predicted)**:
  - `monotonic_health_loss` from `src/loss.py`:
    - base MSE to target HI sequence,
    - penalty for HI increases over time,
    - second‑order smoothness penalty to reduce jitter.

## 5. Evaluation Plan

The Transformer+Attention encoder should use the **same evaluation and diagnostics pipeline**
as Phase‑4 and WorldModel V3:

- Engine‑level metrics on EOL predictions:
  - RMSE, MAE, Bias,
  - R²,
  - NASA PHM08 score (sum and mean).
- Optional **tail metrics** for true RUL `< 40`:
  - Tail‑RMSE, Tail‑MAE, Tail‑Bias, Tail‑NASA (mean).
- Per‑condition metrics (FD002/FD004):
  - same metrics per `ConditionID`.
- Diagnostics plots:
  - error histogram,
  - true vs. predicted RUL scatter,
  - HI + RUL trajectories for degraded engines.

Additional Transformer‑specific diagnostics:

- **Attention maps over time**:
  - visualize which past cycles contribute most to RUL / HI decisions.
- **Feature / sensor importances**:
  - aggregate attention weights or gradient‑based importances per feature.
- **Condition‑wise behavior**:
  - compare attention patterns between different `ConditionID`s.

## 6. Integration Strategy

To minimize refactor risk:

- Reuse the existing:
  - `build_fd_dataset` / feature engineering,
  - EOL sequence builders,
  - evaluation and diagnostics scripts.
- Introduce the Transformer+Attention encoder as a **new encoder_type**
  (e.g. `"universal_v3_transformer"`), wired into:
  - `src/experiment_configs.py`,
  - `run_experiments.py`,
  - the standard diagnostics pipeline.
- Keep old entry points (LSTM, UniversalEncoderV1/V2, WorldModel V3) working unchanged.

## 7. Next Steps

1. Finalize and freeze Phase‑4 baselines (see `docs/phase4_baselines.md`).
2. Stabilize the shared data + EOL‑sequence pipeline.
3. Implement a first Transformer encoder prototype that:
   - plugs into the existing training / evaluation loops,
   - uses the same metrics and diagnostics,
   - can be toggled via YAML config and `encoder_type`.
4. Iterate on capacity, regularization, and attention visualization based on FD004 results.


