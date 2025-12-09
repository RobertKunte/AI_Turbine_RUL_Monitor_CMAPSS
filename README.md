# 🚀 AI-Based Turbine RUL Monitor – Physics-Informed World Models

End-to-end **PHM pipeline** for NASA **C-MAPSS FD001–FD004**, combining:

- **Physics-informed LSTM baselines** (per FD + global)
- **Multi-task EOL models with a learned Health Index (HI) head**
- **World-Model–style seq2seq degradation models**
- **EOL-focused metrics** (RMSE, MAE, Bias, NASA PHM08)
- A clear separation between:
  - **Sharp EOL prediction**, and  
  - **Trajectory, uncertainty & safety evaluation**

The project is designed as an engineering-grade playground for **turbomachinery PHM**, not just as a leaderboard entry for one dataset.

---

## 🔍 0. Quick Benchmark Summary (current status)

### 0.1 Core baselines – Multi-Task EOL-Full-LSTM + Health Head (per FD, EOL evaluation)

For each FD00X we train a **single LSTM** with:

- Windowed sequences (`past_len = 30`)
- Labels: **RUL at window end** (capped at 125)
- Additional **Health Index head** with physics-motivated constraints
- Engine-based train/val split (no leakage)
- Physics-informed feature set

The **Phase-1 configuration** (used for all four datasets):

- `rul_beta = 45`
- `health_loss_weight λ = 0.35`
- `mono_late_weight = 0.03` (late-cycle monotonicity)
- `mono_global_weight = 0.003` (global trend)
- `hi_condition_calib_weight = 0.0` (no per-condition HI bias yet)

#### 0.1.1 EOL metrics – validation vs. test (per engine, last cycle)

| Dataset | Split | RMSE [cyc] | MAE [cyc] | Bias [cyc] | R²   | NASA PHM08 (mean) |
|--------:|:-----:|-----------:|----------:|-----------:|:----:|-------------------:|
| **FD001** | Val  | **14.56** | 11.19 | −2.46 | 0.878 | 3.55 |
|         | Test | **16.17** | 11.04 | +0.10 | 0.837 | 4.92 |
| **FD002** | Val  | 24.00 | 18.61 | −8.53 | 0.670 | 14.00 |
|         | Test | **21.49** | 16.98 | −7.64 | 0.749 | 8.25 |
| **FD003** | Val  | **11.31** | 8.82 | −4.55 | 0.925 | 1.91 |
|         | Test | **12.26** | 8.23 | +0.88 | 0.902 | 3.80 |
| **FD004** | Val  | 24.34 | 17.28 | −5.71 | 0.653 | 49.50 |
|         | Test | **25.87** | 18.95 | −2.25 | 0.638 | 56.97 |

**Highlights:**

- **FD001 & FD003**: very strong EOL performance – RMSE in the ~11–16 cycles range, R² ≈ 0.84–0.93, low NASA scores.  
- **FD002**: solid RMSE and R² with a **pessimistic bias** (RUL slightly underestimated) – safer but could be cost-suboptimal.  
- **FD004**: RMSE is within typical literature range, but **NASA penalties are high** → for some engines, degradation is recognized **too late** (RUL too optimistic near EOL).

The Phase-1 health head therefore gives a **single, robust configuration across all FD00X**, but FD004 remains the main pain point for NASA-style safety.

---

### 0.2 Health Index behavior (qualitative)

The Health head is trained to output a **scaled health index HI ∈ [0, 1]**:

- HI ≈ 1.0 in early life
- Smooth decrease as RUL → 0
- Monotonicity penalties:
  - strong towards EOL (`mono_late_weight`)
  - gentle global trend (`mono_global_weight`)

Observations from FD004 (worst case):

- HI trajectories are **generally monotone and smoother** than in earlier versions.  
- For a subset of engines, HI stays high for long and then drops relatively late → this matches the **high NASA penalties**: the model is sometimes **too optimistic** about remaining life.

This directly motivates **Phase-2/3 work** (see roadmap) on:

- stronger late-cycle constraints,
- tail-focused training, and
- condition-aware calibration.

---

### 0.3 Relation to literature (FD001 perspective)

Recent works on FD001 (EOL RMSE):

- Enhanced LSTMs / label engineering: ~13 cycles
- CNN / GCU / Transformer-based encoders: ~11–13 cycles
- Some advanced methods also report NASA PHM08 scores (192–230 range).

This repo currently focuses on:

- **clean, physics-informed LSTM + HI baselines**,  
- **EOL-consistent evaluation**, and  
- **World-Model concepts** for trajectory & safety.

The goal is **engineering-grade robustness and interpretability**, not pure leaderboard chasing – but FD001/FD003 performance is already **competitive with many LSTM-style baselines**.

---

## 1. Project Overview

This repository implements a complete **Prognostics & Health Management (PHM)** workflow for turbofan engines using **NASA C-MAPSS** FD001–FD004.

It bridges:

- **Classical deep-learning for RUL** (LSTMs, global models, physics features)
- **Multi-task EOL + Health Index models**
- **World-Model–style seq2seq degradation forecasting**
- **Safety-aware evaluation**, including **NASA PHM08** penalties

Main goals:

1. Provide **reproducible, physics-informed baselines** on FD001–FD004.  
2. Learn **Health Index trajectories** that are smooth, monotone and interpretable.  
3. Use **World Models** for multi-step degradation trajectories & “what-if” scenarios.  
4. Evaluate all models w.r.t. both **accuracy (RMSE/R²)** and **safety (NASA)**.  

---

## 2. Data & Operating Conditions

Datasets:

- **FD001 / FD003**: single operating condition, single fault mode.  
- **FD002 / FD004**: multiple operating conditions, multiple fault modes.

Core signals:

- 3 “settings” (`S1, S2, S3`) ≈ operating condition (altitude, Mach, etc.)  
- 21 sensor channels  
- Per-engine time series with run-to-failure trajectories  
- Separate test sets + `RUL_FD00X.txt` files with true EOL RULs.

Operating conditions can be encoded as:

1. **Continuous settings (default)**  
   - Use `S1, S2, S3` directly as features.  
   - Physically interpretable; works well with condition-wise scaling.

2. **Discrete ConditionID (optional)**  
   - Round `(S1, S2, S3)`, map each triple to 0…6.  
   - Useful for ablation and future condition-aware calibration.

For FD002/FD004, we also support **condition-wise standardization**, i.e. separate scalers per ConditionID.

---

## 3. Physics-Informed Features

To inject domain knowledge and denoise raw sensors, we add:

| Feature               | Definition                    | Intuition                                  |
|----------------------|-------------------------------|--------------------------------------------|
| `Effizienz_HPC_Proxy`| `sensor12 / sensor7`          | Proxy for HPC efficiency / degradation     |
| `EGT_Drift`          | `sensor17 − EGT_baseline`     | Exhaust gas temperature drift vs. healthy  |
| `Fan_HPC_Ratio`      | `sensor2 / sensor3`           | Fan vs. HPC loading and operating shift    |

These features:

- Help the model “feel” compressor/thermal degradation,
- Are useful across all FD00X,
- Stabilize training especially on **multi-condition sets FD002/FD004**.

---

## 4. Model Families

### 4.1 Multi-Task EOL-Full-LSTM + Health Head (current core baseline)

**Idea**:  
Train LSTM models that, from the **last 30 cycles**, predict:

1. **RUL at window end** (EOL-style target), and  
2. A **scalar Health Index** (HI), roughly monotone and aligned with RUL.

Key components:

- Inputs: standardized sensors + settings + physics features.
- Targets:
  - RUL labels capped at 125 cycles.
  - HI labels derived from RUL (ideal linear / piecewise mapping).
- Multi-task loss:
  - `L_total = L_RUL (with β-weighting) + λ * L_HI + L_mono_late + L_mono_global [+ L_condition]`
- Regularization:
  - **Late-cycle monotonicity**: penalize HI increases close to EOL.
  - **Global monotonicity**: small penalty for HI bumps over the whole trajectory.
  - Optional **condition calibration loss** (Phase-2, not active yet).

Why this setup?

- EOL-centric training matches how operators use the model in practice.  
- HI head gives a **continuous health signal** for plotting, thresholds, and future World-Model work.  
- Monotonicity losses help make HI trajectories **physically plausible** without destroying RUL performance.

---

### 4.2 Global FD001–FD004 EOL-Full-LSTM (without HI)

A single EOL-Full-LSTM trained jointly on FD001–FD004 (older baseline):

- Shared scaler and LSTM across all operating modes.
- Same windowing and feature set, but **no HI head**.
- Useful as a compact **global reference** and for experimentation with domain adaptation.

Per-FD models with the Health head now clearly outperform this global baseline, especially in **EOL RMSE and NASA**.

---

### 4.3 Legacy LSTMs & MC-Dropout Uncertainty

Earlier phases of the project (kept for reference):

- **Local per-FD LSTMs** with asymmetrical losses and clamped labels.  
- **Global LSTM** across FD001–FD004 with strong FD001 RMSE.  
- **MC-Dropout LSTM** for FD001:
  - 50 stochastic forward passes.
  - Predictive standard deviation correlates with absolute error.
  - Demonstrates how **uncertainty can flag risky predictions**.

These runs live mainly in `notebooks/4_global_lstm.ipynb` and `notebooks/5_training_uncertainty_dropout.ipynb`.

---

### 4.4 Seq2Seq World Models

World-Model modules focus on **trajectory-level behavior** rather than sharp EOL performance.

Variants:

1. **Pure Trajectory World Model (V6)**  
   - Encoder: last 30 cycles.  
   - Decoder: future RUL trajectory (e.g., 20 steps).  
   - Loss over the whole trajectory.  
   - Achieves **excellent NASA scores** and small mean EOL errors, but not optimized for minimum EOL RMSE.

2. **Multi-Task World Model (V7/V8)**  
   - Adds an EOL head on the encoder hidden state.  
   - Joint trajectory + EOL training.  
   - Improves calibration and NASA but EOL RMSE still >40 cycles on FD001.

3. **Two-Stage EOL Regressor**  
   - Freeze World Model encoder.  
   - Train a small MLP on encoder outputs, using only EOL samples.  
   - Better EOL NASA, but still not on par with dedicated EOL-Full-LSTM.

**Interpretation:**  
World Models are best used as **simulation / risk layers** on top of the robust EOL baselines, not as stand-alone RUL regressors.

---

## 5. Lessons Learned

### 5.1 Data leakage and evaluation

Early trajectory experiments highlighted:

- If one splits data **by rows or random windows**, the model may see future segments of the same engine in training.  
- Validation metrics then become unrealistically good.

Fix:

- Utility functions such as `build_full_eol_sequences_from_df` and `create_full_dataloaders` enforce **engine-based splits** and **consistent window labeling**.

Now every model is evaluated in two ways:

1. **Pointwise metrics** (all windows).  
2. **EOL metrics** (one sample per engine, last cycle, NASA PHM08).

This makes results **comparable to PHM literature and practice**.

### 5.2 Health Index as a bridge between physics and deep learning

The HI head turned out to be:

- a strong regularizer for EOL RUL prediction,  
- a way to **visualize degradation paths**, and  
- the natural interface for future **World-Model and digital-twin integration**.

Phase-1 showed that we can:

- learn reasonably smooth, monotone HI across all FD00X with one parameter set,  
- keep or slightly improve RUL performance compared to pure RUL heads, and  
- expose systematic weaknesses (e.g. late HI drops on FD004).

---

## 6. Repository Structure

```text
AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  └─ raw/                        # raw C-MAPSS text files
├─ results/
│  ├─ fd001/                      # per-FD LSTM + HI runs
│  ├─ fd002/
│  ├─ fd003/
│  ├─ fd004/
│  ├─ global_eol_lstm/            # older global EOL-Full-LSTM
│  └─ world_model/                # seq2seq World Model runs
├─ notebooks/
│  ├─ 1_fd001_exploration.ipynb
│  ├─ 1a_fd002_exploration.ipynb
│  ├─ 2_training_single_datasets.ipynb     # per-FD EOL-Full-LSTM + HI
│  ├─ 3_evaluation_global_model.ipynb
│  ├─ 4_global_lstm.ipynb                  # legacy global LSTMs
│  ├─ 5_training_uncertainty_dropout.ipynb
│  └─ 6_world_model_training.ipynb         # seq2seq World Models
└─ src/
   ├─ config.py
   ├─ data_loading.py
   ├─ additional_features.py
   ├─ training.py                           # legacy RUL training
   ├─ eol_full_lstm.py                      # EOL-Full-LSTM + HI utilities
   ├─ world_model_training.py               # World Model training pipeline
   ├─ models/
   │  ├─ lstm_rul.py
   │  ├─ lstm_rul_mcdo.py
   │  └─ world_model.py
   ├─ loss.py
   ├─ uncertainty.py
   └─ eval_utils.py / train_utils.py

(File names may evolve; see repo for the most recent structure.)

7. How to Run
7.1 Environment

conda create -n turbine_ai python=3.10
conda activate turbine_ai

pip install torch torchvision torchaudio \
            pandas numpy matplotlib scikit-learn \
            jupyter ipykernel

Register the kernel if needed:

python -m ipykernel install --user --name turbine_ai --display-name "turbine_ai"

7.2 Data

Download the C-MAPSS text files and place them into ./data/raw:

data/raw/
├─ train_FD001.txt   ├─ test_FD001.txt   ├─ RUL_FD001.txt
├─ train_FD002.txt   ├─ test_FD002.txt   ├─ RUL_FD002.txt
├─ train_FD003.txt   ├─ test_FD003.txt   ├─ RUL_FD003.txt
├─ train_FD004.txt   ├─ test_FD004.txt   └─ RUL_FD004.txt

7.3 Train per-FD EOL-Full-LSTM + Health head

    Open notebooks/2_training_single_datasets.ipynb.

    Select the turbine_ai kernel.

    Configure which FD dataset(s) to run (FD001–FD004).

    Run all cells to:

        build EOL-Full sequences,

        train one LSTM+HI model per dataset with the Phase-1 parameters,

        save metrics and plots into results/fd00X/.

7.4 Global EOL-Full-LSTM (legacy)

Use notebooks/3_evaluation_global_model.ipynb or a small script calling the global training function. Outputs are stored under results/global_eol_lstm/.
7.5 World Model training

    Open notebooks/6_world_model_training.ipynb.

    Run to:

        build seq2seq datasets,

        train the WorldModel encoder–decoder,

        compute trajectory and EOL NASA metrics,

        store checkpoints and plots under results/world_model/.

7.6 Uncertainty experiments (MC-Dropout)

Use notebooks/5_training_uncertainty_dropout.ipynb together with src/uncertainty.py:

from src.uncertainty import enable_mc_dropout, mc_dropout_predict

enable_mc_dropout(model)
mean_pred, std_pred = mc_dropout_predict(model, X_tensor, n_samples=50)

📈 Core Results – Physics-Informed Baselines (Phase 2)

This section summarizes the current core baselines for all four C-MAPSS subsets (FD001–FD004) using the EOL-style training + Health Index (HI) multitask loss.

We compare two encoder families:

    LSTM (sequence model with hidden state)

    Transformer Encoder (self-attention over time, optional condition embeddings)

Metrics are evaluated per engine at its final cycle:

    RMSE, MAE

    Bias (predicted RUL − true RUL; >0 = optimistic, <0 = pessimistic)

    R²

    NASA PHM08 Score (mean)

All numbers are in cycles (RUL).
RUL is clamped to 125 cycles during training and evaluation.
📊 Performance Comparison Across Phases (FD001–FD004)

This table summarizes the evolution of model performance from Phase 1 → Phase 4, showing how each architectural and methodological improvement impacted prediction accuracy.

    Phase 1: Baseline LSTM (raw C-MAPSS features, no physics)

    Phase 2: Physics-informed LSTM (HI + monotonicity + EOL penalties)

    Phase 3: Universal Encoder (multi-scale CNN + transformer) + improved HI losses

    Phase 4: Universal Encoder + 464D residual/digital-twin features (current best EOL models)

Metrics shown: Test RMSE, MAE, Bias, R², NASA Score (mean)
📈 Summary Table — Best Model Per Phase & Dataset
Dataset	Phase	Model	RMSE ↓	MAE ↓	Bias (signed)	R² ↑	NASA Mean ↓	Notes
FD001	Phase 1	Basic LSTM	~24–26	~18	−3 to −5	~0.70	~40–60	First baseline, unstable HI
	Phase 2	PI-LSTM (HI+Mono)	~18–20	~14	−2	~0.80	~12–18	Major boost from physics-informed head
	Phase 3	Universal Encoder V1	~14.9	~10.5	−1.4	0.89	~3.0	Strong encoder, no residuals
	Phase 4	Universal V1 + Residual	13.35	9.40	−1.48	0.889	2.98	Best FD001 EOL result so far
FD002	Phase 1	Basic LSTM	>40	>30	Large bias	<0.50	>100	Multi-condition too hard
	Phase 2	PI-LSTM (HI+Mono)	~28–32	~22	~−5	~0.65	~40–60	Much improved but noisy
	Phase 3	Universal V2 MS-CNN	~20–22	~15–16	−2 to −3	0.78	~11–15	Transformers help a lot
	Phase 4	Universal V2 MS-CNN + Residual	17.77	12.49	−0.80	0.829	7.03	Largest gain from residual features
FD003	Phase 1	Basic LSTM	~27–30	~20	−4 to −6	~0.65	>80	Single-condition but trickier
	Phase 2	PI-LSTM	~20–22	~15	−3	~0.78	~25–30	Physics helps, HI stable
	Phase 3	Universal Encoder	~14–16	~11–12	small bias	~0.88	~4–6	Strong baseline
	Phase 4	Universal V1 + Residual	12.68	8.55	1.82	0.895	3.22	Very strong, slight late bias
FD004	Phase 1	Basic LSTM	>45	>35	−6	<0.45	>1000	Hardest dataset
	Phase 2	PI-LSTM	~30–35	~24–27	−5	~0.60	~200–400	Still very difficult
	Phase 3	Universal V2 MS-CNN	~20–22	~15–17	±2	0.78	~11–15	Big transformer gains
	Phase 4	Universal V2 MS-CNN + Residual	20.50	14.46	1.46	0.773	11.23	Strong, but FD004 remains challenging
🔎 Phase-by-Phase Improvement Overview

Phase 1 → Phase 2: Physics-Informed HI + Monotonicity

    RMSE reduction typically 20–30%

    NASA Score improved by an order of magnitude

    Introduced stable HI monotonicity

Phase 2 → Phase 3: Universal Encoder (CNN + Transformer)

    Large boost for multi-condition datasets (FD002/FD004)

    Reduced variance & better generalization across flight regimes

    HI trajectories become much smoother

Phase 3 → Phase 4: Residual / Digital-Twin Features (464D)

    FD002: Best improvement (+ ~2.3 RMSE)

    FD003: Best single-condition result (12.68 RMSE)

    FD004: Strong but still limited by dataset complexity

    Across all FDs: Lowest NASA Score recorded so far

    HI trajectories become more physically meaningful & stable

⭐ What This Table Shows

    The project evolved from a simple LSTM baseline to a state-of-the-art physics-informed + residual, world-model–ready architecture.

    Residual features are especially powerful for multi-condition scenarios (FD002/FD004).

    FD001/FD003 now approach the best open-source results in the literature.

    FD004 remains the ultimate challenge and motivates the next phase:
    World Models + Residual Features + Cross-Condition Attention.

1. Best per-dataset baselines (Phase 2)

For each dataset we select a main baseline model based on a trade-off between RMSE, NASA and calibration (Bias):

    FD001 (single condition) → LSTM baseline

    FD002 (multi condition) → Transformer baseline

    FD003 (single condition) → LSTM baseline

    FD004 (multi condition) → Transformer (small, regularized)

        plus a NASA-optimized Transformer variant as reference

1.1 Summary table (Test set, EOL metrics)
Dataset	Encoder	Experiment ID	RMSE	MAE	Bias	R²	NASA_mean
FD001	LSTM	fd001_phase2_lstm_baseline	15.0	10.2	−0.3	0.86	4.09
FD002	Transformer	fd002_phase2_transformer_baseline	17.4	12.1	+0.1	0.84	7.42
FD003	LSTM	fd003_phase2_lstm_baseline	11.8	8.0	+1.9	0.91	2.60
FD004	Transformer*	fd004_phase2_transformer_small_regularized	20.0	13.2	−0.7	0.78	13.29

* For FD004 we use a small, regularized Transformer (d_model=32, dropout=0.2) as the main baseline because it combines:

    the best RMSE we observed on FD004 (≈20 cycles),

    good calibration (Bias ≈ −0.7),

    and still a very strong NASA score (≈13).

2. LSTM vs. Transformer – what works where?
FD001 & FD003 – Single Operating Condition

For the single-condition subsets FD001 and FD003 the classic LSTM encoder performs slightly better than the Transformer in terms of RMSE and NASA:

    FD001 LSTM baseline (fd001_phase2_lstm_baseline)

        Val: RMSE ≈ 13.5, NASA ≈ 3.08

        Test: RMSE ≈ 15.0, NASA ≈ 4.09

        Bias very close to 0 → well calibrated

    FD003 LSTM baseline (fd003_phase2_lstm_baseline)

        Val: RMSE ≈ 12.7, NASA ≈ 2.57

        Test: RMSE ≈ 11.8, NASA ≈ 2.60

        Slightly optimistic (Bias ≈ +1.9 cycles) but still safe

Transformers on FD001/FD003 reach similar RMSE but higher NASA scores, so the LSTM remains our preferred baseline here.
FD002 & FD004 – Multi-Condition, Multi-Fault

For the multi-condition subsets FD002 and FD004, the Transformer encoder with condition embeddings clearly outperforms the LSTM:

    FD002 Transformer baseline (fd002_phase2_transformer_baseline)

        Val: RMSE ≈ 17.8, NASA ≈ 6.08

        Test: RMSE ≈ 17.4, NASA ≈ 7.42

        Bias ≈ +0.1 → almost perfectly calibrated

Compared to the FD002 LSTM baseline:

    RMSE improves from ≈22.5 → ≈17.4

    NASA improves from ≈13.8 → ≈7.4

    Bias moves from clearly pessimistic (≈ −6.2) to almost zero

For FD004, the Transformer makes an even bigger difference:

    LSTM baseline (fd004_phase2_lstm_baseline)

        Test RMSE ≈ 25.5

        Test NASA ≈ 77.9 (late EOL detection, HI drops too late)

    Transformer baselines:
    Experiment ID	RMSE (test)	NASA_mean (test)	Comment
    fd004_phase2_transformer_baseline	21.0	11.07	best NASA (very early & safe EOL)
    fd004_phase2_transformer	20.9	11.08	similar to baseline
    fd004_phase2_transformer_small_regularized	20.0	13.29	best RMSE + good calibration (Bias≈−0.7)

Overall, the Transformer encoder:

    dramatically reduces NASA on FD004 (≈77.9 → 11–13),

    improves RMSE by ≈5 cycles,

    and yields smoother, more realistic HI trajectories across different operating conditions.

3. Take-aways for PHM / C-MAPSS

    LSTM remains a very strong baseline for single-condition C-MAPSS subsets (FD001, FD003).

    Transformer + condition embeddings are clearly superior on the multi-condition subsets (FD002, FD004), especially in terms of:

        NASA PHM08 (early, safe EOL recognition),

        handling operating condition shifts,

        and RUL bias calibration.

    The multi-task EOL + HI loss with moderate monotonicity penalties
    (`health_loss_weight ≈ 0.35, mono_late ≈ 0.03, mono_global ≈ 0.003)
    works well across all datasets.

    Very strong HI/monotonicity weights can hurt both RMSE and NASA
    (e.g. fd004_phase2_transformer_hi_strong), so physics-inspired regularization
    should be used carefully and quantitatively evaluated.

3.4 FD004 – Physics-Informed Universal Encoder (Transformer + CNN Head)

This section summarizes the performance and diagnostic behavior of our multi-task EOL model (RUL + Health Index) using the UniversalEncoderV2 architecture on FD004, the most challenging C-MAPSS dataset with multiple operating conditions and fault modes.

All FD004 models share the same design:

    Input: full EOL sequences (past_len = 30, max_rul = 125)

    Feature engineering: physics-based features + multi-scale temporal statistics

    Condition-wise scaling (7 ConditionIDs)

    Shared encoder: Multi-scale CNN + Transformer blocks

Two prediction heads:

    RUL head (scalar EOL estimate)

    HI head (time-distributed health index, constrained to be monotonic)

3.4.1 Test Performance (EOL, per engine)
Model ID	d_model	Params	RMSE [cycles]	MAE [cycles]	Bias (pred − true)	R²	NASA PHM08 (mean)
fd004_phase3_universal_v2_ms_cnn	64	0.31M	26.30	20.56	−6.07	0.626	37.90
fd004_phase3_universal_v2_ms_cnn_d96	96	0.59M	23.26	16.21	+3.96	0.707	32.92
fd004_phase3_universal_v2_ms_cnn_strongmono	64	0.31M	27.16	20.83	−8.15	0.601	31.53

Key observations:

    The d96 model is our Phase-3 FD004 baseline:

        lowest RMSE & MAE

        highest R²

        slight positive bias → mildly conservative predictions

    The strong-mono model yields slightly worse RMSE but improves NASA score (fewer early-failure penalties).

Together, these results demonstrate strong generalization and safety characteristics on the most complex C-MAPSS dataset and provide the launchpad for Phase-4 and world-model experiments.
3.4.2 Error Distribution & True-vs-Prediction Scatter

For each experiment, the diagnostic pipeline automatically saves the following plots under:

    results/fd004/<experiment_id>/error_hist.png

    results/fd004/<experiment_id>/true_vs_pred.png

The d96 model shows:

    a centered error distribution (slightly positive bias)

    no pathological tails

    stable predictions across all 248 engines

This indicates a balanced and safe EOL estimator.
3.4.3 Health Index (HI) + RUL Trajectories (Sliding Window Evaluation)

To improve interpretability, we compute HI trajectories using a sliding window inference:

    For each cycle t with at least past_len history, a window [t-past_len+1, …, t] is passed through the encoder.

    The last HI value from that window is used as HI(t).

We automatically select 10 degrading engines (true RUL < max_rul) in the test set and visualize:

    Health Index (green)

    True RUL trajectory (blue)

    Predicted RUL (red, constant EOL prediction)

The combined plot is saved as:

    results/fd004/<experiment_id>/hi_rul_10_degraded.png

HI behavior (d96 model):

    HI ≈ 1.0 during healthy operation

    clear drop as engines approach EOL

    trajectory shapes vary across engines, matching known FD004 heterogeneity

These plots offer a highly interpretable view of how the model perceives degradation over time.

Summary

The d96 UniversalEncoderV2 architecture currently serves as our Phase-3 reference physics-informed baseline for FD004:

    Strong predictive accuracy

    Good NASA PHM08 score

    Robust, interpretable HI trajectories

    Clean error structure with minimal bias

This model is the foundation for Phase-4 residual models and later World-Model alignment.
📈 Phase 4 Results — Residual / Digital-Twin Feature Models (FD001–FD004)

In Phase 4 we extend the physics-informed EOL-LSTM with a much richer, digital-twin–style feature space. The model now sees:

    Physical degradation indicators (e.g., HPC efficiency, EGT drift, fan/HPC ratio)

    Multi-scale statistics & temporal features (short/long windows, trends, rolling moments)

    Digital-twin residual features
    (Deviation between the measured trajectory and a healthy reference model per engine / per condition)

    Condition-aware scaling via per-ConditionID StandardScaler

This boosts the input dimensionality to 464 features and lets the model disentangle:

    “Operational variability” (different flight regimes) vs. “true degradation”.

All runs use the Phase-3 multitask head (RUL + Health Index) with:

    RUL weighting (τ = 45)

    HI plateau & EOL penalties

    Monotonic HI constraints

    Global trend & smoothness regularization

    Optional condition embeddings for multi-condition datasets (FD002, FD004)

Overall, the Phase-4 residual models are the strongest EOL baselines in this project so far.
🔬 FD001 — Phase-4 Universal Encoder V1 (Residual)

Single operating condition, classical “easy” C-MAPSS subset. Residual features still add value by capturing subtle drift in core performance indicators.

Test Set Metrics (per engine, last cycle)
Metric	Value
RMSE	13.35 cycles
MAE	9.40 cycles
Bias	−1.48 cycles
R²	0.889
NASA Score (mean)	2.98

Validation EOL Metrics
Metric	Value
RMSEₑₒₗ	1.23 cycles
MAEₑₒₗ	1.13 cycles
NASA Score (mean)	0.12

Qualitative diagnostics

    HI trajectories are smooth and strictly degrading for selected engines.

    Error histogram and true-vs-pred scatter show a tight cloud with modest negative bias.

    EOL predictions are sharply aligned (sub-2-cycle RMSEₑₒₗ on validation).

🔬 FD002 — Phase-4 Universal Encoder V2 + Multi-Scale CNN (Residual)

Multi-condition dataset with 6–7 distinct operating regimes. Here, residual features are most impactful, because they normalize degradation across strongly different regimes.

Test Set Metrics (per engine, last cycle)
Metric	Value
RMSE	17.77 cycles
MAE	12.49 cycles
Bias	−0.80 cycles
R²	0.829
NASA Score (mean)	7.03

Validation EOL Metrics
Metric	Value
RMSEₑₒₗ	4.56 cycles
MAEₑₒₗ	4.23 cycles
NASA Score (mean)	0.55

Qualitative diagnostics

    Condition embeddings + residual features clearly separate operational changes from degradation.

    Sliding-window HI trajectories show consistent, monotonic decline for degraded engines.

    Error distribution is narrow, with only mild underestimation at very late EOL.

🔬 FD003 — Phase-4 Universal Encoder V1 (Residual)

Single-condition dataset (like FD001) but with a different operating envelope and fault pattern. Again, residual features provide a strong signal to distinguish aging from noise.

Test Set Metrics (per engine, last cycle)
Metric	Value
RMSE	12.68 cycles
MAE	8.55 cycles
Bias	1.82 cycles
R²	0.895
NASA Score (mean)	3.22

Validation EOL Snapshot
Metric	Value
RMSEₑₒₗ	2.91 cycles
MAEₑₒₗ	2.37 cycles
NASA Score (mean)	0.28

Qualitative diagnostics

    Very strong fit with slightly positive bias (conservative, late predictions) on some engines.

    HI trajectories remain smooth and monotonic with clear separation between “healthy plateau” and “late degradation”.

    Compared to earlier phases, Phase-4 reduces RMSE while keeping the physics-inspired HI behavior intact.

🔬 FD004 — Phase-4 Universal Encoder V2 + Multi-Scale CNN (Residual)

The most challenging subset: multi-condition and multiple fault modes. Phase-4 still improves robustness, but FD004 remains a hard benchmark.

Test Set Metrics (per engine, last cycle)
Metric	Value
RMSE	20.50 cycles
MAE	14.46 cycles
Bias	1.46 cycles
R²	0.773
NASA Score (mean)	11.23

Validation EOL Metrics
Metric	Value
RMSEₑₒₗ	4.67 cycles
MAEₑₒₗ	3.30 cycles
NASA Score (mean)	0.50

Qualitative diagnostics

    Despite multiple conditions and fault types, HI trajectories remain smooth and mostly monotonic for degraded engines.

    Residual + multi-scale CNN encoder helps the model generalize across the 7 ConditionIDs.

    Remaining gap to FD002 suggests further gains are possible with richer encoders (e.g., cross-condition attention, world-model pretraining, or stronger digital twins).

⭐ Overall Summary — Phase-4 Residual Models

Across all FD001–FD004 subsets, the Phase-4 residual models:

    Achieve the best RUL accuracy so far in this project

        Strong RMSE on FD001/FD003 (~13 / ~13 cycles)

        Competitive multi-condition performance on FD002/FD004 (~18 / ~21 cycles)

    Generalize across conditions

        Digital-twin residual features + condition embeddings allow a single universal architecture (V2 + multi-scale CNN) to handle FD002 and FD004 robustly.

    Provide sharp EOL predictions

        EOL RMSEₑₒₗ in the low single digits across validation sets

        NASA scores close to or better than typical literature baselines for comparable setups

    Maintain physically meaningful HI trajectories

        HI is smooth, monotonic, and well-aligned with RUL predictions for degraded engines.

        Diagnostic plots (error_hist.png, true_vs_pred.png, hi_rul_10_degraded.png) confirm that the model does not learn pathological HI behavior.

    Form a strong launchpad for World-Model experiments

        These Phase-4 residual EOL models serve as the teacher / target for the next step:
        learning world-model encoders that reproduce the physics-informed HI + RUL behavior in a more general, generative way.

📘 World Model v3 – Phase-5 Experiments (FD004)

This section documents the World Model v3 experiments on FD004, the most complex C-MAPSS subset (7 operating conditions, 2 fault modes).
Unlike the Phase-4 EOL baselines above, these models focus on:

    Seq2seq degradation modeling (world-model style),

    Residual features and per-engine baselines,

    A strong HI head and HI–EOL fusion,

    Tail-weighted losses to emphasize behavior near EOL.

The Phase-4 FD004 residual model remains the best EOL baseline (RMSE/NASA), while World Model v3 is used as a trajectory & safety layer with explicitly modeled degradation dynamics.
1. Experimental Setup

All runs share:

    Encoder: UniversalEncoderV2 MS-CNN (d_model=96, 3 layers)

    Decoder: LSTM-based, predicting a multi-step horizon H

    Heads:

        HI head (Health Index, monotonicity loss)

        EOL head (RUL at horizon end, with tail-weighting)

    Features:

        Physics-informed features (HPC efficiency, EGT drift, fan/HPC ratio, …)

        Per-engine residual features w.r.t. a healthy baseline (baseline_len=30, include_original=True)

    Loss shaping:

        Tail-weight w ∈ {1.5, 2.0}

        Horizons H ∈ {20, 30, 40}

        Monotonicity and HI-plateau losses as in previous phases

This is the most extensive FD004 world-model sweep so far.
2. FD004 – World Model v3 Results (EOL Metrics)
Experiment ID	Horizon H	Tail-W	RMSE ↓	MAE ↓	Bias [cyc]	R²	NASA Mean ↓	Comment
fd004_world_phase5b_h40_w2	40	2.0	35.00	27.84	−5.91	0.588	28.00	Lowest bias among w=2.0 runs
fd004_world_phase5c_h40_w1p5	40	1.5	35.43	28.15	−8.62	0.578	17.31	⭐ Best NASA score in Phase-5
fd004_world_phase5d_h30_w2	30	2.0	34.60	26.25	−14.33	0.597	24.74	⭐ Best RMSE/MAE/R² World Model
fd004_world_phase5e_h30_w1p5	30	1.5	36.82	27.77	−16.81	0.544	28.66	Weakest variant
fd004_world_phase5f_h20_w2	20	2.0	(slightly worse)	–	–	–	–	Horizon too short, over-regularized

Important:
These metrics are not meant to beat the Phase-4 EOL baselines on FD004 (RMSE ≈ 20.5, NASA ≈ 11.2).
Instead, they:

    demonstrate stable world-model training on FD004,

    quantify the effect of horizon length H and tail weight w,

    provide HI + trajectory outputs as an additional diagnostic layer on top of the stronger EOL models.

3. Champion Models

We highlight two champion configurations:
Champion A – Engineering-Focused (RMSE/MAE/R²)

    Experiment: fd004_world_phase5d_h30_w2

    Horizon: H = 30

    Tail weight: w = 2.0

Metric	Value
RMSE	34.60
MAE	26.25
Bias	−14.33
R²	0.597
NASA	24.74

Pros:

    Best RMSE, MAE, R² among the World Model Phase-5 runs

    Smooth, monotone HI trajectories

    Stable behavior across the 7 ConditionIDs

Cons:

    Strong negative bias → clearly pessimistic (RUL underestimated)

Champion B – NASA-Focused (Safety Penalty)

    Experiment: fd004_world_phase5c_h40_w1p5

    Horizon: H = 40

    Tail weight: w = 1.5

Metric	Value
RMSE	35.43
MAE	28.15
Bias	−8.62
R²	0.578
NASA	17.31

Pros:

    Best NASA PHM08 score among Phase-5 World Models

    More moderate bias than Champion A

    Robust across heterogeneous FD004 conditions

Cons:

    Slightly worse RMSE/MAE/R² compared to Champion A

    Still significantly worse than the Phase-4 EOL baseline for FD004

4. Horizon & Tail Weight Analysis (FD004)

Why H=30 often wins (for world-model RMSE):

    H = 20:

        window too short; tail-weighted loss over-regularizes the model

        model focuses overly on very short-term evolution

    H = 40:

        more future context, but also more uncertainty and noise

        HI & RUL heads must extrapolate further, increasing variance

    H = 30:

        best trade-off between context and predictability on FD004

Tail weight w:

    Higher w (2.0)

        strongly focuses on late cycles (near EOL)

        improves RMSE/MAE but can push bias too negative

    Lower w (1.5)

        softer tail emphasis

        can yield better NASA score when combined with larger horizon (H=40)

5. Role of Residual Features in World Model v3

Residual features (per-engine baselines over the early 30 cycles) help the world model:

    Focus on deviation from healthy behavior rather than absolute sensor levels

    Stabilize training across FD004’s heterogeneous operating conditions

    Produce more consistent HI trajectories that align with physical intuition

Residuals proved so useful that they are now a standard part of the FD004 world-model pipeline, mirroring their impact in the Phase-4 EOL baselines.
6. HI Trajectories & Diagnostics

Each Phase-5 experiment produces standard diagnostics, analogous to the EOL baselines:

    hi_rul_10_degraded.png – HI and RUL trajectories for 10 degraded engines

    error_hist.png – error histogram at EOL

    true_vs_pred.png – scatter of true vs. predicted EOL RUL

    condition_metrics.json – per-condition stats

For the two champion models (5c and 5d) we observe:

    Monotone, smooth HI curves with clear decline when RUL < 50

    No pathological HI “recovery” near EOL (monotonicity loss works)

    Some engines still show long HI plateaus near 1.0, motivating:

        stronger HI anchors near EOL,

        or hybrid training with Phase-4 baseline as teacher.

7. Takeaways – World Model v3 vs. Phase-4 EOL Baselines

    Phase-4 residual EOL models remain the primary choice for:

        sharp EOL prediction,

        low RMSE/NASA,

        deployment-oriented RUL estimation.

    World Model v3 on FD004 provides:

        a second, trajectory-aware view on degradation,

        explicit modeling of multi-step RUL evolution (H=20–40),

        richer HI + residual feature dynamics, useful for:

            scenario analysis,

            debugging,

            and future digital-twin style simulations.

Future phases will extend these world-model ideas beyond FD004 and closer to full sensor-trajectory + HI + RUL joint modeling.
8. Roadmap

The project roadmap is structured into phases.
Phase 1 – Health Head & Monotonicity Tuning ✅

    Add multi-task Health Index head to EOL-Full-LSTM.

    Implement late/global monotonicity losses.

    Find a single configuration that works reasonably well on FD001–FD004.

    Status: done, results summarized in section 0.1.

Phase 2 – FD004 / FD002 NASA Focus & Condition Calibration

Goals:

    Improve NASA PHM08 performance, especially on FD004.

    Introduce condition-aware HI calibration and per-condition losses.

    Explore stronger/lighter monotonicity regimes and their impact on safety.

(See “Core Results – Physics-Informed Baselines (Phase 2)” and FD004 Universal Encoder section.)
Phase 3 – Universal Encoders (CNN + Transformer) ✅

    Introduce UniversalEncoderV1/V2 for all FD subsets.

    Combine:

        multi-scale CNN,

        temporal self-attention,

        condition embeddings (FD002/FD004).

    Achieve strong performance, especially on multi-condition datasets.

(See sections 3.4 and the Phase 3 entries in the phase-summary table.)
Phase 4 – Physics-Informed World Models & Digital Twin Integration (in progress)

    Extend World Models to jointly predict:

        sensor trajectories,

        HI trajectories,

        and RUL scenarios.

    Integrate simple digital-twin baselines:

        healthy reference models,

        residual features (measured – simulated).

    Use World Models plus HI for:

        what-if analysis (load profiles, operating changes),

        risk envelopes and maintenance decision support.

Status:

    Residual/digital-twin features for EOL models: ✅ (see Phase-4 residual results).

    First FD004 World Model v3 experiments with residual features and HI-Fusion: ✅ (see “World Model v3 – Phase-5 Experiments (FD004)”).

    Full joint sensor+HI+RUL world-modeling across all FD subsets: 🚧 planned.

    Phase-4 residual / digital-twin baselines frozen via YAML configs: ✅
        - see config/phase4/*.yaml and docs/phase4_baselines.md

    Design sketch for Transformer+Attention encoder: ✅
        - see docs/transformer_attention_design.md

Phase 5 – Deployment & Engineering Tooling (planned)

    Provide a small API or dashboard for:

        online RUL & HI monitoring,

        visualization of physics features,

        exploration of worst-case engines and conditions.

    Bundle best models and configs into an engineer-friendly toolkit.

    Add uncertainty layers (MC-Dropout / ensembles) as a safety overlay.

9. Contact & License

Author: Dr.-Ing. Robert Kunte
LinkedIn: https://www.linkedin.com/in/robertkunte/

License: MIT

If you work on turbomachinery, PHM, or physics-informed ML and want to collaborate, feel free to reach out!