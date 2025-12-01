# 🚀 AI-Based Turbine RUL Monitor – Physics-Informed World Models

End-to-end **PHM pipeline** for NASA C-MAPSS FD001–FD004, combining:

- **Physics-informed LSTM baselines** (per FD + global)
- **World-Model–style seq2seq degradation models**
- **EOL-focused metrics** (RMSE, MAE, Bias, NASA PHM08)
- A clear separation between:
  - **Sharp EOL prediction** and  
  - **Trajectory, uncertainty & safety evaluation**

---

## 🔍 0. Quick Benchmark Summary

### 0.1 Current core baselines – EOL-Full-LSTM (per FD, EOL evaluation)

Single LSTM model trained per dataset (FD001–FD004), with:

- Windowed sequences (past_len = 30)
- Labels = **RUL at the last cycle** of each window (capped at 125)
- Engine-based train/val split (no leakage)
- Physics-informed features enabled

**Pointwise metrics (all validation samples):**

| Dataset | Engines | Samples | RMSE (point) | MAE (point) | R² (point) | NASA (point, mean) |
|--------:|--------:|--------:|-------------:|------------:|-----------:|--------------------:|
| FD001   | 100     | 17 731  | **16.36**    | 11.87       | 0.845      | 8.37               |
| FD002   | 260     | 46 219  | **19.18**    | 14.52       | 0.789      | 9.73               |
| FD003   | 100     | 21 820  | **13.64**    | 9.47        | 0.891      | 4.99               |
| FD004   | 249     | 54 028  | **21.24**    | 14.99       | 0.736      | 29.04              |

**EOL metrics (one sample per engine, last cycle):**

| Dataset | Engines | RMSE (EOL) | MAE (EOL) | Bias (EOL) | NASA (EOL, mean) |
|--------:|--------:|-----------:|----------:|-----------:|------------------:|
| FD001   | 100     | **2.74**   | 2.58      | +2.58      | 0.30             |
| FD002   | 260     | **4.59**   | 4.13      | +4.13      | 0.54             |
| FD003   | 100     | **3.13**   | 2.92      | +2.92      | 0.35             |
| FD004   | 249     | **9.30**   | 8.46      | +8.46      | 1.54             |

👉 **Takeaway:**  
Per-FD EOL-Full-LSTM models are **strong, physics-informed baselines**: good pointwise RMSE, **very low NASA scores at EOL** (safety), and near-zero or slightly positive bias.

---

### 0.2 Global FD001–FD004 EOL-Full-LSTM (single model for all FDs)

A single EOL-Full-LSTM trained jointly on FD001–FD004:

- Shared scaler + shared LSTM across all operating modes
- Same physics features and windowing
- Engine-based split on the **combined dataset**

**Global validation metrics:**

- **Pointwise (all samples):**
  - RMSE ≈ **21.53** cycles
  - MAE ≈ 15.82
  - Bias ≈ −2.56
  - R² ≈ 0.73

- **EOL (per engine, last cycle across val engines):**
  - RMSE_eol ≈ **17.47** cycles  
  - MAE_eol ≈ 12.90  
  - Bias_eol ≈ +11.62  
  - NASA_mean ≈ **19.1**

👉 **Takeaway:**  
The global model is a **unified baseline** across all FD sets, but **per-FD models are clearly stronger** – especially in EOL RMSE and NASA. This matches the intuition that FD002/FD004 multi-condition / multi-fault behavior is easier to handle with specialized models (or more advanced encoders).

---

### 0.3 World Models vs. LSTM baselines (FD001, qualitative)

From earlier experiments (V6–V8):

| Model                              | FD001 RMSE (EOL) | NASA (EOL mean) | Notes                                                  |
|------------------------------------|------------------|------------------|--------------------------------------------------------|
| Local physics LSTM (old V4)        | ~17–18           | ~300–400         | Early per-FD baseline                                 |
| Global LSTM (old V5)               | ~14–15           | ~400–450         | Joint FD001–FD004 model                               |
| **Seq2Seq World Model (V6)**       | ≫40              | **≪ LSTM NASA**  | Trained on trajectories only (no explicit EOL head)   |
| Multi-task World Model (V7/V8)     | ~41–42           | 280–320          | Joint traj + EOL training                             |
| World Model + EOL Regressor (2-stg)| ~41              | ~220             | Calibrated EOL head on frozen encoder                 |

👉 **Story in one sentence:**  
LSTM baselines achieve **better RMSE**, while **World Models** shine in **trajectory modeling, safety (NASA), and interpretability**, especially when used as calibrated, simulation-style models rather than pure EOL predictors.

---

## 1. Project Overview

This repository implements a complete **Prognostics & Health Management (PHM)** workflow for turbofan engines using the NASA **C-MAPSS** datasets FD001–FD004.

The project bridges:

- **Classical deep-learning for RUL** (LSTM, global models, physics-informed features)
- **World-Model–style degradation forecasting** (seq2seq dynamics, multi-step RUL trajectories)
- **Multi-task models** (trajectory + EOL)
- **Safety-focused scoring** with the NASA PHM08 exponential penalty for late predictions

The main goals:

1. Build **reproducible, physics-informed baselines** on FD001–FD004  
2. Add **World Models** for multi-step degradation trajectories & “what-if” scenarios  
3. Evaluate models with both **RMSE** and **NASA** to balance accuracy and safety  
4. Use **engineer-friendly features** to keep the model interpretable

---

## 2. Data & Operating Conditions

- **FD001 / FD003:** single operating condition, single fault mode  
- **FD002 / FD004:** multiple operating conditions, multiple fault modes

For FD002/FD004, operating conditions are provided as:

- `Setting1`, `Setting2`, `Setting3` (e.g., altitude, Mach, operational condition)

Two encodings were tested:

1. **Continuous settings (default)**  
   Directly use the three setting values as continuous features  
   → **physically realistic** and slightly better NASA score.

2. **Discrete ConditionID (7 clusters)**  
   Round `(S1, S2, S3)` and map each triple to a small integer ID  
   → Similar RMSE / MAE, but less physically interpretable.

**Default:** Continuous settings; `ConditionID` remains an optional experimental feature.

---

## 3. Physics-Informed Features

Three engineered features capture key turbomachinery degradation trends:

| Feature               | Formula              | Meaning                                  |
|----------------------|----------------------|------------------------------------------|
| `Effizienz_HPC_Proxy`| Sensor12 / Sensor7   | Proxy for high-pressure compressor loss  |
| `EGT_Drift`          | Sensor17 − EGT_base  | Exhaust gas temperature drift vs. healthy|
| `Fan_HPC_Ratio`      | Sensor2 / Sensor3    | Fan vs. HPC loading shift                |

They:

- Reduce noise in raw sensors
- Align model behavior with physical intuition
- Stabilize training, especially across FD002/FD004

---

## 4. Model Families

### 4.1 Current Core: EOL-Full-LSTM (per FD)

**Idea:**  
Train on **all windows along the trajectory**, but always predict the **RUL at the last cycle of the window** (EOL-centric label). This is closer to how an operator would use the model “today”.

Key characteristics:

- Inputs: past 30 cycles (standardized sensors + settings + physics features)
- Label: capped RUL at window end (0–125)
- Split: **engine-based** (no leakage across windows)
- Loss: MSE on RUL
- Evaluation:
  - pointwise metrics over all validation windows
  - EOL metrics over **one sample per engine** (last cycle)  
  - NASA PHM08 on EOL predictions

These models form the **main baselines** reported in section 0.1.

---

### 4.2 Global FD001–FD004 EOL-Full-LSTM

Single LSTM over all four FD sets:

- Same windowing and feature set
- Shared model & scaler
- Learned across different fault modes and operating envelopes

Result:

- More **compact** and unified, but
- **Weaker** than per-FD models on EOL metrics  
  → especially important for safety-critical deployment.

This is still a useful **reference point** for research on:

- Domain adaptation
- Multi-condition encoders
- Condition-aware attention mechanisms

---

### 4.3 Legacy Baselines: Classical Global LSTM & MC-Dropout

From earlier experiments (pre-EOL-Full pipeline):

- **Local per-FD LSTM (V2–V4)**  
  - Asymmetric loss to penalize over-optimistic RUL  
  - RUL clamping  
  - FD001 RMSE ≈ 17–18 cycles

- **Global LSTM (V5)**  
  - Single model across FD001–FD004  
  - FD001 RMSE ≈ 14–15 cycles, strong performance especially on FD002/FD004  
  - Acts as a first global physics-informed baseline

- **MC-Dropout LSTM on FD001**  
  - Inference with dropout active (50 stochastic passes)  
  - RMSE improved to ≈ 12.9 cycles in that setup  
  - Predictive std correlates well with absolute error  
  - Good demonstration of **uncertainty as a risk indicator** (“double-check this engine”).

These runs are **kept as historical references** and as a bridge to the newer EOL-Full pipeline.

---

### 4.4 Seq2Seq World Models (V6–V8) & EOL Regressor

#### V6 – Pure Trajectory World Model

- Encoder–Decoder LSTM:
  - Encoder: past 30 cycles
  - Decoder: future 20-step RUL trajectory
- Loss: MSE over entire predicted trajectory
- Evaluated by:
  - trajectory-level RMSE
  - EOL NASA score based on the last step of the predicted trajectory

Outcome (global test set):

- Very low **NASA mean** (~0.33 per engine)
- Small mean EOL error (~1.9 cycles)
- But **not optimized** for classical per-FD EOL RMSE

#### V7/V8 – Multi-Task World Model (Trajectory + EOL Head)

Adds:

- EOL head on encoder hidden state
- Joint loss (trajectory + EOL)
- NASA-based evaluation and bias monitoring

Characteristics:

- FD001 EOL RMSE still ~41–42 cycles
- NASA significantly better than naive LSTM baselines
- Stable calibration, but EOL accuracy limited by encoder representation

#### Two-Stage EOL Regressor

To push NASA further:

1. Freeze the World Model encoder  
2. Train a separate MLP on encoder outputs, **using only EOL windows**  
3. Evaluate directly on EOL metrics

FD001 (historical run):

- RMSE ≈ 40.9 cycles  
- NASA mean ≈ 220  
- Bias clearly improved w.r.t. the raw multi-task head

👉 **Interpretation:**  
World Models excel as **simulation & safety layers** (trajectory, NASA), while sharp EOL prediction is handled better by dedicated EOL-Full-LSTM models.

---

## 5. Benchmark vs. Literature (FD001)

### 5.1 Internal vs. Literature RMSE (FD001, EOL-style)

| Model / Method                        | FD001 RMSE | NASA (if reported) | Comment                                  |
|--------------------------------------|-----------:|--------------------:|------------------------------------------|
| **EOL-Full-LSTM (this repo, FD001)** | **16.36**  | ~282                | Current physics-informed baseline        |
| Legacy local LSTM (V4)               | ~17–18     | ~300–400            | Older baseline, clamped RUL              |
| Global LSTM (V5, FD001 only)         | ~14–15     | ~400–450            | Early cross-FD experiment                |
| World Model V7/V8                    | ~41–42     | 280–320             | Safety-focused, trajectory-aware         |
| World Model + EOL Regressor          | ~41        | ~220                | Best NASA among World Models             |

**Recent literature (typical FD001 EOL RMSE):**

| Architecture / Paper                            | FD001 RMSE | NASA / Score | Summary                                           |
|-------------------------------------------------|-----------:|-------------:|---------------------------------------------------|
| Reweighted Transformer (Kim 2024)               | **≈11.3**  | ≈192         | Transformer + sample reweighting                 |
| GCU-Transformer variants                        | 11–12      | –            | Gated conv front-end + Transformer encoder       |
| Health-aware / CTVAE-style Transformers         | ≈12.4      | –            | Latent health state + attention                  |
| CNN-based RUL models (e.g., Li 2018, follow-ups)| 12–15      | –            | 1D CNN front-end, often better than plain LSTM   |
| Enhanced LSTM with improved labeling            | ≈13.2      | ≈230         | Better RUL target generation & loss shaping      |
| XGBoost + generative RUL labels                 | ≈12.9      | –            | Strong classic ML with heavy feature engineering |

👉 **Positioning of this repo:**  

- **Not** trying to chase absolute SOTA RMSE on FD001  
- Instead focusing on:
  - clean, **physics-informed LSTM baselines**
  - **EOL-consistent evaluation**
  - **World-Model** concepts (trajectory, safety, uncertainty) for engineering applications

---

## 6. Lessons Learned & Conceptual Evolution

### 6.1 From “Toy LSTM” to Engineering-Grade Baseline

**Early approach:**

- Single or per-FD LSTM on clamped RUL labels  
- Sometimes mixed or sample-based splits (risk of leakage)  
- Evaluation often **pointwise-only**, not strictly EOL-based  
- Result: nice-looking RMSE, but not always comparable to literature / PHM practice

**Current approach:**

- **Engine-based splits** only → no shared windows between train and val  
- Clear separation between:
  - **Pointwise metrics** (all windows)  
  - **EOL metrics** (one sample per engine, last cycle)  
- Explicit **RUL capping** to 125 and consistent labeling  
- Per-FD and global variants, fully reproducible

### 6.2 Why the Early Trajectory Approaches Misbehaved

When we initially tried trajectory-based / multi-step targets:

- Window generation, target alignment, and splitting were **tightly coupled**
- A small mistake (e.g. splitting by rows instead of engines) easily created **hidden leakage**
- The model “saw” future parts of the same engine during training → **unrealistic validation scores**

The turning points:

1. **Engine-based splitting utilities**  
   (`create_full_dataloaders`, `build_full_eol_sequences_from_df`)  
   → enforce that **no engine** appears in both train and val.

2. **EOL-centric labeling**  
   Training on windows but predicting the RUL at the **end of the window**  
   → matches the operational use case and the NASA evaluation.

3. **Separate evaluation paths** for:
   - trajectory prediction (World Models)
   - EOL prediction (LSTM baselines, EOL-Full-LSTM)

### 6.3 World Model vs. EOL-Full-LSTM – Conceptual Roles

- **EOL-Full-LSTM:**  
  - “One-shot” RUL estimate from the last 30 cycles  
  - Good RMSE & NASA when evaluated EOL-style  
  - Directly comparable to most literature baselines

- **World Model:**  
  - Learns a latent **dynamics model** of degradation  
  - Can simulate multiple future steps and scenario variations  
  - Better suited for:
    - what-if analysis
    - safety envelopes
    - integration with digital twins / physics models

👉 In a real plant, you would **combine** them:

- World Model for **trajectory & risk-aware simulation**  
- EOL-Full-LSTM (or a refined EOL head) for **sharp decision thresholds** (e.g. maintenance scheduling).

---

## 7. Project Structure

```text
AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  └─ raw/                        # raw C-MAPSS text files
├─ results/
│  ├─ fd001/                      # per-FD EOL-Full-LSTM runs
│  ├─ fd002/
│  ├─ fd003/
│  ├─ fd004/
│  ├─ eol_full_lstm/              # global FD001–FD004 EOL-Full-LSTM
│  └─ world_model/                # V6–V8 World Model checkpoints & diagnostics
├─ notebooks/
│  ├─ 1_fd001_exploration.ipynb
│  ├─ 1a_fd002_exploration.ipynb
│  ├─ 2_training_single_datasets.ipynb   # per-FD EOL-Full-LSTM
│  ├─ 3_evaluation_global_model.ipynb
│  ├─ 4_global_lstm.ipynb                # legacy global LSTM baselines
│  ├─ 5_training_uncertainty_dropout.ipynb
│  └─ 6_world_model_training.ipynb       # V6–V8 World Models
└─ src/
   ├─ config.py
   ├─ data_loading.py
   ├─ additional_features.py
   ├─ training.py                         # legacy LSTM training utils
   ├─ world_model_training.py             # World Model + EOL-Full-LSTM pipeline
   ├─ models/
   │  ├─ lstm_rul.py                      # LSTM baselines
   │  ├─ lstm_rul_mcdo.py                 # MC-Dropout model
   │  └─ world_model.py                   # WorldModelEncoderDecoder
   ├─ loss.py
   ├─ uncertainty.py
   └─ eval_utils.py / train_utils.py

8. How to Run
8.1 Environment

conda create -n turbine_ai python=3.10
conda activate turbine_ai

pip install torch torchvision torchaudio \
            pandas matplotlib scikit-learn jupyter ipykernel

8.2 Data

Download C-MAPSS into ./data/raw:

data/raw/
├─ train_FD001.txt   ├─ test_FD001.txt   ├─ RUL_FD001.txt
├─ train_FD002.txt   ├─ test_FD002.txt   ├─ RUL_FD002.txt
├─ train_FD003.txt   ├─ test_FD003.txt   ├─ RUL_FD003.txt
├─ train_FD004.txt   ├─ test_FD004.txt   └─ RUL_FD004.txt

8.3 Train per-FD EOL-Full-LSTM baselines

    Open notebooks/2_training_single_datasets.ipynb

    Select the turbine_ai kernel

    Run all cells to:

        Build EOL-Full sequences per FD

        Train one LSTM per dataset

        Save checkpoints & metrics into results/fd00X/

8.4 Train global FD001–FD004 EOL-Full-LSTM

    Either via a dedicated notebook (e.g. 3_evaluation_global_model.ipynb)
    or via a script calling train_eol_full_lstm with multi_fd=True.

    Outputs go to results/eol_full_lstm/.

8.5 Train World Models

    Open notebooks/6_world_model_training.ipynb

    Run to:

        Build seq2seq datasets

        Train WorldModelEncoderDecoder

        Save the best World Model to results/world_model/

        Evaluate:

            trajectory-level RMSE

            EOL NASA score

8.6 Uncertainty Experiments (MC-Dropout, legacy)

    Use notebooks/5_training_uncertainty_dropout.ipynb
    and src/uncertainty.py:

        enable_mc_dropout(model)

        mc_dropout_predict(model, X_tensor, n_samples=50)

9. Roadmap

This repository is part of a broader “Mechanical Engineer AI Assistant” vision.

Planned next steps:

    Better encoders for FD002/FD004

        CNN + LSTM + attention front-ends

        Lightweight Transformer encoders (e.g. GCU-style)

    Physics-informed World Models

        Multi-step prediction for sensors + RUL

        Integration with physics constraints / digital twins

        Residual modeling (sensor vs. “healthy” reference)

    Uncertainty & calibration

        Extend MC-Dropout / deep ensembles to World Models

        Calibration plots and risk-aware decision rules

    Deployment

        Simple REST API or Streamlit dashboard for:

            Online RUL monitoring per engine

            Physics feature visualization

            Scenario analysis with the World Model

10. Contact & License

Author: Dr.-Ing. Robert Kunte
LinkedIn: https://www.linkedin.com/in/robertkunte/


License: MIT

If you work on turbomachinery, PHM, or physics-informed ML and want to collaborate, feel free to reach out!