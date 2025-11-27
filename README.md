# 🤖 AI-Based Turbine RUL Monitor  
## Physics-Informed LSTM for C-MAPSS FD001–FD004

## 💡 Project Overview

This repository implements a **Prognostics and Health Management (PHM)** pipeline to predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA **C-MAPSS** datasets (FD001–FD004).

Unplanned failures in gas turbines and similar assets are extremely costly. The goal of this project is:

> **Predict the remaining life of an engine early and reliably so that maintenance can be planned instead of reacting to failures.**

The project currently provides:

- A **modular PyTorch pipeline** for all four C-MAPSS subsets
- **Physics-informed LSTM models** (local per FD + global across FD001–FD004)
- **Risk-aware RUL loss** (asymmetric, RUL-weighted)
- **Uncertainty estimation** via Monte Carlo Dropout
- Detailed evaluation:
  - RMSE / MAE / Bias / NASA PHM08 Score
  - Per-FD and global metrics
  - Correlations with physically meaningful features

---

## 📊 Datasets & Objective

### C-MAPSS subsets

| Subset | Train Engines | Test Engines | Conditions | Fault Modes               |
|--------|---------------|-------------|------------|---------------------------|
| FD001  | 100           | 100         | 1          | 1 (HPC)                   |
| FD002  | 260           | 259         | 6          | 1 (HPC)                   |
| FD003  | 100           | 100         | 1          | 2 (HPC + Fan)             |
| FD004  | 248           | 249         | 6          | 2 (HPC + Fan)             |

Each record contains:

- `UnitNumber`, `TimeInCycles`
- 3 operating settings: `Setting1`, `Setting2`, `Setting3`
- 21 sensors: `Sensor1` … `Sensor21`

### Prediction Task

For each engine in the test set:

- Predict the **RUL at the last observed cycle**
- Ground truth is taken from `RUL_FD00X.txt`
- RUL is **clamped** to a maximum of `MAX_RUL = 125` cycles to focus on the degradation regime and avoid unbounded “early-life” RUL.

---

## 🔬 Preprocessing & Physics-Informed Features

### Standard Preprocessing

Per subset FD00X:

1. **Load data**

   - `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`

2. **RUL computation (train)**

   For each engine:

   ```python
   RUL = MaxTime - TimeInCycles
   MAX_RUL = 125
   RUL = np.minimum(RUL, MAX_RUL)

    Feature selection

        Drop uninformative / constant sensors (e.g. Sensor1, Sensor5, Sensor10)

        For FD001 / FD003 (single condition): drop Setting3

        MaxTime is only used to compute RUL and then removed

        TimeInCycles is kept for sorting, but not used as model input

    Scaling

        All continuous input features scaled to [0, 1] (MinMaxScaler)

        Scaler fitted on train data and reused for test data

        For the global model a single global scaler is fitted over all FDs

    Sequence generation

        Sequence length: SEQUENCE_LENGTH = 30

        Training:

            Sliding windows per engine

            Label = RUL at last step of the window

        Testing:

            If length ≥ 30: last 30 steps

            Otherwise: pad with first row to length 30

    Final shapes per FD:

        X_train: (N_sequences, 30, n_features)

        y_train: (N_sequences,)

⚙️ Physics-Informed Features

On top of selected sensors, three physically motivated features are added:

    HPC Efficiency Proxy – Effizienz_HPC_Proxy

Effizienz_HPC_Proxy = Sensor12 / Sensor7

Crude proxy for compressor efficiency / thermodynamic state at the HPC/HPT interface.

Exhaust Gas Temperature Drift – EGT_Drift

EGT_base (per Unit) = mean(Sensor17 over first 10 cycles)
EGT_Drift = Sensor17 - EGT_base

Captures the temperature rise relative to the healthy baseline; strongly correlated with degradation.

Fan–HPC Degradation Ratio – Fan_HPC_Ratio

    Fan_HPC_Ratio = Sensor2 / Sensor3

    Changes in this ratio may indicate aerodynamic or mechanical degradation in the flow path.

On FD001, for example:

    EGT_Drift vs True RUL: strong negative correlation

    Fan_HPC_Ratio vs True RUL: strong positive correlation

    Predicted RUL shows similar correlations → the model actually uses these physics-informed signals.

🧠 Model Architecture & Loss
LSTM-Based RUL Predictor

Reference model (local and global):

    Architecture

        2-layer LSTM, batch_first=True

        Hidden size: typically HIDDEN_SIZE = 50 (locals), 64 (global variant)

        Final fully connected layer: hidden_size → 1 for scalar RUL

    Input

        (batch_size, 30, n_features)

        n_features = selected sensors + settings + 3 physics features

    Output

        Scalar RUL prediction per sequence

        During evaluation predictions are clamped to [0, MAX_RUL]

Asymmetric, RUL-Weighted Loss

To reflect safety requirements, over-optimistic predictions (too large RUL) should be penalized more than conservative ones.

Key ideas:

    Overestimation (pred > target) gets a higher penalty

    Low RUL (near end-of-life) receives higher weight than large RUL

Implementation (simplified):

def rul_asymmetric_weighted_loss(
    pred, target,
    over_factor=2.0,
    min_rul_weight=1.0,
    max_rul_weight=0.3,
):
    pred = pred.view(-1)
    target = target.view(-1)

    error = pred - target
    over  = torch.clamp(error, min=0.0)   # overestimation
    under = torch.clamp(-error, min=0.0)  # underestimation

    base_loss = over_factor * over**2 + under**2

    t_norm = target / (target.max() + 1e-6)  # [0, 1]
    weights = max_rul_weight + (min_rul_weight - max_rul_weight) * (1.0 - t_norm)

    return (weights * base_loss).mean()

    Optimizer: Adam (lr = 1e-3)

    Epochs: typically ~25 for local models, similar for global runs

🌍 Operating Conditions: Continuous Settings vs. ConditionID

For the multi-condition subsets FD002 and FD004, settings are provided as:

    Setting1, Setting2, Setting3 (e.g. altitude, throttle, environment)

Two encodings were compared:

    Continuous settings (default)

        Use Setting1/2/3 directly as continuous input features

        Closest to physical reality (continuous operating envelope)

    Discrete ConditionID (7 clusters)

        Round (Setting1, Setting2, Setting3) and map triples to a small set of discrete IDs

        Model sees a single numeric ConditionID instead of three settings

Result:
RMSE / MAE are very similar, but continuous settings yield slightly better NASA scores and are physically more realistic.
Therefore, continuous settings are used as the default encoding, and ConditionID remains an optional experimental variant.
📊 Key Results
1. Local Per-FD Models (Physics-Informed LSTM)

Local LSTM models are trained separately per FD with physics-informed features and asymmetric RUL-loss.

Test metrics (latest modular run, clamped RUL):
FD	MSE	RMSE	MAE	Bias (pred−true)	NASA PHM08
FD001	304.10	17.44	13.29	−7.88	342.57
FD002	371.03	19.26	15.03	−5.55	1713.73
FD003	205.97	14.35	10.54	−2.58	278.03
FD004	528.53	22.99	16.94	−9.40	2631.68

Interpretation:

    FD001 & FD003: good performance; FD003 is the best subset (low RMSE, low NASA score)

    FD002 & FD004: multi-condition subsets remain more challenging; RMSE reasonable, but NASA scores higher

2. Global LSTM Model (FD001–FD004 Combined)

A single global LSTM is trained on all four subsets simultaneously:

    Shared feature scaler across FD001–FD004

    Inputs: sensor subset, 3 continuous settings, 3 physics features

    Same architecture (2-layer LSTM) and loss as local models

    Evaluation is still per FD + aggregated

Global model (baseline, without MC-Dropout uncertainty)

From a previous global run:
FD	Local RMSE	Global RMSE	Local NASA	Global NASA
FD001	17.44	14.84	342.6	447.4
FD002	19.26	17.33	1713.7	1395.6
FD003	14.35	15.21	278.0	428.5
FD004	22.99	18.08	2631.7	1622.1

    Strong gains on FD002 and FD004 (multi-condition, multi-fault)

    Slight degradation on FD003 (simpler subset where specialization helps)

    Overall global RMSE improves vs. naïve per-FD models

Conclusion:

    A single global physics-informed LSTM can learn universal degradation patterns and significantly improve performance on complex operating conditions (FD002 / FD004).

🌫️ Uncertainty Estimation with Dropout
3.1 MC-Dropout Model for FD001

For FD001, a dedicated architecture LSTMRUL_MCDO is used:

    Same LSTM structure, but with Dropout layers kept active at inference (MC-Dropout)

    A small helper in src/uncertainty.py:

        enable_mc_dropout(model): forces all Dropout layers into “train” mode

        mc_dropout_predict(model, X_tensor, n_samples=50):
        runs multiple stochastic forward passes and returns:

            mean prediction

            predictive standard deviation

Results (FD001):

    RMSE improves to ≈ 12.9 cycles, better than the standard local LSTM run (~17.4 in the modular setup, ~16–13 in earlier FD001-only experiments)

    Predictive standard deviation correlates strongly with the absolute error

    Engines with:

        Clear degradation → lower uncertainty

        Long periods of “flat” RUL (many cycles close to MAX_RUL = 125 due to clamping) → higher uncertainty

Interpretation:

    MC-Dropout successfully identifies where the model is less sure (e.g. under-informative sequences)

    Uncertainty can be used as a risk indicator for engineers (e.g. “double-check this asset”)

3.2 Global LSTM with Dropout (Regularization Variant)

A second global run was performed with a Dropout-enhanced LSTM (regularization, single deterministic prediction per sample).

Global Dropout Model – Per-FD Metrics:
FD	MSE	RMSE	MAE	Bias (pred−true)	NASA PHM08
FD001	251.97	15.87	12.06	−4.33	365.06
FD002	434.36	20.84	16.09	−11.52	1836.59
FD003	236.08	15.37	11.27	−2.97	447.60
FD004	463.59	21.53	15.48	−9.29	2662.79

Global metrics across all FDs:

    MSE: 390.77

    RMSE: 19.77

    MAE: 14.62

    Bias: −8.51 (overall slightly pessimistic)

    NASA PHM08: 5312.04

Observations:

    Compared to the best global model without dropout, RMSE is somewhat higher, especially on FD002/FD004.

    The global Dropout model shows a consistently negative Bias (pessimistic RUL), which is in line with the asymmetric loss that discourages over-optimistic predictions.

    This variant provides a useful regularized baseline and a starting point to extend MC-Dropout uncertainty from FD001 to the full global setup.

🧱 Project Structure

AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  └─ raw/                # raw C-MAPSS text files
├─ results/
│  ├─ FD001/              # per-FD local model outputs
│  ├─ FD002/
│  ├─ FD003/
│  ├─ FD004/
│  └─ global/             # global model predictions & weights
├─ notebooks/
│  ├─ 1_fd001_exploration.ipynb
│  ├─ 2_training_local_models.ipynb
│  ├─ 3_local_evaluation.ipynb
│  └─ 3_global_model_evaluation.ipynb
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ data_loading.py
   ├─ additional_features.py
   ├─ training.py
   ├─ models/
   │  ├─ lstm_rul.py
   │  └─ lstm_rul_mcdo.py
   └─ uncertainty.py

🚀 How to Run
1. Environment

conda create -n turbine_ai python=3.10
conda activate turbine_ai

pip install torch torchvision torchaudio pandas matplotlib scikit-learn jupyter ipykernel

2. Data

Download the C-MAPSS files into ./data/raw:

data/raw/
├─ train_FD001.txt   ├─ test_FD001.txt   ├─ RUL_FD001.txt
├─ train_FD002.txt   ├─ test_FD002.txt   ├─ RUL_FD002.txt
├─ train_FD003.txt   ├─ test_FD003.txt   ├─ RUL_FD003.txt
├─ train_FD004.txt   ├─ test_FD004.txt   └─ RUL_FD004.txt

3. Train Local Models (FD001–FD004)

    Open notebooks/2_training_local_models.ipynb

    Select the turbine_ai kernel

    Run all cells:

        Train one LSTM per FD

        Save predictions & weights under results/FD00X/

4. Evaluate & Analyze

    Open notebooks/3_local_evaluation.ipynb and 3_global_model_evaluation.ipynb

    Run all cells to:

        Load prediction CSVs

        Compute metrics (RMSE / MAE / Bias / NASA)

        Inspect RUL-bins, per-unit errors

        Visualize correlations and degradation behaviour

🔭 Roadmap / Future Work

This repository is part of a broader “Mechanical Engineer AI Assistant” vision.

Planned next steps:

    Seq2Seq / World Models

        Encoder–decoder architectures (LSTM/GRU) to predict full future sensor trajectories and/or RUL sequences

        Basis for “what-if” simulations and scenario analysis

    Hyperparameter Optimization (Optuna + PyTorch Lightning)

        Systematic tuning of:

            LSTM size, depth, dropout

            Loss weights (asymmetry, RUL vs. sensor)

            Sequence length and feature sets

    Extended Uncertainty Quantification

        Apply MC-Dropout and/or deep ensembles to the global model

        Add calibration plots and risk-aware decision rules

    Deployment

        Simple REST API or Streamlit dashboard for:

            Online RUL monitoring per asset

            Physics-feature plots and trend analysis for engineers

📞 Contact & License

    Author: Dr.-Ing. Robert Kunte

    LinkedIn: https://www.linkedin.com/in/robertkunte/

    License: MIT

If you are working on turbomachinery, PHM or physics-informed ML and would like to exchange ideas, feel free to reach out!