# 🤖 AI-Based Turbine RUL Monitor  
## Physics-Informed LSTM for C-MAPSS FD001–FD004

## 💡 Project Overview & Problem Statement

This project implements a **Prognostics and Health Management (PHM)** pipeline to predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA **C-MAPSS** datasets.

Unplanned failures in critical energy assets are extremely costly – they cause forced outages, expensive emergency repairs, and revenue loss. The goal here is to:

> **Predict the remaining life of an engine early and reliably, so that maintenance can be planned instead of reacting to failures.**

This repository currently focuses on:

- The **NASA C-MAPSS subsets FD001–FD004**
- A **physics-informed LSTM model** implemented in PyTorch
- A **modular training & evaluation pipeline** with:
  - Per-unit error analysis
  - NASA PHM08 scoring
  - Correlations with physically meaningful features

---

## 📊 Datasets & Problem Setup

### C-MAPSS subsets

The project supports all four standard C-MAPSS subsets:

| Subset | Train Engines | Test Engines | Conditions | Fault Modes                    |
|--------|---------------|-------------|------------|--------------------------------|
| FD001  | 100           | 100         | 1          | 1 (HPC degradation)            |
| FD002  | 260           | 259         | 6          | 1 (HPC degradation)            |
| FD003  | 100           | 100         | 1          | 2 (HPC + Fan degradation)      |
| FD004  | 248           | 249         | 6          | 2 (HPC + Fan degradation)      |

Each dataset includes:

- `UnitNumber` (engine ID)  
- `TimeInCycles` (discrete time step)  
- 3 operating settings (`Setting1`, `Setting2`, `Setting3`)  
- 21 sensor signals (`Sensor1` … `Sensor21`)

### Objective

For each engine in the test set (FD00X):

- Predict the **RUL at the last observed cycle**
- Ground truth labels come from `RUL_FD00X.txt`
- RUL is **clamped** to a maximum value to focus on the degradation regime (see below)

---

## 🔬 Preprocessing & Physics-Informed Features

### 1. Standard Preprocessing

For each subset FD00X:

1. **Train/Test loading**

   - Training: `train_FD00X.txt`  
   - Test: `test_FD00X.txt`  
   - Labels: `RUL_FD00X.txt`

2. **RUL computation (train set)**

   For each engine:

   ```python
   RUL = MaxTime - TimeInCycles
   MAX_RUL = 125
   RUL = np.minimum(RUL, MAX_RUL)

Motivation:
Early life often shows no visible degradation – predicting arbitrarily large RUL values is not useful. Clamping RUL forces the model to focus on the critical wear-out phase.

    Feature selection

        Remove constant or uninformative sensors (e.g. Sensor1, Sensor5, Sensor10)

        For FD001/FD003 (single condition), Setting3 is also dropped

        MaxTime is dropped after computing RUL

        TimeInCycles is kept in the DataFrame (for sorting), but not used as a model input feature

    Scaling

        All continuous input features are scaled to [0, 1] using MinMaxScaler

        The scaler is fitted on the training data and reused for the test data

    Sequence generation

        Sequence length: SEQUENCE_LENGTH = 30 cycles

        Training:

            Sliding windows over each engine:

                For each engine, build all possible windows of length 30

                The RUL label is the RUL at the last time step of the window

        Testing:

            For each engine, build one sequence:

                If the engine has ≥ 30 time steps: the last 30 steps

                If shorter: pad the beginning by repeating the first row until length 30

    Final training input shape (per FD):

    X_train: (N_sequences, 30, n_features)
    y_train: (N_sequences,)

2. Physics-Informed Features

On top of the raw sensors, three physically motivated features are added:
a) HPC Efficiency Proxy (Effizienz_HPC_Proxy)

Based on total pressure and total temperature:

Effizienz_HPC_Proxy = Sensor12 / Sensor7

Interpretation:
A crude proxy for compressor efficiency / thermodynamic state at the HPC/HPT interface.
b) Exhaust Gas Temperature Drift (EGT_Drift)

    Sensor17 is the Exhaust Gas Temperature (EGT)

For each engine, we compute a baseline over the first cycles (healthy reference):

EGT_base = mean(Sensor17 for TimeInCycles <= 10, per UnitNumber)
EGT_Drift = Sensor17 - EGT_base

Interpretation:
Captures the temperature increase relative to the healthy state – strongly correlated with degradation.
c) Fan–HPC Degradation Ratio (Fan_HPC_Ratio)

Ratio of fan speed to high-pressure compressor speed:

Fan_HPC_Ratio = Sensor2 / Sensor3

Interpretation:
Changes in this ratio may indicate aerodynamic or mechanical degradation in the flow path.
Correlation Insights (FD001 test set)

    EGT_Drift vs TrueRUL: strong negative correlation (~ −0.65)

    Fan_HPC_Ratio vs TrueRUL: strong positive correlation (~ +0.61)

    PredRUL shows similarly strong correlations to these features

👉 The trained model is clearly using the physics-informed features, not just the raw sensors.
🧠 Model Architecture & Training
1. LSTM Model

The current reference model is a plain LSTM with physics-informed inputs:

    Architecture:

        2-layer LSTM with batch_first=True

        Hidden size: HIDDEN_SIZE = 50

        Final fully connected layer: hidden_size → 1 for scalar RUL prediction

    Input:

        Sequences of shape (batch_size, 30, n_features)

        n_features = selected sensors + 3 physics-informed features

    Output:

        Scalar RUL prediction per sequence

        During evaluation, predictions are clamped to [0, MAX_RUL]

2. Loss Function: Asymmetric, RUL-weighted

Two loss variants were tested:

    Baseline: nn.MSELoss()

    Current best: custom asymmetric, RUL-weighted loss

    Overestimation (pred > target) is penalized more strongly than underestimation (safety)

    Low-RUL samples (close to failure) get higher weight than high-RUL samples

def rul_asymmetric_weighted_loss(
    pred, target,
    over_factor=2.0,
    min_rul_weight=1.0,
    max_rul_weight=0.3
):
    """
    Custom loss for RUL:
    - Overestimation (pred > target) is penalized stronger than underestimation.
    - Low RUL values are weighted higher than large RUL values.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    error = pred - target
    over  = torch.clamp(error, min=0.0)   # overestimation
    under = torch.clamp(-error, min=0.0)  # underestimation

    # Asymmetric penalty (MSE-like but harsher on overestimation)
    base_loss = over_factor * over**2 + under**2

    # Higher weights for small RUL (end-of-life is more critical)
    t_norm = target / (target.max() + 1e-6)  # [0, 1]
    weights = max_rul_weight + (min_rul_weight - max_rul_weight) * (1.0 - t_norm)

    weighted_loss = weights * base_loss
    return weighted_loss.mean()

    Optimizer: Adam with lr = 1e-3

    Epochs: typically 25 for the final runs

    Batching: standard DataLoader mini-batches

📈 Evaluation Setup

The evaluation is done via a dedicated training & analysis pipeline:
1. Modular Training (per FD subset)

src/training.py provides:

res_fd001, metrics_fd001 = train_and_evaluate_fd(
    fd_id="FD001",
    model_class=LSTMRULPredictor,
    loss_fn=rul_asymmetric_weighted_loss,
)

For each subset FD00X, this function:

    Loads and preprocesses the data

    Adds physics-informed features

    Builds sequences

    Trains an LSTM model with the asymmetric loss

    Evaluates on the test set and computes:

        MSE, RMSE, MAE, Bias (Pred − True)

        NASA PHM08 score (asymmetric RUL metric)

    Saves:

        Predictions as CSV

        Model weights as .pt file

Typical output CSV (per FD):

results/FD001/FD001_predictions_local.csv
results/FD002/FD002_predictions_local.csv
...

with at least:

    UnitNumber

    TimeInCycles (last observed cycle)

    TrueRUL

    PredRUL

    Effizienz_HPC_Proxy

    EGT_Drift

    Fan_HPC_Ratio

    FD_ID

    ModelType ("local" for per-FD models)

2. Detailed Evaluation & Analysis

A dedicated evaluation notebook (e.g. notebooks/3_local_evaluation.ipynb) provides:

    Global metrics per dataset:

        RMSE, MAE, Bias, NASA score

    RUL-bin analysis:

        Bins like 0–25, 25–50, 50–100, 100–200 cycles

        RMSE, MAE, Bias per bin

    Per-unit error analysis:

        RMSE / MAE / Bias per UnitNumber

        Top-10 worst engines (outliers)

    Correlation matrix between:

        TrueRUL, PredRUL

        Effizienz_HPC_Proxy, EGT_Drift, Fan_HPC_Ratio

    Visualizations:

        TrueRUL vs PredRUL scatter plots

        Residuals (Pred − True) vs TrueRUL

        Feature vs RUL scatter plots

This yields a much richer picture than a single scalar metric.
🎯 Key Results
1. FD001-focused experiments (history)

On C-MAPSS FD001 test data:
Configuration	RMSE (cycles)	Comment
Simple LSTM, no physics features, MSE loss	~44.7	Initial baseline
LSTM + physics features (EGT_Drift, Fan/HPC, HPC proxy)	~19.1	Large gain from physics-informed features
LSTM + physics features + asymmetric RUL-weighted loss	~16.0	FD001-only “best run” (pre-modular version)

Additional observations for the FD001-only best run:

    Correlation between TrueRUL and PredRUL ≈ 0.93

    Physics features are clearly used:

        EGT_Drift and Fan_HPC_Ratio strongly correlated with predictions

    Worst-case units still show errors on the order of 30–48 cycles

    Global average error is significantly reduced compared to the simple baseline

2. Local models for all FD subsets (current modular V4)

Using the modular pipeline with physics-informed features and the asymmetric RUL-weighted loss, local LSTM models are trained per FD subset (same architecture, per-FD training).

Global metrics (test sets, clamped RUL, latest run):
FD	MSE	RMSE (cycles)	MAE (cycles)	Bias (pred−true)	NASA PHM08 Score
FD001	304.10	17.44	13.29	−7.88	342.57
FD002	371.03	19.26	15.03	−5.55	1713.73
FD003	205.97	14.35	10.54	−2.58	278.03
FD004	528.53	22.99	16.94	−9.40	2631.68

Interpretation:

    FD001: solid performance, in line with prior FD001-only experiments; NASA-score is low (~343)

    FD003: best-performing subset (RMSE ≈ 14.35, NASA ≈ 278), despite two fault modes – physics features help distinguish degradation patterns

    FD002 & FD004: multi-condition subsets (6 operating conditions) remain challenging:

        RMSE still reasonable, but NASA-score significantly higher

        Indicates some late / over-optimistic predictions in certain operating regimes

This provides a strong local baseline against which future global models (joint FD001–FD004 training with condition encodings) can be compared.
🧪 Additional Experiment: LSTM with Temporal Attention (FD001)

As an extension to the baseline LSTM, an attention mechanism over the time dimension was tested:

    Same input features and training setup as the physics-informed LSTM

    Architecture: LSTM followed by a simple additive attention layer over all time steps

    Goal: allow the model to focus on the most informative parts of the 30-cycle history instead of using only the last hidden state

Results (FD001)
Version	Model	RMSE (cycles)	MAE (cycles)	Bias (pred–true)	NASA PHM08 Score
V2	LSTM (no attention)	~13–17	~9–13	small	~340–360
V3	LSTM + time attention	higher	higher	more positive	significantly ↑

Observations:

    The attention model showed higher RMSE and MAE

    Larger positive bias → more optimistic predictions (late failures)

    NASA PHM08 score increased significantly

    Outlier units (engines with the largest per-unit RMSE) were not improved

Conclusion:
In this configuration, the added attention did not provide a benefit and even slightly degraded performance and risk metrics.
For this reason, the repository uses the plain LSTM with physics-informed features and asymmetric RUL-weighted loss as the reference model (V2/V4), while the attention-based variant remains an optional experimental architecture.
🧾 Changelog / Version History
v4 – Modular Multi-Subset Pipeline (current)

    Added support for all four C-MAPSS subsets FD001–FD004

    Introduced a modular Python package structure under src/:

        config.py – hyperparameters & dataset definitions

        data_loading.py – loading & RUL computation per FD

        additional_features.py – physics-informed features

        training.py – generic training & evaluation per FD

        model.py, losses.py – model and custom loss (optional)

    Implemented per-FD local LSTM models with:

        Physics-informed features

        Asymmetric RUL-weighted loss

        NASA PHM08 scoring

    Produced a consistent metrics table for FD001–FD004 (see above)

v3 – Temporal Attention Experiment (FD001)

    Added a simple time attention mechanism on top of the LSTM

    Experimentally evaluated on FD001

    Result: no improvement, sometimes worse RMSE / NASA

    Kept as an experimental architecture, not as the default

v2 – Physics-Informed RUL Model (FD001)

    Added physics-inspired features:

        EGT_Drift (Exhaust Gas Temperature Drift)

        Fan_HPC_Ratio (Fan/HPC speed ratio)

        Effizienz_HPC_Proxy (pressure/temperature proxy)

    Introduced RUL clamping at 125 cycles to focus on degradation regime

    Implemented asymmetric, RUL-weighted loss:

        Overestimation penalized more than underestimation

        Higher weight for low-RUL samples (near end-of-life)

    Added evaluation notebook to:

        analyze per-unit errors

        compute RUL-bin metrics

        visualize feature correlations

    Achieved RMSE ≈ 16 cycles on FD001 (previously ~44.7)

v1 – Initial Baseline Model (FD001)

    Basic 2-layer LSTM model in PyTorch

    Used raw sensor data from C-MAPSS FD001 with minimal feature selection

    Standard MSE loss, trained with Adam

    Achieved RMSE ≈ 44.7 cycles on FD001

    Provided initial data loading, preprocessing, and training notebook

🛠️ Reproduction & Setup
Prerequisites

    Python 3.10

    Recommended: Miniconda for environment management

Environment Setup

# Create and activate environment
conda create -n turbine_ai python=3.10
conda activate turbine_ai

# Install packages
pip install torch torchvision torchaudio pandas matplotlib scikit-learn jupyter ipykernel

Data

Download the C-MAPSS text files into ./data/raw:

data/raw/
├─ train_FD001.txt
├─ test_FD001.txt
├─ RUL_FD001.txt
├─ train_FD002.txt
├─ test_FD002.txt
├─ RUL_FD002.txt
├─ train_FD003.txt
├─ test_FD003.txt
├─ RUL_FD003.txt
├─ train_FD004.txt
├─ test_FD004.txt
└─ RUL_FD004.txt

Project structure (simplified):

AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  └─ raw/
├─ results/
│  ├─ FD001/
│  ├─ FD002/
│  ├─ FD003/
│  └─ FD004/
├─ notebooks/
│  ├─ 1_fd001_exploration.ipynb
│  ├─ 2_training_local_models.ipynb
│  └─ 3_local_evaluation.ipynb
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ data_loading.py
   ├─ additional_features.py
   ├─ training.py
   ├─ model.py          # optional, if you move LSTMRULPredictor here
   └─ losses.py         # optional, for custom loss

Execution
A) Train & export predictions (local models FD001–FD004)

    Open notebooks/2_training_local_models.ipynb

    Select the turbine_ai kernel

    Run all cells to:

        Train one model per FD subset

        Save per-FD predictions and model weights under results/FD00X/

B) Evaluation & analysis

    Open notebooks/3_local_evaluation.ipynb

    Run all cells to:

        Load the per-FD prediction files

        Compute metrics (RMSE / MAE / Bias / NASA)

        Inspect RUL-bin metrics

        Visualize worst engines & correlations with physics-informed features

🔭 Future Work

This project is part of a broader “Mechanical Engineer Assistant” vision and can be extended in several directions:

    Seq2Seq / World Models

        Predict full multi-step future sensor trajectories, not just final RUL

        Use this for scenario simulation and “what-if” analysis

    Uncertainty Quantification

        Monte Carlo Dropout / Deep Ensembles for predictive intervals

        Risk-aware decision-making based on confidence levels

    Global Model Across FD001–FD004

        Train a single LSTM on all subsets jointly

        Encode operating conditions and FD-ID (e.g. via embeddings)

        Compare global vs. local models per FD

    Deployment

        Wrap the model into a REST API or a Streamlit dashboard for:

            live RUL monitoring per asset

            feature & trend visualizations for engineers

📞 Contact & License

    Author: Dr.-Ing. Robert Kunte

    LinkedIn: https://www.linkedin.com/in/robertkunte/

    License: MIT License

If you’re working on turbomachinery, PHM, or physics-informed ML and want to exchange ideas, feel free to reach out!