 🤖 AI-Based Turbine RUL Monitor: Physics-Informed LSTM Predictor (FD001)

## 💡 Project Overview & Problem Statement

This project implements a **Prognostics and Health Management (PHM)** pipeline to predict the **Remaining Useful Life (RUL)** of gas turbine engines.

Unplanned failures in critical energy assets are extremely costly – they cause forced outages, expensive emergency repairs, and revenue loss. The goal here is to:

> **Predict the remaining life of a turbine early and reliably, so that maintenance can be planned instead of reacting to failures.**

This repository focuses on:

- The **NASA C-MAPSS FD001** dataset.
- A **physics-informed LSTM model** implemented in PyTorch.
- An **evaluation workflow** that looks beyond a single RMSE number: per-unit errors, bias, and correlations to physically meaningful features.

---

## 📊 Dataset & Problem Setup

- **Dataset:** NASA C-MAPSS – subset **FD001**  
  Each engine is simulated until failure under a single operating condition.  
  Data contains:
  - `UnitNumber` (engine ID)  
  - `TimeInCycles`  
  - 3 operating settings  
  - 21 sensor signals

- **Objective:** Predict the **RUL at the last observed cycle** for each engine in the test set:
  - Ground truth RUL labels come from `RUL_FD001.txt`.
  - RUL is **clamped** to a maximum value (see below).

---

## 🔬 Preprocessing & Physics-Informed Features

### 1. Standard Preprocessing

1. **Train/Test split**

   - Training: `train_FD001.txt`
   - Test: `test_FD001.txt`
   - Labels: `RUL_FD001.txt`

2. **RUL computation**

   For each engine:

   ```python
   RUL = MaxTime - TimeInCycles
   MAX_RUL = 125
   RUL = np.minimum(RUL, MAX_RUL)

Motivation: early life is often “flat” (no observable degradation). We train the model to focus on the critical wear-out phase rather than predicting very long, uncertain horizons.

    Feature selection

        Constant or uninformative sensors (e.g., Sensor1, Sensor5, Sensor10) and some settings are removed.

        The remaining sensors + new physics features form the input feature vector.

    Scaling

        All continuous features are scaled with MinMaxScaler to [0, 1].

        The scaler is fitted on the training data and reused on the test data.

    Sequence generation

        For each engine, sequences of length SEQUENCE_LENGTH = 30 cycles are built.

        For training, multiple sequences per engine are used (sliding window).

        For testing, only the last 30 cycles of each engine are used to predict its final RUL.

2. Physics-Informed Features

To make the model more interpretable and robust, three physically motivated features are added on top of the raw sensor data:

    HPC Efficiency Proxy (Effizienz_HPC_Proxy)

    Based on total pressure and total temperature:

Effizienz_HPC_Proxy = Sensor12 / Sensor7

Interpreted as a crude proxy for compressor efficiency / thermodynamic state.

Exhaust Gas Temperature Drift (EGT_Drift)

    Sensor17 is the Exhaust Gas Temperature (EGT).

    For each engine, a baseline is computed as the mean EGT over the first 10 cycles:

    EGT_base = mean(Sensor17 for TimeInCycles <= 10, per UnitNumber)
    EGT_Drift = Sensor17 - EGT_base

    Captures the temperature increase relative to the healthy state, which correlates strongly with degradation.

Fan–HPC Degradation Ratio (Fan_HPC_Ratio)

Ratio of fan speed to high-pressure compressor speed:

    Fan_HPC_Ratio = Sensor2 / Sensor3

    Changes in this ratio may indicate aerodynamic or mechanical degradation.

📌 Correlation insights (test set):

    EGT_Drift vs True RUL: strong negative correlation (~ −0.65)

    Fan_HPC_Ratio vs True RUL: strong positive correlation (~ +0.61)

    The trained model’s predictions strongly correlate with these features as well, indicating that the model is actually using the physics-informed information.

🧠 Model Architecture & Training
1. Model

The current best model is:

    Architecture:

        2-layer LSTM with batch_first=True

        Hidden size: 50–64 units (depending on experiment)

        Final Fully Connected Layer: hidden_size → 1 for scalar RUL prediction

    Input:

        Sequences of shape (batch_size, 30, n_features)
        where n_features ≈ remaining sensors + 3 physics features.

    Output:

        Scalar RUL (clamped to [0, MAX_RUL] during evaluation).

2. Loss Function

Two loss variants were tested:

    Baseline:

        nn.MSELoss() (Mean Squared Error)

    Current best: Asymmetric, RUL-weighted loss

Overestimation is penalized more than underestimation (safety),
and low-RUL samples (close to failure) get higher weight:

def rul_asymmetric_weighted_loss(pred, target,
                                 over_factor=2.0,
                                 min_rul_weight=1.0,
                                 max_rul_weight=0.3):
    """
    Custom loss for RUL:
    - Overestimation (pred > target) is penalized stronger than underestimation.
    - Low RUL values are weighted higher than large RUL values.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    error = pred - target
    over  = torch.clamp(error, min=0.0)      # overestimation
    under = torch.clamp(-error, min=0.0)     # underestimation

    # Asymmetric penalty (MSE-like but harsher on overestimation)
    base_loss = over_factor * over**2 + under**2

    # Higher weights for small RUL (end-of-life is more critical)
    t_norm = target / (target.max() + 1e-6)  # [0, 1]
    weights = max_rul_weight + (min_rul_weight - max_rul_weight) * (1.0 - t_norm)

    weighted_loss = weights * base_loss
    return weighted_loss.mean()

    Optimizer: Adam with lr = 1e-3

    Epochs: 25

    Batching: standard mini-batches from DataLoader.

📈 Evaluation Setup

Model evaluation is done in two stages:
1. Prediction file generation (1_data_analysis.ipynb)

The notebook trains the LSTM model and then:

    Loads the test set (test_FD001.txt)

    Builds the last 30-cycle sequence for each engine

    Predicts RUL for all engines

    Clamps predictions to MAX_RUL

    Saves a CSV:

results/fd001_predictions_physical_features.csv

with columns:

    UnitNumber

    TimeInCycles (last observed cycle)

    TrueRUL

    PredRUL

    Effizienz_HPC_Proxy

    EGT_Drift

    Fan_HPC_Ratio

2. Detailed evaluation & analysis (2_model_evaluation.ipynb)

This notebook:

    Computes global metrics:

        RMSE, MAE, Bias (Pred − True)

    Groups engines by RUL bins, e.g. 0–25, 25–50, 50–100, 100–200 cycles:

        RMSE, MAE, Bias per bin

    Computes per-unit error statistics:

        RMSE / MAE / Bias for each UnitNumber

        Identifies top-10 worst engines (outliers)

    Computes a correlation matrix between:

        TrueRUL, PredRUL

        Effizienz_HPC_Proxy, EGT_Drift, Fan_HPC_Ratio

    Provides scatterplots:

        TrueRUL vs PredRUL

        Residuals (Pred − True) vs TrueRUL

This gives a much richer picture than a single scalar metric.
🎯 Key Results
Baseline vs Physics-Informed vs Improved Loss

On C-MAPSS FD001 test data:
Configuration	RMSE (cycles)	Comments
Simple LSTM, no physics features, MSE loss	~44.7	Initial baseline
LSTM + physics features (EGT_Drift, Fan/HPC, HPC proxy)	~19.1	Large gain from physics-informed features
LSTM + physics features + asymmetric RUL-weighted loss	15.98	Current best model (this repo’s main result)

Additional observations (best model):

    Correlation between TrueRUL and PredRUL ≈ 0.93

    Physics features are strongly used:

        EGT_Drift and Fan_HPC_Ratio both show strong correlation to predictions.

    Worst-case units still show errors on the order of 30–48 cycles,
    but global average error is significantly reduced compared to the initial version.

    In practical terms: the model can anticipate the end-of-life of engines in FD001 with an average error of about 16 cycles, grounded in interpretable physics features.

🧾 Changelog / Version History

v2 – Physics-Informed RUL Model (current)

    Added physics-inspired features:

        EGT_Drift (Exhaust Gas Temperature Drift)

        Fan_HPC_Ratio (Fan/HPC speed ratio)

        Effizienz_HPC_Proxy (pressure/temperature proxy)

    Introduced RUL clamping at 125 cycles to focus on the degradation regime.

    Implemented asymmetric, RUL-weighted loss:

        Overestimation penalized more than underestimation.

        Higher weight for low-RUL samples (near end-of-life).

    Added a dedicated evaluation notebook (2_model_evaluation.ipynb) to:

        analyze per-unit errors,

        compute RUL-bin metrics,

        and visualize feature correlations.

    Achieved RMSE ≈ 15.98 cycles on FD001 (previously ~44.7).

v1 – Initial Baseline Model

    Basic 2-layer LSTM model in PyTorch.

    Used raw sensor data from C-MAPSS FD001 with minimal feature selection.

    Standard MSE loss, trained with Adam.

    Achieved RMSE ≈ 44.7 cycles on FD001.

    Provided initial data loading, preprocessing, and training notebook (1_data_analysis.ipynb).

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

Download the C-MAPSS FD001 files into ./data:

    train_FD001.txt

    test_FD001.txt

    RUL_FD001.txt

Resulting structure:

AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  ├─ train_FD001.txt
│  ├─ test_FD001.txt
│  └─ RUL_FD001.txt
├─ results/
│  └─ fd001_predictions_physical_features.csv  # created by notebook
├─ 1_data_analysis.ipynb
├─ 2_model_evaluation.ipynb
└─ ...

Execution

    Training & prediction export

        Open 1_data_analysis.ipynb in VS Code / Cursor / Jupyter.

        Select the turbine_ai kernel.

        Run all cells to:

            preprocess data,

            train the LSTM model,

            generate results/fd001_predictions_physical_features.csv.

    Evaluation & analysis

        Open 2_model_evaluation.ipynb.

        Run all cells to:

            compute RMSE/MAE/Bias,

            inspect per-RUL-bin metrics,

            visualize worst engines & correlations.
    
## 🧪 Additional Experiment: LSTM with Temporal Attention

As an extension to the baseline LSTM, an **attention mechanism over the time dimension** was tested:

- Architecture: same input features and training setup as the final model,  
  but with an LSTM followed by a **simple additive attention layer** over all time steps.
- Goal: allow the model to focus on the most informative parts of the 30-cycle history instead of only using the last hidden state.

### Results

On FD001, this **attention-based model (V3)** did **not** outperform the baseline LSTM (V2):

| Version | Model                 | Global RMSE (cycles) | MAE (cycles) | Bias (pred–true) | NASA PHM08 Score |
|--------|------------------------|----------------------:|-------------:|-----------------:|-----------------:|
| V2     | LSTM (no attention)    | ~13.4                 | ~9.5         | +2.1             | ~359             |
| V3     | LSTM + time attention  | ~15.4                 | ~10.7        | +3.8             | ~560             |

Observations:

- The attention model showed **higher RMSE and MAE** and a **larger positive bias** (more optimistic predictions).
- The **NASA PHM08 score increased significantly**, indicating a higher penalty, especially for late (over-optimistic) predictions.
- Outlier units (engines with the largest per-unit RMSE) were **not improved**, and in some cases became worse.

**Conclusion:**  
In this configuration, the added attention did not provide a benefit and even slightly degraded performance and practical risk metrics. For this reason, the repository uses the **plain LSTM with physics-informed features and asymmetric RUL-weighted loss (V2)** as the **final reference model**, while the attention-based variant remains included as an optional experimental architecture.

🔭 Future Work

This project is part of a broader “Mechanical Engineer Assistant” vision and can be extended in several directions:

    Seq2Seq / World Models

        Predict full multi-step future sensor trajectories, not just final RUL.

        Use this for scenario simulation and “what-if” analysis.

    Uncertainty Quantification

        Monte Carlo Dropout / Deep Ensembles for predictive intervals.

        Risk-aware decision-making based on confidence levels.

    NASA Scoring Metric (done)

        Implement the official asymmetric NASA RUL score
        (heavier penalties for late predictions) to evaluate practical risk.

    Deployment

        Wrap the model into a REST API or a simple Streamlit dashboard for:

            live RUL monitoring per asset,

            feature & trend visualizations for engineers.

    Multi-subset Extension

        Extend from FD001 to FD002–FD004 with varying operating conditions and fault modes.

        Compare generalization across subsets.

📞 Contact & License

    Author: Dr.-Ing. Robert Kunte

    LinkedIn: https://www.linkedin.com/in/robertkunte/

    License: MIT License

If you’re working on turbomachinery, PHM, or physics-informed ML and want to exchange ideas, feel free to reach out!