
# 🚀 AI-Based Turbine RUL Monitor – Physics-Informed World Models

<<<<<<< HEAD
### *Deep Learning for Turbofan Prognostics on NASA C-MAPSS (FD001–FD004)*
=======
This repository implements a physics-informed **Remaining Useful Life (RUL)** monitoring pipeline for turbofan engines based on the NASA **C-MAPSS FD001–FD004** datasets.  

The project has evolved in several stages:

- **Local physics-informed LSTM models (per FD)** with asymmetric, risk-aware RUL loss  
- A **single global LSTM** trained jointly on FD001–FD004  
- **Uncertainty estimation** via Monte Carlo Dropout  
- **V6 (new): a global Seq2Seq “World Model”** that predicts full future RUL trajectories

### High-Level Model Comparison (latest runs)

| Model Type                            | Scope                 | Error Metric (test)                               | NASA PHM08 (test)                          |
|--------------------------------------|-----------------------|---------------------------------------------------|--------------------------------------------|
| Local physics-informed LSTM          | FD001–FD004 (separate)| Per-FD RMSE ≈ 14–23 cycles                        | Sum over FDs ≈ **4.97×10³**                |
| Global physics-informed LSTM         | FD001–FD004 (joint)   | Overall global RMSE ≈ **16.4** cycles             | Sum over FDs ≈ **3.89×10³**                |
| **Global Seq2Seq World Model (V6)**  | FD001–FD004 (joint)   | End-of-life error (pred–true) ≈ **1.9** cycles*   | Mean NASA ≈ **0.33** per engine (653 units)|

\* World Model metric is computed at the **end-of-life horizon** of the predicted RUL trajectory (Seq2Seq rollout), not as a single-step RMSE like the LSTM predictors.

> **Key takeaway:**  
> The global Seq2Seq World Model learns a smooth, well-calibrated RUL trajectory across all four datasets, with very low end-of-life error and a strongly risk-sensitive NASA score. It complements the direct LSTM RUL predictors and prepares the ground for full physics-informed “world models” of turbomachinery.
>>>>>>> 03aaff4e0eee0e1cd30fd003ddabb32c8d0bf692

---

## 🔍 **Quick Benchmark Summary**

This repository implements and compares several classes of RUL prediction models:

### **Your results (FD001, EOL evaluation):**

| Model                           | RMSE     | NASA    | Notes                                          |
| ------------------------------- | -------- | ------- | ---------------------------------------------- |
| **Local LSTM (V4)**             | **17.4** | 342     | Strong classical baseline                      |
| **Global LSTM (V5)**            | ~14.8    | ~447    | Best RMSE among internal models                |
| **World Model V7/V8 (MT)**      | 41–42    | 280–320 | Great safety (NASA), weak RMSE                 |
| **World Model + EOL Regressor** | ~40.9    | 220     | Best NASA among World Models after calibration |

### **Recent literature (FD001, typical RMSE):**

| Architecture                                                        | FD001 RMSE | Sources |
| ------------------------------------------------------------------- | ---------- | ------- |
| **Transformer variants (GCU-Transformer, reweighted Transformers)** | **11–13**  |         |
| **CNN/LSTM hybrids, improved labeling**                             | 12–15      |         |
| **XGBoost + engineered RUL labels**                                 | ~12.9      |         |

**Interpretation:**

* Your **LSTM baselines** are competitive with older DL methods (≈15–20 RMSE).
* Your **World Models** excel at **trajectory prediction, uncertainty, and NASA**, but are not optimized for RMSE.
* Literature SOTA achieves **11–13 cycles RMSE**, usually via Transformers + convolution front-ends + sample reweighting.

This README explains how your work fits into this landscape and how it extends classical PHM approaches toward *trajectory-aware, physics-informed, safety-first World Models*.

---

# 📘 1. Overview

This repository implements a complete **Prognostics & Health Management (PHM)** workflow for turbofan engines using NASA C-MAPSS FD001–FD004.

The project bridges:

* **Classical deep-learning RUL prediction** (LSTM, global models, physics-informed features)
* **World-Model-based degradation forecasting** (seq2seq dynamics, safety metrics, uncertainty)
* **Multi-task learning** (trajectory + End-of-Life prediction)
* **NASA PHM08 exponential scoring** (overprediction-penalized)

The goal is to create an architecture that can:

1. Predict **sharp RUL** at End-of-Life (EOL)
2. Produce **20-step degradation trajectories**
3. Offer **uncertainty** and **safety-focused** outputs
4. Use **physics-informed features** for interpretability

---

# 📁 2. Repository Structure

```
src/
  models/
    lstm_baseline.py
    world_model.py
    eol_regressor.py
  data_loading.py
  additional_features.py
  world_model_training.py
  evaluation.py

notebooks/
  V6, V7, V8 experiments

results/
  checkpoints/
  metrics/
  plots/
```

---

# ⚙️ 3. Physics-Informed Features

To enhance interpretability and improve signal-to-noise ratio, three key degradation indicators were engineered:

| Feature                 | Formula                   | Meaning                       |
| ----------------------- | ------------------------- | ----------------------------- |
| **Effizienz_HPC_Proxy** | Sensor12 / Sensor7        | Proxy for HPC efficiency loss |
| **EGT_Drift**           | Sensor17 − EGT_base(unit) | Exhaust gas temperature drift |
| **Fan_HPC_Ratio**       | Sensor2 / Sensor3         | Fan vs. HPC loading shift     |

These are inspired by real turbomachinery degradation physics and significantly stabilize training.

---

# 🧠 4. Implemented Models

## **V2–V4 — Local LSTM per FD Set**

* Fully literature-conform EOL setup (1 sample per engine)
* Strong RMSE on FD001 (~17.4)

## **V5 — Global LSTM**

* Single model across FD001–FD004
* Better condition generalization for FD002/FD004
* FD001 RMSE ≈ 14.8

---

## **V6 — Seq2Seq World Model (Trajectory Only)**

* LSTM Encoder → LSTM Decoder
* Predicts full 20-step RUL trajectory
* Excellent temporal stability, but **poor EOL accuracy**
  (it was never trained for EOL).

---

## **V7 — Multi-Task World Model (Trajectory + EOL Head)**

Adds:

* EOL prediction head on encoder hidden state
* Joint trajectory + EOL loss
* Bias monitoring
* NASA score integrated in evaluation

**Raw FD001 metrics:**

```
RMSE: 41.8
NASA: 321
Bias: +12
```

### **Calibrated version (linear correction):**

```
RMSE: 40.1
Bias: ~0
NASA mean: 105.8 (excellent)
```

World Models deliver **great NASA** (safety) but not SOTA RMSE.

---

## **V8 — Multi-Task + Physics Features + Stabilization**

Improvements:

* Stronger EOL head (3-layer MLP)
* Physics features
* Better loss balancing
* Smoothed trajectories

But:

* FD001 RMSE remains ~41–42
* NASA remains ~280–320

The bottleneck is the **encoder representation**, not the head.

---

## **EOL Regressor (Two-Stage Model)**

To achieve sharp EOL RMSE:

1. **Freeze World Model Encoder**
2. Train an independent **EOL MLP** on EOL windows only
3. Evaluate using literature-standard EOL RMSE, MAE, Bias, NASA

**FD001 results:**

```
RMSE: 40.9
NASA: 219.9
Bias: +8.1
```

NASA improves significantly; RMSE remains World-Model-limited.

---

# 📚 5. Benchmark Comparison (Internal vs. Literature)

### **Internal Models**

| Model / Version          | FD001 RMSE | FD001 NASA | Notes                      |
| ------------------------ | ---------- | ---------- | -------------------------- |
| **Local LSTM (V4)**      | **17.4**   | 342        | Good classic baseline      |
| **Global LSTM (V5)**     | ~14.8      | ~447       | Best RMSE internally       |
| **World Model V6**       | ~78        | 2160       | Trajectory-only model      |
| **World Model V7**       | 41.8       | 321        | Strong NASA                |
| **V7 (Calibrated)**      | 40.1       | **105.8**  | Best NASA                  |
| **World Model V8**       | 41–42      | 280–320    | Stable but RMSE still high |
| **World Model + EOLReg** | 40.9       | 219.9      | Better NASA, similar RMSE  |

---

### **Literature Benchmarks (FD001)**

| Paper / Method                                    | RMSE      | Score/NASA | Summary                                |   |
| ------------------------------------------------- | --------- | ---------- | -------------------------------------- | - |
| **Reweighted Transformer (Kim 2024)**             | **11.36** | 192.2      | Transformer + sample reweighting       |   |
| **GCU-Transformer**                               | **11–12** | –          | Hybrid gated convolution + transformer |   |
| **CTVAE / Health-Aware Transformer (Sadek 2025)** | 12.41     | –          | VAE + transformer, uncertainty-aware   |   |
| **CNN for RUL (Li 2018)**                         | 12–15     | –          | Early evidence CNN > LSTM              |   |
| **Enhanced LSTM (Elsherif 2025)**                 | 13.22     | 232.2      | Better RUL target generation           |   |
| **XGBoost + RUL pre-generation**                  | ~12.9     | –          | Classic ML with engineered RUL labels  |   |

**Conclusion:**
Literature SOTA for FD001 sits consistently at **11–13 RMSE**, mostly achieved by:

* Transformers with CNN front-ends
* Attention-enhanced LSTMs
* Improved RUL labeling or reweighting
* Variational latent-state models

Your project targets *World Model explainability + safety (NASA)* rather than raw RMSE.

---

# 🧪 6. Why RMSE and NASA Behave Differently

* **RMSE** penalizes squared error → sensitive to systematic trends.
* **NASA PHM08 Score** penalizes only **overprediction** exponentially (late warnings).

World Models tend to:

* Underpredict at high RUL → improves NASA.
* Overpredict near failure → controlled by calibration.

Thus:

* **Low NASA is easier**
* **Low RMSE requires a specialized EOL model**

---

# 🚀 7. Roadmap (Phase 1–3)

## **Phase 1 — Split the Problem**

* World Model → **Trajectory, uncertainty, safety**
* EOL Regressor → **Sharp RMSE**

## **Phase 2 — Encoder Upgrades**

* CNN + LSTM + Attention front-end
* Lightweight Transformer encoders (GCU/health-aware)

## **Phase 3 — Physics-Informed World Model**

* Multi-head forecasting (sensors + RUL)
* Variational latent-state dynamics (Deep State Space)
* Mode-specific encoding for FD002/FD004
* Full uncertainty propagation

---

# 👤 8. Contact & License

**Author:** Dr.-Ing. Robert Kunte
**LinkedIn:** [https://www.linkedin.com/in/robertkunte/](https://www.linkedin.com/in/robertkunte/)
**License:** MIT

If you work on turbomachinery, PHM, or physics-informed ML and want to collaborate, feel free to reach out!

<<<<<<< HEAD
=======
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

    RMSE / MAE are very similar,

    but continuous settings yield slightly better NASA scores and are physically more realistic.

Therefore, continuous settings are used as the default encoding, and ConditionID remains an optional experimental variant.
📊 Key Results – LSTM Models
1. Local Per-FD Models (Physics-Informed LSTM)

Local LSTM models are trained separately per FD with physics-informed features and the asymmetric RUL loss.

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

    Same architecture and loss as local models

    Evaluation is still per FD + aggregated

Global model (baseline, without MC-Dropout uncertainty):
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

For FD001, a dedicated architecture LSTMRUL_MCDropout is used:

    Same LSTM structure, but with Dropout layers kept active at inference (MC-Dropout)

    Helper functions in src/uncertainty.py:

        enable_mc_dropout(model)

        mc_dropout_predict(model, X_tensor, n_samples=50)

Results (FD001):

    RMSE improves to ≈ 12.9 cycles, better than the standard local LSTM run

    Predictive standard deviation correlates strongly with the absolute error

        Clear degradation → lower uncertainty

        Long periods of “flat” RUL (due to clamping) → higher uncertainty

Interpretation:
MC-Dropout successfully identifies where the model is less sure (e.g. under-informative sequences). Uncertainty can be used as a risk indicator for engineers (“double-check this asset”).
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

    This variant provides a useful regularized baseline and the starting point to extend MC-Dropout uncertainty from FD001 to the full global setup.

🌐 V6: Global Seq2Seq World Model (RUL Trajectory Prediction)
Architecture (src/models/world_model.py)

The World Model is implemented as a sequence-to-sequence encoder–decoder:

    Encoder: multi-layer LSTM over past sequences (past_len = 30 cycles)

    Decoder: multi-layer LSTM that rolls out a future RUL trajectory (horizon = 20 cycles)

    A small projection head maps hidden states to a scalar RUL at each future step

    Teacher forcing during training, open-loop rollout during evaluation

class WorldModelEncoderDecoder(nn.Module):
    # Encoder: past sequence  (30 cycles)
    # Decoder: predicts future RUL (20 cycles)
    # Implemented in src/models/world_model.py

Training (src/world_model_training.py, notebooks/6_world_model_training.ipynb)

Global training is performed over all four subsets (FD001–FD004):

    Inputs: same feature set as the global LSTM

        selected sensors

        3 operating settings

        3 physics-informed features (Effizienz_HPC_Proxy, EGT_Drift, Fan_HPC_Ratio)

    Targets: future RUL sequences of length horizon = 20

    Loss: MSE over the entire predicted RUL trajectory

    Training utilities:

        build_seq2seq_samples_from_df(...)

        build_world_model_dataset_from_df(...)

        train_world_model_global(...) with

            train/val split

            best-model tracking (min. val loss)

            optional early stopping

            checkpoint saving to results/world_model/world_model_global_best.pt

Example training log from V6:

[WorldModel] Epoch 1/25  - train: 6450.12, val: 4425.84
...
[WorldModel] Epoch 7/25  - train:    8.04, val:    7.23
[WorldModel] Epoch 10/25 - train:    4.74, val:    3.58
[WorldModel] Epoch 15/25 - train:    2.57, val:    3.00  <-- best model
Early stopping triggered after 20 epochs (no improvement for 5 epochs).
Loaded best model with val_loss=3.0046

Evaluation

Two key evaluation modes are implemented in world_model_training.py:

    Trajectory-Level Error (all future steps)

        MSE / RMSE over all (engine, future_step) samples

        Example (train/global rollout):

            MSE ≈ 21.96, RMSE ≈ 4.69 over ~147k future RUL samples

    End-of-Life NASA Score (test set)

        For each engine:

            Take the rolled-out future RUL trajectory at the end of the horizon

            Compare the last predicted RUL to the true RUL at failure

            Compute per-engine error and NASA PHM08 penalty

        Implemented in compute_nasa_score_from_world_model(...)

V6 Test-Set NASA for the World Model:

{
 'num_engines': 653,
 'mean_error':      1.65,   # mean(pred - true) at end-of-life
 'mean_abs_error':  1.90,   # |pred - true|
 'nasa_score_sum':  215.32,
 'nasa_score_mean': 0.33,   # per engine
}

Interpretation:

    The World Model is slightly optimistic at end-of-life (+1.65 cycles), but very close to unbiased.

    Average end-of-life error is ≈ 1.9 cycles.

    The mean NASA score ≈ 0.33 per engine indicates very low risk of late RUL predictions, especially compared to the LSTM baselines (which have NASA sums in the O(10³–10⁴) range across all FDs).

This makes the V6 World Model a strong candidate for:

    Scenario simulation (“what happens to RUL if operation continues like this?”)

    Downstream decision-making, where both the shape of the RUL trajectory and the risk at end-of-life matter.

🧱 Project Structure

AI_Turbine_RUL_Monitor_CMAPSS/
├─ data/
│  └─ raw/                # raw C-MAPSS text files
├─ results/
│  ├─ FD001/              # per-FD local model outputs
│  ├─ FD002/
│  ├─ FD003/
│  ├─ FD004/
│  ├─ global/             # global LSTM model predictions & weights
│  └─ world_model/        # V6 world model checkpoints & diagnostics
├─ notebooks/
│  ├─ 1_fd001_exploration.ipynb
│  ├─ 1a_fd002_exploration.ipynb
│  ├─ 2_training_single_datasets.ipynb
│  ├─ 3_evaluation_global_model.ipynb
│  ├─ 3a_evaluation_all_datasets.ipynb
│  ├─ 4_global_lstm.ipynb
│  ├─ 4a_global_lstm_dropout.ipynb
│  ├─ 5_training_uncertainty_dropout_analysis.ipynb
│  └─ 6_world_model_training.ipynb          # V6 Seq2Seq world model
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ data_loading.py
   ├─ additional_features.py
   ├─ training.py
   ├─ world_model_training.py               # dataset builders + training + eval
   ├─ models/
   │  ├─ lstm_rul.py
   │  ├─ lstm_rul_mcdo.py
   │  └─ world_model.py                     # WorldModelEncoderDecoder
   ├─ loss.py
   ├─ model.py
   ├─ uncertainty.py
   └─ eval_utils.py / train_utils.py        # helper utilities (optional)

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

    Open notebooks/2_training_single_datasets.ipynb

    Select the turbine_ai kernel

    Run all cells to:

        Train one LSTM per FD

        Save predictions & weights under results/FD00X/

4. Train Global LSTM Model

    Open notebooks/4_global_lstm.ipynb

    Run all cells to:

        Train the global physics-informed LSTM

        Save global predictions & weights under results/global/

5. Train V6 World Model (Seq2Seq)

    Open notebooks/6_world_model_training.ipynb

    Run all cells to:

        Build global Seq2Seq datasets

        Train the WorldModelEncoderDecoder

        Save best checkpoint to results/world_model/world_model_global_best.pt

        Evaluate end-of-life error and NASA score on the global test set

6. Evaluate & Analyze

    Use notebooks/3_local_evaluation.ipynb and 3a_evaluation_all_datasets.ipynb for local/global LSTMs

    Use notebooks/6_world_model_training.ipynb for:

        RUL trajectory visualizations

        End-of-life NASA evaluation for the World Model

🔭 Roadmap / Future Work

This repository is part of a broader “Mechanical Engineer AI Assistant” vision.

Planned next steps:

    Extended World Models

        Predict full multi-step sensor trajectories (not just RUL)

        Combine with physics constraints / PINNs

    Hyperparameter Optimization

        Optuna + PyTorch Lightning for joint tuning of:

            LSTM size, depth, dropout

            Loss weights (RUL vs. sensor vs. physics constraints)

            Sequence length and feature sets

    Extended Uncertainty Quantification

        Apply MC-Dropout and/or deep ensembles to the global and world models

        Add calibration plots and risk-aware decision rules

    Deployment

        Simple REST API or Streamlit dashboard for:

            Online RUL monitoring per asset

            Physics feature plots and trend analysis for engineers

📞 Contact & License

    Author: Dr.-Ing. Robert Kunte

    LinkedIn: https://www.linkedin.com/in/robertkunte/

    License: MIT

If you are working on turbomachinery, PHM or physics-informed ML and would like to exchange ideas, feel free to reach out!
>>>>>>> 03aaff4e0eee0e1cd30fd003ddabb32c8d0bf692
