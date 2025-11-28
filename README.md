
# 🚀 AI-Based Turbine RUL Monitor – Physics-Informed World Models

### *Deep Learning for Turbofan Prognostics on NASA C-MAPSS (FD001–FD004)*

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

