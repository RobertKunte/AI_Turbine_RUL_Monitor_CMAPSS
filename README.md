🤖 AI-Based Turbine RUL Monitor: Physics-Informed LSTM Predictor

💡 Project Overview & Problem Statement

This project presents a Prognostics and Health Management (PHM) approach to predicting the Remaining Useful Life (RUL) of gas turbines. Unplanned failures in critical energy systems are extremely costly and lead to significant revenue loss.

Goal: Development of a Deep Learning model that accurately predicts the time of functional failure based on operational turbine data, enabling Predictive Maintenance.

⚙️ Technical Approach & Architecture

The solution is based on an LSTM (Long Short-Term Memory) architecture implemented in PyTorch, which is ideally suited for analyzing sequential, time-dependent sensor data.

1. Data and Preprocessing

    Dataset: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), Subset FD001.

    Feature Selection: Removal of constant or redundant sensors (e.g., Sensor 1, 5, 10) to minimize model noise.

    Scaling: Normalization of all input features (sensors) using MinMaxScaler to the range [0, 1].

2. Physics-Informed Clamping

To address the inherent uncertainty of long-term forecasts (phases where wear is minimal), a Physics-Informed RUL Clamping technique was implemented:

    The RUL target value was capped at 125 cycles (MAX_RUL = 125). The model is thus only trained to predict the end-of-life once critical wear begins.

3. Model

    Architecture: 2-layer LSTM (Num Layers = 2), followed by a Fully Connected Layer.

    Loss Function: Mean Squared Error (MSE), optimized using Adam.

🎯 Key Results (Test Performance)

The trained model was validated on a completely unseen test dataset (test_FD001.txt).
Metric	Value	Interpretation
Test MSE	1993.86	The mean squared deviation from the true RUL.
Test RMSE	44.65 Cycles	The average prediction error, measured in operating cycles.
The result shows that the monitor is capable of predicting the critical wear time with an average accuracy of 44 cycles.

🛠️ Reproduction and Setup

To reproduce this project locally, you will need Python 3.10 and the following packages:

Prerequisites

    Miniconda (recommended for environment management).

    PyTorch (LSTM Engine).

Environment Setup

Run the following commands in your Terminal/Anaconda Prompt:
Bash

# Create and activate environment
conda create -n turbine_ai python=3.10
conda activate turbine_ai

# Install packages
pip install torch torchvision torchaudio pandas matplotlib scikit-learn jupyter ipykernel

Execution

    Download the files train_FD001.txt, test_FD001.txt, and RUL_FD001.txt into your project folder (or into a /data subdirectory).

    Open the notebook 1_data_analysis.ipynb in Cursor/VS Code.

    Select the turbine_ai kernel.

    Execute the cells sequentially.

🔭 Future Work (World Models)

This project serves as a foundation for more complex applications within the "Mechanical Engineer Assistant" concept. Planned next steps include:

    Seq2Seq Model: Prediction of the entire future sensor time-series (not just the RUL value).

    Uncertainty Quantification (UQ): Implementation of methods for quantifying prediction uncertainty (e.g., via Monte Carlo Dropout).

    Integration: Embedding the model into a REST API or a Streamlit application for visualization.

📞 Contact & License

License	MIT License
Author	Dr.-Ing. Robert Kunte
Contact https://www.linkedin.com/in/robertkunte/