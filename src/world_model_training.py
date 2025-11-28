
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt  # falls noch nicht importiert

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for training routines. Please install torch."
    ) from exc

try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for preprocessing. Please install scikit-learn."
    ) from exc

from .config import (
    CMAPSS_DATASETS,
    MAX_RUL,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    GLOBAL_FEATURE_COLS,
    GLOBAL_DROP_COLS 
)

from .data_loading import load_cmapps_subset, get_feature_drop_cols, load_cmapps_global
from .additional_features import create_physical_features
from .model import LSTMRULPredictor
from .loss import rul_asymmetric_weighted_loss
from .uncertainty import mc_dropout_predict
from .models.lstm_rul_mcdo import LSTMRULPredictorMCDropout
from .models.world_model import WorldModelEncoderDecoder, WorldModelEncoderDecoderMultiTask
from .training import build_eol_sequences_from_df
from .eval_utils import compute_nasa_score_pairwise
from src.models.eol_regressor import EOLRegressor


# -------------------------------------------------------------------
# Helper: Sequence building
# -------------------------------------------------------------------

def build_seq2seq_samples_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    past_len: int = 30,
    horizon: int = 20,
):
    """
    Build (X_past, Y_future) samples for a seq2seq world model.

    - df should be a single-FD, single-UnitNumber DataFrame or already grouped.
    - feature_cols = input features (sensors + settings + physics).
    - target_cols  = target features to predict (e.g. ['RUL'] or subset of sensors).
    """
    X_list, Y_list = [], []

    # ensure numeric and float32
    values_in = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    values_out = df[target_cols].to_numpy(dtype=np.float32, copy=True)
    T = len(df)

    for t_past_end in range(past_len - 1, T - horizon):
        t_past_start = t_past_end + 1 - past_len
        t_future_start = t_past_end + 1
        t_future_end = t_future_start + horizon

        X_past = values_in[t_past_start : t_past_end + 1]     # (L_past, F)
        Y_future = values_out[t_future_start : t_future_end]  # (H, F_out)

        X_list.append(X_past)
        Y_list.append(Y_future)

    if not X_list:
        # wichtig: leerer Rückgabefall
        return np.empty((0, past_len, len(feature_cols)), dtype=np.float32), \
               np.empty((0, horizon, len(target_cols)), dtype=np.float32)

    return np.stack(X_list), np.stack(Y_list)


def build_world_model_dataset_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    past_len: int = 30,
    horizon: int = 20,
    unit_col: str = "UnitNumber",
) -> TensorDataset:
    """
    Baut ein TensorDataset für das World Model.
    - Gruppiert df nach UnitNumber, damit keine Sequenzen über Engine-Grenzen springen.
    - Targets sind hier (erstmal) nur RUL-Sequenzen.
    """
    X_list, Y_list = [], []

    for _, df_unit in df.groupby(unit_col):
        X_np, Y_np = build_seq2seq_samples_from_df(
            df=df_unit,
            feature_cols=feature_cols,
            target_cols=[target_col],
            past_len=past_len,
            horizon=horizon,
        )
        if X_np.shape[0] == 0:
            continue
        X_list.append(X_np)
        Y_list.append(Y_np)

    if not X_list:
        raise ValueError("No seq2seq samples could be built from the given DataFrame.")

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)

    X = torch.tensor(X_all, dtype=torch.float32)
    Y = torch.tensor(Y_all, dtype=torch.float32)
    return TensorDataset(X, Y)


def train_world_model_global(
    df_train_global,
    feature_cols: List[str],
    target_col: str = "RUL",
    past_len: int = 30,
    horizon: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 20,
    lr: float = 1e-3,
    val_split: float = 0.1,
    early_stopping_patience: int = 5,
    checkpoint_dir: str | None = "results/world_model",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Training eines globalen World Models (Seq2Seq auf RUL) über alle FDs.

    Features:
    - Train/Val-Split
    - Speichert das Modell mit dem besten Val-Loss (best_state_dict)
    - Optional: Early Stopping über early_stopping_patience
    - Optional: Checkpoint-Speicherung im Filesystem
    """

    # 1) Dataset bauen (inkl. gruppieren nach UnitNumber)
    full_dataset = build_world_model_dataset_from_df(
        df=df_train_global,
        feature_cols=feature_cols,
        target_col=target_col,
        past_len=past_len,
        horizon=horizon,
    )

    # 2) Train/Val-Split
    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3) Modell
    model = WorldModelEncoderDecoder(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,  # nur RUL-Sequenz als Output
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # MSE über die RUL-Trajektorie

    history = {"train_loss": [], "val_loss": []}

    # Early-Stopping / Best-Model-Tracking
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    # Optional: Checkpoint-Dir vorbereiten
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "world_model_global_best.pt")
    else:
        checkpoint_path = None

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)  # (B, L_past, F)
            Y_batch = Y_batch.to(device)  # (B, H, 1)

            optimizer.zero_grad()

            # Teacher Forcing im Training
            preds = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                teacher_forcing_ratio=0.5,
            )  # (B, H, 1) or tuple for multi-task
            
            # Handle tuple returns from multi-task models
            if isinstance(preds, tuple):
                preds = preds[0]  # Use trajectory predictions

            loss = loss_fn(preds, Y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / n_train

        # ---- Validation ----
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                # reiner Rollout im Val-Mode
                preds = model(
                    encoder_inputs=X_batch,
                    decoder_targets=Y_batch,
                    teacher_forcing_ratio=0.0,
                )
                
                # Handle tuple returns from multi-task models
                if isinstance(preds, tuple):
                    preds = preds[0]  # Use trajectory predictions

                loss = loss_fn(preds, Y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / n_val

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"[WorldModel] Epoch {epoch+1}/{num_epochs} "
            f"- train: {epoch_train_loss:.4f}, val: {epoch_val_loss:.4f}"
        )

        # ---- Best-Model / Early Stopping ----
        if epoch_val_loss < best_val_loss - 1e-6:
            best_val_loss = epoch_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(
                f"--> New best model at epoch {epoch+1} "
                f"with val_loss={epoch_val_loss:.4f}"
            )

            # optional: Checkpoint speichern
            if checkpoint_path is not None:
                torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and early_stopping_patience > 0:
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {epoch+1} epochs "
                        f"(no improvement for {epochs_without_improvement} epochs)."
                    )
                    break

    # Nach Training: bestes Modell laden (falls vorhanden)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Loaded best model with val_loss={best_val_loss:.4f}")

    return model, history


def evaluate_world_model_global(
    model: WorldModelEncoderDecoder,
    df_eval_global: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    past_len: int = 30,
    horizon: int = 20,
    unit_col: str = "UnitNumber",
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Evaluiert das World Model global auf einem DataFrame (z.B. df_train_global oder df_test_global).

    - Baut (X_past, Y_future)-Samples über alle Units
    - Lässt das Modell im reinen Rollout-Modus (teacher_forcing_ratio=0.0) vorhersagen
    - Berechnet MSE / RMSE über alle Future-Steps und Samples
    - Optional: kann später erweitert werden (z.B. Schritt-spezifische RMSE)

    Returns:
        metrics: dict mit 'MSE', 'RMSE', 'num_samples'
    """
    model = model.to(device)
    model.eval()

    # Dataset auf Basis des gesamten DataFrames (mit Unit-Gruppierung)
    dataset = build_world_model_dataset_from_df(
        df=df_eval_global,
        feature_cols=feature_cols,
        target_col=target_col,
        past_len=past_len,
        horizon=horizon,
        unit_col=unit_col,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_sq_error = 0.0
    total_count = 0

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)          # (B, past_len, F)
            Y_batch = Y_batch.to(device)          # (B, horizon, 1)

            # Reiner Rollout → kein Teacher Forcing
            preds = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                teacher_forcing_ratio=0.0,
            )  # (B, horizon, 1) or tuple for multi-task
            
            # Handle tuple returns from multi-task models
            if isinstance(preds, tuple):
                preds = preds[0]  # Use trajectory predictions

            sq_error = (preds - Y_batch) ** 2     # (B, horizon, 1)
            total_sq_error += sq_error.sum().item()
            total_count += Y_batch.numel()        # B * horizon * 1

    mse = total_sq_error / total_count
    rmse = math.sqrt(mse)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "num_samples": len(dataset),
    }

    print(
        f"[WorldModel-Eval] MSE: {mse:.4f}, RMSE: {rmse:.4f}, "
        f"samples: {len(dataset)}"
    )

    return metrics

def plot_world_model_rul_rollout_for_unit(
    model: WorldModelEncoderDecoder,
    df_global: pd.DataFrame,
    feature_cols: List[str],
    fd_id: str,
    unit_number: int,
    past_len: int = 30,
    horizon: int = 20,
    unit_col: str = "UnitNumber",
    fd_col: str = "FD_ID",
    target_col: str = "RUL",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualisiert True vs Pred RUL-Trajektorie für eine konkrete Engine
    (z.B. FD001, Unit 1) über den angegebenen Horizon.

    - Nutzt das letzte mögliche Fenster:
        vergangene past_len Schritte → zukünftige horizon Schritte
    - Macht einen reinen Rollout (teacher_forcing_ratio=0.0).
    """

    # 1) Spezifische Engine aus globalem DF filtern
    mask = (df_global[fd_col] == fd_id) & (df_global[unit_col] == unit_number)
    df_unit = df_global[mask].sort_values("TimeInCycles")

    if df_unit.empty:
        raise ValueError(f"Keine Daten gefunden für {fd_id}, Unit {unit_number}")

    T = len(df_unit)
    if T < past_len + horizon:
        raise ValueError(
            f"Unit {unit_number} in {fd_id} ist zu kurz für past_len={past_len} und "
            f"horizon={horizon} (T={T})."
        )

    # 2) Letzte mögliche Position wählen:
    #    t_past_end + horizon <= T-1  → t_past_end = T - horizon - 1
    t_past_end = T - horizon - 1
    t_past_start = t_past_end + 1 - past_len
    if t_past_start < 0:
        raise ValueError(
            f"Berechnete Vergangenheit startet < 0 (t_past_start={t_past_start}). "
            f"Passe past_len/horizon an."
        )

    t_future_start = t_past_end + 1
    t_future_end = t_future_start + horizon

    # 3) NumPy-Arrays bauen
    values_in = df_unit[feature_cols].to_numpy(dtype=np.float32, copy=True)
    values_out = df_unit[[target_col]].to_numpy(dtype=np.float32, copy=True)

    X_past = values_in[t_past_start : t_past_end + 1]     # (past_len, F)
    Y_future_true = values_out[t_future_start : t_future_end]  # (horizon, 1)

    # 4) Tensor bauen & Modell aufrufen
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_past[None, ...], dtype=torch.float32, device=device)   # (1, past_len, F)
    Y_tensor = torch.tensor(Y_future_true[None, ...], dtype=torch.float32, device=device)  # (1, horizon, 1)

    with torch.no_grad():
        Y_pred = model(
            encoder_inputs=X_tensor,
            decoder_targets=Y_tensor,
            teacher_forcing_ratio=0.0,
        )
        
    # Handle both single tensor and tuple returns (for multi-task models)
    if isinstance(Y_pred, tuple):
        # Multi-task model returns (traj_outputs, eol_pred)
        Y_pred = Y_pred[0]  # Use trajectory predictions
    
    Y_pred = Y_pred.squeeze(0).squeeze(-1).cpu().numpy()   # (horizon,)
    Y_true = Y_future_true.squeeze(-1)                     # (horizon,)

    # 5) Plotten
    plt.figure(figsize=(8, 5))
    plt.plot(range(horizon), Y_true, marker="o", label="True RUL")
    plt.plot(range(horizon), Y_pred, marker="x", linestyle="--", label="Predicted RUL")
    plt.xlabel("Future step (Δ cycles)")
    plt.ylabel("RUL [cycles]")
    plt.title(f"World Model RUL Rollout – {fd_id}, Unit {unit_number}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: einfache Kennzahlen ausgeben
    mse = np.mean((Y_pred - Y_true) ** 2)
    rmse = math.sqrt(mse)
    print(f"[WorldModel-Rollout] {fd_id}, Unit {unit_number} – MSE: {mse:.4f}, RMSE: {rmse:.4f}")

def compute_nasa_score_from_world_model(
    model,
    df_global,
    feature_cols,
    past_len=30,
    horizon=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    model.to(device)

    errors = []

    grouped = df_global.groupby(["FD_ID", "UnitNumber"], sort=True)

    for (fd_id, unit), df_unit in grouped:
        df_unit = df_unit.sort_values("TimeInCycles")

        if len(df_unit) < past_len + horizon:
            continue

        # Vergangenheit + Zukunft wie im Plot
        past   = df_unit.iloc[-(past_len + horizon):-horizon]
        future = df_unit.iloc[-horizon:]

        X = torch.tensor(
            past[feature_cols].values[np.newaxis, :, :],
            dtype=torch.float32,
            device=device,
        )

        Y_future_true = future[["RUL"]].values  # (H, 1)
        Y_tensor = torch.tensor(
            Y_future_true[np.newaxis, :, :],     # (1, H, 1)
            dtype=torch.float32,
            device=device,
        )

        true_rul_last = float(Y_future_true[-1, 0])  # sollte ~0 sein

        with torch.no_grad():
            Y_pred = model(
                encoder_inputs=X,
                decoder_targets=Y_tensor,
                teacher_forcing_ratio=0.0,
            )
            
        # Handle both single tensor and tuple returns (for multi-task models)
        if isinstance(Y_pred, tuple):
            # Multi-task model returns (traj_outputs, eol_pred)
            Y_pred = Y_pred[0]  # Use trajectory predictions
        
        pred_rul_last = float(Y_pred[0, -1, 0])  # letzter Horizon-Schritt

        error = pred_rul_last - true_rul_last
        errors.append(error)

    errors = np.array(errors)

    def nasa_term(e):
        if e < 0:
            return np.exp(-e / 13.0) - 1.0
        else:
            return np.exp(e / 10.0) - 1.0

    nasa_scores = np.array([nasa_term(e) for e in errors])

    return {
        "num_engines": int(len(errors)),
        "mean_error": float(errors.mean()),
        "mean_abs_error": float(np.abs(errors).mean()),
        "nasa_score_sum": float(nasa_scores.sum()),
        "nasa_score_mean": float(nasa_scores.mean()),
    }

def evaluate_world_model_pointwise(
    model: "nn.Module",
    df_global: "pd.DataFrame",
    feature_cols: list[str],
    past_len: int = 30,
    horizon: int = 20,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Evaluates the world model as a *point* RUL predictor:
    - Build sliding (past, future) windows per engine (FD_ID, UnitNumber)
    - For each window, use the model to predict the future RUL sequence
    - Take ONLY the *last* future step as "RUL point"
    - Compute RMSE / MAE / Bias / NASA over all these points.

    This is closer to the LSTM-style evaluation, but uses many time windows.
    """
    model = model.to(device)
    model.eval()

    all_y_true = []
    all_y_pred = []

    grouped = df_global.groupby(["FD_ID", "UnitNumber"], sort=True)

    with torch.no_grad():
        for (fd_id, unit), df_unit in grouped:
            df_unit = df_unit.sort_values("TimeInCycles")

            # baue (X_past, Y_future) für genau diese Engine
            X_np, Y_np = build_seq2seq_samples_from_df(
                df=df_unit,
                feature_cols=feature_cols,
                target_cols=["RUL"],
                past_len=past_len,
                horizon=horizon,
            )

            if X_np.size == 0:
                continue

            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
            # wir können in Batches vorgehen, damit es nicht zu groß wird
            for start in range(0, X_tensor.size(0), batch_size):
                end = start + batch_size
                X_batch = X_tensor[start:end]

                # roll-out ohne teacher forcing
                preds = model(
                    encoder_inputs=X_batch,
                    decoder_targets=None,
                    teacher_forcing_ratio=0.0,
                    horizon=horizon,
                )  # (B, H, 1) or tuple for multi-task
                
                # Handle tuple returns from multi-task models
                if isinstance(preds, tuple):
                    preds = preds[0]  # Use trajectory predictions

                preds_np = preds.cpu().numpy()[:, -1, 0]  # letzter Future-Step
                true_np = Y_np[start:end, -1, 0]          # true RUL beim letzten Future-Step

                all_y_pred.append(preds_np)
                all_y_true.append(true_np)

    if not all_y_true:
        raise ValueError("No valid samples generated for pointwise evaluation.")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    nasa_stats = compute_nasa_score_pairwise(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
    }
    metrics.update(nasa_stats)
    return metrics

def evaluate_world_model_trajectory(
    model: "nn.Module",
    df_global: "pd.DataFrame",
    feature_cols: list[str],
    past_len: int = 30,
    horizon: int = 20,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Evaluates the world model over the *full* predicted RUL trajectories:
    - For each engine and each window, predict the future RUL sequence.
    - Compare **all** future steps (flattened) to the true RUL sequence.
    - Compute MSE / RMSE / MAE / Bias / NASA over all (engine, step) pairs.
    """
    model = model.to(device)
    model.eval()

    all_y_true = []
    all_y_pred = []

    grouped = df_global.groupby(["FD_ID", "UnitNumber"], sort=True)

    with torch.no_grad():
        for (fd_id, unit), df_unit in grouped:
            df_unit = df_unit.sort_values("TimeInCycles")

            X_np, Y_np = build_seq2seq_samples_from_df(
                df=df_unit,
                feature_cols=feature_cols,
                target_cols=["RUL"],
                past_len=past_len,
                horizon=horizon,
            )

            if X_np.size == 0:
                continue

            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
            for start in range(0, X_tensor.size(0), batch_size):
                end = start + batch_size
                X_batch = X_tensor[start:end]

                preds = model(
                    encoder_inputs=X_batch,
                    decoder_targets=None,
                    teacher_forcing_ratio=0.0,
                    horizon=horizon,
                )  # (B, H, 1) or tuple for multi-task
                
                # Handle tuple returns from multi-task models
                if isinstance(preds, tuple):
                    preds = preds[0]  # Use trajectory predictions

                preds_np = preds.cpu().numpy().reshape(-1)   # B*H
                true_np = Y_np[start:end].reshape(-1)        # B*H

                all_y_pred.append(preds_np)
                all_y_true.append(true_np)

    if not all_y_true:
        raise ValueError("No valid samples generated for trajectory evaluation.")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    nasa_stats = compute_nasa_score_pairwise(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
    }
    metrics.update(nasa_stats)
    return metrics

def _build_eol_input_for_unit(
    df_unit: "pd.DataFrame",
    feature_cols: list[str],
    past_len: int = 30,
) -> np.ndarray:
    """
    Helper: Baut für genau EINE Engine (UnitNumber) das Input-Window
    für die EOL-Evaluation:
    - sortiert nach TimeInCycles
    - nimmt die letzten `past_len` Zeilen
    - falls die Engine kürzer ist, pad mit der ersten Zeile.
    
    Rückgabe: np.ndarray der Form (past_len, n_features)
    """
    df_unit = df_unit.sort_values("TimeInCycles")

    n = len(df_unit)
    if n >= past_len:
        window_df = df_unit.iloc[-past_len:]
    else:
        # mit der ersten Zeile nach oben auffüllen
        first_row = df_unit.iloc[[0]].copy()
        pad_rows = [first_row] * (past_len - n)
        window_df = pd.concat(pad_rows + [df_unit], ignore_index=True)

    # nur die gewünschten Features
    X = window_df[feature_cols].values.astype(np.float32)  # (past_len, F)
    return X


def evaluate_world_model_eol(
    model: "nn.Module",
    df_global_test: "pd.DataFrame",
    feature_cols: list[str],
    past_len: int = 30,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Literatur-konforme EOL-Evaluation des World Models auf C-MAPSS:

    - Genau EIN Punkt pro Engine im Testset.
    - Für jede Engine:
        - nimm die letzten `past_len` Zyklen (mit Padding falls nötig)
        - baue Input X_past (1, past_len, F)
        - lass das World Model horizon=1 Schritt in die Zukunft rollen
        - y_pred = RUL-Vorhersage für den nächsten Schritt
        - y_true = RUL aus df_global_test an der letzten Zeile dieser Engine
    - Aggregiere:
        - RMSE, MAE, Bias
        - NASA PHM08 Score (Summe & Mean) nach Standardformel.

    Args:
        model:             trainiertes WorldModelEncoderDecoder
        df_global_test:    globaler Test-DataFrame (alle FDs) mit Spalte "RUL"
        feature_cols:      Liste der Inputfeatures (Sensors + Settings + Physikfeatures)
        past_len:          Länge des History-Fensters vor EOL
        device:            "cuda" oder "cpu"

    Returns:
        dict mit:
            - "MSE", "RMSE", "MAE", "Bias"
            - "num_engines"
            - "mean_error", "mean_abs_error"
            - "nasa_score_sum", "nasa_score_mean"
    """
    # Handle both dict and DataFrame inputs
    if isinstance(df_global_test, dict):
        # Concatenate all test DataFrames from the dict
        df_list = []
        for fd_id, df_test in df_global_test.items():
            df_test = df_test.copy()
            if "FD_ID" not in df_test.columns:
                df_test["FD_ID"] = fd_id
            df_list.append(df_test)
        df_global_test = pd.concat(df_list, ignore_index=True)
    elif not isinstance(df_global_test, pd.DataFrame):
        raise TypeError(f"df_global_test must be a pd.DataFrame or dict, got {type(df_global_test)}")

    model = model.to(device)
    model.eval()

    y_true_all = []
    y_pred_all = []
    engine_info = []  # Store (FD_ID, UnitNumber) for each prediction

    # Gruppierung wie üblich: jede Engine ist (FD_ID, UnitNumber)
    grouped = df_global_test.groupby(["FD_ID", "UnitNumber"], sort=True)

    with torch.no_grad():
        for (fd_id, unit), df_unit in grouped:
            # sicherstellen, dass RUL vorhanden ist
            if "RUL" not in df_unit.columns:
                raise ValueError("df_global_test must contain a 'RUL' column for EOL evaluation.")

            # true RUL am letzten beobachteten Zyklus
            df_unit = df_unit.sort_values("TimeInCycles")
            true_rul_last = float(df_unit["RUL"].iloc[-1])

            # Input-Window bauen
            X_past_np = _build_eol_input_for_unit(
                df_unit=df_unit,
                feature_cols=feature_cols,
                past_len=past_len,
            )  # (past_len, F)

            X_past = torch.tensor(
                X_past_np[np.newaxis, :, :],  # (1, past_len, F)
                dtype=torch.float32,
                device=device,
            )

            # World Model: horizon=1, kein Teacher Forcing
            preds = model(
                encoder_inputs=X_past,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
                horizon=1,
            )  # (1, 1, 1) or tuple for multi-task
            
            # Handle tuple returns from multi-task models
            # For EOL, prefer eol_pred if available, otherwise use trajectory
            if isinstance(preds, tuple):
                traj_preds, eol_pred = preds
                # Use EOL prediction if available, otherwise use trajectory
                if eol_pred is not None:
                    pred_rul = float(eol_pred[0, 0].cpu().item())
                else:
                    pred_rul = float(traj_preds[0, 0, 0].cpu().item())
            else:
                pred_rul = float(preds[0, 0, 0].cpu().item())

            y_true_all.append(true_rul_last)
            y_pred_all.append(pred_rul)
            engine_info.append((fd_id, unit))

    if len(y_true_all) == 0:
        raise ValueError("No engines found for EOL evaluation. Check df_global_test grouping.")

    y_true = np.array(y_true_all, dtype=np.float32)
    y_pred = np.array(y_pred_all, dtype=np.float32)

    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    nasa_stats = compute_nasa_score_pairwise(y_true, y_pred)

    # Create results DataFrame
    results_df = pd.DataFrame({
        "FD_ID": [info[0] for info in engine_info],
        "UnitNumber": [info[1] for info in engine_info],
        "true_rul": y_true,
        "pred_rul": y_pred,
        "error": errors,
    })

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "num_engines": len(y_true),
        "mean_error": float(errors.mean()),
        "mean_abs_error": float(np.abs(errors).mean()),
        "nasa_score_sum": float(nasa_stats["nasa_score_sum"]),
        "nasa_score_mean": float(nasa_stats["nasa_score_mean"]),
        "results_df": results_df,  # Include results DataFrame in metrics
    }
    return metrics

def train_world_model_global_multitask(
    df_train_global,
    feature_cols: List[str],
    target_col: str = "RUL",
    past_len: int = 30,
    horizon: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 25,
    lr: float = 5e-4,
    val_split: float = 0.1,
    lambda_traj: float = 1.0,
    lambda_eol: float = 1.0,
    lambda_bias: float = 0.1,
    lambda_nasa: float = 0.0,  # Commented out: lambda_nasa: float = 0.05,
    early_stopping_patience: int = 5,
    checkpoint_dir: str = "results/world_model_multitask",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Globales Multi-Task World Model Training:
    - Hauptaufgabe: RUL-Trajektorie (Seq2Seq)
    - Nebenaufgabe: EOL-Point (RUL_{t+1} aus Encoder-Summary)

    df_train_global: globaler Trainings-DataFrame (alle FDs, inkl. RUL & physikalischen Features)
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 1) Dataset bauen (wie vorher) ---
    X_np, Y_np = build_seq2seq_samples_from_df(
        df=df_train_global,
        feature_cols=feature_cols,
        target_cols=[target_col],
        past_len=past_len,
        horizon=horizon,
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)   # (N, H, 1)

    full_dataset = TensorDataset(X, Y)

    # --- 2) Train/Val-Split ---
    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    
    # Ensure validation set has at least 1 sample if dataset is large enough
    if n_total > 1 and n_val == 0:
        n_val = 1
        n_train = n_total - n_val
    elif n_total == 0:
        raise ValueError("Dataset is empty. Cannot train model.")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty. Dataset size: {n_total}, "
            f"val_split: {val_split}, n_val: {n_val}. "
            f"Try reducing val_split or using a larger dataset."
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 3) Modell, Optimizer, Loss ---
    model = WorldModelEncoderDecoderMultiTask(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_traj_loss": [],
        "val_eol_loss": [],
        "val_traj_loss_per_step": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, "world_model_multitask_best.pt")

    # --- 4) Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_traj_loss = 0.0
        running_train_eol_loss = 0.0
        running_train_bias_loss = 0.0
        running_train_nasa_loss = 0.0
        n_train_samples = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            traj_pred, eol_pred = model(
                encoder_inputs=X_batch,
                decoder_targets=Y_batch,
                horizon=horizon,
            )

            # 1) Trajektorien-Loss
            loss_traj = mse_loss(traj_pred, Y_batch)

            # 2) EOL-Loss
            target_eol = Y_batch[:, 0, 0]
            eol_pred_flat = eol_pred.squeeze(-1)
            loss_eol = mse_loss(eol_pred_flat, target_eol)

            # 3) Bias-Loss
            errors = eol_pred_flat - target_eol
            batch_bias = errors.mean()
            loss_bias = batch_bias ** 2

            # 4) NASA-ähnlicher Loss (geclampte Errors)
            errors_clamped = torch.clamp(errors, min=-100.0, max=100.0)
            nasa_like = torch.where(
                errors_clamped >= 0,
                torch.exp(errors_clamped / 10.0) - 1.0,
                torch.exp(-errors_clamped / 13.0) - 1.0,
            ).mean()

            # 5) Gesamt-Loss
            loss = (
                lambda_traj * loss_traj
                + lambda_eol * loss_eol
                + lambda_bias * loss_bias
                # + lambda_nasa * nasa_like
            )

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_train_loss      += loss.item()      * batch_size
            running_train_traj_loss += loss_traj.item() * batch_size
            running_train_eol_loss  += loss_eol.item()  * batch_size
            running_train_bias_loss += loss_bias.item() * batch_size
            running_train_nasa_loss += nasa_like.item() * batch_size
            n_train_samples         += batch_size

        epoch_train_loss = running_train_loss / n_train_samples
        # Optional: separat loggen, wenn du willst
        # history.setdefault("train_traj_loss", []).append(running_train_traj_loss / n_train_samples)
        # history.setdefault("train_eol_loss", []).append(running_train_eol_loss / n_train_samples)
        # history.setdefault("train_bias_loss", []).append(running_train_bias_loss / n_train_samples)
        # history.setdefault("train_nasa_loss", []).append(running_train_nasa_loss / n_train_samples)


        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        running_val_traj_loss = 0.0
        running_val_eol_loss = 0.0
        running_val_bias_loss = 0.0
        running_val_nasa_loss = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                traj_pred, eol_pred = model(
                    encoder_inputs=X_batch,
                    decoder_targets=Y_batch,
                    teacher_forcing_ratio=0.0,
                    horizon=horizon,
                )

                # 1) Trajektorien-Loss über alle 20 Schritte
                loss_traj = mse_loss(traj_pred, Y_batch)

                # 2) EOL-Loss: erster Zeitschritt, erstes Feature
                target_eol = Y_batch[:, 0, 0]                       # [B]
                eol_pred_flat = eol_pred.squeeze(-1)                # [B]
                loss_eol = mse_loss(eol_pred_flat, target_eol)

                # 3) Bias-Loss (mittlerer Fehler^2)
                errors = eol_pred_flat - target_eol                 # [B]
                batch_bias = errors.mean()
                loss_bias = batch_bias ** 2

                # 4) NASA-ähnlicher Loss (asymmetrisch, sanft)
                errors_clamped = torch.clamp(errors, min=-100.0, max=100.0)
                nasa_like = torch.where(
                    errors_clamped >= 0,
                    torch.exp(errors_clamped / 10.0) - 1.0,
                    torch.exp(-errors_clamped / 13.0) - 1.0,
                ).mean()

                # 5) Gesamt-Loss wie im Training
                loss = (
                    lambda_traj * loss_traj
                    + lambda_eol * loss_eol
                    + lambda_bias * loss_bias
                    # + lambda_nasa * nasa_like
                )

                batch_size = X_batch.size(0)
                running_val_loss       += loss.item()       * batch_size
                running_val_traj_loss  += loss_traj.item()  * batch_size
                running_val_eol_loss   += loss_eol.item()   * batch_size
                running_val_bias_loss  += loss_bias.item()  * batch_size
                running_val_nasa_loss  += nasa_like.item()  * batch_size
                n_val_samples          += batch_size

        if n_val_samples == 0:
            n_val_samples = len(val_dataset)
            if n_val_samples == 0:
                raise ValueError("Validation dataset is empty. Check val_split and dataset size.")

        epoch_val_loss = running_val_loss / n_val_samples
        epoch_val_traj_loss = running_val_traj_loss / n_val_samples
        epoch_val_eol_loss = running_val_eol_loss / n_val_samples
        epoch_val_bias_loss = running_val_bias_loss / n_val_samples
        epoch_val_nasa_loss = running_val_nasa_loss / n_val_samples
        epoch_val_traj_loss_per_step = epoch_val_traj_loss / horizon

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_traj_loss"].append(epoch_val_traj_loss)
        history["val_eol_loss"].append(epoch_val_eol_loss)
        history["val_traj_loss_per_step"].append(epoch_val_traj_loss_per_step)
        history.setdefault("val_bias_loss", []).append(epoch_val_bias_loss)
        history.setdefault("val_nasa_loss", []).append(epoch_val_nasa_loss)

        print(
            f"[WorldModel-MT] Epoch {epoch+1}/{num_epochs} "
            f"- train: {epoch_train_loss:.4f}, val: {epoch_val_loss:.4f}, "
            f"traj: {epoch_val_traj_loss:.4f} ({epoch_val_traj_loss_per_step:.4f}/step), "
            f"eol: {epoch_val_eol_loss:.4f}, "
            f"bias: {epoch_val_bias_loss:.6f}, "
            f"nasa: {epoch_val_nasa_loss:.4f}"
        )


        # Early Stopping + Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best multitask model at epoch {epoch+1} with val_loss={epoch_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping (multitask) triggered after {epoch+1} epochs "
                    f"(no improvement for {early_stopping_patience} epochs)."
                )
                break

    # Bestes Modell laden
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    return model, history 


def train_eol_regressor_from_world_model(
    world_model,
    df_train,
    feature_cols,
    past_len: int,
    hidden_size: int,
    device,
    max_rul: int = None,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_epochs: int = 50,
    lr: float = 1e-3,
    random_seed: int = 42,
):
    """
    Train a separate EOL regressor on top of a frozen WorldModel encoder,
    using exactly ONE EOL-style sample per engine.

    Args:
        world_model: pretrained WorldModelEncoderDecoderMultiTask
        df_train: pandas DataFrame with training data (multiple engines)
        feature_cols: list of feature column names (e.g. GLOBAL_FEATURE_COLS)
        past_len: encoder sequence length (e.g. 30)
        hidden_size: encoder hidden size (world_model.hidden_size)
        device: torch.device
        max_rul: optional RUL clamp (e.g. MAX_RUL)
        val_split: fraction of engines used for validation
        batch_size: batch size for train/val loaders
        num_epochs: number of epochs
        lr: learning rate
        random_seed: random seed for reproducibility

    Returns:
        eol_reg: trained EOLRegressor (best val RMSE)
        history: dict with loss / RMSE curves
    """
    try:
        import torch  # type: ignore[import]
        import torch.nn as nn  # type: ignore[import]
        from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for this notebook. Please install torch."
        ) from exc

    # 1) Build ONE EOL sample per engine
    # mode="train": pick random truncation point BEFORE failure (like NASA test setting)
    X_eol, y_eol = build_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        mode="train",
        random_seed=random_seed,
    )

    print(f"[EOLReg] Built EOL sequences: X={X_eol.shape}, y={y_eol.shape}")
    print(
        f"[EOLReg] RUL stats: min={float(y_eol.min()):.2f}, "
        f"max={float(y_eol.max()):.2f}, mean={float(y_eol.mean()):.2f}"
    )

    class EOLDataset(Dataset):
        def __init__(self, X, y):
            self.X = X.float()
            self.y = y.float().view(-1)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    full_ds = EOLDataset(X_eol, y_eol)

    # 2) Train/Val split auf Engine-Ebene
    n_total = len(full_ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    print(f"[EOLReg] Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3) World Model einfrieren
    world_model.to(device)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    # 4) EOLRegressor instanziieren
    eol_reg = EOLRegressor(
        in_dim=hidden_size,
        hidden_dims=(128, 64),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(eol_reg.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
    }

    best_val_rmse = float("inf")
    best_state = None

    def extract_encoder_summary(batch_inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs only the encoder of the world model and returns encoder summary.
        batch_inputs: [B, L_past, F]
        Returns: [B, hidden_size]
        """
        with torch.no_grad():
            enc_out, (h_n, c_n) = world_model.encoder(batch_inputs)
            enc_summary = h_n[-1]  # last layer, shape [B, hidden_size]
        return enc_summary

    for epoch in range(num_epochs):
        # --------------------
        # TRAIN
        # --------------------
        eol_reg.train()
        running_train_loss = 0.0
        n_train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            enc_summary = extract_encoder_summary(X_batch).to(device)
            pred = eol_reg(enc_summary).squeeze(-1)

            loss = mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_train_loss += loss.item() * batch_size
            n_train_samples += batch_size

        epoch_train_loss = running_train_loss / max(1, n_train_samples)
        history["train_loss"].append(epoch_train_loss)

        # --------------------
        # VALIDATION
        # --------------------
        eol_reg.eval()
        running_val_loss = 0.0
        running_val_sqerr = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float()

                enc_summary = extract_encoder_summary(X_batch).to(device)
                pred = eol_reg(enc_summary).squeeze(-1)

                loss = mse_loss(pred, y_batch)
                errors = pred - y_batch

                batch_size = X_batch.size(0)
                running_val_loss += loss.item() * batch_size
                running_val_sqerr += torch.sum(errors ** 2).item()
                n_val_samples += batch_size

        epoch_val_loss = running_val_loss / max(1, n_val_samples)
        epoch_val_rmse = (running_val_sqerr / max(1, n_val_samples)) ** 0.5

        history["val_loss"].append(epoch_val_loss)
        history["val_rmse"].append(epoch_val_rmse)

        print(
            f"[EOLReg] Epoch {epoch+1}/{num_epochs} "
            f"- train_loss: {epoch_train_loss:.4f}, "
            f"val_loss: {epoch_val_loss:.4f}, "
            f"val_RMSE: {epoch_val_rmse:.4f}"
        )

        if epoch_val_rmse < best_val_rmse:
            best_val_rmse = epoch_val_rmse
            best_state = {
                "model": eol_reg.state_dict(),
                "val_rmse": best_val_rmse,
                "epoch": epoch + 1,
            }

    if best_state is not None:
        eol_reg.load_state_dict(best_state["model"])
        print(
            f"[EOLReg] Best model at epoch {best_state['epoch']} "
            f"with val_RMSE={best_state['val_rmse']:.4f}"
        )

    return eol_reg, history
