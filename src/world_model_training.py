
import os
from typing import List
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt  # falls noch nicht importiert

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    from torch.utils.data import TensorDataset, DataLoader, random_split  # type: ignore[import]
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
from .models.world_model import WorldModelEncoderDecoder
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
            )  # (B, H, 1)

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
            )  # (B, horizon, 1)

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
        )  # (1, horizon, 1)

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
            )  # (1, H, 1)

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
