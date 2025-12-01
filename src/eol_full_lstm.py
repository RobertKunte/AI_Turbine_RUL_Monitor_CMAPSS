"""
Full-Trajectory LSTM for EOL prediction.

This module provides a LSTM-based regressor that processes complete engine
trajectories (not just tail samples) for Remaining Useful Life (RUL) prediction.
Includes NASA PHM08-style evaluation metrics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def build_full_eol_sequences_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    past_len: int = 30,
    max_rul: int = 125,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    rul_col: str = "RUL",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Baut Sliding-Window-Sequenzen über die KOMPLETTE Lebensdauer jeder Engine.

    Für jede Engine:
      - sortiere nach cycle_col
      - laufe t = past_len .. T
      - X_seq = Features[t-past_len:t, :]
      - y_seq = min(RUL_t, max_rul)

    Args:
        df: DataFrame mit mindestens feature_cols, unit_col, cycle_col, rul_col
        feature_cols: Liste der Feature-Spalten
        past_len: Länge des Vergangenheitsfensters
        max_rul: Maximum RUL (wird gecappt, Default: 125)
        unit_col: Name der Unit/Engine-Spalte
        cycle_col: Name der Cycle/Time-Spalte
        rul_col: Name der RUL-Spalte

    Returns:
        X: FloatTensor [N, past_len, num_features] - Input-Sequenzen
        y: FloatTensor [N] - RUL-Targets (gecappt auf max_rul)
        unit_ids: IntTensor [N] - Unit-IDs für jedes Sample
    """
    X_list = []
    y_list = []
    unit_id_list = []

    unit_ids = df[unit_col].unique()

    print("============================================================")
    print("[build_full_eol_sequences_from_df] Summary")
    print("============================================================")
    print(f"Num units: {len(unit_ids)}")
    print(f"Using past_len={past_len}, max_rul={max_rul}")
    print(f"Num feature cols: {len(feature_cols)}")

    for uid in unit_ids:
        df_u = (
            df[df[unit_col] == uid]
            .sort_values(cycle_col)
            .reset_index(drop=True)
        )

        if len(df_u) < past_len:
            continue

        values = df_u[feature_cols].to_numpy(dtype=np.float32)
        rul_values = df_u[rul_col].to_numpy(dtype=np.float32)

        # Sliding window über den gesamten Lebenslauf
        for i in range(past_len - 1, len(df_u)):
            window = values[i - past_len + 1 : i + 1]  # [past_len, num_features]
            rul_i = rul_values[i]
            rul_capped = min(rul_i, max_rul)  # Cap RUL at max_rul

            X_list.append(window)
            y_list.append(rul_capped)
            unit_id_list.append(uid)

    if len(X_list) == 0:
        raise ValueError(
            "[build_full_eol_sequences_from_df] No samples built – "
            "check past_len and data."
        )

    X = torch.from_numpy(np.stack(X_list))  # [N, past_len, num_features]
    y = torch.from_numpy(np.array(y_list, dtype=np.float32))  # [N]
    unit_ids_tensor = torch.from_numpy(np.array(unit_id_list, dtype=np.int32))  # [N]

    print(f"X shape: {X.shape}, y shape: {y.shape}, unit_ids shape: {unit_ids_tensor.shape}")
    print(
        f"RUL stats (capped at {max_rul}): min={y.min().item():.2f}, "
        f"max={y.max().item():.2f}, mean={y.mean().item():.2f}, "
        f"std={y.std().item():.2f}"
    )
    print("============================================================")

    return X, y, unit_ids_tensor


class SequenceDatasetWithUnits:
    """Dataset that includes unit_ids for each sample."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, unit_ids: torch.Tensor):
        self.X = X
        self.y = y
        self.unit_ids = unit_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.unit_ids[idx]


def create_full_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    unit_ids: torch.Tensor,
    batch_size: int = 256,
    engine_train_ratio: float = 0.8,
    shuffle_engines: bool = True,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, StandardScaler, torch.Tensor, torch.Tensor]:
    """
    Erstellt Train- und Validation-Dataloader mit engine-basiertem Split.

    - Split auf ENGINE-Ebene: zufällige 80% der Engines -> train, rest -> val
    - Feature-Scaling NUR aus Train (StandardScaler)
    - Liefert train_loader, val_loader, scaler, train_unit_ids, val_unit_ids

    Args:
        X: Input-Sequenzen, shape (N, past_len, F)
        y: RUL-Targets, shape (N,)
        unit_ids: Unit-IDs für jedes Sample, shape (N,)
        batch_size: Batch-Größe
        engine_train_ratio: Anteil der Engines für Training (Default: 0.8)
        shuffle_engines: Ob Engines zufällig gemischt werden sollen
        random_seed: Random Seed für Reproduzierbarkeit

    Returns:
        train_loader: DataLoader für Training
        val_loader: DataLoader für Validation
        scaler: Fitted StandardScaler (nur auf Train-Daten)
        train_unit_ids: Unit-IDs der Train-Engines
        val_unit_ids: Unit-IDs der Val-Engines
    """
    # Engine-basierter Split
    unique_units = torch.unique(unit_ids)
    n_units = len(unique_units)
    n_train_units = int(n_units * engine_train_ratio)
    if n_units > 1 and n_train_units == 0:
        n_train_units = 1
    n_val_units = n_units - n_train_units

    # Shuffle Engines
    if shuffle_engines:
        gen = torch.Generator().manual_seed(random_seed)
        perm = torch.randperm(n_units, generator=gen)
        train_unit_ids = unique_units[perm[:n_train_units]]
        val_unit_ids = unique_units[perm[n_train_units:]]
    else:
        train_unit_ids = unique_units[:n_train_units]
        val_unit_ids = unique_units[n_train_units:]

    # Masken für Train/Val
    train_mask = torch.isin(unit_ids, train_unit_ids)
    val_mask = torch.isin(unit_ids, val_unit_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # Feature-Scaling: Fit nur auf Train-Daten
    # Reshape für Scaler: (N_train * past_len, num_features)
    N_train, past_len, num_features = X_train.shape
    X_train_flat = X_train.numpy().reshape(-1, num_features)
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    # Transform Train
    X_train_scaled = scaler.transform(X_train_flat)
    X_train_scaled = torch.from_numpy(X_train_scaled.reshape(N_train, past_len, num_features))

    # Transform Val (mit gleichem Scaler)
    N_val = X_val.shape[0]
    X_val_flat = X_val.numpy().reshape(-1, num_features)
    X_val_scaled = scaler.transform(X_val_flat)
    X_val_scaled = torch.from_numpy(X_val_scaled.reshape(N_val, past_len, num_features))

    # Speichere unit_ids für Train/Val
    train_unit_ids_samples = unit_ids[train_mask]
    val_unit_ids_samples = unit_ids[val_mask]

    train_dataset = SequenceDatasetWithUnits(X_train_scaled, y_train, train_unit_ids_samples)
    val_dataset = SequenceDatasetWithUnits(X_val_scaled, y_val, val_unit_ids_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("============================================================")
    print("[create_full_dataloaders] Engine-based split")
    print("============================================================")
    print(f"Total units: {n_units}")
    print(f"Train units: {n_train_units}, Val units: {n_val_units}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Feature scaling: StandardScaler (fitted on train only)")
    print("============================================================")

    return train_loader, val_loader, scaler, train_unit_ids, val_unit_ids


class EOLFullLSTM(nn.Module):
    """
    LSTM-basierter Regressor für Full-Trajectory EOL-Prediction.

    Verarbeitet komplette Engine-Trajektorien (nicht nur Tail-Samples)
    und gibt RUL als Skalar pro Sample zurück.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 1),
        )

        # Initialisierung
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, F] - Input sequences

        Returns:
            y_hat: [B] - RUL predictions in cycles
        """
        # LSTM-Output
        out, _ = self.lstm(x)  # out: [B, T, H * num_directions]

        # Verwende Hidden State des letzten Zeitschritts
        last = out[:, -1, :]  # [B, H * num_directions]

        y_hat = self.head(last).squeeze(-1)  # [B]
        return y_hat


def nasa_phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 Score.

    e = y_pred - y_true
    score_i = exp(-e/13) - 1   falls e < 0  (zu pessimistisch)
             = exp( e/10) - 1   falls e >= 0 (zu optimistisch)
    Gesamtscore = Sum(score_i)

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values

    Returns:
        NASA score (sum over all samples)
    """
    e = y_pred - y_true
    score = np.where(
        e < 0.0,
        np.exp(-e / 13.0) - 1.0,
        np.exp(e / 10.0) - 1.0,
    )
    return float(score.sum())


def evaluate_eol_full_lstm(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates the Full-Trajectory LSTM model.

    Sammelt für alle Val-Samples:
      - y_true_all, y_pred_all, unit_ids_all

    Berechnet:
      - Pointwise: MSE, RMSE, MAE, Bias, R²

    EOL/NASA:
      - Für jede Engine: wähle Sample mit minimalem true RUL (letzter Zyklus)
      - RMSE_eol, MAE_eol, Bias_eol
      - NASA-Score_eol (Sum + Mean)

    Args:
        model: Trained EOLFullLSTM model
        val_loader: Validation DataLoader
        device: torch.device

    Returns:
        Dictionary mit allen Metriken
    """
    model.eval()
    model.to(device)

    y_true_all = []
    y_pred_all = []
    unit_ids_all = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:  # SequenceDatasetWithUnits
                X_batch, y_batch, unit_ids_batch = batch
                unit_ids_all.append(unit_ids_batch.numpy())
            else:  # Fallback für TensorDataset
                X_batch, y_batch = batch
                unit_ids_all.append(None)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)

            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # Pointwise Metriken
    errors = y_pred_all - y_true_all
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    r2 = 1 - np.sum(errors ** 2) / np.sum((y_true_all - y_true_all.mean()) ** 2)
    r2 = float(r2)

    # EOL-Style Metriken: Für jede Engine den letzten Zyklus (minimales RUL)
    if unit_ids_all[0] is not None:
        unit_ids_all = np.concatenate(unit_ids_all)
        
        # Für jede Engine: finde Sample mit minimalem true RUL
        unique_units = np.unique(unit_ids_all)
        y_true_eol = []
        y_pred_eol = []
        
        for uid in unique_units:
            mask = unit_ids_all == uid
            y_true_unit = y_true_all[mask]
            y_pred_unit = y_pred_all[mask]
            
            # Finde Index mit minimalem RUL (letzter Zyklus)
            min_rul_idx = np.argmin(y_true_unit)
            y_true_eol.append(y_true_unit[min_rul_idx])
            y_pred_eol.append(y_pred_unit[min_rul_idx])
        
        y_true_eol = np.array(y_true_eol)
        y_pred_eol = np.array(y_pred_eol)
        
        # EOL Metriken
        errors_eol = y_pred_eol - y_true_eol
        rmse_eol = float(np.sqrt(np.mean(errors_eol ** 2)))
        mae_eol = float(np.mean(np.abs(errors_eol)))
        bias_eol = float(np.mean(errors_eol))
        
        # NASA Score (EOL)
        nasa_score_sum_eol = nasa_phm_score(y_true_eol, y_pred_eol)
        nasa_score_mean_eol = nasa_score_sum_eol / len(y_true_eol)
        
        has_eol_metrics = True
    else:
        rmse_eol = None
        mae_eol = None
        bias_eol = None
        nasa_score_sum_eol = None
        nasa_score_mean_eol = None
        has_eol_metrics = False

    print("============================================================")
    print("[evaluate_eol_full_lstm] Pointwise Metrics")
    print("============================================================")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f} cycles")
    print(f"MAE: {mae:.4f} cycles")
    print(f"Bias: {bias:.4f} cycles")
    print(f"R²: {r2:.4f}")
    print("============================================================")
    
    if has_eol_metrics:
        print("[evaluate_eol_full_lstm] EOL/NASA Metrics (per Engine, last cycle)")
        print("============================================================")
        print(f"RMSE_eol: {rmse_eol:.4f} cycles")
        print(f"MAE_eol: {mae_eol:.4f} cycles")
        print(f"Bias_eol: {bias_eol:.4f} cycles")
        print(f"NASA Score (sum): {nasa_score_sum_eol:.2f}")
        print(f"NASA Score (mean): {nasa_score_mean_eol:.4f}")
        print(f"Num engines: {len(y_true_eol)}")
    else:
        print("[evaluate_eol_full_lstm] EOL/NASA Metrics")
        print("============================================================")
        print("[NOTE] EOL metrics require unit_ids per sample.")
        print("[NOTE] Using pointwise metrics as approximation.")
    
    print("============================================================")

    # NASA Score (pointwise)
    nasa_score_sum = nasa_phm_score(y_true_all, y_pred_all)
    nasa_score_mean = nasa_score_sum / len(y_true_all)

    metrics = {
        "pointwise": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r2": r2,
        },
        "nasa_pointwise": {
            "score_sum": nasa_score_sum,
            "score_mean": nasa_score_mean,
        },
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }
    
    if has_eol_metrics:
        metrics["eol"] = {
            "rmse": rmse_eol,
            "mae": mae_eol,
            "bias": bias_eol,
            "nasa_score_sum": nasa_score_sum_eol,
            "nasa_score_mean": nasa_score_mean_eol,
            "num_engines": len(y_true_eol),
            "y_true_eol": y_true_eol,
            "y_pred_eol": y_pred_eol,
        }

    return metrics


def train_eol_full_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 80,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: torch.device | str = "cpu",
    results_dir: Path | str = "../results/eol_full_lstm",
    run_name: str = "fd001_fd004",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Training-Loop für Full-Trajectory LSTM.

    - MSELoss
    - Adam (lr, weight_decay)
    - LR-Scheduler (ReduceLROnPlateau auf val_loss)
    - Early Stopping (patience)
    - Speichere bestes Modell als 'eol_full_lstm_best.pt'
    - Logge Train/Val MSE + Val RMSE pro Epoch

    Args:
        model: EOLFullLSTM model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Anzahl Epochen
        lr: Learning Rate
        weight_decay: L2-Regularisierung
        patience: Early Stopping Patience
        device: torch.device
        results_dir: Verzeichnis für Checkpoints und Plots
        run_name: Name des Runs (für Dateinamen)

    Returns:
        model: Bestes Modell (geladen)
        history: Dictionary mit Trainings-Verlauf
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "lr": [],
    }

    print("============================================================")
    print("[train_eol_full_lstm] Training Configuration")
    print("============================================================")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(f"Device: {device}")
    print("============================================================")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            if len(batch) == 3:  # SequenceDatasetWithUnits
                X_batch, y_batch, _ = batch
            else:  # Fallback für TensorDataset
                X_batch, y_batch = batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        val_losses = []
        val_targets = []
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:  # SequenceDatasetWithUnits
                    X_batch, y_batch, _ = batch
                else:  # Fallback für TensorDataset
                    X_batch, y_batch = batch
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)

                val_losses.append(loss.item())
                val_targets.append(y_batch.cpu())
                val_preds.append(preds.cpu())

        val_loss = float(np.mean(val_losses))
        val_targets = torch.cat(val_targets)
        val_preds = torch.cat(val_preds)
        val_rmse = torch.sqrt(torch.mean((val_preds - val_targets) ** 2)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(current_lr)

        # Scheduler
        scheduler.step(val_loss)

        # Best Model Tracking & Early Stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            checkpoint_path = results_dir / f"eol_full_lstm_best_{run_name}.pt"
            torch.save(best_model_state, checkpoint_path)
        else:
            epochs_no_improve += 1

        print(
            f"[EOL-Full-LSTM] Epoch {epoch}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"val_RMSE: {val_rmse:.4f}, lr: {current_lr:.2e}"
        )
        if val_loss < best_val_loss:
            print(f"  --> New best val_loss: {val_loss:.4f}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print(
                f"[EOL-Full-LSTM] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {epochs_no_improve} epochs)."
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"[EOL-Full-LSTM] Loaded best model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    # Plot training curves
    plot_training_curves(history, results_dir / f"training_curves_{run_name}.png")

    print("============================================================")
    print("[train_eol_full_lstm] Training Complete")
    print("============================================================")
    print(f"Best val_loss: {best_val_loss:.4f} (at epoch {best_epoch})")
    print("============================================================")

    return model, history


def plot_training_curves(history: Dict[str, Any], save_path: Path | str):
    """
    Plottet Trainingskurven.

    Args:
        history: Dictionary mit train_loss, val_loss, val_rmse, lr
        save_path: Pfad zum Speichern des Plots
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Train/Val Loss
    axes[0].plot(epochs, history["train_loss"], label="Train MSE", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val MSE", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Full-Trajectory LSTM Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Val RMSE + LR
    axes[1].plot(epochs, history["val_rmse"], label="Val RMSE", linewidth=2, color="green")
    if "lr" in history:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["lr"], label="Learning Rate", linewidth=1, color="orange", linestyle="--")
        ax2.set_ylabel("Learning Rate", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (cycles)", color="green")
    axes[1].tick_params(axis="y", labelcolor="green")
    axes[1].set_title("Full-Trajectory LSTM Validation RMSE & Learning Rate")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved training curves to {save_path}")
    plt.close()

