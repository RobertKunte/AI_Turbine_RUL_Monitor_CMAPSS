# src/training.py
import os
import numpy as np
import pandas as pd

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
    NUM_EPOCHS
)
from .data_loading import load_cmapps_subset, get_feature_drop_cols
from .additional_features import create_physical_features
from .model import LSTMRULPredictor
from .loss import rul_asymmetric_weighted_loss

# -------------------------------------------------------------------
# Helper: NASA PHM08 scoring function
# -------------------------------------------------------------------
def nasa_phm08_score(y_true, y_pred):
    """
    Compute the NASA PHM08 scoring function for RUL predictions.

    Args:
        y_true (array-like): true RUL values, shape (n_samples,)
        y_pred (array-like): predicted RUL values, shape (n_samples,)

    Returns:
        float: total NASA score (lower is better, 0 is perfect)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    d = y_pred - y_true  # positive: late (too optimistic), negative: early (too pessimistic)

    score = np.zeros_like(d, dtype=float)

    # Early predictions (d < 0): milder penalty
    mask_early = d < 0
    score[mask_early] = np.exp(-d[mask_early] / 13.0) - 1.0

    # Late predictions (d >= 0): stronger penalty
    mask_late = ~mask_early
    score[mask_late] = np.exp(d[mask_late] / 10.0) - 1.0

    return float(np.sum(score))


# -------------------------------------------------------------------
# Helper: Sequence building
# -------------------------------------------------------------------
def build_sequences_from_df(df, feature_cols, sequence_length):
    """
    Build sliding window sequences for training from a unit-wise DataFrame.

    Args:
        df (pd.DataFrame): must contain columns 'UnitNumber', 'TimeInCycles', 'RUL' and feature_cols
        feature_cols (list of str): columns to be used as model input features
        sequence_length (int): length of the sequence (e.g. 30)

    Returns:
        X (np.ndarray): shape (N, sequence_length, F)
        y (np.ndarray): shape (N,)
    """
    X_list, y_list = [], []

    for unit_id, df_unit in df.groupby("UnitNumber"):
        df_unit_sorted = df_unit.sort_values("TimeInCycles")
        feat = df_unit_sorted[feature_cols].values
        rul = df_unit_sorted["RUL"].values

        if len(feat) >= sequence_length:
            for i in range(len(feat) - sequence_length + 1):
                X_list.append(feat[i : i + sequence_length])
                y_list.append(rul[i + sequence_length - 1])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def build_last_sequences_from_df(df, feature_cols, sequence_length):
    """
    Build one sequence per unit: the last sequence_length time steps.
    For units shorter than sequence_length, pad at the beginning
    by repeating the first row.

    Returns:
        X (np.ndarray): shape (n_units, sequence_length, F)
        unit_ids (list): list of UnitNumbers
    """
    X_list, unit_ids = [], []

    for unit_id, df_unit in df.groupby("UnitNumber"):
        df_unit_sorted = df_unit.sort_values("TimeInCycles")
        feat = df_unit_sorted[feature_cols].values
        n = len(feat)

        if n >= sequence_length:
            seq = feat[-sequence_length:]
        else:
            pad_len = sequence_length - n
            # repeat the first row pad_len times at the beginning
            pad_block = np.repeat(feat[0:1, :], pad_len, axis=0)
            seq = np.vstack([pad_block, feat])

        X_list.append(seq)
        unit_ids.append(unit_id)

    X = np.array(X_list)
    return X, unit_ids



# -------------------------------------------------------------------
# Helper: Train / Val loop
# -------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=NUM_EPOCHS,
    device=None,
    fd_id="FDXXX",
):
    """
    Generic training loop for RUL model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)  # (B,)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        epoch_train_loss = train_loss / n_train

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).squeeze(-1)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        epoch_val_loss = val_loss / n_val

        print(
            f"[{fd_id}] Epoch [{epoch+1}/{num_epochs}] "
            f"- Train: {epoch_train_loss:.4f}  Val: {epoch_val_loss:.4f}"
        )

    print(f"Training for {fd_id} completed.")
    return model


# -------------------------------------------------------------------
# Helper: Evaluate on test sequences
# -------------------------------------------------------------------
def predict_on_test(model, X_test, device=None, max_rul=MAX_RUL):
    """
    Run model on test sequences and clamp predictions to max_rul.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_t).cpu().numpy().reshape(-1)

    y_pred = np.minimum(y_pred, max_rul)
    return y_pred


# -------------------------------------------------------------------
# Helper: Basic metrics
# -------------------------------------------------------------------
def compute_basic_metrics(y_true, y_pred):
    """
    Compute MSE, RMSE, MAE, Bias and NASA score.
    """
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    nasa = nasa_phm08_score(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NASA_score": nasa,
    }
    return metrics


# -------------------------------------------------------------------
# Main: Train and evaluate model for a single FD subset
# -------------------------------------------------------------------
def train_and_evaluate_fd(
    fd_id,
    model_class,
    loss_fn,
    num_epochs=NUM_EPOCHS,
    max_rul=MAX_RUL,
    results_root="results",
    batch_size=64,
    random_seed=42,
):
    """
    Train a local RUL model for a given C-MAPSS subset (FD001–FD004)
    and save predictions + metrics.

    Args:
        fd_id (str): e.g. "FD001"
        model_class: model constructor, e.g. LSTMRULPredictor
        loss_fn: loss function, e.g. rul_asymmetric_weighted_loss
        num_epochs (int)
        max_rul (int)
        results_root (str): base directory for saving results
        batch_size (int)
        random_seed (int)

    Returns:
        results_df (pd.DataFrame), metrics (dict)
    """
    assert fd_id in CMAPSS_DATASETS, f"Unknown subset: {fd_id}"

    print(f"\n=== Training subset {fd_id}: {CMAPSS_DATASETS[fd_id]['desc']} ===")

    # Set seeds for reproducibility (optional)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load raw data
    df_train_raw, df_test_raw, y_test_true = load_cmapps_subset(
        fd_id, max_rul=max_rul
        )


    # 2) Physics-informed features
    df_train_phys = create_physical_features(df_train_raw.copy())
    df_test_phys = create_physical_features(df_test_raw.copy())

    # 3) Drop columns according to FD
    drop_cols = get_feature_drop_cols(fd_id)
    df_train_feat = df_train_phys.drop(columns=drop_cols, errors="ignore")
    df_test_feat = df_test_phys.drop(columns=drop_cols, errors="ignore")

    non_feature_cols = ["UnitNumber", "RUL", "TimeInCycles"]
    feature_cols = [c for c in df_train_feat.columns if c not in non_feature_cols]

    print("Feature columns used for training:")
    print(feature_cols)

    # 4) Scaling
    scaler = MinMaxScaler()
    df_train_feat[feature_cols] = scaler.fit_transform(df_train_feat[feature_cols])
    df_test_feat[feature_cols] = scaler.transform(df_test_feat[feature_cols])

    # 5) Build sequences for training
    X_train, y_train = build_sequences_from_df(
        df_train_feat, feature_cols, sequence_length=SEQUENCE_LENGTH
    )

    # 6) Torch datasets and loaders (simple 80/20 split)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    full_ds = TensorDataset(X_train_t, y_train_t)
    n_total = len(full_ds)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 7) Model, optimizer, loss
    input_size = X_train.shape[2]
    model = model_class(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = loss_fn

    # 8) Training
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        fd_id=fd_id,
    )

    # 9) Test sequences (last window per unit)
    X_test, unit_ids_test = build_last_sequences_from_df(
        df_test_feat, feature_cols, sequence_length=SEQUENCE_LENGTH
    )
    y_test_pred = predict_on_test(model, X_test, device=device, max_rul=max_rul)

    # 10) Build results DataFrame using last row of df_test_phys per unit
    df_last = (
        df_test_phys.sort_values(["UnitNumber", "TimeInCycles"])
        .groupby("UnitNumber")
        .tail(1)
        .sort_values("UnitNumber")
    )

    # Make sure lengths match
    assert len(df_last) == len(y_test_true) == len(y_test_pred), (
        f"Length mismatch for {fd_id}: "
        f"df_last={len(df_last)}, y_test_true={len(y_test_true)}, y_test_pred={len(y_test_pred)}"
    )

    results_df = pd.DataFrame(
        {
            "UnitNumber": df_last["UnitNumber"].values,
            "TimeInCycles": df_last["TimeInCycles"].values,
            "TrueRUL": y_test_true,
            "PredRUL": y_test_pred,
            "Effizienz_HPC_Proxy": df_last["Effizienz_HPC_Proxy"].values,
            "EGT_Drift": df_last["EGT_Drift"].values,
            "Fan_HPC_Ratio": df_last["Fan_HPC_Ratio"].values,
            "FD_ID": fd_id,
            "ModelType": "local",
        }
    )

    # 11) Metrics
    metrics = compute_basic_metrics(y_test_true, y_test_pred)
    print(f"[{fd_id}] Evaluation Metrics:")
    for k, v in metrics.items():
        if k == "NASA_score":
            print(f"  {k:>10}: {v:.3f}")
        else:
            print(f"  {k:>10}: {v:.3f}")

    # 12) Save results
    out_dir = os.path.join(results_root, fd_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fd_id}_predictions_local.csv")
    results_df.to_csv(out_path, index=False)
    print(f"[{fd_id}] Saved predictions to: {out_path}")

    # Optional: Modellgewichte speichern
    model_path = os.path.join(out_dir, f"{fd_id}_lstm_local.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{fd_id}] Saved model weights to: {model_path}")

    return results_df, metrics
