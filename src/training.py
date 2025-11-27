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
    NUM_EPOCHS,
    GLOBAL_FEATURE_COLS,
    GLOBAL_DROP_COLS 
)
from .data_loading import load_cmapps_subset, get_feature_drop_cols, load_cmapps_global
from .additional_features import create_physical_features
from .model import LSTMRULPredictor
from .loss import rul_asymmetric_weighted_loss
from src.uncertainty import mc_dropout_predict
from .models.lstm_rul_mcdo import LSTMRULPredictorMCDropout

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

def build_sequences_from_df(
    df: pd.DataFrame,
    feature_cols,
    sequence_length: int,
    group_cols=("UnitNumber",),
):
    """
    Build sliding-window sequences from a DataFrame.

    group_cols: tuple of columns to group by, e.g. ("FD_ID", "UnitNumber")
    """
    X_list, y_list = [], []

    for _, df_unit in df.groupby(list(group_cols)):
        df_unit_sorted = df_unit.sort_values("TimeInCycles")
        feat = df_unit_sorted[feature_cols].values
        rul = df_unit_sorted["RUL"].values

        if len(df_unit_sorted) < sequence_length:
            continue

        for i in range(len(df_unit_sorted) - sequence_length + 1):
            X_list.append(feat[i:i + sequence_length])
            y_list.append(rul[i + sequence_length - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y


def build_last_sequences_from_df(
    df: pd.DataFrame,
    feature_cols,
    sequence_length: int,
    group_cols=("UnitNumber",),
):
    """
    Build only the LAST sequence per unit (for final RUL prediction).

    - Ensures *every* unit produces exactly one sequence.
    - If a unit has fewer than `sequence_length` rows, we pad at the *beginning*
      by repeating the first row (healthy state assumption).
    """
    X_list = []
    unit_keys = []

    for key, df_unit in df.groupby(list(group_cols)):
        df_unit_sorted = df_unit.sort_values("TimeInCycles")
        feat = df_unit_sorted[feature_cols].values  # shape: (T, F)

        T = feat.shape[0]
        if T < sequence_length:
            pad_len = sequence_length - T
            # Repeat the first row (early, healthy state) pad_len times
            pad = np.repeat(feat[:1, :], pad_len, axis=0)
            feat_padded = np.vstack([pad, feat])
        else:
            feat_padded = feat

        # Take the last `sequence_length` rows
        X_list.append(feat_padded[-sequence_length:])
        unit_keys.append(key)

    X = np.array(X_list, dtype=np.float32)
    return X, unit_keys

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
    results_root="../results",
    batch_size=64,
    random_seed=42,
    use_condition_id: bool = False,
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

    base_cols = df_train_feat.columns.tolist()
    if use_condition_id:
        # ConditionID-Ansatz:
        # - Settings raus
        # - ConditionID bleibt als Feature drin
        exclude = ["UnitNumber", "TimeInCycles", "RUL", "MaxTime",
                   "Setting1", "Setting2", "Setting3"]
        feature_cols = [c for c in base_cols if c not in exclude]
        print(f"[{fd_id}] Using CONDITION-ID feature set")
    else:
        # Physikalischer Ansatz:
        # - Settings drin
        # - ConditionID nur für Logging/Analyse, NICHT als Feature
        exclude = ["UnitNumber", "TimeInCycles", "RUL", "MaxTime",
                   "ConditionID"]
        feature_cols = [c for c in base_cols if c not in exclude]
        print(f"[{fd_id}] Using PHYSICAL (continuous settings) feature set")
    if use_condition_id and fd_id in ["FD001", "FD003"]:
        print(f"[WARN] use_condition_id=True for {fd_id}, but this is a single-condition dataset.")
        print("       ConditionID will be constant 0 and adds no information.")

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

    # 13) Alles Wichtige in einem Dict bündeln
    results = {
        "fd_id": fd_id,
        "results_df": results_df,      # das alte DataFrame
        "model": model,                # trainiertes PyTorch-Modell
        "scaler": scaler,              # MinMaxScaler der Features
        "feature_cols": feature_cols,  # Feature-Spalten, die ins Modell gehen
        "y_test_true": y_test_true,    # True RUL für Test
        "y_test_pred": y_test_pred,    # Predicted RUL für Test
    }

    return results, metrics


def train_and_evaluate_global(
    fd_ids=("FD001", "FD002", "FD003", "FD004"),
    model_class=LSTMRULPredictor,
    loss_fn=rul_asymmetric_weighted_loss,
    num_epochs=NUM_EPOCHS,
    batch_size=64,
    results_root="../results/global",
    use_mc_dropout: bool = False,
    n_mc_samples: int = 50,
):
    import os
    os.makedirs(results_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load all subsets
    df_train_raw, test_dfs_raw, test_ruls = load_cmapps_global(
        fd_ids=fd_ids, max_rul=MAX_RUL
    )

    # 2) Add physics features
    df_train_phys = create_physical_features(df_train_raw.copy())
    test_dfs_phys = {
        fd: create_physical_features(df.copy())
        for fd, df in test_dfs_raw.items()
    }

    # 3) Drop unused columns (global)
    df_train_feat = df_train_phys.drop(columns=GLOBAL_DROP_COLS, errors="ignore")
    test_dfs_feat = {
        fd: df.drop(columns=GLOBAL_DROP_COLS, errors="ignore")
        for fd, df in test_dfs_phys.items()
    }

    # 4) Scale features globally
    feature_cols = GLOBAL_FEATURE_COLS
    scaler = MinMaxScaler()
    df_train_feat[feature_cols] = scaler.fit_transform(df_train_feat[feature_cols])

    for fd, df in test_dfs_feat.items():
        df[feature_cols] = scaler.transform(df[feature_cols])

    # 5) Build global training sequences
    X_train, y_train = build_sequences_from_df(
        df_train_feat, feature_cols,
        sequence_length=SEQUENCE_LENGTH,
        group_cols=("FD_ID", "UnitNumber")
    )

    # convert to tensors + DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 6) Init model
    input_size = X_train.shape[2]
    model = model_class(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7) Training loop (nur Train, Val optional global)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"[GLOBAL] Epoch [{epoch+1}/{num_epochs}] - Train: {epoch_loss:.4f}")

    print("Global training completed.")

    # 8) Evaluation per FD subset
    all_true = []
    all_pred = []
    all_fd = []

    model.eval()  # Grundzustand (MC-Dropout schalten wir in uncertainty.py)
    for fd_idx, fd_id in enumerate(fd_ids):
        df_test_fd = test_dfs_feat[fd_id]
        y_true_fd = test_ruls[fd_id]

        X_test_fd, unit_keys = build_last_sequences_from_df(
            df_test_fd, feature_cols,
            sequence_length=SEQUENCE_LENGTH,
            group_cols=("FD_ID", "UnitNumber")
        )

        X_test_tensor = torch.tensor(X_test_fd)

        # --- MC-Dropout oder normale Vorhersage ---
        if use_mc_dropout:
            y_mean_fd, y_std_fd = mc_dropout_predict(
                model,
                X_test_tensor,
                n_samples=n_mc_samples,
                device=device,
                max_rul=MAX_RUL,
            )
            y_pred_fd = y_mean_fd
        else:
            with torch.no_grad():
                y_pred_fd = model(X_test_tensor.to(device)) \
                    .cpu().numpy().reshape(-1)
            y_pred_fd = np.minimum(y_pred_fd, MAX_RUL)

        # Truth ebenfalls clampen
        y_true_fd = np.minimum(y_true_fd, MAX_RUL)

        # Metrics pro FD
        metrics_fd = compute_basic_metrics(y_true_fd, y_pred_fd)
        print(f"[GLOBAL] Metrics for {fd_id}: {metrics_fd}")

        # sammeln für globale Metrik
        all_true.append(y_true_fd)
        all_pred.append(y_pred_fd)
        all_fd.extend([fd_id] * len(y_true_fd))

        # CSV pro FD (inkl. Unsicherheit, falls vorhanden)
        out_data = {
            "FD": fd_id,
            "UnitIndex": np.arange(len(y_true_fd)),
            "TrueRUL": y_true_fd,
            "PredRUL_mean": y_pred_fd,
        }
        if use_mc_dropout:
            out_data["PredRUL_std"] = y_std_fd

        out_df = pd.DataFrame(out_data)
        out_path = os.path.join(results_root, f"{fd_id}_global_predictions.csv")
        out_df.to_csv(out_path, index=False)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    global_metrics = compute_basic_metrics(all_true, all_pred)
    print("[GLOBAL] Overall metrics across all FDs:", global_metrics)

    # Save global metrics
    metrics_df = pd.DataFrame(global_metrics, index=[0])
    metrics_df.to_csv(os.path.join(results_root, "global_metrics.csv"), index=False)

    return model, scaler, global_metrics