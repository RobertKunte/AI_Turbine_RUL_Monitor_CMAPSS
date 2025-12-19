import pickle
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_x_scaler(X_train_np: np.ndarray, max_rows: int = 2_000_000, random_state: int = 42) -> StandardScaler:
    """
    Fit a StandardScaler on flattened (N,T,D) input windows.

    Notes:
    - Caller is responsible for passing TRAIN windows only (no val/test leakage).
    - For RAM safety, we subsample at most `max_rows` time-rows from the flattened array.
    """
    if X_train_np.ndim != 3:
        raise ValueError(f"fit_x_scaler expects X_train_np shape (N,T,D), got {X_train_np.shape}")
    N, T, D = X_train_np.shape
    X_flat = X_train_np.reshape(-1, D)

    if X_flat.shape[0] > int(max_rows):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(X_flat.shape[0], size=int(max_rows), replace=False)
        X_fit = X_flat[idx]
    else:
        X_fit = X_flat

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_fit)
    return scaler


def transform_x(scaler: StandardScaler, X_np: np.ndarray) -> np.ndarray:
    """
    Apply scaler to X windows. X_np must be (N,T,D).
    """
    if X_np.ndim != 3:
        raise ValueError(f"transform_x expects X_np shape (N,T,D), got {X_np.shape}")
    N, T, D = X_np.shape
    X_flat = X_np.reshape(-1, D)
    Xs = scaler.transform(X_flat).reshape(N, T, D)
    return Xs.astype(np.float32, copy=False)


def save_scaler(path: str, scaler: StandardScaler) -> None:
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

