# z.B. in deinem Notebook oder in src/uncertainty.py
import numpy as np
try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    from torch.utils.data import TensorDataset, DataLoader, random_split  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this notebook. Please install torch."
    ) from exc

def enable_mc_dropout(model: nn.Module):
    """
    Schaltet alle Dropout-Layer in den 'train'-Modus,
    damit sie auch beim Inference samplen.
    (Andere Layer, z.B. BatchNorm, bleiben im eval()-Modus)
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    X: torch.Tensor,
    n_samples: int = 50,
    device: torch.device | None = None,
    max_rul: float | None = None,
):
    """
    Generische MC-Dropout-Inferenz:

    Args:
        model: PyTorch-Modell mit mindestens einem Dropout-Layer.
        X:     Input-Tensor shape (N, seq_len, n_features) oder (N, D).
        n_samples: Anzahl der Monte-Carlo-Samples.
        device:    Torch-Device (wird automatisch gewählt, falls None).
        max_rul:   Optional, Clamping der Mittelwerte auf [0, max_rul].

    Returns:
        y_mean: np.ndarray, shape (N,)
        y_std:  np.ndarray, shape (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    X = X.to(device)

    # Wir starten im eval(), schalten aber die Dropout-Layer gezielt auf train().
    model.eval()
    preds = []

    for _ in range(n_samples):
        enable_mc_dropout(model)
        with torch.no_grad():
            y = model(X)                   # (N, 1) oder (N,)
        y = y.view(-1).cpu().numpy()
        preds.append(y)

    preds = np.stack(preds, axis=0)        # (n_samples, N)
    y_mean = preds.mean(axis=0)
    y_std = preds.std(axis=0)

    if max_rul is not None:
        y_mean = np.minimum(y_mean, max_rul)

    return y_mean, y_std