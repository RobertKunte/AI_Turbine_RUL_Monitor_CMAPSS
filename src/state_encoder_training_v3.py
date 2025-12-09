from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.models.transformer_state_encoder_v3 import TransformerStateEncoderV3
from src.eol_full_lstm import build_full_eol_sequences_from_df
from src.additional_features import group_feature_columns


def build_hi_from_rul(
    rul_array: np.ndarray,
    max_rul: float,
    plateau_threshold: float,
    eol_threshold: float,
) -> np.ndarray:
    """
    Simple analytic HI:
        - HI = 1.0 for RUL >= plateau_threshold
        - HI = 0.0 for RUL <= eol_threshold
        - linearly decreasing between.

    Input:
        rul_array: RUL in cycles (numpy, any shape)
    """
    hi = np.ones_like(rul_array, dtype=np.float32)

    mask_eol = rul_array <= eol_threshold
    mask_plateau = rul_array >= plateau_threshold
    mask_mid = (~mask_eol) & (~mask_plateau)

    hi[mask_eol] = 0.0
    # Linear between plateau_threshold and eol_threshold
    denom = max(plateau_threshold - eol_threshold, 1e-6)
    hi[mask_mid] = (rul_array[mask_mid] - eol_threshold) / denom
    hi[mask_mid] = np.clip(hi[mask_mid], 0.0, 1.0)

    return hi


def _engine_based_split(
    X: torch.Tensor,
    y_rul: torch.Tensor,
    cond_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    train_frac: float,
    random_state: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Simple engine-based train/val split using unit_ids.
    """
    rng = np.random.RandomState(random_state)
    unique_units = np.unique(unit_ids.numpy())
    rng.shuffle(unique_units)

    n_total = len(unique_units)
    n_train = int(train_frac * n_total)
    if n_total > 1 and n_train == 0:
        n_train = 1
    train_units = unique_units[:n_train]
    val_units = unique_units[n_train:]

    train_mask = np.isin(unit_ids.numpy(), train_units)
    val_mask = np.isin(unit_ids.numpy(), val_units)

    X_tr = X[train_mask]
    y_rul_tr = y_rul[train_mask]
    cond_tr = cond_ids[train_mask]

    X_val = X[val_mask]
    y_rul_val = y_rul[val_mask]
    cond_val = cond_ids[val_mask]

    return (X_tr, y_rul_tr, cond_tr), (X_val, y_rul_val, cond_val)


def build_state_encoder_dataloaders_fd004(cfg: SimpleNamespace):
    """
    Build train/val dataloaders for the state encoder on FD004.

    Uses the same ms+DT feature set (e.g., 349 features) as the EOL Transformer encoder.
    """
    # df_train_fe / df_test_fe + feature_cols werden idealerweise schon in run_experiments
    # mit genau derselben Pipeline wie für die ms+DT-Encoder erzeugt.
    # Für die Standalone-Variante erwarten wir hier, dass cfg.data.df_train_fe,
    # cfg.data.feature_cols etc. bereits gesetzt sind.
    if not hasattr(cfg.data, "df_train_fe") or cfg.data.df_train_fe is None:
        raise ValueError(
            "[build_state_encoder_dataloaders_fd004] cfg.data.df_train_fe is not set. "
            "Please ensure the feature pipeline is run before calling train_state_encoder_v3."
        )

    df_train = cfg.data.df_train_fe
    feature_cols = cfg.data.feature_cols
    past_len = int(cfg.data.past_len)
    max_rul = float(cfg.data.max_rul)

    # Build full EOL-style sliding windows (like EOL encoder / diagnostics)
    X_full, y_rul_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
        df=df_train,
        feature_cols=feature_cols,
        past_len=past_len,
        max_rul=int(max_rul),
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        rul_col="RUL",
        cond_col="ConditionID",
    )

    # Engine-based train/val split
    (X_tr, y_rul_tr, cond_ids_tr), (X_val, y_rul_val, cond_ids_val) = _engine_based_split(
        X_full,
        y_rul_full,
        cond_ids_full,
        unit_ids_full,
        train_frac=cfg.data.train_frac,
        random_state=cfg.seed,
    )

    # Derive continuous condition feature indices from feature_cols via group_feature_columns
    groups = group_feature_columns(feature_cols)
    cond_cols = groups.get("cond", [])
    cond_feature_idxs = [feature_cols.index(c) for c in cond_cols]

    def extract_cond_seq(X: torch.Tensor) -> Optional[torch.Tensor]:
        if not cond_feature_idxs:
            return None
        # X: [N, T, F]
        return X[:, :, cond_feature_idxs]

    cond_tr_seq = extract_cond_seq(X_tr)
    cond_val_seq = extract_cond_seq(X_val)

    # Normalize RUL to [0,1]
    y_rul_tr_norm = y_rul_tr / max_rul
    y_rul_val_norm = y_rul_val / max_rul

    # Analytic HI labels from RUL
    hi_tr_np = build_hi_from_rul(
        y_rul_tr.numpy(),
        max_rul=max_rul,
        plateau_threshold=cfg.hi.plateau_threshold,
        eol_threshold=cfg.hi.eol_threshold,
    )
    hi_val_np = build_hi_from_rul(
        y_rul_val.numpy(),
        max_rul=max_rul,
        plateau_threshold=cfg.hi.plateau_threshold,
        eol_threshold=cfg.hi.eol_threshold,
    )

    # To tensors
    X_tr_t = X_tr.float()
    X_val_t = X_val.float()
    y_rul_tr_t = y_rul_tr_norm.float()
    y_rul_val_t = y_rul_val_norm.float()
    hi_tr_t = torch.from_numpy(hi_tr_np).float()
    hi_val_t = torch.from_numpy(hi_val_np).float()

    if cond_tr_seq is not None:
        cond_tr_t = cond_tr_seq.float()
        cond_val_t = cond_val_seq.float()
        train_ds = TensorDataset(X_tr_t, cond_tr_t, y_rul_tr_t, hi_tr_t)
        val_ds = TensorDataset(X_val_t, cond_val_t, y_rul_val_t, hi_val_t)
    else:
        cond_tr_t = cond_val_t = None
        train_ds = TensorDataset(X_tr_t, y_rul_tr_t, hi_tr_t)
        val_ds = TensorDataset(X_val_t, y_rul_val_t, hi_val_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # Expose cond_feature_idxs back to cfg if needed later
    cfg.data.cond_feature_idxs = cond_feature_idxs

    return train_loader, val_loader, max_rul


def train_state_encoder_v3(cfg: SimpleNamespace) -> Dict[str, Any]:
    """
    Train TransformerStateEncoderV3 on FD004 (or similar FD) with:
      - main HI loss (MSE),
      - auxiliary RUL loss (MSE on normalized RUL with small weight).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, max_rul = build_state_encoder_dataloaders_fd004(cfg)

    input_dim = int(cfg.model.input_dim)
    use_cond_encoder = bool(cfg.model.use_cond_encoder)
    # Infer cond_in_dim from detected Cond_* feature indices if not explicitly set
    if use_cond_encoder:
        if hasattr(cfg.data, "cond_feature_idxs") and cfg.data.cond_feature_idxs:
            cond_in_dim = len(cfg.data.cond_feature_idxs)
        elif getattr(cfg.model, "cond_in_dim", None) is not None:
            cond_in_dim = int(cfg.model.cond_in_dim)
        else:
            cond_in_dim = 0
    else:
        cond_in_dim = 0

    model = TransformerStateEncoderV3(
        input_dim=input_dim,
        d_model=int(cfg.model.d_model),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        dropout=float(cfg.model.dropout),
        use_cond_encoder=use_cond_encoder,
        cond_in_dim=cond_in_dim,
        cond_emb_dim=int(cfg.model.d_model),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_state = None
    patience = int(cfg.training.patience)
    patience_counter = 0

    for epoch in range(1, int(cfg.training.num_epochs) + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                x_b, cond_b, rul_b, hi_b = batch
                cond_b = cond_b.to(device)
            else:
                x_b, rul_b, hi_b = batch
                cond_b = None

            x_b = x_b.to(device)
            rul_b = rul_b.to(device).view(-1)
            hi_b = hi_b.to(device).view(-1)

            optimizer.zero_grad()

            # Mixed-precision only if CUDA is available (use legacy autocast API)
            use_amp = device.type == "cuda"
            with autocast(enabled=use_amp):
                rul_raw, hi_raw, _ = model(x_b, cond_seq=cond_b)

                hi_pred = torch.sigmoid(hi_raw.view(-1))
                rul_pred_norm = torch.sigmoid(rul_raw.view(-1))  # in [0,1]

                hi_loss = F.mse_loss(hi_pred, hi_b)
                rul_loss = F.mse_loss(rul_pred_norm, rul_b)

                loss = (
                    float(cfg.loss.hi_weight) * hi_loss
                    + float(cfg.loss.rul_weight) * rul_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(1, num_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x_b, cond_b, rul_b, hi_b = batch
                    cond_b = cond_b.to(device)
                else:
                    x_b, rul_b, hi_b = batch
                    cond_b = None

                x_b = x_b.to(device)
                rul_b = rul_b.to(device).view(-1)
                hi_b = hi_b.to(device).view(-1)

                rul_raw, hi_raw, _ = model(x_b, cond_seq=cond_b)

                hi_pred = torch.sigmoid(hi_raw.view(-1))
                rul_pred_norm = torch.sigmoid(rul_raw.view(-1))

                hi_loss = F.mse_loss(hi_pred, hi_b)
                rul_loss = F.mse_loss(rul_pred_norm, rul_b)
                loss = (
                    float(cfg.loss.hi_weight) * hi_loss
                    + float(cfg.loss.rul_weight) * rul_loss
                )

                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(1, val_batches)

        print(
            f"[StateEncoderV3] Epoch {epoch}/{cfg.training.num_epochs} "
            f"- train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "max_rul": max_rul,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[StateEncoderV3] Early stopping at epoch {epoch}")
                break

    # Save best model
    results_dir = getattr(cfg.paths, "result_dir", f"results/fd004/{cfg.experiment_name}")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f"{cfg.experiment_name}_state_encoder_v3.pt")
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[StateEncoderV3] Saved best model to {ckpt_path}")
    else:
        print("[StateEncoderV3] WARNING: no best_state saved.")

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": ckpt_path,
    }


