from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.eol_full_lstm import build_full_eol_sequences_from_df
from src.additional_features import group_feature_columns
from src.models.transformer_state_encoder_v3_physics import TransformerStateEncoderV3_Physics


def build_physics_hi_from_components(
    resid_norm: np.ndarray,
    egt_drift: np.ndarray,
    hpc_eff_proxy: np.ndarray,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> np.ndarray:
    """
    Build physics-informed HI from residual norm + EGT drift + HPC efficiency proxy.

    Args:
        resid_norm:      [N] residual norm (>=0)
        egt_drift:       [N] Exhaust Gas Temperature drift proxy
        hpc_eff_proxy:   [N] HPC efficiency proxy (higher ~ healthier)

    Returns:
        hi: [N] in [0,1]

    Simple formula:
        HI_raw = 1 - w1 * resid_norm_norm - w2 * egt_norm - w3 * hpc_loss_norm
        HI = clip(HI_raw, 0, 1)

    where *_norm are clamped to [0,1] using robust scales.
    """
    eps = 1e-6

    # Robust scales (95%-Quantile) to normalise to ~[0,1]
    def _norm(x: np.ndarray) -> np.ndarray:
        scale = np.quantile(np.abs(x), 0.95) + eps
        x_n = np.clip(np.abs(x) / scale, 0.0, 1.0)
        return x_n.astype(np.float32)

    resid_norm_n = _norm(resid_norm)
    # For EGT drift we care about positive overheating; negative values are benign.
    egt_pos = np.maximum(egt_drift, 0.0)
    egt_norm = _norm(egt_pos)
    # HPC efficiency loss: lower efficiency = more loss. Assume proxy ~1.0 healthy -> 0.0 degraded.
    # We treat (1 - proxy) as "loss" and normalise.
    hpc_loss = 1.0 - hpc_eff_proxy
    hpc_loss = np.maximum(hpc_loss, 0.0)
    hpc_loss_norm = _norm(hpc_loss)

    hi_raw = 1.0 - (w1 * resid_norm_n + w2 * egt_norm + w3 * hpc_loss_norm)
    hi = np.clip(hi_raw, 0.0, 1.0).astype(np.float32)
    return hi


def _engine_based_split_physics(
    X: torch.Tensor,
    y_rul: torch.Tensor,
    hi_phys: torch.Tensor,
    unit_ids: torch.Tensor,
    train_frac: float,
    random_state: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Engine-based train/val split using unit_ids.
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
    hi_tr = hi_phys[train_mask]

    X_val = X[val_mask]
    y_rul_val = y_rul[val_mask]
    hi_val = hi_phys[val_mask]

    return (X_tr, y_rul_tr, hi_tr, unit_ids[train_mask]), (X_val, y_rul_val, hi_val, unit_ids[val_mask])


def build_state_encoder_dataloaders_fd004_physics(cfg: SimpleNamespace):
    """
    Build train/val dataloaders for the physics-informed state encoder on FD004.

    Uses the same ms+DT feature set as the Transformer ms+DT encoder, but:
      - builds physics-informed HI labels from the precomputed HI_phys_final
        trajectory (see src.data.physics_hi.add_physics_hi),
      - does NOT use integer ConditionIDs (only continuous Cond_* features).

    For backwards compatibility, if HI_phys_final is not present in the
    feature-engineered DataFrame, the older residual/EGT/HPC-based HI
    construction is used as a fallback.
    """
    if not hasattr(cfg.data, "df_train_fe") or cfg.data.df_train_fe is None:
        raise ValueError(
            "[build_state_encoder_dataloaders_fd004_physics] cfg.data.df_train_fe is not set. "
            "Please ensure the feature pipeline is run before calling train_state_encoder_v3_physics."
        )

    df_train = cfg.data.df_train_fe
    feature_cols: List[str] = cfg.data.feature_cols
    past_len = int(cfg.data.past_len)
    max_rul = float(cfg.data.max_rul)

    # Build full sliding windows (no scaling here; we rely on engineered features only)
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

    # Group features into blocks to find Cond_* feature indices (for the
    # continuous condition encoder). Residuals are no longer needed here for
    # HI label construction if HI_phys_final is present.
    groups = group_feature_columns(feature_cols)
    cond_cols = groups.get("cond", [])
    cond_feature_idxs = [feature_cols.index(c) for c in cond_cols]

    # ------------------------------------------------------------------
    # Physics-based HI labels: prefer the new hybrid target if present
    # (HI_target_hybrid from add_physics_hi_v2). As a fallback, we use
    # HI_phys_v2 or, if neither is available, the legacy component-based HI.
    # ------------------------------------------------------------------
    if "HI_target_hybrid" in df_train.columns:
        hi_col = "HI_target_hybrid"
    elif "HI_phys_v2" in df_train.columns:
        hi_col = "HI_phys_v2"
    else:
        hi_col = None

    unit_col = "UnitNumber"
    cycle_col = "TimeInCycles"

    if hi_col is not None:
        hi_values: List[float] = []
        unit_ids_df = df_train[unit_col].unique()

        for uid in unit_ids_df:
            df_u = (
                df_train[df_train[unit_col] == uid]
                .sort_values(cycle_col)
                .reset_index(drop=True)
            )
            if len(df_u) < past_len:
                continue

            hi_seq = df_u[hi_col].to_numpy(dtype=np.float32)
            for i in range(past_len - 1, len(df_u)):
                hi_values.append(float(hi_seq[i]))

        if len(hi_values) != X_full.shape[0]:
            raise RuntimeError(
                "[build_state_encoder_dataloaders_fd004_physics] Mismatch between "
                f"number of HI labels derived from '{hi_col}' and number of "
                f"sequences: got {len(hi_values)} vs X_full.shape[0]={X_full.shape[0]}"
            )
        hi_phys = torch.from_numpy(np.array(hi_values, dtype=np.float32))  # [N]
    else:
        # Legacy fallback: build HI from residual norm + EGT_Drift + Effizienz_HPC_Proxy
        print(
            "[build_state_encoder_dataloaders_fd004_physics] WARNING: 'HI_phys_final' "
            "not found in df_train_fe. Falling back to legacy component-based HI."
        )
        residual_cols = groups.get("residual", [])
        residual_idxs = [feature_cols.index(c) for c in residual_cols]

        hpc_col = "Effizienz_HPC_Proxy"
        egt_col = "EGT_Drift"
        if hpc_col not in feature_cols or egt_col not in feature_cols:
            raise KeyError(
                f"[build_state_encoder_dataloaders_fd004_physics] Required physics columns "
                f"'{hpc_col}' and '{egt_col}' not found in feature_cols."
            )
        hpc_idx = feature_cols.index(hpc_col)
        egt_idx = feature_cols.index(egt_col)

        X_last = X_full[:, -1, :]  # [N,F]
        if residual_idxs:
            resid_last = X_last[:, residual_idxs]  # [N,R]
            resid_norm = torch.sqrt(torch.mean(resid_last ** 2, dim=-1)).cpu().numpy()  # [N]
        else:
            resid_norm = np.zeros(X_last.shape[0], dtype=np.float32)

        egt_drift = X_last[:, egt_idx].cpu().numpy()
        hpc_eff = X_last[:, hpc_idx].cpu().numpy()

        hi_phys_np = build_physics_hi_from_components(
            resid_norm=resid_norm,
            egt_drift=egt_drift,
            hpc_eff_proxy=hpc_eff,
            w1=0.5,
            w2=0.3,
            w3=0.2,
        )
        hi_phys = torch.from_numpy(hi_phys_np)  # [N]

    # Engine-based train/val split (no cond_ids usage here)
    (X_tr, y_rul_tr, hi_tr, _), (X_val, y_rul_val, hi_val, _) = _engine_based_split_physics(
        X_full,
        y_rul_full,
        hi_phys,
        unit_ids_full,
        train_frac=cfg.data.train_frac,
        random_state=cfg.seed,
    )

    # Normalised RUL (auxiliary label)
    y_rul_tr_norm = y_rul_tr / max_rul
    y_rul_val_norm = y_rul_val / max_rul

    # Continuous condition sequences (Cond_*) from full X
    def extract_cond_seq(X: torch.Tensor) -> Optional[torch.Tensor]:
        if not cond_feature_idxs:
            return None
        return X[:, :, cond_feature_idxs]

    cond_tr_seq = extract_cond_seq(X_tr)
    cond_val_seq = extract_cond_seq(X_val)

    # Tensors for dataloader
    X_tr_t = X_tr.float()
    X_val_t = X_val.float()
    y_rul_tr_t = y_rul_tr_norm.float()
    y_rul_val_t = y_rul_val_norm.float()
    hi_tr_t = hi_tr.float()
    hi_val_t = hi_val.float()

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

    # Expose cond_feature_idxs for potential later use
    cfg.data.cond_feature_idxs = cond_feature_idxs

    return train_loader, val_loader, max_rul


def train_state_encoder_v3_physics(cfg: SimpleNamespace) -> Dict[str, Any]:
    """
    Train TransformerStateEncoderV3_Physics on FD004 (or similar FD) with:
      - main HI loss (MSE) using physics-informed HI labels,
      - auxiliary RUL loss (MSE on normalized RUL with small weight 0.1),
      - no EOL / monotonicity / NASA losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, max_rul = build_state_encoder_dataloaders_fd004_physics(cfg)

    input_dim = int(cfg.model.input_dim)
    # Infer cond_in_dim from Cond_* indices
    if hasattr(cfg.data, "cond_feature_idxs") and cfg.data.cond_feature_idxs:
        cond_in_dim = len(cfg.data.cond_feature_idxs)
    else:
        cond_in_dim = int(getattr(cfg.model, "cond_in_dim", 0) or 0)

    # Optional cumulative damage head configuration
    use_damage_head = bool(getattr(cfg.model, "use_damage_head", False))
    L_ref = float(getattr(cfg.model, "L_ref", 300.0))
    alpha_base = float(getattr(cfg.model, "alpha_base", 0.1))

    model = TransformerStateEncoderV3_Physics(
        input_dim=input_dim,
        cond_in_dim=cond_in_dim,
        d_model=int(cfg.model.d_model),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        dropout=float(cfg.model.dropout),
        use_damage_head=use_damage_head,
        L_ref=L_ref,
        alpha_base=alpha_base,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = GradScaler()

    # Optional physics-HI scalers (trained in the feature pipeline) for later
    # diagnostics / inference. If present, we persist them alongside the model.
    hi_scalers = getattr(getattr(cfg, "paths", SimpleNamespace()), "hi_scalers", None)

    # Loss weights and optional RUL–HI alignment (all configurable via cfg.loss)
    loss_cfg = getattr(cfg, "loss", SimpleNamespace())
    hi_weight = float(getattr(loss_cfg, "hi_weight", 1.0))
    rul_weight = float(getattr(loss_cfg, "rul_weight", 0.1))
    align_weight = float(getattr(loss_cfg, "align_weight", 0.0))
    use_align = bool(
        getattr(loss_cfg, "use_rul_hi_alignment_loss", False) or align_weight > 0.0
    )

    # Damage-head specific weights (only used if use_damage_head=True)
    hi_phys_weight = float(getattr(loss_cfg, "hi_phys_weight", 1.0))
    hi_aux_weight = float(getattr(loss_cfg, "hi_aux_weight", 0.3))
    rul_weight_damage = float(getattr(loss_cfg, "rul_weight", 0.3))
    mono_weight = float(getattr(loss_cfg, "mono_weight", 0.01))
    smooth_weight = float(getattr(loss_cfg, "smooth_weight", 0.01))

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
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
            hi_b = hi_b.to(device)

            optimizer.zero_grad()

            use_amp = device.type == "cuda"
            with autocast(enabled=use_amp):
                if use_damage_head:
                    outputs = model(x_b, cond_seq=cond_b, return_dict=True)
                    hi_raw = outputs["hi_raw"]
                    rul_raw = outputs["rul_raw"]
                    hi_seq_phys = outputs["hi_seq_phys"]

                    hi_pred_scalar = torch.sigmoid(hi_raw.view(-1))
                    rul_pred = torch.sigmoid(rul_raw.view(-1))

                    if hi_seq_phys is None:
                        raise RuntimeError(
                            "[train_state_encoder_v3_physics] hi_seq_phys is None "
                            "while use_damage_head=True."
                        )
                    # Ensure hi_target_seq shape [B, T]
                    if hi_b.dim() == 1:
                        hi_target_seq = hi_b.view(-1, 1).expand(-1, hi_seq_phys.size(1))
                        hi_scalar_target = hi_b.view(-1)
                    else:
                        hi_target_seq = hi_b
                        hi_scalar_target = hi_b.mean(dim=-1)

                    hi_loss_phys = F.mse_loss(hi_seq_phys, hi_target_seq)
                    hi_loss_aux = F.mse_loss(hi_pred_scalar, hi_scalar_target)
                    rul_loss = F.mse_loss(rul_pred, rul_b)

                    # Monotonicity and smoothness penalties on HI_phys trajectory
                    hi_diffs = hi_seq_phys[:, 1:] - hi_seq_phys[:, :-1]  # [B, T-1]
                    mono_penalty = F.relu(hi_diffs).pow(2).mean()
                    smooth_penalty = hi_diffs.abs().mean()

                    loss = (
                        hi_phys_weight * hi_loss_phys
                        + hi_aux_weight * hi_loss_aux
                        + rul_weight_damage * rul_loss
                        + mono_weight * mono_penalty
                        + smooth_weight * smooth_penalty
                    )
                else:
                    hi_b_flat = hi_b.view(-1)
                    hi_raw, rul_raw, _ = model(x_b, cond_seq=cond_b)

                    hi_pred = torch.sigmoid(hi_raw.view(-1))
                    rul_pred = torch.sigmoid(rul_raw.view(-1))

                    hi_loss = F.mse_loss(hi_pred, hi_b_flat)
                    rul_loss = F.mse_loss(rul_pred, rul_b)

                    loss = hi_weight * hi_loss + rul_weight * rul_loss

                    if use_align and align_weight > 0.0:
                        # Simple physics-inspired mapping: HI ≈ 1 - RUL_norm
                        hi_from_rul = 1.0 - rul_b
                        align_loss = F.mse_loss(hi_pred, hi_from_rul)
                        loss = loss + align_weight * align_loss

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
                hi_b = hi_b.to(device)

                if use_damage_head:
                    outputs = model(x_b, cond_seq=cond_b, return_dict=True)
                    hi_raw = outputs["hi_raw"]
                    rul_raw = outputs["rul_raw"]
                    hi_seq_phys = outputs["hi_seq_phys"]

                    hi_pred_scalar = torch.sigmoid(hi_raw.view(-1))
                    rul_pred = torch.sigmoid(rul_raw.view(-1))

                    if hi_seq_phys is None:
                        raise RuntimeError(
                            "[train_state_encoder_v3_physics] hi_seq_phys is None "
                            "while use_damage_head=True (val loop)."
                        )
                    if hi_b.dim() == 1:
                        hi_target_seq = hi_b.view(-1, 1).expand(-1, hi_seq_phys.size(1))
                        hi_scalar_target = hi_b.view(-1)
                    else:
                        hi_target_seq = hi_b
                        hi_scalar_target = hi_b.mean(dim=-1)

                    hi_loss_phys = F.mse_loss(hi_seq_phys, hi_target_seq)
                    hi_loss_aux = F.mse_loss(hi_pred_scalar, hi_scalar_target)
                    rul_loss = F.mse_loss(rul_pred, rul_b)

                    hi_diffs = hi_seq_phys[:, 1:] - hi_seq_phys[:, :-1]
                    mono_penalty = F.relu(hi_diffs).pow(2).mean()
                    smooth_penalty = hi_diffs.abs().mean()

                    batch_loss = (
                        hi_phys_weight * hi_loss_phys
                        + hi_aux_weight * hi_loss_aux
                        + rul_weight_damage * rul_loss
                        + mono_weight * mono_penalty
                        + smooth_weight * smooth_penalty
                    )
                else:
                    hi_b_flat = hi_b.view(-1)
                    hi_raw, rul_raw, _ = model(x_b, cond_seq=cond_b)

                    hi_pred = torch.sigmoid(hi_raw.view(-1))
                    rul_pred = torch.sigmoid(rul_raw.view(-1))

                    hi_loss = F.mse_loss(hi_pred, hi_b_flat)
                    rul_loss = F.mse_loss(rul_pred, rul_b)

                    batch_loss = hi_weight * hi_loss + rul_weight * rul_loss

                    if use_align and align_weight > 0.0:
                        hi_from_rul = 1.0 - rul_b
                        align_loss = F.mse_loss(hi_pred, hi_from_rul)
                        batch_loss = batch_loss + align_weight * align_loss

                val_loss += batch_loss.item()
                val_batches += 1

        val_loss /= max(1, val_batches)

        print(
            f"[StateEncoderV3_Physics] Epoch {epoch}/{cfg.training.num_epochs} "
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
                print(f"[StateEncoderV3_Physics] Early stopping at epoch {epoch}")
                break

    # Save best model
    results_dir = getattr(cfg.paths, "result_dir", f"results/fd004/{cfg.experiment_name}")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f"{cfg.experiment_name}_state_encoder_v3_physics.pt")
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[StateEncoderV3_Physics] Saved best model to {ckpt_path}")
        # Persist HI scalers if available
        if isinstance(hi_scalers, dict) and hi_scalers:
            import pickle

            hi_scaler_path = os.path.join(results_dir, "hi_phys_scalers.pkl")
            with open(hi_scaler_path, "wb") as f:
                pickle.dump(hi_scalers, f)
            print(f"[StateEncoderV3_Physics] Saved HI_phys scalers to {hi_scaler_path}")
    else:
        print("[StateEncoderV3_Physics] WARNING: no best_state saved.")

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": ckpt_path,
    }


