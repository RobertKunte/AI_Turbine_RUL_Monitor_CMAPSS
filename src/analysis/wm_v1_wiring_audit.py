from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data_loading import load_cmapps_subset
from src.experiment_configs import get_experiment_by_name
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.models.transformer_world_model_v1 import TransformerWorldModelV1
from src.tools.x_scaler import load_scaler, transform_x, clip_x


@dataclass(frozen=True)
class AuditPaths:
    results_dir: Path
    checkpoint_path: Path
    encoder_summary_path: Path


def _infer_encoder_experiment_from_ckpt_path(ckpt_path: str) -> Optional[str]:
    """
    Given something like:
      results/fd004/<encoder_experiment>/eol_full_lstm_best_<...>.pt
    return <encoder_experiment>.
    """
    try:
        p = Path(ckpt_path)
        parts = [x for x in p.parts]
        # Find ".../results/<dataset>/<exp>/(file)"
        if "results" in parts:
            i = parts.index("results")
            if i + 2 < len(parts):
                return parts[i + 2]
    except Exception:
        pass
    return None


def _resolve_paths(dataset: str, experiment: str, cfg: dict) -> AuditPaths:
    results_dir = Path("results") / dataset.lower() / experiment
    # Prefer the canonical WM-V1 best checkpoint name
    cand = results_dir / f"transformer_world_model_v1_best_{experiment}.pt"
    if not cand.exists():
        # Fallback: any "*best*.pt"
        pts = sorted(results_dir.glob("*best*.pt"))
        if pts:
            cand = pts[0]
    if not cand.exists():
        raise FileNotFoundError(f"WM-V1 checkpoint not found under {results_dir}")

    encoder_ckpt = str(cfg.get("world_model_params", {}).get("encoder_checkpoint") or "")
    enc_exp = _infer_encoder_experiment_from_ckpt_path(encoder_ckpt) or ""
    enc_summary = Path("results") / dataset.lower() / enc_exp / "summary.json"
    if not enc_summary.exists():
        raise FileNotFoundError(
            "Could not find encoder summary.json needed to reconstruct feature pipeline:\n"
            f"- encoder_checkpoint={encoder_ckpt}\n"
            f"- expected={enc_summary}"
        )
    return AuditPaths(results_dir=results_dir, checkpoint_path=cand, encoder_summary_path=enc_summary)


def _select_feature_cols(df_train: "Any") -> List[str]:
    feature_cols = [
        c
        for c in df_train.columns
        if c not in ["UnitNumber", "TimeInCycles", "RUL", "RUL_raw", "MaxTime", "ConditionID"]
    ]
    from src.feature_safety import remove_rul_leakage

    feature_cols, _ = remove_rul_leakage(feature_cols)
    feature_cols = [c for c in feature_cols if c not in ["HI_phys_final", "HI_target_hybrid", "HI_phys_v2", "HI_phys_v3"]]
    return feature_cols


def _rebuild_feature_pipeline_from_encoder_summary(
    dataset: str,
    encoder_summary_path: Path,
    *,
    max_rul: float = 125.0,
) -> Tuple["Any", "Any", np.ndarray, List[str], Dict[str, Any]]:
    """
    Rebuild the ms+DT + residual + Cond_* pipeline from the encoder experiment summary.json.
    This mirrors `src/rul_decoder_training_v1.prepare_fd004_ms_dt_encoder_data` but returns dataframes.
    """
    with open(encoder_summary_path, "r") as f:
        summary_cfg = json.load(f)

    # Raw CMAPSS
    df_train, df_test, y_test_true = load_cmapps_subset(
        dataset,
        max_rul=None,
        clip_train=False,
        clip_test=True,
    )

    from src.additional_features import (
        PhysicsFeatureConfig,
        ResidualFeatureConfig,
        TemporalFeatureConfig,
        FeatureConfig,
        create_physical_features,
        create_all_features,
        create_twin_features,
        build_condition_features,
    )

    name_lower = str(summary_cfg.get("experiment_name", "")).lower()
    is_phase4_residual = (("phase4" in name_lower or "phase5" in name_lower) and "residual" in name_lower) or (
        "residual" in name_lower or "resid" in name_lower
    )
    physics_config = PhysicsFeatureConfig(
        use_core=True,
        use_extended=False,
        use_residuals=is_phase4_residual,
        use_temporal_on_physics=False,
        residual=ResidualFeatureConfig(
            enabled=is_phase4_residual,
            mode="per_engine",
            baseline_len=30,
            include_original=True,
        )
        if is_phase4_residual
        else ResidualFeatureConfig(enabled=False),
    )

    phys_opts = summary_cfg.get("phys_features", {})
    use_phys_condition_vec = bool(phys_opts.get("use_condition_vector", False))
    use_twin_features = bool(phys_opts.get("use_twin_features", phys_opts.get("use_digital_twin_residuals", False)))
    twin_baseline_len = int(phys_opts.get("twin_baseline_len", 30))
    condition_vector_version = int(phys_opts.get("condition_vector_version", 2))

    features_cfg = summary_cfg.get("features", {})
    ms_cfg = features_cfg.get("multiscale", {})
    use_temporal_features = bool(features_cfg.get("use_multiscale_features", True))
    windows_short = tuple(ms_cfg.get("windows_short", (5, 10)))
    windows_medium = tuple(ms_cfg.get("windows_medium", ()))
    windows_long = tuple(ms_cfg.get("windows_long", (30,)))
    combined_long = windows_medium + windows_long

    temporal_cfg = TemporalFeatureConfig(
        base_cols=None,
        short_windows=windows_short,
        long_windows=combined_long if combined_long else (30,),
        add_rolling_mean=True,
        add_rolling_std=False,
        add_trend=True,
        add_delta=True,
        delta_lags=(5, 10),
    )
    feature_config = FeatureConfig(
        add_physical_core=True,
        add_temporal_features=use_temporal_features,
        temporal=temporal_cfg,
    )

    # 1) physics/residuals
    df_train = create_physical_features(df_train, physics_config, "UnitNumber", "TimeInCycles")
    df_test = create_physical_features(df_test, physics_config, "UnitNumber", "TimeInCycles")

    # 2) Cond_* continuous condition
    if use_phys_condition_vec:
        df_train = build_condition_features(df_train, unit_col="UnitNumber", cycle_col="TimeInCycles", version=condition_vector_version)
        df_test = build_condition_features(df_test, unit_col="UnitNumber", cycle_col="TimeInCycles", version=condition_vector_version)

    # 3) Twin_/Resid_ from twin model
    if use_twin_features:
        df_train, twin_model = create_twin_features(
            df_train,
            unit_col="UnitNumber",
            cycle_col="TimeInCycles",
            baseline_len=twin_baseline_len,
            condition_vector_version=condition_vector_version,
        )
        df_test = twin_model.add_twin_and_residuals(df_test)

    # 4) multiscale temporal
    df_train = create_all_features(df_train, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)
    df_test = create_all_features(df_test, "UnitNumber", "TimeInCycles", feature_config, inplace=False, physics_config=physics_config)

    feature_cols = _select_feature_cols(df_train)
    # Ensure deterministic ordering like run_experiments (same list order from df columns)
    feature_cols = list(feature_cols)

    # Keep a few key cfg bits for reporting
    info = {
        "encoder_summary_path": str(encoder_summary_path),
        "use_phys_condition_vec": use_phys_condition_vec,
        "use_twin_features": use_twin_features,
        "use_temporal_features": use_temporal_features,
        "twin_baseline_len": twin_baseline_len,
        "condition_vector_version": condition_vector_version,
        "windows_short": list(windows_short),
        "windows_medium": list(windows_medium),
        "windows_long": list(windows_long),
        "max_rul_used": float(max_rul),
    }
    return df_train, df_test, np.asarray(y_test_true, dtype=float), feature_cols, info


def _grad_norm_from_named_params(named_params: List[Tuple[str, torch.nn.Parameter]], prefix: str) -> float:
    s = 0.0
    for n, p in named_params:
        if not n.startswith(prefix):
            continue
        if (p.grad is None) or (not p.requires_grad):
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            continue
        s += float((g * g).sum().cpu())
    return float(np.sqrt(max(s, 0.0)))


def _grad_norm_excluding_prefix(named_params: List[Tuple[str, torch.nn.Parameter]], prefix: str) -> float:
    s = 0.0
    for n, p in named_params:
        if n.startswith(prefix):
            continue
        if (p.grad is None) or (not p.requires_grad):
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            continue
        s += float((g * g).sum().cpu())
    return float(np.sqrt(max(s, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description="WM-V1 wiring audit (1 train batch + 1 test batch).")
    ap.add_argument("--experiment", type=str, default="fd004_wm_v1_infwin_wiringcheck_k0")
    ap.add_argument("--dataset", type=str, default="FD004")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None, help="Optional explicit output json path")
    args = ap.parse_args()

    dataset = str(args.dataset)
    exp = str(args.experiment)
    device = torch.device(args.device)

    cfg = get_experiment_by_name(exp)
    paths = _resolve_paths(dataset, exp, cfg)

    df_train, df_test, y_test_true, feature_cols, feat_info = _rebuild_feature_pipeline_from_encoder_summary(
        dataset, paths.encoder_summary_path, max_rul=float(cfg.get("world_model_params", {}).get("max_rul", 125.0))
    )

    # --- Rebuild training windows exactly like WM-V1 trainer ---
    from sklearn.preprocessing import StandardScaler
    from src.data.windowing import WindowConfig, TargetConfig, build_sliding_windows

    world_model_params = cfg.get("world_model_params", {})
    past_len = int(world_model_params.get("past_len", 30))
    horizon = int(world_model_params.get("future_horizon", world_model_params.get("horizon", 30)))
    max_rul = float(world_model_params.get("max_rul", 125.0))

    # Sensors for teacher forcing (even if sensor_loss_weight==0)
    sensor_cols = [c for c in feature_cols if c.startswith("Sensor")]
    target_sensor_cols = sensor_cols[: int(world_model_params.get("num_sensors_out", 21))] if sensor_cols else []

    sensor_scaler = StandardScaler()
    if target_sensor_cols:
        sensor_values = df_train[target_sensor_cols].to_numpy(dtype=np.float32, copy=True)
        sensor_scaler.fit(sensor_values)
        df_train_scaled = df_train.copy()
        df_train_scaled[target_sensor_cols] = sensor_scaler.transform(sensor_values)
    else:
        df_train_scaled = df_train.copy()

    cond_cols = [c for c in feature_cols if c.startswith("Cond_")]
    future_cols = target_sensor_cols + cond_cols
    wc = WindowConfig(past_len=past_len, horizon=horizon, stride=1, require_full_horizon=False, pad_mode="clamp")
    tc = TargetConfig(max_rul=int(max_rul), cap_targets=True, eol_target_mode="future0", clip_eval_y_true=False)

    built = build_sliding_windows(
        df=df_train_scaled,
        feature_cols=feature_cols,
        target_col="RUL",
        future_feature_cols=future_cols if future_cols else None,
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=wc,
        target_cfg=tc,
        return_mask=True,
    )

    X_train_np = built["X"].astype(np.float32)
    future_rul_cycles = built["Y_seq"].squeeze(-1).astype(np.float32)  # (N,H)
    Y_rul_np = (future_rul_cycles / max(max_rul, 1e-6)).astype(np.float32)
    Y_hi_np = np.clip(future_rul_cycles / max(max_rul, 1e-6), 0.0, 1.0).astype(np.float32)
    fut = built["future_features"].astype(np.float32)
    Y_sens_np = fut[:, :, : len(target_sensor_cols)].astype(np.float32) if target_sensor_cols else np.zeros((X_train_np.shape[0], horizon, 0), dtype=np.float32)
    future_cond_np = fut[:, :, len(target_sensor_cols) :].astype(np.float32) if cond_cols else np.zeros((X_train_np.shape[0], horizon, 0), dtype=np.float32)

    # Engine split like trainer (deterministic)
    unit_ids_np = built["unit_ids"].astype(np.int64)
    cond_ids_np_full = built["cond_ids"].astype(np.int64) if "cond_ids" in built else np.zeros((X_train_np.shape[0],), dtype=np.int64)
    unit_ids_t = torch.from_numpy(unit_ids_np)
    uniq = torch.unique(unit_ids_t)
    gen = torch.Generator().manual_seed(42)
    perm = uniq[torch.randperm(len(uniq), generator=gen)]
    n_val_units = max(1, int(0.2 * len(uniq)))
    val_units = perm[:n_val_units]
    train_units = perm[n_val_units:]
    train_mask = torch.isin(unit_ids_t, train_units)
    train_indices = torch.nonzero(train_mask, as_tuple=False).view(-1).cpu().numpy()

    # Apply informative sampling if enabled (same definition as trainer)
    inf_enable = bool(world_model_params.get("informative_sampling_enable", False))
    eps_inf = float(world_model_params.get("informative_eps_norm", 1e-6))
    mode_inf = str(world_model_params.get("informative_sampling_mode", "future_min_lt_cap"))
    keep_p = float(world_model_params.get("keep_prob_noninformative", 0.1))
    if inf_enable and train_indices.size > 0:
        y_tr_norm = np.clip(Y_rul_np[train_indices], 0.0, 1.0)
        future_min = y_tr_norm.min(axis=1)
        if mode_inf == "future_has_zero":
            is_inf = future_min <= eps_inf
        else:
            is_inf = future_min < (1.0 - eps_inf)
        rng = np.random.default_rng(42)
        keep = is_inf | (rng.random(is_inf.shape[0]) < np.clip(keep_p, 0.0, 1.0))
        train_indices = train_indices[keep]

    # Load persisted X scaler from run results_dir (ensures same scaling as training)
    x_scaler_path = paths.results_dir / "world_model_v1_x_scaler.pkl"
    x_scaler = load_scaler(str(x_scaler_path))
    X_tr = transform_x(x_scaler, X_train_np[train_indices])
    X_tr, _ = clip_x(X_tr, clip=10.0)

    # Scale future_cond using Cond_* stats from X scaler (same as trainer)
    cond_idx = np.array([i for i, c in enumerate(feature_cols) if c.startswith("Cond_")], dtype=np.int64)
    future_cond_tr = future_cond_np[train_indices].astype(np.float32, copy=False)
    if cond_idx.size > 0 and future_cond_tr.size > 0:
        cond_mean = x_scaler.mean_[cond_idx]
        cond_scale = x_scaler.scale_[cond_idx]
        future_cond_tr = (future_cond_tr - cond_mean[None, None, :]) / cond_scale[None, None, :]
        future_cond_tr = np.clip(future_cond_tr, -10.0, 10.0).astype(np.float32, copy=False)

    # Pick a single train batch
    B = int(cfg.get("training_params", {}).get("batch_size", 256))
    b = min(B, X_tr.shape[0])
    X_b = torch.from_numpy(X_tr[:b]).float().to(device)
    Y_rul_b = torch.from_numpy(Y_rul_np[train_indices][:b]).float().to(device)
    Y_hi_b = torch.from_numpy(Y_hi_np[train_indices][:b]).float().to(device)
    Y_sens_b = torch.from_numpy(Y_sens_np[train_indices][:b]).float().to(device)
    future_cond_b = torch.from_numpy(future_cond_tr[:b]).float().to(device) if future_cond_tr.size > 0 else None
    cond_ids_b = torch.from_numpy(cond_ids_np_full[train_indices][:b]).long().to(device)

    # --- Build model and load checkpoint ---
    input_dim = len(feature_cols)
    enc_kwargs = cfg.get("encoder_kwargs", {})
    encoder = EOLFullTransformerEncoder(
        input_dim=input_dim,
        d_model=int(enc_kwargs.get("d_model", 64)),
        num_layers=int(enc_kwargs.get("num_layers", 3)),
        n_heads=int(enc_kwargs.get("nhead", 4)),
        dim_feedforward=int(enc_kwargs.get("dim_feedforward", 256)),
        dropout=float(enc_kwargs.get("dropout", 0.1)),
        use_condition_embedding=True,
        num_conditions=7,
        cond_emb_dim=4,
        max_seq_len=300,
    ).to(device)

    wm = TransformerWorldModelV1(
        encoder=encoder,
        input_dim=input_dim,
        num_sensors_out=int(world_model_params.get("num_sensors_out", 21)),
        cond_dim=int(len(cond_cols) if cond_cols else world_model_params.get("cond_dim", 9)),
        future_horizon=horizon,
        decoder_hidden_dim=int(world_model_params.get("decoder_hidden_dim", 256)),
        num_layers_decoder=int(world_model_params.get("num_layers_decoder", 1)),
        dropout=float(enc_kwargs.get("dropout", 0.1)),
        predict_hi=True,
        predict_rul=True,
        target_mode=str(world_model_params.get("target_mode", "latent_hi_rul")),
        init_from_rul_hi=bool(world_model_params.get("init_from_rul_hi", False)),
        use_latent_history=bool(world_model_params.get("use_latent_history", False)),
        use_hi_anchor=bool(world_model_params.get("use_hi_anchor", False)),
        use_future_conds=bool(world_model_params.get("use_future_conds", False)),
        use_eol_fusion=bool(world_model_params.get("use_eol_fusion", False)),
        eol_fusion_mode=str(world_model_params.get("eol_fusion_mode", "token")),
        predict_latent=bool(world_model_params.get("predict_latent", False)),
        latent_decoder_type=str(world_model_params.get("latent_decoder_type", "transformer")),
        latent_decoder_num_layers=int(world_model_params.get("latent_decoder_num_layers", 2)),
        latent_decoder_nhead=int(world_model_params.get("latent_decoder_nhead", 4)),
    ).to(device)

    ckpt = torch.load(paths.checkpoint_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    wm.load_state_dict(sd, strict=True)

    wm.train()
    cond_vec = torch.zeros((X_b.size(0), int(wm.cond_dim)), dtype=torch.float32, device=device)
    current_rul_b = Y_rul_b[:, 0]
    current_hi_b = Y_hi_b[:, 0]

    out = wm(
        past_seq=X_b,
        cond_vec=cond_vec,
        cond_ids=cond_ids_b,
        future_horizon=horizon,
        teacher_forcing_targets=None,
        current_rul=current_rul_b if bool(world_model_params.get("use_hi_anchor", False)) else None,
        current_hi=current_hi_b if bool(world_model_params.get("use_hi_anchor", False)) else None,
        future_conds=future_cond_b if bool(world_model_params.get("use_future_conds", False)) else None,
    )
    pred_sens, pred_hi, pred_rul, pred_eol = out if (isinstance(out, (tuple, list)) and len(out) == 4) else (*out, None)

    # Basic wiring assertions
    assert pred_rul is not None and pred_rul.size(1) == horizon, "pred_rul horizon mismatch"
    assert pred_hi is not None and pred_hi.size(1) == horizon, "pred_hi horizon mismatch"
    assert torch.isfinite(pred_rul).all() and torch.isfinite(pred_hi).all(), "non-finite preds"
    assert torch.isfinite(Y_rul_b).all() and torch.isfinite(Y_hi_b).all(), "non-finite targets"

    # Loss computation (mirrors main trainer: future losses + optional eol scalar loss)
    sensor_w = float(world_model_params.get("sensor_loss_weight", 0.0))
    hi_w = float(world_model_params.get("hi_future_loss_weight", 0.0))
    rul_w = float(world_model_params.get("rul_future_loss_weight", 0.0))
    eol_scalar_loss_weight = float(world_model_params.get("eol_scalar_loss_weight", 0.0))

    loss_sens = F.mse_loss(pred_sens, Y_sens_b) if (pred_sens is not None and Y_sens_b.numel() > 0) else torch.tensor(0.0, device=device)
    loss_hi = F.mse_loss(pred_hi.squeeze(-1), Y_hi_b) if pred_hi is not None else torch.tensor(0.0, device=device)

    # RUL future loss: use the same masked MSE path as the trainer's "new" mode (no cap-mask logic here beyond simple cap mask)
    pred_rul_seq_norm = pred_rul.squeeze(-1).clamp(0.0, 1.0)
    true_rul_seq_norm = Y_rul_b.clamp(0.0, 1.0)
    cap_thr = float(world_model_params.get("rul_cap_threshold", 0.999999))
    mask = (true_rul_seq_norm < cap_thr).float()  # (B,H)
    diff2 = (pred_rul_seq_norm - true_rul_seq_norm) ** 2
    loss_rul = (diff2 * mask).sum() / (mask.sum() + 1e-6)

    # Late mask/weights (as used in training)
    late_enable = bool(world_model_params.get("late_weight_enable", False))
    late_factor = float(world_model_params.get("late_weight_factor", 1.0))
    late_eps = float(world_model_params.get("late_weight_eps_norm", 1e-6))
    late_mask = (true_rul_seq_norm.min(dim=1).values <= late_eps) if (late_enable and late_factor > 1.0) else torch.zeros((X_b.size(0),), device=device, dtype=torch.bool)
    w_late = (1.0 + late_mask.float() * (late_factor - 1.0)) if (late_enable and late_factor > 1.0) else torch.ones((X_b.size(0),), device=device)

    loss_rul_per_sample = (diff2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    loss_rul_weighted = (w_late * loss_rul_per_sample).mean()

    loss_eol = F.mse_loss(pred_eol.view(-1), current_rul_b.view(-1)) if (pred_eol is not None and eol_scalar_loss_weight > 0.0) else torch.tensor(0.0, device=device)

    total = sensor_w * loss_sens + hi_w * loss_hi + rul_w * loss_rul_weighted + eol_scalar_loss_weight * loss_eol

    # Grad-flow check
    wm.zero_grad(set_to_none=True)
    total.backward()
    named = list(wm.named_parameters())
    gn_enc = _grad_norm_from_named_params(named, "encoder.")
    gn_dec = _grad_norm_excluding_prefix(named, "encoder.")

    # -------------------------
    # Build ONE test batch (same window definition as evaluate_transformer_world_model_v1_on_test)
    # -------------------------
    wm.eval()
    X_test_list: List[np.ndarray] = []
    Y_rul_test_list: List[np.ndarray] = []
    Y_hi_test_list: List[np.ndarray] = []
    cond_id_test_list: List[int] = []
    future_cond_test_list: List[np.ndarray] = []
    unit_id_test_list: List[int] = []

    df_test_local = df_test.copy()
    # Ensure RUL column exists (clipped to max_rul, as evaluation expects)
    if "RUL" not in df_test_local.columns:
        df_max_time = (
            df_test_local.groupby("UnitNumber")["TimeInCycles"]
            .max()
            .reset_index()
            .rename(columns={"TimeInCycles": "MaxTime"})
        )
        df_test_local = df_test_local.merge(df_max_time, on="UnitNumber", how="left")
        df_test_local["RUL_raw"] = df_test_local["MaxTime"] - df_test_local["TimeInCycles"]
        df_test_local["RUL"] = np.minimum(df_test_local["RUL_raw"], float(max_rul))
    else:
        df_test_local["RUL"] = df_test_local["RUL"].clip(lower=0.0, upper=float(max_rul))

    cond_cols = [c for c in feature_cols if c.startswith("Cond_")]
    for unit_id, df_unit in df_test_local.groupby("UnitNumber"):
        df_unit = df_unit.sort_values("TimeInCycles").reset_index(drop=True)
        num_rows = len(df_unit)
        if num_rows < past_len + horizon:
            continue
        cond_id_unit = int(df_unit["ConditionID"].iloc[0]) if "ConditionID" in df_unit.columns else 0
        for start in range(0, num_rows - past_len - horizon + 1):
            past = df_unit.iloc[start : start + past_len]
            future = df_unit.iloc[start + past_len : start + past_len + horizon]
            X_test_list.append(past[feature_cols].to_numpy(dtype=np.float32, copy=True))
            rul_future = future["RUL"].to_numpy(dtype=np.float32)
            Y_rul_test_list.append((rul_future / max(float(max_rul), 1e-6)).astype(np.float32))
            Y_hi_test_list.append(np.clip(rul_future / max(float(max_rul), 1e-6), 0.0, 1.0).astype(np.float32))
            cond_id_test_list.append(cond_id_unit)
            unit_id_test_list.append(int(unit_id))
            if cond_cols:
                future_cond_test_list.append(future[cond_cols].to_numpy(dtype=np.float32, copy=True))
            else:
                future_cond_test_list.append(np.zeros((horizon, int(wm.cond_dim)), dtype=np.float32))
            if len(X_test_list) >= b:
                break
        if len(X_test_list) >= b:
            break

    if X_test_list:
        X_test_np = np.stack(X_test_list, axis=0).astype(np.float32)
        # Apply SAME X scaler
        X_test_np = transform_x(x_scaler, X_test_np)
        X_test_np, _ = clip_x(X_test_np, clip=10.0)
        # Scale future conds using X scaler's Cond_* stats
        future_cond_test_np = np.stack(future_cond_test_list, axis=0).astype(np.float32)
        if cond_idx.size > 0 and future_cond_test_np.size > 0:
            cond_mean = x_scaler.mean_[cond_idx]
            cond_scale = x_scaler.scale_[cond_idx]
            future_cond_test_np = (future_cond_test_np - cond_mean[None, None, :]) / cond_scale[None, None, :]
            future_cond_test_np = np.clip(future_cond_test_np, -10.0, 10.0).astype(np.float32, copy=False)

        X_tb = torch.from_numpy(X_test_np).float().to(device)
        Y_rul_tb = torch.from_numpy(np.stack(Y_rul_test_list, axis=0)).float().to(device)  # (B,H)
        Y_hi_tb = torch.from_numpy(np.stack(Y_hi_test_list, axis=0)).float().to(device)    # (B,H)
        cond_tb = torch.from_numpy(np.asarray(cond_id_test_list, dtype=np.int64)).long().to(device)
        future_cond_tb = torch.from_numpy(future_cond_test_np).float().to(device) if future_cond_test_np.size > 0 else None

        with torch.no_grad():
            cond_vec_t = torch.zeros((X_tb.size(0), int(wm.cond_dim)), dtype=torch.float32, device=device)
            out_t = wm(
                past_seq=X_tb,
                cond_vec=cond_vec_t,
                cond_ids=cond_tb,
                future_horizon=horizon,
                teacher_forcing_targets=None,
                current_rul=Y_hi_tb[:, 0],
                current_hi=Y_hi_tb[:, 0],
                future_conds=future_cond_tb if bool(world_model_params.get("use_future_conds", False)) else None,
            )
            pred_sens_t, pred_hi_t, pred_rul_t, pred_eol_t = out_t if (isinstance(out_t, (tuple, list)) and len(out_t) == 4) else (*out_t, None)

        pred_rul_last_cycles = (pred_rul_t[:, -1, 0].clamp(0.0, 1.0) * float(max_rul)).detach().cpu().numpy().astype(float)
        true_rul_last_cycles = (Y_rul_tb[:, -1].clamp(0.0, 1.0) * float(max_rul)).detach().cpu().numpy().astype(float)
        report_test = {
            "B": int(X_tb.size(0)),
            "shapes": {
                "X_b": list(X_tb.shape),
                "future_cond_b": list(future_cond_tb.shape) if future_cond_tb is not None else None,
                "Y_rul_b": list(Y_rul_tb.shape),
                "Y_hi_b": list(Y_hi_tb.shape),
                "pred_rul": list(pred_rul_t.shape) if pred_rul_t is not None else None,
                "pred_hi": list(pred_hi_t.shape) if pred_hi_t is not None else None,
                "pred_eol": list(pred_eol_t.shape) if pred_eol_t is not None else None,
            },
            "metric_wiring": {
                "rmse_last_source": "pred_rul_seq_norm[:, -1] vs true_rul_seq_norm[:, -1] (converted to cycles)",
                "uses_pred_eol_for_metrics": False,
            },
            "pred_rul_last_cycles_mean": float(np.mean(pred_rul_last_cycles)),
            "true_rul_last_cycles_mean": float(np.mean(true_rul_last_cycles)),
        }
    else:
        report_test = {"note": "Could not build a full-horizon test batch (past_len+horizon too large?)"}

    report: Dict[str, Any] = {
        "experiment": exp,
        "dataset": dataset,
        "paths": {
            "results_dir": str(paths.results_dir),
            "checkpoint_path": str(paths.checkpoint_path),
            "encoder_summary_path": str(paths.encoder_summary_path),
        },
        "feature_pipeline": feat_info,
        "feature_cols_count": int(len(feature_cols)),
        "windowing": {"past_len": past_len, "horizon": horizon, "pad_mode": "clamp", "cap": True, "eol_mode": "future0"},
        "train_batch": {
            "B": int(X_b.size(0)),
            "shapes": {
                "X_b": list(X_b.shape),
                "future_cond_b": list(future_cond_b.shape) if future_cond_b is not None else None,
                "Y_rul_b": list(Y_rul_b.shape),
                "Y_hi_b": list(Y_hi_b.shape),
                "pred_rul": list(pred_rul.shape) if pred_rul is not None else None,
                "pred_hi": list(pred_hi.shape) if pred_hi is not None else None,
                "pred_eol": list(pred_eol.shape) if pred_eol is not None else None,
            },
            "stats": {
                "true_rul_min_mean_max": [
                    float(true_rul_seq_norm.min().detach().cpu()),
                    float(true_rul_seq_norm.mean().detach().cpu()),
                    float(true_rul_seq_norm.max().detach().cpu()),
                ],
                "pred_rul_min_mean_max": [
                    float(pred_rul_seq_norm.min().detach().cpu()),
                    float(pred_rul_seq_norm.mean().detach().cpu()),
                    float(pred_rul_seq_norm.max().detach().cpu()),
                ],
                "late_mask_frac": float(late_mask.float().mean().detach().cpu()),
                "w_late_min_mean_max": [
                    float(w_late.min().detach().cpu()),
                    float(w_late.mean().detach().cpu()),
                    float(w_late.max().detach().cpu()),
                ],
                "cap_mask_keep_frac": float(mask.mean().detach().cpu()),
                "pred_rul_requires_grad": bool(pred_rul.requires_grad) if pred_rul is not None else False,
                "pred_hi_requires_grad": bool(pred_hi.requires_grad) if pred_hi is not None else False,
            },
            "losses": {
                "sensor_w": float(sensor_w),
                "hi_w": float(hi_w),
                "rul_w": float(rul_w),
                "eol_scalar_loss_weight": float(eol_scalar_loss_weight),
                "loss_sensors": float(loss_sens.detach().cpu()),
                "loss_hi": float(loss_hi.detach().cpu()),
                "loss_rul_masked_mean": float(loss_rul.detach().cpu()),
                "loss_rul_weighted_mean": float(loss_rul_weighted.detach().cpu()),
                "loss_eol": float(loss_eol.detach().cpu()),
                "loss_total": float(total.detach().cpu()),
            },
            "grads": {
                "encoder_grad_norm": float(gn_enc),
                "decoder_grad_norm": float(gn_dec),
            },
        },
        "test_batch": report_test,
    }

    out_path = Path(args.out) if args.out else (paths.results_dir / "wiring_audit_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # stdout summary (markdown-ish)
    print("\n## WM-V1 Wiring Audit Summary")
    print(f"- **experiment**: `{exp}`")
    print(f"- **checkpoint**: `{paths.checkpoint_path}`")
    print(f"- **features**: {len(feature_cols)}")
    print(f"- **train batch B**: {int(X_b.size(0))}")
    print(f"- **late_mask_frac**: {report['train_batch']['stats']['late_mask_frac']:.6f}")
    print(f"- **cap_mask_keep_frac**: {report['train_batch']['stats']['cap_mask_keep_frac']:.6f}")
    print(f"- **pred_rul_requires_grad**: {report['train_batch']['stats']['pred_rul_requires_grad']}")
    print(f"- **pred_hi_requires_grad**: {report['train_batch']['stats']['pred_hi_requires_grad']}")
    print(f"- **loss_total**: {report['train_batch']['losses']['loss_total']:.6f}")
    print(f"- **saved**: `{out_path}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

