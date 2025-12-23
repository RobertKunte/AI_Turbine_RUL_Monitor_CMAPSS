from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.windowing import WindowConfig, TargetConfig, build_test_windows_last
from src.data_loading import load_cmapps_subset
from src.experiment_configs import get_experiment_by_name
from src.models.transformer_eol import EOLFullTransformerEncoder
from src.models.transformer_world_model_v1 import TransformerWorldModelV1
from src.tools.x_scaler import load_scaler, transform_x, clip_x

# Reuse feature-pipeline reconstruction from the wiring audit
from src.analysis.wm_v1_wiring_audit import _rebuild_feature_pipeline_from_encoder_summary, _resolve_paths  # type: ignore


def _predict_last_window(
    *,
    model: TransformerWorldModelV1,
    X_last_np: np.ndarray,
    cond_ids_np: np.ndarray,
    feature_cols: List[str],
    max_rul: float,
    horizon: int,
    device: torch.device,
    results_dir: Path,
    use_future_conds: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict RUL for LAST-OBSERVED windows using y_pred = pred_rul_seq_norm[:,0]*max_rul (aligned to y_test_true).
    Returns (y_pred_cycles, pred_rul_seq_norm_laststep_cycles) for inspection.
    """
    x_scaler_path = results_dir / "world_model_v1_x_scaler.pkl"
    x_scaler = load_scaler(str(x_scaler_path))

    X_scaled = transform_x(x_scaler, X_last_np.astype(np.float32))
    X_scaled, _ = clip_x(X_scaled, clip=10.0)

    X_t = torch.from_numpy(X_scaled).float().to(device)
    cond_ids_t = torch.from_numpy(cond_ids_np.astype(np.int64)).long().to(device)

    # Build future_conds by repeating the (scaled) Cond_* vector from the last timestep.
    cond_idx = np.array([i for i, c in enumerate(feature_cols) if c.startswith("Cond_")], dtype=np.int64)
    future_cond_t = None
    if use_future_conds and cond_idx.size > 0:
        cond_last = X_scaled[:, -1, cond_idx]  # already scaled in X space
        future_cond = np.repeat(cond_last[:, None, :], repeats=int(horizon), axis=1).astype(np.float32, copy=False)
        future_cond_t = torch.from_numpy(future_cond).float().to(device)

    model.eval()
    with torch.no_grad():
        B = int(X_t.size(0))
        cond_vec = torch.zeros((B, int(getattr(model, "cond_dim", 9))), dtype=torch.float32, device=device)
        out = model(
            past_seq=X_t,
            cond_vec=cond_vec,
            cond_ids=cond_ids_t,
            future_horizon=int(horizon),
            teacher_forcing_targets=None,
            current_rul=None,
            current_hi=None,
            future_conds=future_cond_t if use_future_conds else None,
        )
        if isinstance(out, (tuple, list)) and len(out) == 4:
            _, _pred_hi, pred_rul, _pred_eol = out
        else:
            _, _pred_hi, pred_rul = out
        pred_rul_seq_norm = pred_rul[:, :, 0].clamp(0.0, 1.0).detach().cpu().numpy().astype(float)  # (N,H)

    # For LAST-OBSERVED target y_test_true: use step0 as "current-ish" (clamp mode makes this meaningful near end).
    y_pred_cycles = (pred_rul_seq_norm[:, 0] * float(max_rul)).astype(float)
    y_pred_end_cycles = (pred_rul_seq_norm[:, -1] * float(max_rul)).astype(float)
    return y_pred_cycles, y_pred_end_cycles


def main() -> int:
    ap = argparse.ArgumentParser(description="WM-V1 eval alignment check (LAST windows vs y_test_true).")
    ap.add_argument("--experiment", type=str, default="fd004_wm_v1_infwin_wiringcheck_k0")
    ap.add_argument("--dataset", type=str, default="FD004")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_units", type=int, default=20)
    args = ap.parse_args()

    dataset = str(args.dataset)
    exp = str(args.experiment)
    device = torch.device(args.device)

    cfg = get_experiment_by_name(exp)
    paths = _resolve_paths(dataset, exp, cfg)

    # Rebuild feature pipeline from encoder summary to match training features
    df_train, df_test, y_test_true, feature_cols, feat_info = _rebuild_feature_pipeline_from_encoder_summary(
        dataset,
        paths.encoder_summary_path,
        max_rul=float(cfg.get("world_model_params", {}).get("max_rul", 125.0)),
    )

    world_model_params = cfg.get("world_model_params", {})
    past_len = int(world_model_params.get("past_len", 30))
    horizon = int(world_model_params.get("future_horizon", world_model_params.get("horizon", 30)))
    max_rul = float(world_model_params.get("max_rul", 125.0))
    use_future_conds = bool(world_model_params.get("use_future_conds", False))

    # Build LAST windows per unit (aligned with NASA y_test_true)
    wc = WindowConfig(past_len=past_len, horizon=horizon, stride=1, require_full_horizon=False, pad_mode="clamp")
    tc = TargetConfig(max_rul=int(max_rul), cap_targets=True, eol_target_mode="current_from_df", clip_eval_y_true=True)
    built_last = build_test_windows_last(
        df_test=df_test,
        y_test_true=y_test_true,
        feature_cols=feature_cols,
        unit_col="UnitNumber",
        time_col="TimeInCycles",
        cond_col="ConditionID",
        window_cfg=wc,
        target_cfg=tc,
    )

    X_last = built_last["X"]  # (N,P,F)
    y_true_last = built_last["y_true"].astype(float)  # (N,)
    unit_ids = built_last["unit_ids"].astype(int)
    cond_ids = built_last["cond_ids"].astype(int)

    # Subselect first N units (deterministic)
    n = min(int(args.n_units), int(X_last.shape[0]))
    X_last = X_last[:n]
    y_true_last = y_true_last[:n]
    unit_ids = unit_ids[:n]
    cond_ids = cond_ids[:n]

    # Load WM-V1 checkpoint
    ckpt_path = paths.results_dir / f"transformer_world_model_v1_best_{exp}.pt"
    if not ckpt_path.exists():
        pts = sorted(paths.results_dir.glob("*best*.pt"))
        if not pts:
            raise FileNotFoundError(f"No WM-V1 best checkpoint found under {paths.results_dir}")
        ckpt_path = pts[0]

    enc_kwargs = cfg.get("encoder_kwargs", {})
    encoder = EOLFullTransformerEncoder(
        input_dim=len(feature_cols),
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
        input_dim=len(feature_cols),
        num_sensors_out=int(world_model_params.get("num_sensors_out", 21)),
        cond_dim=int(len([c for c in feature_cols if c.startswith("Cond_")]) or world_model_params.get("cond_dim", 9)),
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
        use_future_conds=bool(use_future_conds),
        use_eol_fusion=bool(world_model_params.get("use_eol_fusion", False)),
        eol_fusion_mode=str(world_model_params.get("eol_fusion_mode", "token")),
        predict_latent=bool(world_model_params.get("predict_latent", False)),
        latent_decoder_type=str(world_model_params.get("latent_decoder_type", "transformer")),
        latent_decoder_num_layers=int(world_model_params.get("latent_decoder_num_layers", 2)),
        latent_decoder_nhead=int(world_model_params.get("latent_decoder_nhead", 4)),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    wm.load_state_dict(sd, strict=True)

    y_pred_last, y_pred_end = _predict_last_window(
        model=wm,
        X_last_np=X_last,
        cond_ids_np=cond_ids,
        feature_cols=feature_cols,
        max_rul=max_rul,
        horizon=horizon,
        device=device,
        results_dir=paths.results_dir,
        use_future_conds=use_future_conds,
    )

    # Alignment metrics
    err = y_pred_last - y_true_last
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
    mae = float(np.mean(np.abs(err))) if err.size else 0.0
    bias = float(np.mean(err)) if err.size else 0.0
    frac_pred_gt_true = float(np.mean(y_pred_last > y_true_last)) if err.size else 0.0

    rows = []
    for i in range(int(n)):
        rows.append(
            {
                "unit_id": int(unit_ids[i]),
                "true_last_cycles": float(y_true_last[i]),
                "pred_step0_cycles": float(y_pred_last[i]),
                "pred_seq_end_cycles": float(y_pred_end[i]),
                "pred_gt_true": bool(y_pred_last[i] > y_true_last[i]),
            }
        )

    out: Dict[str, Any] = {
        "experiment": exp,
        "dataset": dataset,
        "results_dir": str(paths.results_dir),
        "checkpoint": str(ckpt_path),
        "window_cfg": {"past_len": past_len, "horizon": horizon, "pad_mode": "clamp"},
        "target_definition": {
            "y_true_last": "NASA y_test_true (remaining cycles at last observed cycle; capped to max_rul)",
            "y_pred_last": "pred_rul_seq_norm[:,0] * max_rul (interpreted as current/next-step RUL for last window)",
        },
        "n_units_checked": int(n),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "frac_pred_gt_true": frac_pred_gt_true,
        "rows": rows,
        "feature_pipeline": feat_info,
    }

    out_json = paths.results_dir / "alignment_check.json"
    out_md = paths.results_dir / "alignment_check.md"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Minimal markdown summary + small table
    lines = []
    lines.append("## WM‑V1 Eval Alignment Check")
    lines.append(f"- experiment: `{exp}`")
    lines.append(f"- checkpoint: `{ckpt_path}`")
    lines.append(f"- n_units_checked: **{n}**")
    lines.append(f"- RMSE: **{rmse:.2f}**  MAE: **{mae:.2f}**  Bias: **{bias:.2f}**")
    lines.append(f"- frac(pred>true): **{frac_pred_gt_true:.3f}**")
    lines.append("")
    lines.append("| unit | true_last | pred_step0 | pred_seq_end | pred>true |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for r in rows[: min(10, len(rows))]:
        lines.append(
            f"| {r['unit_id']} | {r['true_last_cycles']:.2f} | {r['pred_step0_cycles']:.2f} | {r['pred_seq_end_cycles']:.2f} | {str(r['pred_gt_true'])} |"
        )
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

