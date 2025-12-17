"""
Compare "true_vs_pred.png" (diagnostics path) vs "true_vs_pred_mu.png" (inference risk path).

Why this exists
---------------
The project currently generates two similar-looking scatter plots via two different code paths:

1) Diagnostics path (used right after training):
   - src/analysis/diagnostics.py -> run_diagnostics_for_run -> build_eval_data -> evaluate_on_test_data
   - Produces: results/<dataset>/<run_name>/true_vs_pred.png

2) Inference path (used by fd004_worst20_diagnosis_v1.py):
   - src/analysis/inference.py -> run_inference_for_experiment
   - Produces: results/<dataset>/<run_name>/true_vs_pred_mu.png (if residual risk head exists)

If these plots differ materially for the same run, it usually means one of:
- Different feature pipelines / feature column ordering
- Different scaler usage
- Different engine<->true RUL mapping / ordering

This script computes per-engine (unit_id) true/pred from both paths and reports diffs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import torch

# Ensure project root is on sys.path (so `import src...` works when run as a script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.inference import load_model_from_experiment, run_inference_for_experiment
from src.analysis.diagnostics import build_eval_data
from src.additional_features import FeatureConfig, TemporalFeatureConfig
from src.config import PhysicsFeatureConfig, ResidualFeatureConfig
from src.eol_full_lstm import evaluate_on_test_data


def _load_summary(experiment_dir: Path) -> dict:
    p = experiment_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"summary.json not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_scaler(experiment_dir: Path):
    p = experiment_dir / "scaler.pkl"
    if not p.exists():
        raise FileNotFoundError(f"scaler.pkl not found at {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def _is_phase4_residual_from_name(name: str) -> bool:
    n = (name or "").lower()
    return ("resid" in n) or ("residual" in n) or ("phase4" in n) or ("phase5" in n) or ("world" in n)


def _build_feature_config_from_summary(config: dict) -> Tuple[FeatureConfig, PhysicsFeatureConfig, Dict[str, Any]]:
    # Match the logic used in src/analysis/diagnostics.py:run_diagnostics_for_run
    experiment_name = config.get("experiment_name", "")
    is_phase4_residual = _is_phase4_residual_from_name(experiment_name)

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

    features_cfg = config.get("features", {}) or {}
    ms_cfg = features_cfg.get("multiscale", {}) or {}
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

    # phys_features config (condition vector + twin residuals) if present
    phys_features_cfg = config.get("phys_features", None)
    if phys_features_cfg is None:
        # Best-effort fallback: ms_dt style experiments expect this.
        name_lower = (experiment_name or "").lower()
        if "transformer_encoder_ms_dt_v" in name_lower or "ms_dt" in name_lower:
            phys_features_cfg = {
                "use_condition_vector": True,
                "use_twin_features": True,
                "twin_baseline_len": 30,
                "condition_vector_version": 3,
            }

    return feature_config, physics_config, (phys_features_cfg or {})


def _metrics_to_arrays(eol_metrics) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by unit_id for stable alignment
    eol_metrics_sorted = sorted(eol_metrics, key=lambda m: int(m.unit_id))
    unit_ids = np.array([int(m.unit_id) for m in eol_metrics_sorted], dtype=int)
    y_true = np.array([float(m.true_rul) for m in eol_metrics_sorted], dtype=float)
    y_pred = np.array([float(m.pred_rul) for m in eol_metrics_sorted], dtype=float)
    return unit_ids, y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="FD004")
    ap.add_argument("--run_name", type=str, required=True, help="results/<dataset>/<run_name>/")
    ap.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Optional explicit experiment directory path (overrides dataset/run_name).",
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    run_name = args.run_name
    device = torch.device(args.device)
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else (Path("results") / dataset.lower() / run_name)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {experiment_dir}")

    print(f"[compare] run_dir={experiment_dir} device={device}")

    # ------------------------------------------------------------------
    # A) Inference path (creates true_vs_pred_mu.png when risk head exists)
    # ------------------------------------------------------------------
    eol_metrics_inf, _ = run_inference_for_experiment(
        experiment_dir=experiment_dir,
        split="test",
        device=device,
        return_hi_trajectories=False,
    )
    u_inf, y_true_inf, y_pred_inf = _metrics_to_arrays(eol_metrics_inf)
    print(f"[compare] inference: n={len(u_inf)} true_mean={y_true_inf.mean():.2f} pred_mean={y_pred_inf.mean():.2f}")

    # ------------------------------------------------------------------
    # B) Diagnostics path (used to create true_vs_pred.png)
    # ------------------------------------------------------------------
    # Load model + config the same way run_diagnostics_for_run does.
    model, config = load_model_from_experiment(
        experiment_dir=experiment_dir,
        device=device,
        input_dim=None,
        num_conditions=None,
    )
    feature_config, physics_config, phys_features_cfg = _build_feature_config_from_summary(config)
    max_rul = int(config.get("max_rul", 125))
    past_len = int(config.get("past_len", config.get("sequence_length", 30)))

    # Build DF + feature_cols via the diagnostics helper (same as run_diagnostics_for_run)
    _, _, y_test_true, unit_ids_test, _, _, feature_cols, df_test_fe = build_eval_data(
        dataset_name=dataset,
        max_rul=max_rul,
        past_len=past_len,
        feature_config=feature_config,
        physics_config=physics_config,
        phys_features=phys_features_cfg,
    )

    scaler_loaded = _load_scaler(experiment_dir)
    metrics_diag = evaluate_on_test_data(
        model=model,
        df_test=df_test_fe,
        y_test_true=np.asarray(y_test_true, dtype=float),
        feature_cols=feature_cols,
        scaler=scaler_loaded,
        past_len=past_len,
        max_rul=max_rul,
        unit_col="UnitNumber",
        cycle_col="TimeInCycles",
        device=device,
    )
    u_diag = np.asarray(metrics_diag.get("unit_ids"), dtype=int)
    y_true_diag = np.asarray(metrics_diag.get("y_true"), dtype=float)
    y_pred_diag = np.asarray(metrics_diag.get("y_pred"), dtype=float)

    # Sort diag arrays by unit_id for stable alignment
    idx = np.argsort(u_diag)
    u_diag = u_diag[idx]
    y_true_diag = y_true_diag[idx]
    y_pred_diag = y_pred_diag[idx]
    print(f"[compare] diagnostics: n={len(u_diag)} true_mean={y_true_diag.mean():.2f} pred_mean={y_pred_diag.mean():.2f}")

    # ------------------------------------------------------------------
    # Compare alignment and values
    # ------------------------------------------------------------------
    u_all = np.union1d(u_diag, u_inf)
    diag_map = {int(u): (float(t), float(p)) for u, t, p in zip(u_diag, y_true_diag, y_pred_diag)}
    inf_map = {int(u): (float(t), float(p)) for u, t, p in zip(u_inf, y_true_inf, y_pred_inf)}

    rows: List[Dict[str, Any]] = []
    for u in u_all:
        t_d, p_d = diag_map.get(int(u), (np.nan, np.nan))
        t_i, p_i = inf_map.get(int(u), (np.nan, np.nan))
        rows.append(
            {
                "unit_id": int(u),
                "true_diag": t_d,
                "pred_diag": p_d,
                "true_inf": t_i,
                "pred_inf": p_i,
                "d_true": t_d - t_i if np.isfinite(t_d) and np.isfinite(t_i) else np.nan,
                "d_pred": p_d - p_i if np.isfinite(p_d) and np.isfinite(p_i) else np.nan,
                "abs_d_pred": abs(p_d - p_i) if np.isfinite(p_d) and np.isfinite(p_i) else np.nan,
            }
        )

    # Summary stats (where both available)
    d_true = np.array([r["d_true"] for r in rows], dtype=float)
    d_pred = np.array([r["d_pred"] for r in rows], dtype=float)
    m = np.isfinite(d_pred)
    print(f"[compare] common engines: {int(m.sum())}/{len(u_all)}")
    if np.any(m):
        print(f"[compare] pred diff: mean={float(np.nanmean(d_pred)):.4f}  max_abs={float(np.nanmax(np.abs(d_pred))):.4f}")
        print(f"[compare] true diff: mean={float(np.nanmean(d_true)):.4f}  max_abs={float(np.nanmax(np.abs(d_true))):.4f}")

        # Identify top offenders
        abs_d = np.abs(d_pred)
        top_idx = np.argsort(-np.nan_to_num(abs_d, nan=-1.0))[:10]
        print("[compare] Top-10 |pred_diag - pred_inf|")
        for j in top_idx:
            r = rows[int(j)]
            if not np.isfinite(r["abs_d_pred"]):
                continue
            print(
                f"  unit={r['unit_id']:>3d} "
                f"true_diag={r['true_diag']:.1f} pred_diag={r['pred_diag']:.2f} | "
                f"true_inf={r['true_inf']:.1f} pred_inf={r['pred_inf']:.2f} | "
                f"d_pred={r['d_pred']:.2f}"
            )

    # Write CSV for deeper inspection
    out_csv = args.out_csv or str(experiment_dir / "compare_diagnostics_vs_inference.csv")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[compare] Wrote {out_csv}")
    except Exception as e:
        print(f"[compare] Could not write CSV: {e}")


if __name__ == "__main__":
    main()


