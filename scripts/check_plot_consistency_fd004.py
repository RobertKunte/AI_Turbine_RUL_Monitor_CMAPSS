"""
Colab/CLI helper: verify that diagnostics plots (error_hist) and risk plots (overshoot μ vs safe)
are consistent for a given run directory, and regenerate "check" plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Headless plotting (Colab-safe)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so `import src...` works when called as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _plot_hist(
    *,
    values: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
    vline0: bool = True,
    alpha: float = 0.7,
    color: str = "tab:blue",
    label: str | None = None,
) -> None:
    plt.figure(figsize=(9, 4.5))
    v = values[np.isfinite(values)]
    plt.hist(v, bins=bins, alpha=alpha, label=label, color=color)
    if vline0:
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    if label is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_overlay_hist(
    *,
    a: np.ndarray,
    b: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    label_a: str,
    label_b: str,
    bins: int = 60,
) -> None:
    plt.figure(figsize=(9, 4.5))
    va = a[np.isfinite(a)]
    vb = b[np.isfinite(b)]
    # Use shared bin edges so overlays are visually comparable.
    v_all = np.concatenate([va, vb]) if (va.size + vb.size) > 0 else np.array([0.0])
    lo = float(np.nanmin(v_all))
    hi = float(np.nanmax(v_all))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    plt.hist(va, bins=edges, alpha=0.55, label=label_a, color="tab:blue")
    plt.hist(vb, bins=edges, alpha=0.55, label=label_b, color="tab:orange")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _summ(arr: np.ndarray) -> Dict[str, Any]:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def _errors_multiset_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> Tuple[bool, float]:
    """
    Compare two arrays ignoring ordering by sorting.
    Returns (ok, max_abs_diff_sorted).
    """
    aa = np.sort(a[np.isfinite(a)])
    bb = np.sort(b[np.isfinite(b)])
    if aa.shape != bb.shape:
        return False, float("inf")
    return bool(np.max(np.abs(aa - bb)) <= atol), float(np.max(np.abs(aa - bb)))


def _try_load_compare_csv(run_dir: Path) -> Dict[str, np.ndarray] | None:
    """
    If present, load the outputs of scripts/compare_diagnostics_vs_inference_fd004.py.

    Expected columns:
      unit_id,true_diag,pred_diag,true_inf,pred_inf,...
    """
    csv_path = run_dir / "compare_diagnostics_vs_inference.csv"
    if not csv_path.exists():
        return None

    try:
        import csv

        unit_id: list[int] = []
        true_diag: list[float] = []
        pred_diag: list[float] = []
        true_inf: list[float] = []
        pred_inf: list[float] = []

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"unit_id", "true_diag", "pred_diag", "true_inf", "pred_inf"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                return None
            for row in reader:
                unit_id.append(int(float(row["unit_id"])))
                true_diag.append(float(row["true_diag"]))
                pred_diag.append(float(row["pred_diag"]))
                true_inf.append(float(row["true_inf"]))
                pred_inf.append(float(row["pred_inf"]))

        return {
            "unit_id": np.asarray(unit_id, dtype=int),
            "true_diag": np.asarray(true_diag, dtype=float),
            "pred_diag": np.asarray(pred_diag, dtype=float),
            "true_inf": np.asarray(true_inf, dtype=float),
            "pred_inf": np.asarray(pred_inf, dtype=float),
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to results/<dataset>/<run_name> directory.",
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output directory for check artifacts (default: run_dir).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(run_dir / "summary.json")
    max_rul = float(summary.get("max_rul", 125.0))

    # Load artifacts written by diagnostics/inference
    eol_metrics_path = run_dir / "eol_metrics.json"
    overshoot_path = run_dir / "overshoot_metrics_mu_vs_safe.json"
    eol_metrics = _load_json(eol_metrics_path)
    overshoot_metrics = _load_json(overshoot_path) if overshoot_path.exists() else None

    errors_from_eol = np.asarray(eol_metrics.get("errors", []), dtype=float).reshape(-1)
    if errors_from_eol.size == 0:
        raise RuntimeError(f"{eol_metrics_path} has no 'errors' array")

    # Re-run inference to reconstruct μ, risk_q, safe distributions
    import torch
    from src.analysis.inference import run_inference_for_experiment

    device = torch.device(args.device)
    eol_metrics_inf, _ = run_inference_for_experiment(
        experiment_dir=run_dir,
        split="test",
        device=device,
        # Must be True for residual-risk runs: risk_q is only extracted from the full forward output,
        # which is only executed when return_hi_trajectories=True in run_inference_for_experiment.
        return_hi_trajectories=True,
    )
    # Sort by unit_id for deterministic arrays
    eol_sorted = sorted(eol_metrics_inf, key=lambda m: int(m.unit_id))
    y_true = np.array([float(m.true_rul) for m in eol_sorted], dtype=float)
    mu = np.array([float(m.pred_rul) for m in eol_sorted], dtype=float)
    risk_q = np.array(
        [float(m.risk_q) if getattr(m, "risk_q", None) is not None else np.nan for m in eol_sorted],
        dtype=float,
    )
    risk_pred = np.maximum(0.0, np.nan_to_num(risk_q, nan=0.0))
    safe = np.clip(mu - risk_pred, 0.0, max_rul)

    errors_from_inf = (mu - y_true).astype(float)
    overshoot_safe = (safe - y_true).astype(float)

    # Consistency checks
    ok_multiset, max_abs_sorted = _errors_multiset_close(errors_from_eol, errors_from_inf, atol=1e-5)

    # Optional: if compare CSV exists, use it as an additional alignment cross-check.
    compare = _try_load_compare_csv(run_dir)
    compare_section: Dict[str, Any] | None = None
    stale_eol_metrics_hint: str | None = None
    if compare is not None:
        err_csv_diag = (compare["pred_diag"] - compare["true_diag"]).astype(float)
        err_csv_inf = (compare["pred_inf"] - compare["true_inf"]).astype(float)

        ok_csv_diag_inf, max_abs_csv_diag_inf = _errors_multiset_close(err_csv_diag, err_csv_inf, atol=1e-6)
        ok_csv_inf_vs_script_inf, max_abs_csv_inf_vs_script_inf = _errors_multiset_close(err_csv_inf, errors_from_inf, atol=1e-5)
        ok_csv_diag_vs_eol, max_abs_csv_diag_vs_eol = _errors_multiset_close(err_csv_diag, errors_from_eol, atol=1e-5)

        compare_section = {
            "csv_path": str(run_dir / "compare_diagnostics_vs_inference.csv"),
            "n": int(err_csv_diag.size),
            "errors_from_compare_diag": _summ(err_csv_diag),
            "errors_from_compare_inf": _summ(err_csv_inf),
            "csv_diag_vs_csv_inf_multiset_equal_sorted": bool(ok_csv_diag_inf),
            "csv_diag_vs_csv_inf_max_abs_diff_sorted": float(max_abs_csv_diag_inf),
            "csv_inf_vs_script_inference_multiset_equal_sorted": bool(ok_csv_inf_vs_script_inf),
            "csv_inf_vs_script_inference_max_abs_diff_sorted": float(max_abs_csv_inf_vs_script_inf),
            "csv_diag_vs_eol_metrics_multiset_equal_sorted": bool(ok_csv_diag_vs_eol),
            "csv_diag_vs_eol_metrics_max_abs_diff_sorted": float(max_abs_csv_diag_vs_eol),
        }

        # If CSV says diag==inf and CSV matches inference, but eol_metrics differs,
        # then eol_metrics.json is likely stale (generated by older code).
        if ok_csv_diag_inf and ok_csv_inf_vs_script_inf and (not ok_csv_diag_vs_eol):
            stale_eol_metrics_hint = (
                "eol_metrics.json appears inconsistent with current inference/diagnostics. "
                "Most likely it is stale (generated by older code) or belongs to a different artifact snapshot. "
                "Re-run diagnostics to regenerate eol_metrics.json / error_hist.png for this run_dir."
            )

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "max_rul": float(max_rul),
        "errors_from_eol_metrics": _summ(errors_from_eol),
        "errors_from_inference_mu": _summ(errors_from_inf),
        "overshoot_safe": _summ(overshoot_safe),
        "errors_multiset_equal_sorted": bool(ok_multiset),
        "max_abs_diff_sorted_errors": float(max_abs_sorted),
        "compare_diagnostics_vs_inference_csv": compare_section,
        "stale_eol_metrics_hint": stale_eol_metrics_hint,
        "notes": {
            "meaning": "errors are pred-true at last observed cycle (FD004 right-censored).",
            "expectation": "errors_from_eol_metrics should match errors_from_inference_mu as a multiset.",
        },
    }
    _save_json(out_dir / "check_consistency_report.json", report)

    # Regenerate check plots
    _plot_hist(
        values=errors_from_eol,
        out_path=out_dir / "check_error_hist_from_eol_metrics.png",
        title=f"Check: error_hist from eol_metrics.json (mean={report['errors_from_eol_metrics']['mean']:.2f}, std={report['errors_from_eol_metrics']['std']:.2f})",
        xlabel="error = pred - true (cycles)",
        bins=60,
        vline0=True,
    )
    _plot_hist(
        values=errors_from_inf,
        out_path=out_dir / "check_error_hist_from_inference_mu.png",
        title=f"Check: error_hist from inference μ (mean={report['errors_from_inference_mu']['mean']:.2f}, std={report['errors_from_inference_mu']['std']:.2f})",
        xlabel="error = μ - true (cycles)",
        bins=60,
        vline0=True,
    )
    _plot_overlay_hist(
        a=errors_from_eol,
        b=errors_from_inf,
        out_path=out_dir / "check_error_hist_overlay_eol_vs_mu.png",
        title="Check overlay: errors (eol_metrics.json) vs errors (inference μ)",
        xlabel="error (cycles)",
        label_a="eol_metrics.json errors",
        label_b="inference μ - true",
        bins=60,
    )
    _plot_overlay_hist(
        a=errors_from_inf,
        b=overshoot_safe,
        out_path=out_dir / "check_overshoot_hist_mu_vs_safe.png",
        title="Check: overshoot histogram μ vs safe (last observed / right-censored)",
        xlabel="overshoot = pred - true (cycles)",
        label_a="μ overshoot",
        label_b="safe overshoot",
        bins=60,
    )

    # Also record what the existing overshoot_metrics file (if any) says (for quick eyeballing).
    if overshoot_metrics is not None:
        _save_json(out_dir / "check_overshoot_metrics_original.json", overshoot_metrics)

    print("\n=== check_plot_consistency_fd004 ===")
    print("run_dir:", run_dir)
    print("out_dir:", out_dir)
    print("multiset_equal_sorted:", ok_multiset, "max_abs_sorted_diff:", max_abs_sorted)
    print("wrote:", out_dir / "check_consistency_report.json")
    print("wrote check plots:")
    print(" -", out_dir / "check_error_hist_from_eol_metrics.png")
    print(" -", out_dir / "check_error_hist_from_inference_mu.png")
    print(" -", out_dir / "check_error_hist_overlay_eol_vs_mu.png")
    print(" -", out_dir / "check_overshoot_hist_mu_vs_safe.png")


if __name__ == "__main__":
    main()


