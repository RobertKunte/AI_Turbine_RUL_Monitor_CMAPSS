"""
CLI helper to (re)generate diagnostics artifacts for an existing run directory.

Why this exists:
- Some older run folders contain stale `eol_metrics.json` / plots produced by older code
  (e.g. before engine/target alignment fixes).
- This script makes regeneration explicit and repeatable, and records provenance in
  `eol_metrics.json["_meta"]` (written by src/analysis/diagnostics.py).

Usage examples:
  python -u scripts/regenerate_diagnostics.py --run_dir results/fd004/<run_name> --device cpu
  python -u scripts/regenerate_diagnostics.py --dataset FD004 --run_name <run_name> --device cuda --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


# Ensure repo root is on sys.path so `import src...` works when called as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except Exception:
        return None


def _infer_dataset_and_run_name(run_dir: Path) -> Tuple[str, str]:
    """
    Infer dataset and run_name from a run_dir of the form:
      results/<dataset>/<run_name>
    """
    run_dir = run_dir.resolve()
    run_name = run_dir.name
    dataset = run_dir.parent.name  # e.g. "fd004"
    if dataset.lower().startswith("fd"):
        dataset = dataset.upper()
    return dataset, run_name


def _load_eol_metrics_meta(run_dir: Path) -> dict:
    p = run_dir / "eol_metrics.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("_meta", {}) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _pick_checkpoint_file(run_dir: Path) -> Optional[Path]:
    pts = sorted(run_dir.glob("*.pt"))
    if not pts:
        return None
    best = [p for p in pts if "best" in p.name.lower()]
    return best[0] if best else pts[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to results/<dataset>/<run_name> directory.",
    )
    ap.add_argument("--dataset", type=str, default=None, choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--run_name", type=str, default=None, help="Experiment name (folder name under results/<dataset>/)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate diagnostics even if existing eol_metrics.json appears up-to-date.",
    )
    args = ap.parse_args()

    if args.run_dir is None and (args.dataset is None or args.run_name is None):
        raise SystemExit("Provide either --run_dir or both --dataset and --run_name.")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        dataset_name, run_name = _infer_dataset_and_run_name(run_dir)
        exp_dir = run_dir.parent.parent  # results/
    else:
        dataset_name = str(args.dataset)
        run_name = str(args.run_name)
        exp_dir = Path("results")
        run_dir = exp_dir / dataset_name.lower() / run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    current_sha = _get_git_sha()
    meta = _load_eol_metrics_meta(run_dir)
    metrics_sha = meta.get("generated_git_sha")

    needs_regen = args.force or (metrics_sha is None) or (current_sha is not None and metrics_sha != current_sha)

    print("\n=== regenerate_diagnostics ===")
    print("run_dir:", run_dir)
    print("dataset:", dataset_name)
    print("run_name:", run_name)
    print("device:", args.device)
    print("current_git_sha:", current_sha)
    print("eol_metrics._meta.generated_git_sha:", metrics_sha)
    print("force:", bool(args.force))
    print("will_regenerate:", bool(needs_regen))

    if not needs_regen:
        print("OK: diagnostics appear up-to-date; nothing to do.")
        return

    # Preflight: ensure checkpoint file is readable. If it's truncated/corrupted,
    # diagnostics cannot run and we should fail loudly with actionable instructions.
    ckpt_path = _pick_checkpoint_file(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pt checkpoint found in: {run_dir}")
    try:
        import torch

        # Quick load to validate file integrity; we don't keep the object.
        _ = torch.load(ckpt_path, map_location="cpu")
    except EOFError:
        size = None
        try:
            size = ckpt_path.stat().st_size
        except Exception:
            pass
        print("\n[ERROR] Checkpoint file appears truncated/corrupted (EOFError):", ckpt_path)
        if size is not None:
            print("        size_bytes:", int(size))
        print("\nAction:")
        print("  1) Delete the local broken checkpoint file (copy-only-newer sync may not overwrite it):")
        print(f"     rm -f '{ckpt_path}'")
        print("  2) Re-pull the run from Drive (results + artifacts):")
        print("     python -m src.tools.sync_artifacts --pull --run_name "
              f"{run_name} --what both")
        print("  3) Re-run this script.")
        raise SystemExit(2)
    except Exception as e:
        print("\n[ERROR] Could not load checkpoint file:", ckpt_path)
        print("Reason:", repr(e))
        raise SystemExit(2)

    from src.analysis.diagnostics import run_diagnostics_for_run

    device = torch.device(args.device)
    run_diagnostics_for_run(
        exp_dir=Path(exp_dir),
        dataset_name=dataset_name,
        run_name=run_name,
        device=device,
    )

    # Post-check: ensure summary.json and eol_metrics.json agree (same folder truth).
    def _load_json(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    eol_after = _load_json(run_dir / "eol_metrics.json")
    summary_after = _load_json(run_dir / "summary.json")

    def _pick_metrics_from_eol(eol: dict) -> dict:
        keys = ["rmse", "mae", "bias", "r2", "nasa_mean", "nasa_sum", "num_engines"]
        out = {}
        for k in keys:
            if k in eol:
                out[k] = eol[k]
        meta = eol.get("_meta")
        if isinstance(meta, dict):
            out["_meta.generated_git_sha"] = meta.get("generated_git_sha")
            out["_meta.generated_at_utc"] = meta.get("generated_at_utc")
        return out

    def _pick_metrics_from_summary(s: dict) -> dict:
        tm = s.get("test_metrics", {})
        out = dict(tm) if isinstance(tm, dict) else {}
        dm = s.get("diagnostics_meta")
        if isinstance(dm, dict):
            out["diagnostics_meta.generated_git_sha"] = dm.get("generated_git_sha")
            out["diagnostics_meta.generated_at_utc"] = dm.get("generated_at_utc")
        sm = s.get("_meta")
        if isinstance(sm, dict):
            out["summary._meta.generated_git_sha"] = sm.get("generated_git_sha")
        return out

    eol_m = _pick_metrics_from_eol(eol_after)
    sum_m = _pick_metrics_from_summary(summary_after)

    print("\n=== regenerate_diagnostics (post-check) ===")
    print("eol_metrics.json:", eol_m)
    print("summary.json:test_metrics:", sum_m)

    # Hard check for the core metrics (tolerate float formatting)
    for k in ["rmse", "mae", "bias", "r2", "nasa_mean", "nasa_sum"]:
        a = eol_after.get(k)
        b = (summary_after.get("test_metrics") or {}).get(k)
        if a is None or b is None:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-6:
                raise SystemExit(
                    f"[ERROR] Metrics mismatch after regeneration for key '{k}': "
                    f"eol_metrics.json={a} vs summary.json:test_metrics={b}. "
                    f"Run dir: {run_dir}"
                )
        except Exception:
            pass

    print("Done. Regenerated diagnostics in:", run_dir)


if __name__ == "__main__":
    main()


