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

    import torch
    from src.analysis.diagnostics import run_diagnostics_for_run

    device = torch.device(args.device)
    run_diagnostics_for_run(
        exp_dir=Path(exp_dir),
        dataset_name=dataset_name,
        run_name=run_name,
        device=device,
    )

    print("Done. Regenerated diagnostics in:", run_dir)


if __name__ == "__main__":
    main()


