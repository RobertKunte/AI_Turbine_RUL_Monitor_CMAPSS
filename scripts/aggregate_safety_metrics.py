"""
Aggregate safety metrics (μ vs safe) across runs into a single CSV/Markdown table.

Reads per-run files:
  - results/<dataset>/<run_name>/overshoot_metrics_mu_vs_safe.json
  - results/<dataset>/<run_name>/summary.json (optional, for convenience fields)

Usage:
  python -u scripts/aggregate_safety_metrics.py --results-dir results --output results/safety_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}.", v, out)
    else:
        out[prefix[:-1]] = obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output", type=Path, default=Path("results/safety_metrics.csv"))
    ap.add_argument("--output-md", type=Path, default=None, help="Optional Markdown output path")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for run_dir in args.results_dir.rglob("overshoot_metrics_mu_vs_safe.json"):
        exp_dir = run_dir.parent
        # Expected structure: results/<dataset>/<run_name>/
        dataset = exp_dir.parent.name
        run_name = exp_dir.name

        m = _load_json(run_dir) or {}
        s = _load_json(exp_dir / "summary.json") or {}

        row: Dict[str, Any] = {
            "dataset": dataset,
            "run_name": run_name,
            "path": str(exp_dir),
            # Convenience: risk controls
            "risk_tau": m.get("risk_tau"),
            "low_rul_threshold": m.get("low_rul_threshold"),
            "overshoot_threshold": m.get("overshoot_threshold"),
        }

        # Pull canonical μ test metrics from summary if present
        tm = s.get("test_metrics") if isinstance(s.get("test_metrics"), dict) else {}
        if isinstance(tm, dict):
            for k in ["rmse", "mae", "bias", "r2", "nasa_mean", "nasa_sum"]:
                row[f"summary.test_metrics.{k}"] = tm.get(k)

        # Flatten safety metrics JSON
        flat: Dict[str, Any] = {}
        _flatten("risk.", m, flat)
        row.update(flat)

        rows.append(row)

    if not rows:
        raise SystemExit(f"No overshoot_metrics_mu_vs_safe.json found under: {args.results_dir}")

    df = pd.DataFrame(rows)
    # Helpful default ordering
    sort_cols = [c for c in ["dataset", "run_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} (rows={len(df)})")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(df.to_markdown(index=False), encoding="utf-8")
        print(f"Wrote: {args.output_md}")


if __name__ == "__main__":
    main()


