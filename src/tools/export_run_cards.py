from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt_float(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _drive_paths(dataset: str, run_name: str, run_id: str) -> Tuple[str, str]:
    base = "/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS"
    drive_results = f"{base}/results/{dataset.lower()}/{run_name}/"
    drive_artifacts = f"{base}/artifacts/runs/{run_id}/"
    return drive_results, drive_artifacts


def build_run_card_markdown(
    *,
    run_id: str,
    dataset: str,
    run_name: str,
    status: str,
    started_at: str,
    finished_at: Optional[str],
    git_sha: Optional[str],
    results_dir: Optional[str],
    artifact_root: Optional[str],
    config: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
) -> str:
    drive_results, drive_artifacts = _drive_paths(dataset, run_name, run_id)

    # Pull common metrics from summary if available (no invented numbers)
    test_metrics = _safe_get(summary or {}, ["test_metrics"])
    val_metrics = _safe_get(summary or {}, ["val_metrics"])

    # Inputs / dependencies (best-effort; keep stable and readable)
    cfg = config or {}
    dep_keys = [
        "encoder_experiment",
        "encoder_checkpoint",
        "encoder_type",
        "model_type",
        "hi_calibrator_path",
        "scaler_path",
        "base_run",
        "decoder_run",
        "dataset",
    ]
    deps: List[Tuple[str, Any]] = []
    for k in dep_keys:
        if k in cfg and cfg.get(k) not in (None, "", []):
            deps.append((k, cfg.get(k)))

    # Also scan nested config blocks commonly used in this repo
    for k in ["encoder_kwargs", "training_params", "loss_params", "phys_features", "features"]:
        if isinstance(cfg.get(k), dict):
            # keep as a small reference only (not the full blob)
            deps.append((k, "(present)"))

    def _metric_block(m: Any) -> str:
        if not isinstance(m, dict):
            return "n/a"
        # Prefer the standardized LAST keys; fall back to legacy keys if needed.
        if "rmse_last" in m or "mae_last" in m:
            rmse = _fmt_float(m.get("rmse_last"), 3)
            mae = _fmt_float(m.get("mae_last"), 3)
            bias = _fmt_float(m.get("bias_last"), 3)
            r2 = _fmt_float(m.get("r2_last"), 4)
            nasa = _fmt_float(m.get("nasa_last_mean"), 3)
            n_units = m.get("n_units", None)
            n_units_s = f", n_units={int(n_units)}" if n_units is not None else ""
            return f"LAST: RMSE={rmse}, MAE={mae}, Bias={bias}, R²={r2}, NASA_mean={nasa}{n_units_s}"

        rmse = _fmt_float(m.get("rmse"), 3)
        mae = _fmt_float(m.get("mae"), 3)
        bias = _fmt_float(m.get("bias"), 3)
        r2 = _fmt_float(m.get("r2"), 4)
        nasa = _fmt_float(m.get("nasa_mean"), 3)
        return f"RMSE={rmse}, MAE={mae}, Bias={bias}, R²={r2}, NASA_mean={nasa}"

    lines: List[str] = []
    lines.append(f"### Run Card: `{run_name}`")
    lines.append("")
    lines.append(f"- **run_id**: `{run_id}`")
    lines.append(f"- **dataset**: `{dataset}`")
    lines.append(f"- **status**: `{status}`")
    lines.append(f"- **started_at (UTC)**: `{started_at}`")
    lines.append(f"- **finished_at (UTC)**: `{finished_at or 'n/a'}`")
    lines.append(f"- **git_sha**: `{git_sha or 'n/a'}`")
    lines.append("")
    lines.append("#### Locations")
    lines.append(f"- **Drive results**: `{drive_results}`")
    lines.append(f"- **Drive artifacts**: `{drive_artifacts}`")
    lines.append(f"- **Local results_dir**: `{results_dir or 'n/a'}`")
    lines.append(f"- **Local artifact_root**: `{artifact_root or 'n/a'}`")
    lines.append("")
    lines.append("#### Inputs / dependencies (from registry config_json, best-effort)")
    if deps:
        for k, v in deps:
            lines.append(f"- **{k}**: `{str(v)}`")
    else:
        lines.append("- n/a")
    lines.append("")
    lines.append("#### Colab preload suggestion")
    lines.append(f"- `PRELOAD_RESULTS_RUNS = [\"{run_name}\"]`  (if a later run depends on this run’s results)")
    lines.append("")
    lines.append("#### Metrics (from summary.json, if available)")
    lines.append(f"- **val**: {_metric_block(val_metrics)}")
    lines.append(f"- **test**: {_metric_block(test_metrics)}")
    lines.append("")
    lines.append("#### Notes / Gates")
    lines.append("- For FD004 test: remember it is **right-censored** (last observed cycle).")
    lines.append("- If this run is used as an input to another run (decoder/world model), preload its `results/` from Drive in Colab.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export per-run markdown 'Run Cards' from the SQLite registry.")
    parser.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUN_REGISTRY_DB", str(Path("artifacts") / "run_registry.sqlite")),
        help="Path to run_registry sqlite DB",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("docs") / "status" / "run_cards"),
        help="Output directory for run cards (tracked by git)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of latest runs to export")
    args = parser.parse_args(argv)

    from src.tools.run_registry import RunRegistry

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = RunRegistry(db_path)
    try:
        # Use list_runs for ordering and minimal selection
        rows = reg.list_runs(limit=int(args.limit))
        # For full run details, query each run_id
        full = [reg.get_run(r.run_id) for r in rows]
    finally:
        reg.close()

    index_lines: List[str] = []
    index_lines.append("### Run Cards Index")
    index_lines.append("")
    index_lines.append(f"- exported_at (UTC): `{_utc_now_iso()}`")
    index_lines.append(f"- db_path: `{db_path}`")
    index_lines.append("")
    index_lines.append("| started_at | status | dataset | run_name | run_id | card |")
    index_lines.append("| --- | --- | --- | --- | --- | --- |")

    written = 0
    for rec in full:
        run_id = str(rec.get("run_id"))
        run_name = str(rec.get("experiment_name") or "")
        dataset = str(rec.get("dataset") or "")
        status = str(rec.get("status") or "")
        started_at = str(rec.get("started_at") or "")
        finished_at = rec.get("finished_at")
        git_sha = rec.get("git_sha")
        results_dir = rec.get("results_dir")
        artifact_root = rec.get("artifact_root")
        config_json = rec.get("config_json")
        cfg = config_json if isinstance(config_json, dict) else None

        summary_path = rec.get("summary_path")
        summary = _json_load(Path(summary_path)) if isinstance(summary_path, str) and summary_path else None

        md = build_run_card_markdown(
            run_id=run_id,
            dataset=dataset,
            run_name=run_name,
            status=status,
            started_at=started_at,
            finished_at=str(finished_at) if finished_at is not None else None,
            git_sha=str(git_sha) if git_sha is not None else None,
            results_dir=str(results_dir) if results_dir is not None else None,
            artifact_root=str(artifact_root) if artifact_root is not None else None,
            config=cfg,
            summary=summary,
        )

        card_path = out_dir / f"{run_id}.md"
        card_path.write_text(md, encoding="utf-8")
        written += 1

        rel = f"run_cards/{run_id}.md"
        index_lines.append(
            f"| `{started_at}` | `{status}` | `{dataset}` | `{run_name}` | `{run_id}` | [{run_id}]({rel}) |"
        )

    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[export_run_cards] Wrote {written} run cards to {out_dir}")
    print(f"[export_run_cards] Wrote index: {out_dir / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


