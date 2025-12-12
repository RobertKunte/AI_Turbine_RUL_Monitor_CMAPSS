from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _write_md_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("### Run Registry Export\n\nNo runs found.\n", encoding="utf-8")
        return

    cols = ["started_at", "status", "dataset", "experiment_name", "run_id"]
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    lines = [header, sep]
    for r in rows:
        line = "| " + " | ".join(_md_escape(str(r.get(c, ""))) for c in cols) + " |\n"
        lines.append(line)
    path.write_text("### Run Registry Export\n\n" + "".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a lightweight run-registry status snapshot for Git commits.")
    parser.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUN_REGISTRY_DB", str(Path("artifacts") / "run_registry.sqlite")),
        help="Path to run_registry sqlite DB",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("docs") / "status"),
        help="Output directory for exported status files (tracked by git)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of latest runs to export")
    args = parser.parse_args(argv)

    from src.tools.run_registry import RunRegistry

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)

    reg = RunRegistry(db_path)
    try:
        runs = reg.list_runs(limit=int(args.limit))
    finally:
        reg.close()

    rows = [
        {
            "run_id": r.run_id,
            "experiment_name": r.experiment_name,
            "dataset": r.dataset,
            "status": r.status,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "results_dir": r.results_dir,
            "artifact_root": r.artifact_root,
            "summary_path": r.summary_path,
        }
        for r in runs
    ]

    payload = {
        "exported_at": _utc_now_iso(),
        "db_path": str(db_path),
        "limit": int(args.limit),
        "runs": rows,
    }

    _json_dump(out_dir / "run_registry_export.json", payload)
    _write_md_table(out_dir / "run_registry_export.md", rows)
    print(f"[export_status] Wrote: {out_dir / 'run_registry_export.json'}")
    print(f"[export_status] Wrote: {out_dir / 'run_registry_export.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


