from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import sys
import traceback as tb
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        try:
            return float(o)
        except Exception:
            return str(o)

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_default)


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove common secret-like keys before storing in the registry.
    This is best-effort; do not rely on it as the only protection.
    """
    deny_substrings = [
        "api_key",
        "apikey",
        "openai",
        "gemini",
        "token",
        "secret",
        "password",
        "key",
        "auth",
        "bearer",
    ]

    def _should_redact(k: str) -> bool:
        kl = k.lower()
        return any(s in kl for s in deny_substrings)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(k, str) and _should_redact(k):
                    out[k] = "<REDACTED>"
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [_walk(x) for x in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    return _walk(config)


@dataclass(frozen=True)
class RunRow:
    run_id: str
    experiment_name: str
    dataset: str
    status: str
    started_at: str
    finished_at: Optional[str]
    results_dir: Optional[str]
    summary_path: Optional[str]
    artifact_root: Optional[str]


class RunRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self.ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _apply_pragmas(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        self._conn.commit()

    def ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              experiment_name TEXT NOT NULL,
              dataset TEXT NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              host TEXT,
              platform TEXT,
              python TEXT,
              git_sha TEXT,
              results_dir TEXT,
              artifact_root TEXT,
              summary_path TEXT,
              config_json TEXT,
              metrics_json TEXT,
              error_message TEXT,
              traceback TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              path TEXT NOT NULL,
              kind TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )

        # Schema versioning + minimal migrations
        cur.execute("SELECT value FROM meta WHERE key='schema_version';")
        row = cur.fetchone()
        current = int(row["value"]) if row is not None and row["value"] is not None else 0

        # Migration to v2: add runs.artifact_root if missing (ALTER TABLE)
        if current < 2:
            cur.execute("PRAGMA table_info(runs);")
            cols = {str(r["name"]) for r in cur.fetchall()}
            if "artifact_root" not in cols:
                cur.execute("ALTER TABLE runs ADD COLUMN artifact_root TEXT;")
            current = 2

        if row is None:
            cur.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?);", (str(current),))
        else:
            cur.execute("UPDATE meta SET value=? WHERE key='schema_version';", (str(current),))
        self._conn.commit()

    def start_run(
        self,
        *,
        experiment_name: str,
        dataset: str,
        config: Dict[str, Any],
        results_dir: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
        git_sha: Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        host = platform.node()
        plat = platform.platform()
        py = sys.version.replace("\n", " ")

        cfg = sanitize_config(config)
        cfg_json = _json_dumps(cfg)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO runs(
              run_id, experiment_name, dataset, status, started_at,
              host, platform, python, git_sha,
              results_dir, artifact_root, config_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                experiment_name,
                dataset,
                "running",
                started_at,
                host,
                plat,
                py,
                git_sha,
                str(results_dir) if results_dir is not None else None,
                str(artifact_root) if artifact_root is not None else None,
                cfg_json,
            ),
        )
        self._conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        summary_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
        status: str = "success",
    ) -> None:
        finished_at = _utc_now_iso()
        metrics_json = _json_dumps(metrics) if metrics is not None else None
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE runs
            SET status=?,
                finished_at=?,
                results_dir=COALESCE(?, results_dir),
                artifact_root=COALESCE(?, artifact_root),
                summary_path=COALESCE(?, summary_path),
                metrics_json=COALESCE(?, metrics_json)
            WHERE run_id=?;
            """,
            (
                status,
                finished_at,
                str(results_dir) if results_dir is not None else None,
                str(artifact_root) if artifact_root is not None else None,
                str(summary_path) if summary_path is not None else None,
                metrics_json,
                run_id,
            ),
        )
        self._conn.commit()

    def fail_run(
        self,
        run_id: str,
        *,
        error_message: str,
        traceback_str: Optional[str] = None,
        summary_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
    ) -> None:
        finished_at = _utc_now_iso()
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE runs
            SET status='failed',
                finished_at=?,
                results_dir=COALESCE(?, results_dir),
                artifact_root=COALESCE(?, artifact_root),
                summary_path=COALESCE(?, summary_path),
                error_message=?,
                traceback=?
            WHERE run_id=?;
            """,
            (
                finished_at,
                str(results_dir) if results_dir is not None else None,
                str(artifact_root) if artifact_root is not None else None,
                str(summary_path) if summary_path is not None else None,
                error_message,
                traceback_str,
                run_id,
            ),
        )
        self._conn.commit()

    def set_artifact_root(self, run_id: str, artifact_root: Path) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET artifact_root=? WHERE run_id=?;",
            (str(artifact_root), run_id),
        )
        self._conn.commit()

    def log_artifact(self, run_id: str, path: Path, *, kind: Optional[str] = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO artifacts(run_id, path, kind, created_at)
            VALUES(?, ?, ?, ?);
            """,
            (run_id, str(path), kind, _utc_now_iso()),
        )
        self._conn.commit()

    def list_runs(
        self,
        *,
        limit: int = 20,
        dataset: Optional[str] = None,
        experiment: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunRow]:
        where = []
        params: List[Any] = []
        if dataset:
            where.append("dataset = ?")
            params.append(dataset)
        if experiment:
            where.append("experiment_name = ?")
            params.append(experiment)
        if status:
            where.append("status = ?")
            params.append(status)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT run_id, experiment_name, dataset, status, started_at, finished_at, results_dir, artifact_root, summary_path
            FROM runs
            {where_sql}
            ORDER BY started_at DESC
            LIMIT ?;
        """
        params.append(int(limit))
        cur = self._conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        out: List[RunRow] = []
        for r in rows:
            out.append(
                RunRow(
                    run_id=str(r["run_id"]),
                    experiment_name=str(r["experiment_name"]),
                    dataset=str(r["dataset"]),
                    status=str(r["status"]),
                    started_at=str(r["started_at"]),
                    finished_at=str(r["finished_at"]) if r["finished_at"] is not None else None,
                    results_dir=str(r["results_dir"]) if r["results_dir"] is not None else None,
                    artifact_root=str(r["artifact_root"]) if r["artifact_root"] is not None else None,
                    summary_path=str(r["summary_path"]) if r["summary_path"] is not None else None,
                )
            )
        return out

    def find_latest_run_id(
        self,
        *,
        experiment_name: str,
        dataset: str,
        status: str = "running",
    ) -> Optional[str]:
        """
        Find the most recent run_id for a given (experiment_name, dataset, status).
        Used to mark a run as failed from an outer exception handler.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT run_id
            FROM runs
            WHERE experiment_name = ? AND dataset = ? AND status = ?
            ORDER BY started_at DESC
            LIMIT 1;
            """,
            (experiment_name, dataset, status),
        )
        row = cur.fetchone()
        return str(row["run_id"]) if row is not None else None

    def find_latest_run_id_any(
        self,
        *,
        experiment_name: Optional[str] = None,
        dataset: Optional[str] = None,
        prefer_success: bool = True,
    ) -> Optional[str]:
        """
        Find the latest run_id, optionally filtered by experiment_name/dataset.
        If prefer_success=True, try latest success first, then fall back to latest any-status.
        """
        cur = self._conn.cursor()

        where = []
        params: List[Any] = []
        if experiment_name:
            where.append("experiment_name = ?")
            params.append(experiment_name)
        if dataset:
            where.append("dataset = ?")
            params.append(dataset)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        if prefer_success:
            cur.execute(
                f"""
                SELECT run_id FROM runs
                {where_sql} {"AND" if where_sql else "WHERE"} status='success'
                ORDER BY started_at DESC
                LIMIT 1;
                """,
                tuple(params),
            )
            row = cur.fetchone()
            if row is not None:
                return str(row["run_id"])

        cur.execute(
            f"""
            SELECT run_id FROM runs
            {where_sql}
            ORDER BY started_at DESC
            LIMIT 1;
            """,
            tuple(params),
        )
        row2 = cur.fetchone()
        return str(row2["run_id"]) if row2 is not None else None

    def get_run(self, run_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM runs WHERE run_id=?;", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"run_id not found: {run_id}")
        d = dict(row)
        # Best-effort parse JSON blobs
        for k in ["config_json", "metrics_json"]:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d


def _default_db_path() -> Path:
    # Prefer artifacts/, fall back to results/ if artifacts is not desired.
    return Path("artifacts") / "run_registry.sqlite"


def _print_runs_table(runs: Sequence[RunRow]) -> None:
    if not runs:
        print("No runs found.")
        return

    cols = ["started_at", "status", "dataset", "experiment_name", "run_id"]
    widths = {c: len(c) for c in cols}
    for r in runs:
        widths["started_at"] = max(widths["started_at"], len(r.started_at))
        widths["status"] = max(widths["status"], len(r.status))
        widths["dataset"] = max(widths["dataset"], len(r.dataset))
        widths["experiment_name"] = max(widths["experiment_name"], len(r.experiment_name))
        widths["run_id"] = max(widths["run_id"], len(r.run_id))

    def _fmt_row(vals: Dict[str, str]) -> str:
        return " | ".join(vals[c].ljust(widths[c]) for c in cols)

    print(_fmt_row({c: c for c in cols}))
    print("-" * (sum(widths.values()) + 3 * (len(cols) - 1)))
    for r in runs:
        print(
            _fmt_row(
                {
                    "started_at": r.started_at,
                    "status": r.status,
                    "dataset": r.dataset,
                    "experiment_name": r.experiment_name,
                    "run_id": r.run_id,
                }
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SQLite run registry (Phase 1)")
    parser.add_argument("--db", type=str, default=str(_default_db_path()), help="Path to sqlite DB file")
    parser.add_argument("--list", type=int, default=20, help="List latest N runs")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., FD004)")
    parser.add_argument("--experiment", type=str, default=None, help="Filter by experiment_name")
    parser.add_argument("--status", type=str, default=None, help="Filter by status (running/success/failed)")
    parser.add_argument("--show", type=str, default=None, help="Show a run record (use 'latest' or a run_id)")
    parser.add_argument("--export", type=str, default=None, help="Export a run record (use 'latest' or a run_id)")
    parser.add_argument("--format", type=str, default="json", choices=["json"], help="Export format")
    args = parser.parse_args(argv)

    reg = RunRegistry(Path(args.db))
    try:
        if args.show is not None:
            token = str(args.show).strip()
            if token == "latest":
                rid = reg.find_latest_run_id_any(prefer_success=False)
                if rid is None:
                    print("No runs found.")
                    return 0
                token = rid
            rec = reg.get_run(token)
            print(_json_dumps(rec))
            return 0

        if args.export is not None:
            token = str(args.export).strip()
            if token == "latest":
                rid = reg.find_latest_run_id_any(prefer_success=False)
                if rid is None:
                    print("No runs found.")
                    return 0
                token = rid
            rec = reg.get_run(token)
            if args.format == "json":
                print(_json_dumps(rec))
            return 0

        runs = reg.list_runs(
            limit=int(args.list),
            dataset=args.dataset,
            experiment=args.experiment,
            status=args.status,
        )
        _print_runs_table(runs)
    finally:
        reg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


