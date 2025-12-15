from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


DRIVE_ROOT_DEFAULT = Path("/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS")


@dataclass(frozen=True)
class CopyResult:
    copied_files: int
    copied_bytes: int
    skipped_files: int


def _is_newer(src: Path, dst: Path) -> bool:
    """
    Copy-only-newer decision:
      - If dst missing => copy
      - Else compare mtime; if src newer => copy
      - If equal mtime, compare size; if different => copy
    """
    if not dst.exists():
        return True
    try:
        sm = src.stat().st_mtime
        dm = dst.stat().st_mtime
        if sm > dm + 1e-6:
            return True
        if abs(sm - dm) <= 1e-6:
            return src.stat().st_size != dst.stat().st_size
        return False
    except Exception:
        # If we cannot stat reliably, be conservative and copy
        return True


def _copy_tree_only_newer(src_root: Path, dst_root: Path) -> CopyResult:
    if not src_root.exists():
        raise FileNotFoundError(f"Source does not exist: {src_root}")

    copied_files = 0
    copied_bytes = 0
    skipped_files = 0

    for p in src_root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src_root)
        q = dst_root / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if _is_newer(p, q):
            shutil.copy2(p, q)
            copied_files += 1
            try:
                copied_bytes += int(p.stat().st_size)
            except Exception:
                pass
        else:
            skipped_files += 1

    return CopyResult(copied_files=copied_files, copied_bytes=copied_bytes, skipped_files=skipped_files)


def _resolve_registry(db_path: Path):
    from src.tools.run_registry import RunRegistry
    return RunRegistry(db_path)


def resolve_run_id(
    *,
    db_path: Path,
    run_id: Optional[str],
    run_name: Optional[str],
    latest: bool,
) -> str:
    if run_id:
        return run_id

    reg = _resolve_registry(db_path)
    try:
        if run_name:
            rid = reg.find_latest_run_id_any(experiment_name=run_name, prefer_success=True)
            if rid is None:
                rid = reg.find_latest_run_id_any(experiment_name=run_name, prefer_success=False)
            if rid is None:
                raise RuntimeError(f"Could not resolve run_id for run_name='{run_name}'")
            return rid
        if latest:
            rid = reg.find_latest_run_id_any(prefer_success=False)
            if rid is None:
                raise RuntimeError("Could not resolve latest run_id (registry empty?)")
            return rid
        raise RuntimeError("Must provide --run_id, --run_name, or --latest")
    finally:
        reg.close()


def resolve_artifact_root(
    *,
    db_path: Path,
    run_id: str,
) -> Path:
    """
    Prefer artifact_root from registry; otherwise fall back to convention:
      artifacts/runs/<run_id>/
    """
    reg = _resolve_registry(db_path)
    try:
        rec = reg.get_run(run_id)
        ar = rec.get("artifact_root")
        if isinstance(ar, str) and ar.strip():
            return Path(ar)
    except Exception:
        pass
    finally:
        try:
            reg.close()
        except Exception:
            pass
    return Path("artifacts") / "runs" / run_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync per-run artifacts between local and Google Drive (copy-only-newer).")
    parser.add_argument("--db", type=str, default=os.environ.get("RUN_REGISTRY_DB", str(Path("artifacts") / "run_registry.sqlite")))
    parser.add_argument("--push", action="store_true", help="Push local -> Drive")
    parser.add_argument("--pull", action="store_true", help="Pull Drive -> local")
    parser.add_argument("--run_id", type=str, default=None, help="Explicit run_id")
    parser.add_argument("--run_name", type=str, default=None, help="Resolve run_id via registry by experiment_name")
    parser.add_argument("--latest", action="store_true", help="Use latest run in registry")
    # Tolerate older Colab cells that accidentally pass training flags through.
    parser.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--what",
        type=str,
        default="artifacts",
        choices=["artifacts", "results", "both"],
        help="What to sync: artifacts (artifacts/runs/<run_id>), results (results/<dataset>/<run_name>), or both",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Be permissive: the sync tool should not fail because of irrelevant flags.
        print(f"[sync_artifacts] WARNING: ignoring unknown args: {unknown}")

    if args.push == args.pull:
        raise SystemExit("Must specify exactly one of --push or --pull")

    db_path = Path(args.db)
    rid = resolve_run_id(db_path=db_path, run_id=args.run_id, run_name=args.run_name, latest=bool(args.latest))

    # Registry record: needed for results sync
    from src.tools.run_registry import RunRegistry

    reg = RunRegistry(db_path)
    try:
        rec = reg.get_run(rid)
    finally:
        reg.close()

    dataset = str(rec.get("dataset") or "").strip()
    run_name_resolved = str(rec.get("experiment_name") or "").strip()
    if not dataset or not run_name_resolved:
        # Defensive: in practice these exist
        raise SystemExit(f"Registry record missing dataset/experiment_name for run_id={rid}")

    artifact_root_local = Path(resolve_artifact_root(db_path=db_path, run_id=rid))
    results_dir_local = Path("results") / dataset.lower() / run_name_resolved

    drive_root = DRIVE_ROOT_DEFAULT
    artifact_root_drive = drive_root / "artifacts" / "runs" / rid
    results_dir_drive = drive_root / "results" / dataset.lower() / run_name_resolved

    if not Path("/content/drive").exists():
        raise SystemExit("Google Drive is not mounted at /content/drive. Mount Drive first.")

    results: List[Tuple[str, Path, Path, CopyResult]] = []

    def _do_copy(label: str, src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        r = _copy_tree_only_newer(src, dst)
        results.append((label, src, dst, r))

    if args.push:
        if args.what in {"artifacts", "both"}:
            _do_copy("artifacts", artifact_root_local, artifact_root_drive)
        if args.what in {"results", "both"}:
            _do_copy("results", results_dir_local, results_dir_drive)
        direction = "PUSH (local -> Drive)"
    else:
        if args.what in {"artifacts", "both"}:
            _do_copy("artifacts", artifact_root_drive, artifact_root_local)
        if args.what in {"results", "both"}:
            _do_copy("results", results_dir_drive, results_dir_local)
        direction = "PULL (Drive -> local)"

    print("\n=== Artifact Sync Summary ===")
    print(f"Direction:   {direction}")
    print(f"run_id:      {rid}")
    print(f"dataset:     {dataset}")
    print(f"run_name:    {run_name_resolved}")
    for label, src, dst, res in results:
        print(f"\n[{label}]")
        print(f"  Source:      {src}")
        print(f"  Destination: {dst}")
        print(f"  Copied files:  {res.copied_files}")
        print(f"  Skipped files: {res.skipped_files}")
        print(f"  Copied bytes:  {res.copied_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


