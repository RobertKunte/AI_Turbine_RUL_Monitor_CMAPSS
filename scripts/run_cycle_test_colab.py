"""
Single-file Google Colab runner template (registry-first + artifact sync).

Usage in Colab:
1) Upload or open this file in a Colab notebook.
2) Run it (or copy/paste into a cell).
3) Change RUN_NAME/DEVICE in the config section only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


# ============================================================
# USER CONFIG (edit only this section)
# ============================================================

# You can run a single experiment or multiple experiments sequentially.
# Keep this list short in Colab to avoid long runtimes.
RUN_NAMES = [
    "fd004_transformer_worldmodel_v1_cycle_test",
]
DEVICE = "cuda"

# Repo to clone (HTTPS). You can fork and replace this.
REPO_URL = "https://github.com/RobertKunte/AI_Turbine_RUL_Monitor_CMAPSS.git"

# Where to sync artifacts on Drive:
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/AI_Turbine_RUL_Monitor_CMAPSS"

# Optional: preload result folders from Drive into /content before running.
# This is needed when a run depends on previous run artifacts (e.g., pretrained encoder/scaler).
# Example:
# PRELOAD_RESULTS_RUNS = ["fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm"]
PRELOAD_RESULTS_RUNS: list[str] = []

# Optional: persist the registry DB on Drive (recommended for multi-session history).
# If you want the registry history to survive Colab resets, set to True.
PERSIST_REGISTRY_ON_DRIVE = True

# Optional: export lightweight status snapshots into docs/status/ (git-friendly).
EXPORT_STATUS_TO_GIT = True

# Optional: auto-commit/push status snapshots. Disabled by default (requires GitHub auth).
AUTO_GIT_COMMIT = False
AUTO_GIT_PUSH = False


def sh(cmd: str, *, check: bool = True) -> None:
    """
    Run a shell command with Colab-friendly output.

    - Prefer IPython.system (same behavior as `!cmd`) when available, to get live output.
    - Fall back to subprocess with unbuffered python for `python ...` commands.
    """
    cmd_run = cmd.strip()
    if cmd_run.startswith("python "):
        cmd_run = "python -u " + cmd_run[len("python ") :]

    print(f"\n[colab] $ {cmd_run}", flush=True)

    # Stream output line-by-line (closest to Colab `!` behavior).
    # This avoids "silent runs" caused by buffering.
    p = subprocess.Popen(
        cmd_run,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        # Print exactly as produced; avoid double newlines
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {cmd_run}")


def infer_dataset_from_run_name(run_name: str) -> Optional[str]:
    rn = run_name.strip().lower()
    if rn.startswith("fd001"):
        return "FD001"
    if rn.startswith("fd002"):
        return "FD002"
    if rn.startswith("fd003"):
        return "FD003"
    if rn.startswith("fd004"):
        return "FD004"
    return None


def normalize_run_name(run_name: str) -> str:
    """
    RUN_NAMES should contain only experiment names.
    Users sometimes paste CLI fragments like: "<name> --device cuda".
    We strip anything after the first whitespace to avoid:
      - running the wrong experiment (fallback routing)
      - passing garbage into sync_artifacts (--run_name)
    """
    rn = str(run_name).strip()
    if not rn:
        return rn
    parts = rn.split()
    if len(parts) > 1:
        print(f"[colab] WARNING: RUN_NAMES entry contains extra tokens; using '{parts[0]}' and ignoring: {' '.join(parts[1:])}")
    return parts[0]


def copy_results_run_from_drive(run_name: str) -> None:
    """
    Preload a results folder from Drive into /content repo.
    This is a direct copy fallback (does not depend on registry).
    """
    dataset = infer_dataset_from_run_name(run_name)
    if dataset is None:
        print(f"[colab] WARNING: Could not infer dataset from run_name='{run_name}', skipping preload.")
        return
    
    src = Path(DRIVE_PROJECT_ROOT) / "results" / dataset.lower() / run_name
    dst = Path("results") / dataset.lower() / run_name
    if not src.exists():
        print(f"[colab] WARNING: Drive results folder not found: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)

    # Copy-only-newer via our sync tool (registry-free copy path)
    # We use rsync-like semantics by copying files with cp -u (update) where possible.
    # cp -u is widely available in Colab.
    sh(f'mkdir -p "{dst}"', check=False)
    sh(f'cp -u -r "{src}/"* "{dst}/"', check=False)
    print(f"[colab] Preloaded results: {run_name} -> {dst}")


def ensure_hi_calibrator_fd004() -> None:
    """
    Ensure HI_calibrator_FD004.pkl exists for runs that need HI_cal supervision (v4/v5).
    Strategy:
      1) Preload base encoder run results from Drive.
      2) If calibrator still missing, fit it via src.analysis.hi_calibration CLI.
    """
    dataset = "FD004"
    encoder_run = "fd004_transformer_encoder_ms_dt_v2_damage_v3d_delta_two_phase"
    cal_path = Path("results") / dataset.lower() / encoder_run / f"hi_calibrator_{dataset}.pkl"
    if cal_path.exists():
        # Robustness: detect corrupted/partial calibrator (EOFError etc.) and refit.
        try:
            from src.analysis.hi_calibration import load_hi_calibrator

            _ = load_hi_calibrator(cal_path)
            return
        except Exception as e:
            print(f"[colab] HI_calibrator exists but failed to load ({type(e).__name__}: {e})")
            print("[colab] Deleting corrupted calibrator and refitting ...")
            try:
                cal_path.unlink()
            except Exception:
                pass

    print(f"[colab] HI_calibrator missing, attempting to preload base run: {encoder_run}")
    copy_results_run_from_drive(encoder_run)

    if cal_path.exists():
        try:
            from src.analysis.hi_calibration import load_hi_calibrator

            _ = load_hi_calibrator(cal_path)
            return
        except Exception as e:
            print(f"[colab] Preloaded HI_calibrator but failed to load ({type(e).__name__}: {e})")
            print("[colab] Deleting corrupted calibrator and refitting ...")
            try:
                cal_path.unlink()
            except Exception:
                pass

    print("[colab] HI_calibrator still missing -> fitting calibrator (TRAIN-only) ...")
    sh(f"python -m src.analysis.hi_calibration --dataset {dataset} --encoder_run {encoder_run}")
    if not cal_path.exists():
        raise RuntimeError(f"HI_calibrator still missing after fit attempt: {cal_path}")
    # Verify it loads
    from src.analysis.hi_calibration import load_hi_calibrator

    _ = load_hi_calibrator(cal_path)


def get_git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def is_git_dirty(repo_dir: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_dir), "status", "--porcelain"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def main() -> None:
    print("============================================================")
    print("Colab Runner — AI_Turbine_RUL_Monitor_CMAPSS")
    print("============================================================")

    # 1) Mount Drive
    print("\n[1] Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    # 2) Clone repo into /content (not Drive)
    repo_dir = Path("/content/AI_Turbine_RUL_Monitor_CMAPSS")
    print("\n[2] Preparing repo in /content ...")
    if not repo_dir.exists():
        sh(f"git clone {REPO_URL} {repo_dir}")
    else:
        sh(f"git -C {repo_dir} pull")

    os.chdir(str(repo_dir))
    sys.path.insert(0, str(repo_dir))

    # Optional: persist run registry on Drive (so history survives Colab resets)
    if PERSIST_REGISTRY_ON_DRIVE:
        os.environ["RUN_REGISTRY_DB"] = str(
            Path(DRIVE_PROJECT_ROOT) / "artifacts" / "run_registry.sqlite"
        )
        print(f"[colab] RUN_REGISTRY_DB={os.environ['RUN_REGISTRY_DB']}")

    # 3) Print environment info
    print("\n[3] Environment info")
    sha = get_git_sha(repo_dir)
    dirty = is_git_dirty(repo_dir)
    print(f"git sha: {sha}")
    print(f"git dirty: {dirty}")
    print(f"python: {sys.version.replace(os.linesep, ' ')}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[WARN] Could not import torch yet: {e}")

    # 4) Install requirements (cached-friendly)
    print("\n[4] Installing dependencies...")
    if Path("requirements.txt").exists():
        sh("pip -q install -r requirements.txt")
    else:
        sh("pip -q install torch numpy pandas matplotlib scikit-learn tqdm")

    # 5) Execute experiment
    print("\n[5] (Optional) Preloading input run results from Drive -> /content/results")
    for rn in PRELOAD_RESULTS_RUNS:
        # Prefer direct preload (registry-free) for inputs.
        copy_results_run_from_drive(rn)

    # Ensure Drive project folder exists
    sh(f"mkdir -p {DRIVE_PROJECT_ROOT}/artifacts/runs", check=False)
    sh(f"mkdir -p {DRIVE_PROJECT_ROOT}/results", check=False)

    print("\n[6] Running experiments sequentially")
    if not RUN_NAMES:
        raise RuntimeError("RUN_NAMES is empty. Please add at least one experiment name.")

    for idx, run_name in enumerate(RUN_NAMES, 1):
        run_name = normalize_run_name(run_name)
        print("\n" + "=" * 70)
        print(f"[Run {idx}/{len(RUN_NAMES)}] {run_name}")
        print("=" * 70)

        # If the run name suggests HI_cal supervision (v4/v5), ensure calibrator exists.
        if "damage_v4" in run_name.lower() or "damage_v5" in run_name.lower():
            ensure_hi_calibrator_fd004()

        # Execute one experiment at a time for clearer logs and robust syncing
        # Preflight: ensure the experiment name exists in this repo checkout.
        # If this fails, you likely cloned an older git sha that doesn't include the config.
        sh(
            "python -c \"from src.experiment_configs import get_experiment_by_name; "
            f"get_experiment_by_name('{run_name}'); print('OK: experiment exists')\""
        )
        sh(f"python run_experiments.py --experiments {run_name} --device {DEVICE}")

        # Registry visibility
        print("\n[Registry] show latest")
        sh("python -m src.tools.run_registry --show latest")

        # Sync this run (resolve latest run_id for this run_name via registry)
        print("\n[Sync] push results + artifacts for this run_name")
        try:
            sh(f"python -m src.tools.sync_artifacts --push --run_name {run_name} --what both")
        except Exception as e:
            print(f"[colab] WARNING: registry-based sync failed: {e}")
            print("[colab] Falling back to results-only copy by folder name (registry-free).")
            dataset = infer_dataset_from_run_name(run_name) or "FD004"
            src = Path("results") / dataset.lower() / run_name
            dst = Path(DRIVE_PROJECT_ROOT) / "results" / dataset.lower() / run_name
            sh(f'mkdir -p "{dst}"', check=False)
            sh(f'cp -u -r "{src}/"* "{dst}/"', check=False)
            print(f"[colab] Fallback pushed results: {src} -> {dst}")

    # ------------------------------------------------------------------
    # Optional: Regenerate diagnostics for existing runs (stale run folders)
    # ------------------------------------------------------------------
    # Sometimes older run folders contain stale `eol_metrics.json` / plots produced by older
    # code (e.g., before engine/target alignment fixes). Use this helper to rebuild them:
    #
    # Example:
    #   sh("python -u scripts/regenerate_diagnostics.py "
    #      "--run_dir results/fd004/<run_name> --device cpu --force")
    #
    # Or for the most recently run experiment in this script (last RUN_NAMES entry):
    #   last_run = normalize_run_name(RUN_NAMES[-1])
    #   dataset = infer_dataset_from_run_name(last_run) or "FD004"
    #   sh(f"python -u scripts/regenerate_diagnostics.py --run_dir results/{dataset.lower()}/{last_run} --device {DEVICE} --force")

    # Optional diagnostics hook (commented):
    # sh(f"python run_diagnostics.py --experiments {run_name}")

    # Export git-friendly status snapshots (registry overview) into docs/status/
    if EXPORT_STATUS_TO_GIT:
        print("\n[Status] Exporting registry snapshot to docs/status/")
        sh("python -m src.tools.export_status --limit 50")
        print("\n[Status] Exporting per-run Run Cards to docs/status/run_cards/")
        sh("python -m src.tools.export_run_cards --limit 50")

        if AUTO_GIT_COMMIT:
            print("\n[Git] Committing status snapshots (docs/status/)")
            sh("git status --porcelain")
            sh("git add docs/status")
            sh('git commit -m "Update run status snapshots" || true')

            if AUTO_GIT_PUSH:
                print("\n[Git] Pushing commit (requires auth in Colab)")
                # This will only work if you have configured authentication in Colab.
                sh("git push || true", check=False)

    print("\n[Done] Colab run completed.")


if __name__ == "__main__":
    main()
