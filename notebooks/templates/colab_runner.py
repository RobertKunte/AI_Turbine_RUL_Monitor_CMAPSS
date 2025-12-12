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


# ============================================================
# USER CONFIG (edit only this section)
# ============================================================

# You can run a single experiment or multiple experiments sequentially.
# Keep this list short in Colab to avoid long runtimes.
RUN_NAMES = [
    "fd004_transformer_encoder_ms_dt_v2_damage_v5_cond_norm",
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
    print(f"\n[colab] $ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


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
        # Pull the full results directory for the referenced run_name
        sh(f"python -m src.tools.sync_artifacts --pull --run_name {rn} --what results")

    # Ensure Drive project folder exists
    sh(f"mkdir -p {DRIVE_PROJECT_ROOT}/artifacts/runs", check=False)
    sh(f"mkdir -p {DRIVE_PROJECT_ROOT}/results", check=False)

    print("\n[6] Running experiments sequentially")
    if not RUN_NAMES:
        raise RuntimeError("RUN_NAMES is empty. Please add at least one experiment name.")

    for idx, run_name in enumerate(RUN_NAMES, 1):
        print("\n" + "=" * 70)
        print(f"[Run {idx}/{len(RUN_NAMES)}] {run_name}")
        print("=" * 70)

        # Execute one experiment at a time for clearer logs and robust syncing
        sh(f"python run_experiments.py --experiments {run_name} --device {DEVICE}")

        # Registry visibility
        print("\n[Registry] show latest")
        sh("python -m src.tools.run_registry --show latest")

        # Sync this run (resolve latest run_id for this run_name via registry)
        print("\n[Sync] push results + artifacts for this run_name")
        sh(f"python -m src.tools.sync_artifacts --push --run_name {run_name} --what both")

    # Optional diagnostics hook (commented):
    # sh(f\"python run_diagnostics.py --experiments {RUN_NAME}\")

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


