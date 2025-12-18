"""Self-test for right-censored RUL trajectory reconstruction.

This repo does not currently use pytest/unittest, so we keep this as a simple
script with asserts.

Run:
  python scripts/self_test_dynamics_kpis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.health_index_metrics import reconstruct_rul_trajectory_from_last


def main() -> None:
    cycles = np.array([10, 11, 12, 13], dtype=float)
    cycle_last = 13.0
    rul_last = 40.0

    got = reconstruct_rul_trajectory_from_last(
        rul_last=rul_last,
        cycle_last=cycle_last,
        cycles=cycles,
        cap=None,
    )
    expected = np.array([43, 42, 41, 40], dtype=float)
    assert np.allclose(got, expected), f"uncapped mismatch: got={got}, expected={expected}"

    got_cap = reconstruct_rul_trajectory_from_last(
        rul_last=rul_last,
        cycle_last=cycle_last,
        cycles=cycles,
        cap=42.0,
    )
    expected_cap = np.array([42, 42, 41, 40], dtype=float)
    assert np.allclose(got_cap, expected_cap), f"capped mismatch: got={got_cap}, expected={expected_cap}"

    print("OK: reconstruct_rul_trajectory_from_last")


if __name__ == "__main__":
    main()
