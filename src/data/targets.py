from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def build_rul_horizon_targets(
    rul0: np.ndarray,
    horizon: int,
    *,
    clamp_min: float = 0.0,
    return_mask: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build deterministic RUL horizon targets from a scalar "current RUL" value.

    This is the single source of truth for padded/clamped horizon targets.

    Definition:
      y[i, k] = max(clamp_min, rul0[i] - k)

    Mask definition (if enabled):
      mask[i, k] = 1  if (rul0[i] - k) >= clamp_min
                 = 0  otherwise  (i.e. padded/clamped region beyond EOL)

    Args:
        rul0: Array of shape (N,) - current RUL at window end ("now"), in cycles.
        horizon: Forecast horizon H.
        clamp_min: Minimum clamp value (usually 0.0).
        return_mask: If True, also return a float32 mask (N, H).

    Returns:
        y: (N, H) float32
        mask: (N, H) float32 if return_mask else None
    """
    r0 = np.asarray(rul0, dtype=np.float32).reshape(-1)
    if horizon <= 0:
        y = np.zeros((r0.shape[0], 0), dtype=np.float32)
        m = np.zeros_like(y) if return_mask else None
        return y, m

    steps = np.arange(int(horizon), dtype=np.float32)[None, :]  # (1, H)
    y = r0[:, None] - steps  # (N, H)

    if return_mask:
        mask = (y >= float(clamp_min)).astype(np.float32)
    else:
        mask = None

    y = np.maximum(y, float(clamp_min)).astype(np.float32)
    return y, mask


if __name__ == "__main__":
    # Tiny sanity check
    rul0 = np.array([0, 1, 2, 40], dtype=np.float32)
    y, mask = build_rul_horizon_targets(rul0, horizon=5, clamp_min=0.0, return_mask=True)

    exp_y = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [40, 39, 38, 37, 36],
        ],
        dtype=np.float32,
    )
    exp_m = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert np.allclose(y, exp_y), f"y mismatch:\n{y}\n!=\n{exp_y}"
    assert mask is not None
    assert np.allclose(mask, exp_m), f"mask mismatch:\n{mask}\n!=\n{exp_m}"
    print("OK: build_rul_horizon_targets")

