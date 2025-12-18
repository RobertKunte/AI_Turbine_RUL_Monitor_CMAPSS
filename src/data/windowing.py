from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    past_len: int
    horizon: int
    stride: int = 1
    # If True, only produce samples where the full future horizon exists (no end-padding).
    # This matches the legacy seq2seq builder behavior.
    require_full_horizon: bool = False
    # For y_seq padding beyond available future:
    # - "clamp": repeat last available value
    # - "zero": pad with 0.0
    pad_mode: str = "clamp"


@dataclass(frozen=True)
class TargetConfig:
    max_rul: int = 125
    cap_targets: bool = True
    # How to derive the scalar EOL target from y_seq / current RUL:
    # - "future0":      y_seq[:, 0]     (RUL at t+1)
    # - "future_last":  y_seq[:, -1]    (RUL at t+H)
    # - "current_from_df": current RUL at window end t (from df[rul_col] at t_end)
    # - "current_plus_one": min(max_rul, y_seq[:, 0] + 1)  (approx current from future0)
    eol_target_mode: str = "current_from_df"
    # If True, evaluation clips y_true to [0,max_rul] (NASA-capped style).
    clip_eval_y_true: bool = False


def _cap(arr: np.ndarray, max_rul: float, enabled: bool) -> np.ndarray:
    if not enabled:
        return arr
    return np.minimum(arr, float(max_rul))


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "RUL",
    # Optional: additionally return future sequences for these columns
    # using the SAME window indices and pad policy.
    future_feature_cols: Optional[list[str]] = None,
    unit_col: str = "UnitNumber",
    time_col: str = "TimeInCycles",
    cond_col: Optional[str] = "ConditionID",
    window_cfg: WindowConfig,
    target_cfg: TargetConfig,
    return_mask: bool = True,
) -> Dict[str, Any]:
    """
    Central, shared window builder for Train/Val datasets.

    Window policy:
      For each engine and each time index t (t_end):
        x = features[t-P+1 : t+1]  (only if t >= P-1)
        y_seq = target[t+1 : t+H+1]  (future)
        If y_seq shorter than H: pad to H using pad_mode.

    Returns dict with:
      X:      (N, P, D)
      Y_seq:  (N, H, 1)
      Y_eol:  (N,) scalar target derived via TargetConfig
      mask:   (N, H, 1) 1 for observed future steps, 0 for padded
      unit_ids: (N,)
      cond_ids: (N,) (if cond_col provided else zeros)
      meta: dict with debug stats including pad_frac, y_true ranges, configs
    """
    P = int(window_cfg.past_len)
    H = int(window_cfg.horizon)
    stride = int(max(1, window_cfg.stride))
    pad_mode = str(window_cfg.pad_mode).lower()
    require_full = bool(window_cfg.require_full_horizon)
    if pad_mode not in {"clamp", "zero"}:
        raise ValueError(f"Unsupported pad_mode={window_cfg.pad_mode!r}. Use 'clamp' or 'zero'.")

    eol_mode = str(target_cfg.eol_target_mode).lower()
    if eol_mode not in {"future0", "future_last", "current_from_df", "current_plus_one"}:
        raise ValueError(f"Unsupported eol_target_mode={target_cfg.eol_target_mode!r}.")

    X_list: list[np.ndarray] = []
    Yseq_list: list[np.ndarray] = []
    Yeol_list: list[float] = []
    M_list: list[np.ndarray] = []
    Fut_list: list[np.ndarray] = []
    unit_ids_list: list[int] = []
    cond_ids_list: list[int] = []
    t_end_pos_list: list[int] = []

    for uid, df_u in df.groupby(unit_col):
        df_u = df_u.sort_values(time_col).reset_index(drop=True)
        if len(df_u) < P:
            continue

        values = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
        values_future = None
        if future_feature_cols:
            values_future = df_u[future_feature_cols].to_numpy(dtype=np.float32, copy=True)
        rul = df_u[target_col].to_numpy(dtype=np.float32, copy=True).reshape(-1)
        rul = np.maximum(rul, 0.0)
        rul = _cap(rul, target_cfg.max_rul, target_cfg.cap_targets)

        if cond_col is not None and cond_col in df_u.columns:
            cond_id = int(df_u[cond_col].iloc[0])
        else:
            cond_id = 0

        T = len(df_u)
        t_end_stop = T if not require_full else max(P - 1, T - H - 1)
        for t_end in range(P - 1, t_end_stop, stride):
            x = values[t_end - P + 1 : t_end + 1]  # (P, D)

            # Future slice [t_end+1, t_end+H+1)
            f0 = t_end + 1
            f1 = min(T, t_end + H + 1)
            y_obs = rul[f0:f1]  # (<=H,)
            n_obs = int(y_obs.shape[0])

            if H <= 0:
                y_full = np.zeros((0,), dtype=np.float32)
                m = np.zeros((0,), dtype=np.float32)
                fut_full = None
            else:
                if n_obs > 0:
                    if pad_mode == "clamp":
                        pad_value = float(y_obs[-1])
                    else:
                        pad_value = 0.0
                else:
                    # No future available (t_end is last point). Clamp-mode uses current.
                    pad_value = float(rul[t_end]) if pad_mode == "clamp" else 0.0

                y_full = np.full((H,), pad_value, dtype=np.float32)
                if n_obs > 0:
                    y_full[:n_obs] = y_obs.astype(np.float32, copy=False)

                if return_mask:
                    m = np.zeros((H,), dtype=np.float32)
                    if n_obs > 0:
                        m[:n_obs] = 1.0
                else:
                    m = np.ones((H,), dtype=np.float32)

                # Optional future feature sequences (padded identically)
                fut_full = None
                if values_future is not None:
                    f_obs = values_future[f0:f1]  # (<=H, Ff)
                    Ff = int(values_future.shape[1])
                    if n_obs > 0:
                        if pad_mode == "clamp":
                            pad_vec = f_obs[-1]
                        else:
                            pad_vec = np.zeros((Ff,), dtype=np.float32)
                    else:
                        pad_vec = values_future[t_end] if pad_mode == "clamp" else np.zeros((Ff,), dtype=np.float32)

                    fut_full = np.repeat(pad_vec.reshape(1, Ff), H, axis=0).astype(np.float32)
                    if n_obs > 0:
                        fut_full[:n_obs] = f_obs.astype(np.float32, copy=False)

            # Scalar EOL target
            if eol_mode == "future0":
                y_eol = float(y_full[0]) if H > 0 else float(rul[t_end])
            elif eol_mode == "future_last":
                y_eol = float(y_full[-1]) if H > 0 else float(rul[t_end])
            elif eol_mode == "current_plus_one":
                if H > 0:
                    y_eol = float(min(float(target_cfg.max_rul), float(y_full[0] + 1.0)))
                else:
                    y_eol = float(rul[t_end])
            else:
                # current_from_df
                y_eol = float(rul[t_end])

            X_list.append(x)
            Yseq_list.append(y_full.reshape(H, 1))
            Yeol_list.append(y_eol)
            M_list.append(m.reshape(H, 1))
            if values_future is not None and fut_full is not None:
                Fut_list.append(fut_full)  # (H, Ff)
            unit_ids_list.append(int(uid))
            cond_ids_list.append(cond_id)
            t_end_pos_list.append(int(t_end))

    if not X_list:
        raise ValueError("build_sliding_windows: no samples were produced (check past_len/horizon and df contents).")

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y_seq = np.stack(Yseq_list, axis=0).astype(np.float32)
    Y_eol = np.asarray(Yeol_list, dtype=np.float32)
    mask = np.stack(M_list, axis=0).astype(np.float32) if return_mask else None
    future_features = np.stack(Fut_list, axis=0).astype(np.float32) if Fut_list else None
    unit_ids = np.asarray(unit_ids_list, dtype=np.int64)
    cond_ids = np.asarray(cond_ids_list, dtype=np.int64)
    t_end_pos = np.asarray(t_end_pos_list, dtype=np.int64)

    # meta logging / debug
    meta: Dict[str, Any] = {
        "window_cfg": asdict(window_cfg),
        "target_cfg": asdict(target_cfg),
        "num_samples": int(X.shape[0]),
        "num_units": int(len(np.unique(unit_ids))),
        "x_shape": tuple(X.shape),
        "y_seq_shape": tuple(Y_seq.shape),
        "y_eol_shape": tuple(Y_eol.shape),
    }
    if future_features is not None:
        meta["future_features_shape"] = tuple(future_features.shape)
    try:
        meta["y_eol_min"] = float(np.min(Y_eol))
        meta["y_eol_mean"] = float(np.mean(Y_eol))
        meta["y_eol_max"] = float(np.max(Y_eol))
        meta["y_seq_min"] = float(np.min(Y_seq))
        meta["y_seq_max"] = float(np.max(Y_seq))
        if mask is not None:
            meta["pad_frac"] = float(1.0 - float(mask.mean()))
        else:
            meta["pad_frac"] = None
        meta["near_eol_frac_y_eol_le_10"] = float(np.mean(Y_eol <= 10.0))
    except Exception:
        pass

    print(
        f"[windowing] N={meta.get('num_samples')} units={meta.get('num_units')} "
        f"| y_eol(min/mean/max)={meta.get('y_eol_min'):.2f}/{meta.get('y_eol_mean'):.2f}/{meta.get('y_eol_max'):.2f} "
        f"| pad_frac={meta.get('pad_frac') if meta.get('pad_frac') is not None else 'NA'} "
        f"| cfg(past={P},H={H},stride={stride},pad={pad_mode},eol_mode={eol_mode},cap={target_cfg.cap_targets})"
    )

    return {
        "X": X,
        "Y_seq": Y_seq,
        "Y_eol": Y_eol,
        "mask": mask,
        "future_features": future_features,
        "unit_ids": unit_ids,
        "cond_ids": cond_ids,
        "t_end_pos": t_end_pos,
        "meta": meta,
    }


def build_test_windows_last(
    df_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_cols: list[str],
    *,
    unit_col: str = "UnitNumber",
    time_col: str = "TimeInCycles",
    cond_col: Optional[str] = "ConditionID",
    window_cfg: WindowConfig,
    target_cfg: TargetConfig,
) -> Dict[str, Any]:
    """
    Build exactly one window per unit (the last available window), plus aligned y_true.
    """
    P = int(window_cfg.past_len)
    pad_mode = str(window_cfg.pad_mode).lower()
    if pad_mode not in {"clamp", "zero"}:
        raise ValueError(f"Unsupported pad_mode={window_cfg.pad_mode!r}.")

    X_list: list[np.ndarray] = []
    y_true_list: list[float] = []
    unit_ids_list: list[int] = []
    cond_ids_list: list[int] = []

    # Map unit_id -> index in y_test_true (NASA files are 1-indexed by unit ordering)
    unit_id_to_idx = {i + 1: i for i in range(len(y_test_true))}

    for uid, df_u in df_test.groupby(unit_col):
        uid_i = int(uid)
        df_u = df_u.sort_values(time_col).reset_index(drop=True)
        feats = df_u[feature_cols].to_numpy(dtype=np.float32, copy=True)
        if feats.shape[0] < P:
            pad_len = P - feats.shape[0]
            if pad_mode == "clamp":
                pad_row = feats[0:1]
                pad = np.repeat(pad_row, pad_len, axis=0)
            else:
                pad = np.zeros((pad_len, feats.shape[1]), dtype=np.float32)
            feats = np.vstack([pad, feats])
        x = feats[-P:]

        if cond_col is not None and cond_col in df_u.columns:
            cond_id = int(df_u[cond_col].iloc[0])
        else:
            cond_id = 0

        idx = unit_id_to_idx.get(uid_i, uid_i - 1)
        yt = float(y_test_true[idx]) if idx < len(y_test_true) else float(y_test_true[-1])
        if target_cfg.cap_targets:
            yt = float(min(float(target_cfg.max_rul), yt))
        yt = float(max(0.0, yt))

        X_list.append(x)
        y_true_list.append(yt)
        unit_ids_list.append(uid_i)
        cond_ids_list.append(cond_id)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_true = np.asarray(y_true_list, dtype=np.float32)
    unit_ids = np.asarray(unit_ids_list, dtype=np.int64)
    cond_ids = np.asarray(cond_ids_list, dtype=np.int64)

    print(
        f"[windowing-test] N={int(X.shape[0])} units={int(len(np.unique(unit_ids)))} "
        f"| y_true(min/max)={float(np.min(y_true)):.2f}/{float(np.max(y_true)):.2f} "
        f"| cfg(past={P},pad={pad_mode},cap={target_cfg.cap_targets},max_rul={target_cfg.max_rul})"
    )

    return {
        "X": X,
        "y_true": y_true,
        "unit_ids": unit_ids,
        "cond_ids": cond_ids,
        "meta": {"window_cfg": asdict(window_cfg), "target_cfg": asdict(target_cfg)},
    }

