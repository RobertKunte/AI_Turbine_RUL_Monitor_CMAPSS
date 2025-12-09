"""
Feature safety checks to prevent RUL leakage.
"""

from typing import List, Tuple, Sequence, Any, Dict, Optional
import logging

LEAKAGE_KEYWORDS = ["RUL", "rul"]


def remove_rul_leakage(feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Remove RUL-related columns from feature list.
    
    Args:
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (safe_features, leaked_features)
    """
    leaked = [c for c in feature_cols if any(k in c.upper() for k in LEAKAGE_KEYWORDS)]
    safe = [c for c in feature_cols if c not in leaked]
    if leaked:
        logging.warning(
            "[SANITY CHECK] Removing RUL-related columns from features: %s", leaked
        )
    return safe, leaked


def _get_scaler_n_features(scaler: Any) -> Optional[int]:
    """
    Helper to extract the expected feature dimension from a scaler.

    Supports both:
      - single sklearn scaler with attribute `n_features_in_`
      - dict[int, scaler] for condition-wise scaling (all must agree)
    """
    if scaler is None:
        return None

    # Condition-wise dict of scalers
    if isinstance(scaler, dict):
        n_feats: List[int] = []
        for s in scaler.values():
            n = getattr(s, "n_features_in_", None)
            if n is not None:
                n_feats.append(int(n))
        if not n_feats:
            return None
        unique = set(n_feats)
        if len(unique) > 1:
            raise AssertionError(
                "[FeatureDimMismatch] Inconsistent scaler feature dimensions across "
                f"conditions: {sorted(unique)}. All condition-wise scalers must be "
                "fit on the same feature_dim."
            )
        return n_feats[0]

    n = getattr(scaler, "n_features_in_", None)
    return int(n) if n is not None else None


def check_feature_dimensions(
    feature_cols: Sequence[str],
    scaler: Any = None,
    model: Any = None,
    context: str = "",
) -> None:
    """
    Hard safety check to ensure feature dimensionality is consistent between:
      - diagnostics/inference pipeline (len(feature_cols)),
      - fitted scaler (StandardScaler or dict[int, StandardScaler]),
      - model (via optional `input_dim` attribute).

    Raises AssertionError with a clear, user-facing message if a mismatch is detected.

    Args:
        feature_cols: Ordered list of feature column names used to build X.
        scaler: Fitted scaler object or dict of scalers (may be None).
        model: Trained model instance with optional `input_dim` attribute.
        context: Short string describing the caller ("diagnostics", "inference", ...).
    """
    n_features_pipeline = len(feature_cols)

    # --- Scaler vs. pipeline ---
    n_features_scaler = _get_scaler_n_features(scaler)
    if n_features_scaler is not None and n_features_scaler != n_features_pipeline:
        raise AssertionError(
            "Feature dimension mismatch between pipeline and scaler"
            f"{f' in {context}' if context else ''}: "
            f"scaler expects {n_features_scaler} features, "
            f"but pipeline produced {n_features_pipeline}. "
            "Check 'features.use_multiscale_features' and "
            "'phys_features.use_digital_twin_residuals' / 'phys_features.use_twin_features' "
            "and ensure the config is identical to the training run."
        )

    # --- Model vs. pipeline ---
    model_input_dim = getattr(model, "input_dim", None)
    if model_input_dim is not None and model_input_dim != n_features_pipeline:
        raise AssertionError(
            "Feature dimension mismatch between pipeline and model"
            f"{f' in {context}' if context else ''}: "
            f"model.input_dim={model_input_dim}, "
            f"pipeline feature_dim={n_features_pipeline}. "
            "You likely changed the feature configuration (multi-scale / digital-twin / "
            "condition vector) without retraining the model."
        )

