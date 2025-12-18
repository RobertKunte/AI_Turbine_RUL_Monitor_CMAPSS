from __future__ import annotations

"""
Backwards-compatible alias for the shared EOL metric implementation.

The repo historically had multiple EOL metric code paths; the current single
source of truth is `src.eval.eol_eval.evaluate_eol_metrics`.
"""

from src.eval.eol_eval import evaluate_eol_metrics

__all__ = ["evaluate_eol_metrics"]

