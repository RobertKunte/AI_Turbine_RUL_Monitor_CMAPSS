"""
Central registry for named experiments (baselines and new models).

This module maps high-level experiment IDs to:
  - C-MAPSS subset (fd_id),
  - YAML config path,
  - checkpoint path,
  - model class name.

The registry is primarily used by inference / diagnostics utilities and
not by the core training loops (which currently use src.experiment_configs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Phase-4 residual / digital-twin baselines
    # ------------------------------------------------------------------
    "fd001_phase4_universal_v1_residual": {
        "fd_id": "FD001",
        "config_path": Path("config/phase4/fd001_phase4_universal_v1_residual.yaml"),
        "checkpoint_path": Path("results/fd001/fd001_phase4_universal_v1_residual/best_model.pt"),
        "model_class": "RULHIUniversalModelV1",
    },
    "fd002_phase4_universal_v2_ms_cnn_residual": {
        "fd_id": "FD002",
        "config_path": Path("config/phase4/fd002_phase4_universal_v2_ms_cnn_residual.yaml"),
        "checkpoint_path": Path("results/fd002/fd002_phase4_universal_v2_ms_cnn_d96_residual/best_model.pt"),
        "model_class": "RULHIUniversalModelV2",
    },
    "fd003_phase4_universal_v1_residual": {
        "fd_id": "FD003",
        "config_path": Path("config/phase4/fd003_phase4_universal_v1_residual.yaml"),
        "checkpoint_path": Path("results/fd003/fd003_phase4_universal_v1_residual/best_model.pt"),
        "model_class": "RULHIUniversalModelV1",
    },
    "fd004_phase4_universal_v2_ms_cnn_residual": {
        "fd_id": "FD004",
        "config_path": Path("config/phase4/fd004_phase4_universal_v2_ms_cnn_residual.yaml"),
        "checkpoint_path": Path("results/fd004/fd004_phase3_universal_v2_ms_cnn_d96_residual/best_model.pt"),
        "model_class": "RULHIUniversalModelV2",
    },

    # ------------------------------------------------------------------
    # Transformer + Attention (UniversalEncoderV3Attention) – FD004 v1
    # ------------------------------------------------------------------
    "fd004_transformer_attention_v1": {
        "fd_id": "FD004",
        "config_path": Path("config/transformer_attention/fd004_transformer_attention_v1.yaml"),
        "checkpoint_path": Path("results/fd004/fd004_transformer_attention_v1/best_model.pt"),
        "model_class": "UniversalEncoderV3Attention",
    },
    # ------------------------------------------------------------------
    # Pure Transformer Encoder (EOLFullTransformerEncoder) – FD004 v1
    # ------------------------------------------------------------------
    "fd004_transformer_encoder_v1": {
        "fd_id": "FD004",
        "config_path": Path("config/transformer_attention/fd004_transformer_encoder_v1.yaml"),
        "checkpoint_path": Path("results/fd004/fd004_transformer_encoder_v1/best_model.pt"),
        "model_class": "EOLFullTransformerEncoder",
    },
}


