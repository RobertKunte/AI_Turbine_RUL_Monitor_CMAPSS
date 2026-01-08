"""
Cycle Branch Artifact Generation.

Generates diagnostics, plots, and metrics for the Cycle Branch (Mode 1 Factorized).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.utils.cycle_branch_helper import CycleBranchComponents, cycle_branch_forward
from src.world_model_training import CycleBranchConfig

logger = logging.getLogger(__name__)

def generate_cycle_artifacts(
    components: CycleBranchComponents,
    loader: DataLoader,
    encoder: torch.nn.Module,
    cfg: CycleBranchConfig,
    run_dir: Path,
    device: torch.device,
    num_samples: int = 1000,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    default_space: str = "scaled",
) -> Dict[str, Any]:
    """Generate all cycle branch artifacts (metrics, plots, summary).
    
    SPACE CONTRACT:
    - If scaler_stats is provided: compute metrics in scaled space (normalized)
    - If scaler_stats is None and default_space="raw": compute raw metrics
    - Fail-fast if default_space="scaled" but no scaler_stats
    
    Args:
        components: CycleBranchComponents
        loader: DataLoader (validation or test)
        encoder: Trained encoder model (to produce z_t)
        cfg: CycleBranchConfig
        run_dir: Run directory to save artifacts
        device: Device
        num_samples: Number of samples to use for detailed plotting
        scaler_stats: Optional tuple of (mean, std) arrays for condition-wise normalization
        default_space: "scaled" or "raw" - determines metric space
        
    Returns:
        Dict containing computed metrics
    """
    out_dir = run_dir / "diagnostics" / "cycle_branch"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating cycle artifacts in {out_dir}...")
    
    # 1. Collect predictions (now also returns cond_ids)
    preds, targets, m_seqs, eta_noms, eta_effs, ops_seqs, cond_ids = _collect_predictions(
        components, loader, encoder, cfg, device, max_batches=50
    )
    
    if len(preds) == 0:
        logger.warning("No cycle predictions collected.")
        return {}
        
    # 2. Compute Metrics in specified space
    try:
        metrics = _compute_metrics(
            preds, targets, components.target_col_map,
            scaler_stats=scaler_stats,
            cond_ids=cond_ids,
            space=default_space,
        )
    except ValueError as e:
        # Fallback to raw space if scaled config missing
        logger.warning(f"Could not compute scaled metrics: {e}. Falling back to raw space.")
        metrics = _compute_metrics(
            preds, targets, components.target_col_map,
            scaler_stats=None,
            cond_ids=None,
            space="raw",
        )
    
    # Add space annotation to printed output
    from src.utils.cycle_metrics_space import format_cycle_metrics_for_print
    logger.info(f"  {format_cycle_metrics_for_print(metrics)}")
    
    # Save metrics JSON
    with open(out_dir / "metrics_cycle.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        
    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(out_dir / "metrics_cycle.csv")

    # 3. Generate Plots
    _plot_residuals(preds, targets, components.target_col_map, out_dir)
    _plot_cycle_fits(preds, targets, components.target_col_map, out_dir)
    _plot_theta_trajectories(m_seqs, out_dir)
    if "eta_eff" in eta_effs:
        _plot_eta_analysis(eta_noms, eta_effs["eta_eff"], m_seqs, out_dir)
        
    logger.info("Cycle artifacts generated successfully.")
    return metrics


def _collect_predictions(
    components: CycleBranchComponents,
    loader: DataLoader,
    encoder: torch.nn.Module,
    cfg: CycleBranchConfig,
    device: torch.device,
    max_batches: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """Run inference to collect batches of data.
    
    Returns:
        Tuple of (preds, targets, m_seqs, eta_noms, eta_effs, ops_seqs, cond_ids)
        All numpy arrays with shape (N, T, D) or similar
    """
    components.cycle_layer.eval()
    components.nominal_head.eval()
    components.param_head.eval()
    encoder.eval()
    
    all_preds = []
    all_targets = []
    all_m = []
    all_eta_nom = []
    all_ops = []
    all_cond_ids = []  # NEW: collect condition IDs
    
    # Can't easily stack dicts of intermediates, so detailed eta/work analysis 
    # might be limited to what we explicitly extract here or last batch.
    # We'll extract eta_eff from the last batch for plotting.
    last_eta_eff = {}
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
                
            # Unpack batch (assuming standard tuple structure)
            if len(batch) == 2:
                X, _ = batch
                cond_ids = None
            elif len(batch) == 3:
                X, _, cond_ids = batch
            elif len(batch) >= 4: # Standard tuples (X, Y, cond, ...)
                X = batch[0]
                cond_ids = batch[2]
            else:
                continue
                
            X = X.to(device)
            if cond_ids is not None:
                cond_ids = cond_ids.to(device)
            else:
                # Default to condition 0 if not provided
                cond_ids = torch.zeros(X.size(0), dtype=torch.long, device=device)
                
            # Encoder forward (need z_t sequence)
            # Assuming UniversalEncoderV3 style or V2
            # If encoder returns only (B, dim), we can't get sequence z_t easily unless supported
            # But CycleBranch requires z_t (B, T, z_dim) if running in seq mode.
            # Workaround: If encoder returns (B, dim), repeat it? Or check if encoder supports return_sequence=True
            
            # Try to get sequence
            if hasattr(encoder, "encoder"):
                 # WorldModel wrapper
                 if hasattr(encoder.encoder, "forward"):
                      # Check signature or just try
                      try:
                          z_t = encoder.encoder(X, cond_ids=cond_ids, return_sequence=True)
                      except (TypeError, RuntimeError):
                          z_t = encoder.encoder(X, cond_ids=cond_ids)
                          if z_t.dim() == 2:
                              z_t = z_t.unsqueeze(1).expand(-1, X.shape[1], -1)
            else:
                 # Raw encoder
                 try:
                    z_t = encoder(X, cond_ids=cond_ids, return_sequence=True)
                 except:
                    z_t = encoder(X, cond_ids=cond_ids)
                    if z_t.dim() == 2:
                        z_t = z_t.unsqueeze(1).expand(-1, X.shape[1], -1)

            # Cycle Branch Forward
            pred, target, m_t, eta_nom, inter = cycle_branch_forward(
                components, X, z_t, cond_ids, cfg, epoch=999 # no warmup
            )
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_m.append(m_t.cpu().numpy())
            all_eta_nom.append(eta_nom.cpu().numpy())
            all_ops.append(inter["ops_t"].cpu().numpy())
            all_cond_ids.append(cond_ids.cpu().numpy())  # NEW: collect cond_ids
            
            if i == 0 and "eta_eff" in inter:
                # Capture first batch eta dict structure for reference/debug
                # Need to move to cpu
                last_eta_eff = {k: v.cpu().numpy() for k,v in inter["eta_eff"].items()}

    if not all_preds:
        return [], [], [], [], {}, [], []

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_m, axis=0),
        np.concatenate(all_eta_nom, axis=0),
        last_eta_eff, # Return just one batch for detailed eta plots to save memory
        np.concatenate(all_ops, axis=0),
        np.concatenate(all_cond_ids, axis=0),  # NEW: return cond_ids
    )


def _compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    col_map: Dict[str, str],
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cond_ids: Optional[np.ndarray] = None,
    space: str = "scaled",
) -> Dict[str, Dict[str, Any]]:
    """Compute MSE/MAE/RMSE per sensor in specified space.
    
    SPACE CONTRACT:
    - If space="scaled" and scaler_stats+cond_ids provided: normalize raw preds before comparison
    - If space="raw": compare raw preds (requires targets to be denormalized)
    - Fail-fast if space="scaled" but missing normalization config
    
    Args:
        preds: Raw predictions from cycle layer (N, T, n_targets)
        targets: Targets (already scaled if cycle_target_space="scaled")
        col_map: Dict mapping sensor names to column names
        scaler_stats: Tuple of (mean, std) arrays, each (num_conditions, n_targets)
        cond_ids: Condition IDs (N,) for each sample
        space: "scaled" or "raw" - determines output metric space
        
    Returns:
        Dict with per-sensor metrics including space annotation
    """
    from src.utils.cycle_metrics_space import (
        normalize_cycle_pred,
        compute_cycle_metrics,
        validate_cycle_space_config,
    )
    
    sensor_names = list(col_map.keys())
    
    # Validate config
    if space == "scaled":
        validate_cycle_space_config(space, scaler_stats, cond_ids, "cycle_artifacts._compute_metrics")
        # Normalize raw preds to scaled space
        preds_compare = normalize_cycle_pred(preds, cond_ids, scaler_stats)
        targets_compare = targets  # Already scaled
    else:
        # Raw space comparison - targets need to be denormalized
        from src.utils.cycle_metrics_space import denormalize_cycle_target
        if scaler_stats is not None and cond_ids is not None:
            targets_compare = denormalize_cycle_target(targets, cond_ids, scaler_stats)
        else:
            targets_compare = targets
        preds_compare = preds
    
    # Compute metrics using shared helper
    return compute_cycle_metrics(preds_compare, targets_compare, sensor_names, space=space)


def _plot_residuals(preds, targets, col_map, out_dir):
    """Plot residual distributions."""
    sensor_names = list(col_map.keys())
    preds_f = preds.reshape(-1, preds.shape[-1])
    targets_f = targets.reshape(-1, targets.shape[-1])
    
    fig, axes = plt.subplots(1, len(sensor_names), figsize=(5*len(sensor_names), 4))
    if len(sensor_names) == 1: axes = [axes]
    
    for i, name in enumerate(sensor_names):
        res = preds_f[:, i] - targets_f[:, i]
        sns.histplot(res, ax=axes[i], kde=True)
        axes[i].set_title(f"Residuals: {name}")
        axes[i].set_xlabel("Pred - True")
        
    plt.tight_layout()
    plt.savefig(out_dir / "residual_dist.png")
    plt.close()


def _plot_cycle_fits(preds, targets, col_map, out_dir):
    """Plot scatter fits."""
    sensor_names = list(col_map.keys())
    preds_f = preds.reshape(-1, preds.shape[-1])
    targets_f = targets.reshape(-1, targets.shape[-1])
    
    # Sample down if too large
    if preds_f.shape[0] > 5000:
        idx = np.random.choice(preds_f.shape[0], 5000, replace=False)
        preds_f = preds_f[idx]
        targets_f = targets_f[idx]
        
    fig, axes = plt.subplots(1, len(sensor_names), figsize=(5*len(sensor_names), 5))
    if len(sensor_names) == 1: axes = [axes]
    
    for i, name in enumerate(sensor_names):
        p = preds_f[:, i]
        t = targets_f[:, i]
        
        ax = axes[i]
        ax.scatter(t, p, alpha=0.1, s=1)
        
        # Diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_title(f"Fit: {name}")
        
    plt.tight_layout()
    plt.savefig(out_dir / "cycle_fits.png")
    plt.close()


def _plot_theta_trajectories(m_seqs, out_dir):
    """Plot degradation theta trajectories (m(t)) for a few samples."""
    # m_seqs: (N, T, 6)
    N, T, D = m_seqs.shape
    
    # Plot first 10 samples
    n_plot = min(N, 10)
    
    names = ["m_fan", "m_lpc", "m_hpc", "m_hpt", "m_lpt", "m_dp"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(D, 6)):
        ax = axes[i]
        for k in range(n_plot):
            ax.plot(m_seqs[k, :, i], alpha=0.5)
        ax.set_title(names[i])
        ax.set_ylim(0.8, 1.05) # Typical range for degradations
        
    plt.tight_layout()
    plt.savefig(out_dir / "theta_trajectories_sample.png")
    plt.close()


def _plot_eta_analysis(eta_nom, eta_eff_dict, m_seq, out_dir):
    """Overlay eta_nom vs eta_eff."""
    # This expects arrays of same shape, but eta_eff_dict is from a single batch.
    # So we should compare only for that batch.
    # This is a bit complex to generalize without matching indices.
    # For now, skip or implement a simplified version.
    pass
