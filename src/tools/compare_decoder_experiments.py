"""
Compare CONTROL vs Transformer AR decoder experiments.

Reads metrics from summary.json files and prints comparison table.

Usage:
    python -m src.tools.compare_decoder_experiments --dataset FD004
    python -m src.tools.compare_decoder_experiments --dataset FD001 --control <control_run> --treatment <treatment_run>
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_metrics(results_dir: Path) -> dict:
    """Load metrics from summary.json."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary


def print_comparison(control_dir: Path, tf_ar_dir: Path, dataset: str = "FD004"):
    """Print comparison table."""
    control_metrics = load_metrics(control_dir)
    tf_ar_metrics = load_metrics(tf_ar_dir)
    
    if control_metrics is None:
        print(f"[ERROR] CONTROL metrics not found: {control_dir / 'summary.json'}")
        return
    
    if tf_ar_metrics is None:
        print(f"[ERROR] Transformer AR metrics not found: {tf_ar_dir / 'summary.json'}")
        return
    
    print("=" * 80)
    print(f"CONTROL vs Transformer AR Decoder Comparison ({dataset})")
    print("=" * 80)
    print()
    
    # Extract test metrics
    control_test = control_metrics.get("test_metrics", {})
    tf_ar_test = tf_ar_metrics.get("test_metrics", {})
    
    # LAST metrics
    print("LAST_AVAILABLE_PER_UNIT (Primary):")
    print("-" * 80)
    print(f"{'Metric':<20} {'CONTROL (LSTM)':<20} {'TF_AR (Self-Attn)':<20} {'Delta':<20}")
    print("-" * 80)
    
    metrics_to_compare = [
        ("rmse_last", "RMSE"),
        ("mae_last", "MAE"),
        ("bias_last", "Bias"),
        ("r2_last", "R²"),
        ("nasa_last_mean", "NASA (mean)"),
    ]
    
    for key, label in metrics_to_compare:
        c_val = control_test.get(key, None)
        t_val = tf_ar_test.get(key, None)
        
        if c_val is not None and t_val is not None:
            delta = t_val - c_val
            delta_pct = (delta / abs(c_val) * 100) if c_val != 0 else 0.0
            delta_str = f"{delta:+.3f} ({delta_pct:+.1f}%)"
            print(f"{label:<20} {c_val:<20.3f} {t_val:<20.3f} {delta_str:<20}")
        else:
            c_str = f"{c_val:.3f}" if c_val is not None else "N/A"
            t_str = f"{t_val:.3f}" if t_val is not None else "N/A"
            print(f"{label:<20} {c_str:<20} {t_str:<20} {'N/A':<20}")
    
    print()
    
    # ALL metrics (secondary)
    print("ALL windows (Secondary - for transparency):")
    print("-" * 80)
    print(f"{'Metric':<20} {'CONTROL (LSTM)':<20} {'TF_AR (Self-Attn)':<20} {'Delta':<20}")
    print("-" * 80)
    
    all_metrics = [
        ("rmse_all", "RMSE"),
        ("mae_all", "MAE"),
        ("bias_all", "Bias"),
        ("r2_all", "R²"),
        ("nasa_all_mean", "NASA (mean)"),
    ]
    
    for key, label in all_metrics:
        c_val = control_test.get(key, None)
        t_val = tf_ar_test.get(key, None)
        
        if c_val is not None and t_val is not None:
            delta = t_val - c_val
            delta_pct = (delta / abs(c_val) * 100) if c_val != 0 else 0.0
            delta_str = f"{delta:+.3f} ({delta_pct:+.1f}%)"
            print(f"{label:<20} {c_val:<20.3f} {t_val:<20.3f} {delta_str:<20}")
        else:
            c_str = f"{c_val:.3f}" if c_val is not None else "N/A"
            t_str = f"{t_val:.3f}" if t_val is not None else "N/A"
            print(f"{label:<20} {c_str:<20} {t_str:<20} {'N/A':<20}")
    
    print()
    
    # Risk metrics (if available)
    print("Risk Metrics (if available):")
    print("-" * 80)
    
    risk_keys = [
        "overshoot_rate",
        "p95_overshoot",
        "unsafe_fraction",
    ]
    
    control_risk = control_test.get("risk_metrics", {})
    tf_ar_risk = tf_ar_test.get("risk_metrics", {})
    
    if control_risk or tf_ar_risk:
        print(f"{'Metric':<20} {'CONTROL (LSTM)':<20} {'TF_AR (Self-Attn)':<20} {'Delta':<20}")
        print("-" * 80)
        
        for key in risk_keys:
            c_val = control_risk.get(key, None) if control_risk else None
            t_val = tf_ar_risk.get(key, None) if tf_ar_risk else None
            
            if c_val is not None and t_val is not None:
                delta = t_val - c_val
                delta_pct = (delta / abs(c_val) * 100) if c_val != 0 else 0.0
                delta_str = f"{delta:+.3f} ({delta_pct:+.1f}%)"
                print(f"{key:<20} {c_val:<20.3f} {t_val:<20.3f} {delta_str:<20}")
            else:
                c_str = f"{c_val:.3f}" if c_val is not None else "N/A"
                t_str = f"{t_val:.3f}" if t_val is not None else "N/A"
                print(f"{key:<20} {c_str:<20} {t_str:<20} {'N/A':<20}")
    else:
        print("Risk metrics not available in summary.json")
    
    print()
    print("=" * 80)
    
    # Decoder type confirmation
    control_decoder = (
        control_metrics.get("config", {}).get("world_model_params", {}).get("decoder_type")
        or control_metrics.get("world_model_config", {}).get("decoder_type")
        or "unknown"
    )
    tf_ar_decoder = (
        tf_ar_metrics.get("config", {}).get("world_model_params", {}).get("decoder_type")
        or tf_ar_metrics.get("world_model_config", {}).get("decoder_type")
        or "unknown"
    )
    
    print(f"CONTROL decoder_type: {control_decoder}")
    print(f"TF_AR decoder_type: {tf_ar_decoder}")
    print("=" * 80)


def get_default_run_names(dataset: str) -> tuple[str, str]:
    """Get default control and treatment run names for a dataset."""
    dataset_lower = dataset.lower()
    defaults = {
        "fd001": ("fd001_wm_v1_p0_softcap_k3_hm_pad", "fd001_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"),
        "fd002": ("fd002_wm_v1_p0_softcap_k3_hm_pad", "fd002_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"),
        "fd003": ("fd003_wm_v1_p0_softcap_k3_hm_pad", "fd003_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"),
        "fd004": ("fd004_wm_v1_p0_softcap_k3_hm_pad", "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"),
    }
    return defaults.get(dataset_lower, (None, None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CONTROL vs Transformer AR decoder experiments")
    parser.add_argument("--dataset", type=str, default="FD004", help="Dataset name (FD001, FD002, FD003, FD004)")
    parser.add_argument("--control", type=str, default=None, help="Control run name (default: auto-detect from dataset)")
    parser.add_argument("--treatment", type=str, default=None, help="Treatment run name (default: auto-detect from dataset)")
    
    args = parser.parse_args()
    
    dataset = args.dataset.upper()
    dataset_lower = dataset.lower()
    
    # Get default run names if not provided
    if args.control is None or args.treatment is None:
        default_control, default_treatment = get_default_run_names(dataset)
        control_run = args.control or default_control
        treatment_run = args.treatment or default_treatment
        
        if control_run is None or treatment_run is None:
            print(f"[ERROR] No default run names for dataset {dataset}. Please specify --control and --treatment.")
            sys.exit(1)
    else:
        control_run = args.control
        treatment_run = args.treatment
    
    project_root = Path(__file__).parent.parent.parent
    results_base = project_root / "results" / dataset_lower
    
    control_dir = results_base / control_run
    treatment_dir = results_base / treatment_run
    
    print_comparison(control_dir, treatment_dir, dataset=dataset)

