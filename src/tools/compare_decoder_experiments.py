"""
Compare CONTROL vs Transformer AR decoder experiments.

Reads metrics from summary.json files and prints comparison table.
"""

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


def print_comparison(control_dir: Path, tf_ar_dir: Path):
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
    print("CONTROL vs Transformer AR Decoder Comparison")
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
        ("RMSE_LAST", "RMSE"),
        ("MAE_LAST", "MAE"),
        ("Bias_LAST", "Bias"),
        ("R2_LAST", "R²"),
        ("NASA_LAST_MEAN", "NASA (mean)"),
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
        ("RMSE_ALL", "RMSE"),
        ("MAE_ALL", "MAE"),
        ("Bias_ALL", "Bias"),
        ("R2_ALL", "R²"),
        ("NASA_ALL_MEAN", "NASA (mean)"),
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
    control_decoder = control_metrics.get("config", {}).get("world_model_params", {}).get("decoder_type", "lstm")
    tf_ar_decoder = tf_ar_metrics.get("config", {}).get("world_model_params", {}).get("decoder_type", "tf_ar")
    
    print(f"CONTROL decoder_type: {control_decoder}")
    print(f"TF_AR decoder_type: {tf_ar_decoder}")
    print("=" * 80)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    results_base = project_root / "results" / "fd004"
    
    control_dir = results_base / "fd004_wm_v1_p0_softcap_k3_hm_pad"
    tf_ar_dir = results_base / "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"
    
    print_comparison(control_dir, tf_ar_dir)

