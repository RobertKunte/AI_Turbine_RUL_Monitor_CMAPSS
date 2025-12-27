"""
Feature Pipeline Contract: Deterministic reconstruction of effective pipeline config from feature columns.

This module provides utilities to derive the exact feature pipeline configuration used during training
by analyzing the actual feature column names. This ensures diagnostics can reconstruct the identical
feature set.
"""

import re
from typing import Dict, List, Set, Any


def derive_feature_pipeline_config_from_feature_cols(
    feature_cols: List[str],
    dataset: str,
    default_twin_baseline_len: int = 30,
    default_condition_vector_version: int = 3,
) -> Dict[str, Any]:
    """
    Derive effective feature pipeline config from actual feature column names.
    
    This function analyzes feature_cols to determine:
    - Which feature families are present (condition vector, digital twin, residuals, multiscale)
    - Exact window sizes used in multiscale features
    - Explicit list of extra temporal base columns that were selected
    
    Args:
        feature_cols: List of feature column names from training
        dataset: Dataset name (e.g., "FD004")
        default_twin_baseline_len: Default baseline length for digital twin (30)
        default_condition_vector_version: Default condition vector version (3)
    
    Returns:
        Dict with schema_version=2 and complete pipeline config structure
    """
    # Detect presence toggles
    has_cond = any(c.startswith("Cond_") for c in feature_cols)
    has_twin = any(c.startswith("Twin_") for c in feature_cols)
    has_resid = any(c.startswith("Resid_") for c in feature_cols)
    
    # Detect multiscale presence
    has_ms = any(
        "_roll_mean_" in c or "_roll_std_" in c or "_trend_" in c or 
        "_delta_" in c or "_diff_" in c or "_roll_min_" in c or "_roll_max_" in c
        for c in feature_cols
    )
    
    # Extract windows exactly from feature names using regex
    # Pattern: *_roll_mean_<W>, *_roll_std_<W>, *_trend_<W>, *_delta_<W>, *_diff_<W>
    window_pattern = re.compile(r"_(roll_(mean|std|min|max)|trend|delta|diff)_(\d+)$")
    
    windows: Set[int] = set()
    for col in feature_cols:
        match = window_pattern.search(col)
        if match:
            window_size = int(match.group(3))
            windows.add(window_size)
    
    # Sort windows and categorize
    windows_sorted = sorted(windows)
    windows_short = sorted([w for w in windows_sorted if w <= 10])
    windows_medium = sorted([w for w in windows_sorted if 10 < w < 60])
    windows_long = sorted([w for w in windows_sorted if w >= 60])
    
    # If no windows found but multiscale detected, use defaults
    if has_ms and not windows_sorted:
        windows_short = [5, 10]
        windows_long = [30]
    
    # Determine extra temporal base prefixes
    extra_temporal_base_prefixes = []
    if has_twin:
        extra_temporal_base_prefixes.append("Twin_")
    if has_resid:
        extra_temporal_base_prefixes.append("Resid_")
    
    # MOST IMPORTANT: Extract explicit extra temporal base columns selection
    # Find base columns that have derived temporal features (roll_mean, trend, delta, etc.)
    extra_temporal_base_cols_selected: Set[str] = set()
    
    # Pattern to extract base column name from temporal features
    # Examples: "Twin_Sensor10_roll_mean_60" -> "Twin_Sensor10"
    #           "Resid_Sensor3_trend_120" -> "Resid_Sensor3"
    #           "Twin_Sensor10_delta_10" -> "Twin_Sensor10"
    # Pattern matches: (prefix + base_name) followed by temporal suffix
    temporal_suffix_pattern = re.compile(r"_(roll_(?:mean|std|min|max)|trend|delta|diff)_\d+$")
    
    for col in feature_cols:
        # Check if this is a temporal feature derived from Twin_ or Resid_ base
        if any(col.startswith(prefix) for prefix in extra_temporal_base_prefixes):
            # Check if it has a temporal suffix
            match = temporal_suffix_pattern.search(col)
            if match:
                # Extract base column by removing the temporal suffix
                base_col = col[:match.start()]
                extra_temporal_base_cols_selected.add(base_col)
    
    # Sort for deterministic output
    extra_temporal_base_cols_selected_list = sorted(list(extra_temporal_base_cols_selected))
    extra_temporal_base_max_cols = len(extra_temporal_base_cols_selected_list)
    
    # Build config dict
    config = {
        "schema_version": 2,
        "dataset": dataset,
        "features": {
            "use_multiscale_features": has_ms,
            "multiscale": {
                "windows_short": windows_short,
                "windows_medium": windows_medium,
                "windows_long": windows_long,
                "extra_temporal_base_prefixes": extra_temporal_base_prefixes,
                "extra_temporal_base_max_cols": extra_temporal_base_max_cols,
                "extra_temporal_base_cols_selected": extra_temporal_base_cols_selected_list,
            },
        },
        "phys_features": {
            "use_digital_twin_residuals": has_twin,
            "twin_baseline_len": default_twin_baseline_len,
            "use_condition_vector": has_cond,
            "condition_vector_version": default_condition_vector_version if has_cond else 2,
        },
        "use_residuals": has_resid,  # Phase 4 residuals
    }
    
    return config


def validate_feature_pipeline_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate feature pipeline config structure and return list of issues (empty if valid).
    
    Args:
        config: Feature pipeline config dict
    
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if "schema_version" not in config:
        issues.append("Missing schema_version")
    elif config["schema_version"] < 2:
        issues.append(f"Schema version {config['schema_version']} < 2; extra_temporal_base_cols_selected required")
    
    if "features" not in config:
        issues.append("Missing 'features' section")
    else:
        features = config["features"]
        if "multiscale" not in features:
            issues.append("Missing 'features.multiscale' section")
        else:
            ms = features["multiscale"]
            if "extra_temporal_base_cols_selected" not in ms:
                issues.append("Missing 'features.multiscale.extra_temporal_base_cols_selected' (required for schema_version=2)")
    
    if "phys_features" not in config:
        issues.append("Missing 'phys_features' section")
    
    return issues


if __name__ == "__main__":
    """
    Ad-hoc sanity check: load feature_cols.json and derive config.
    
    Usage:
        python -m src.utils.feature_pipeline_contract --run_dir results/fd004/fd004_wm_v1_p0_softcap_k3
    """
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Derive feature pipeline config from feature_cols.json")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment run directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    feature_cols_path = run_dir / "feature_cols.json"
    
    if not feature_cols_path.exists():
        print(f"ERROR: {feature_cols_path} not found")
        exit(1)
    
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
    
    # Try to get dataset from summary.json or infer from path
    dataset = "FD004"  # Default
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
                dataset = summary.get("dataset", dataset)
        except Exception:
            pass
    
    config = derive_feature_pipeline_config_from_feature_cols(feature_cols, dataset)
    
    print(f"Derived config for {dataset}:")
    print(json.dumps(config, indent=2))
    
    issues = validate_feature_pipeline_config(config)
    if issues:
        print("\nVALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Config is valid")

