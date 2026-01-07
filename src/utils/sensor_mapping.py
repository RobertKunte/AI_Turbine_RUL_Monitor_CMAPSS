"""
Sensor Mapping Utilities for CMAPSS Cycle Branch.

Provides resolution of semantic sensor names (T24, T30, P30, T50) to actual
column names in the feature set.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Load sensor configuration from YAML
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "cmapss_sensors.yaml"


def _load_sensor_config() -> Dict:
    """Load the CMAPSS sensor configuration from YAML."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"CMAPSS sensor config not found at {_CONFIG_PATH}. "
            "Please ensure config/cmapss_sensors.yaml exists."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_sensor_index(semantic_name: str) -> int:
    """Get the 1-based sensor index for a semantic name.
    
    Args:
        semantic_name: Semantic sensor name (e.g., 'T24', 'P30')
        
    Returns:
        1-based sensor index (e.g., T24 -> 2)
        
    Raises:
        KeyError: If semantic name not found in config
    """
    config = _load_sensor_config()
    if semantic_name not in config.get("cmapss_sensors", {}):
        available = list(config.get("cmapss_sensors", {}).keys())
        raise KeyError(
            f"Unknown sensor '{semantic_name}'. Available: {available}"
        )
    return config["cmapss_sensors"][semantic_name]["idx"]


def resolve_cycle_target_cols(
    feature_cols: List[str],
    targets: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Resolve semantic target names to actual feature column names.
    
    This function maps semantic sensor names (T24, T30, P30, T50) to the actual
    column names present in the feature set, handling various naming conventions
    (Sensor1, sensor_01, S1, etc.).
    
    Args:
        feature_cols: List of actual feature column names from the dataset
        targets: List of semantic target names. Defaults to ['T24', 'T30', 'P30', 'T50']
        
    Returns:
        Dictionary mapping semantic names to actual column names.
        Example: {'T24': 'Sensor2', 'T30': 'Sensor3', ...}
        
    Raises:
        ValueError: If any target cannot be resolved to a column
    """
    if targets is None:
        config = _load_sensor_config()
        targets = config.get("cycle_branch_targets", ["T24", "T30", "P30", "T50"])
    
    result = {}
    missing = []
    
    for target in targets:
        idx = get_sensor_index(target)
        col_name = _find_sensor_column(feature_cols, idx)
        
        if col_name is None:
            missing.append(f"{target} (idx={idx})")
        else:
            result[target] = col_name
    
    if missing:
        raise ValueError(
            f"Could not resolve cycle targets to columns: {missing}. "
            f"Available columns: {_list_sensor_columns(feature_cols)}"
        )
    
    return result


def _find_sensor_column(feature_cols: List[str], sensor_idx: int) -> Optional[str]:
    """Find the column name for a sensor index.
    
    Handles various naming conventions:
    - Sensor1, Sensor2, ...
    - sensor_01, sensor_02, ...
    - S1, S2, ...
    - sensor1, sensor2, ...
    """
    patterns = [
        rf"^Sensor{sensor_idx}$",           # Sensor1
        rf"^sensor_{sensor_idx:02d}$",      # sensor_01
        rf"^sensor_{sensor_idx}$",          # sensor_1
        rf"^S{sensor_idx}$",                # S1
        rf"^sensor{sensor_idx}$",           # sensor1
        rf"^Sensor{sensor_idx:02d}$",       # Sensor01
    ]
    
    for col in feature_cols:
        for pattern in patterns:
            if re.match(pattern, col, re.IGNORECASE):
                return col
    
    return None


def _list_sensor_columns(feature_cols: List[str]) -> List[str]:
    """List all sensor-like columns for error messages."""
    sensor_pattern = re.compile(r"(?i)(sensor|^s\d)", re.IGNORECASE)
    return [c for c in feature_cols if sensor_pattern.search(c)]


def get_operating_settings_cols(
    feature_cols: List[str],
) -> Dict[str, str]:
    """Resolve operating settings column names.
    
    Returns:
        Dictionary mapping setting names to column names.
        Example: {'TRA': 'Setting1', 'Altitude': 'Setting2', 'Mach': 'Setting3'}
    """
    result = {}
    patterns = {
        "TRA": [r"^Setting1$", r"^setting_1$", r"^setting1$"],
        "Altitude": [r"^Setting2$", r"^setting_2$", r"^setting2$"],
        "Mach": [r"^Setting3$", r"^setting_3$", r"^setting3$"],
    }
    
    for name, pats in patterns.items():
        for col in feature_cols:
            for pat in pats:
                if re.match(pat, col, re.IGNORECASE):
                    result[name] = col
                    break
            if name in result:
                break
    
    return result


def get_default_cycle_targets() -> List[str]:
    """Get the default cycle branch target list."""
    try:
        config = _load_sensor_config()
        return config.get("cycle_branch_targets", ["T24", "T30", "P30", "T50"])
    except FileNotFoundError:
        return ["T24", "T30", "P30", "T50"]
