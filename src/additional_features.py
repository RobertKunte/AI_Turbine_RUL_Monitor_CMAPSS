from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd

from .feature_config import FeatureGroupsConfig, TemporalWindowConfig
from .config import PhysicsFeatureConfig, ResidualFeatureConfig
from .digital_twin import HealthyTwinRegressor


@dataclass
class TemporalFeatureConfig:
    """
    Configuration for temporal (multi-scale) features.
    All booleans can be turned on/off for sensitivity analysis.
    """

    # Which base columns to use for temporal features.
    # If None, a reasonable default will be inferred.
    base_cols: Optional[Sequence[str]] = None

    # Windows for rolling statistics (in cycles)
    short_windows: Tuple[int, ...] = (5, 10)
    long_windows: Tuple[int, ...] = (30,)

    # What to compute
    add_rolling_mean: bool = True
    add_rolling_std: bool = False
    add_trend: bool = True  # approx. slope over the window
    add_delta: bool = True  # simple lagged differences

    # Lags for delta features
    delta_lags: Tuple[int, ...] = (5, 10)


@dataclass
class FeatureConfig:
    """
    High-level feature configuration for sensitivity analysis.
    Controls which engineered features are created.
    """

    # Core physics-informed features (HPC efficiency proxy, EGT drift, Fan/HPC ratio, ...)
    add_physical_core: bool = True

    # Multi-scale temporal features
    add_temporal_features: bool = True
    temporal: TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)


def _infer_default_temporal_base_cols(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
) -> List[str]:
    """
    Heuristic to pick reasonable base columns for temporal features.
    - Keep sensor columns (e.g. 'Sensor2', 'Sensor3', ...) and known physics features
    - Exclude ID / time / RUL columns (and ALL RUL-related columns)
    """
    exclude = {unit_col, cycle_col, "MaxTime"}
    # Exclude ALL RUL-related columns (case-insensitive)
    rul_related = [col for col in df.columns if "RUL" in col.upper()]
    exclude.update(rul_related)
    
    # You can extend this list if you have more meta columns
    physics_candidates = {
        "Effizienz_HPC_Proxy",
        "EGT_Drift",
        "Fan_HPC_Ratio",
    }

    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        # Exclude any column containing RUL (case-insensitive)
        if "RUL" in col.upper():
            continue
        # crude heuristic: sensors often start with 's' or 'S' or are Setting columns
        if (
            col.startswith("s")
            or col.startswith("S")
            or col.startswith("Setting")
            or col in physics_candidates
        ):
            cols.append(col)
    return cols


def add_temporal_features(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    config: Optional[TemporalFeatureConfig] = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Add multi-scale temporal features per engine:
    - Rolling means / stds over short and long windows
    - Approximate trends (slope) over windows
    - Simple lagged deltas

    All components are controlled via TemporalFeatureConfig.
    """
    if config is None:
        config = TemporalFeatureConfig()

    if not inplace:
        df = df.copy()

    # Ensure sorted by engine and cycle
    df = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)

    # Determine which base columns to use
    if config.base_cols is None:
        base_cols = _infer_default_temporal_base_cols(df, unit_col=unit_col, cycle_col=cycle_col)
    else:
        base_cols = list(config.base_cols)

    if len(base_cols) == 0:
        print("Warning: No base columns found for temporal features. Skipping.")
        return df

    # Prepare groupby
    g = df.groupby(unit_col, group_keys=False)

    # Helper to create rolling features
    all_windows = list(config.short_windows) + list(config.long_windows)

    # Collect all new features in a dictionary to avoid DataFrame fragmentation
    new_features = {}

    for col in base_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue

        series_by_unit = g[col]

        # Rolling mean / std / trend
        for w in all_windows:
            window_name = f"{w}"

            # Rolling mean
            if config.add_rolling_mean:
                new_name = f"{col}_roll_mean_{window_name}"
                new_features[new_name] = series_by_unit.transform(
                    lambda x, w=w: x.rolling(window=w, min_periods=1).mean()
                )

            # Rolling std
            if config.add_rolling_std:
                new_name = f"{col}_roll_std_{window_name}"
                rolled_std = series_by_unit.transform(
                    lambda x, w=w: x.rolling(window=w, min_periods=1).std()
                )
                # Fill NaN (occurs when std is computed on single value) with 0
                new_features[new_name] = rolled_std.fillna(0)

            # Trend (approx. slope): (x_t - x_{t-w}) / w
            if config.add_trend:
                new_name = f"{col}_trend_{window_name}"
                # difference within each engine
                shifted = series_by_unit.shift(w)
                trend = (df[col] - shifted) / float(w)
                # Fill NaN (occurs at start of each engine) with 0
                new_features[new_name] = trend.fillna(0)

        # Delta features: x_t - x_{t-lag}
        if config.add_delta and config.delta_lags:
            for lag in config.delta_lags:
                new_name = f"{col}_delta_{lag}"
                delta = series_by_unit.diff(lag)
                # Fill NaN (occurs at start of each engine) with 0
                new_features[new_name] = delta.fillna(0)

    # Add all new features at once using pd.concat to avoid fragmentation
    if len(new_features) > 0:
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        
        # Final check: ensure no NaN values remain in new features
        nan_cols = new_features_df.columns[new_features_df.isna().any()].tolist()
        if len(nan_cols) > 0:
            print(f"Warning: Found NaN in temporal features: {len(nan_cols)} columns. Filling with 0.")
            df[nan_cols] = df[nan_cols].fillna(0)

    return df


def create_all_features(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    config: Optional[FeatureConfig] = None,
    inplace: bool = True,
    physics_config: Optional[PhysicsFeatureConfig] = None,
) -> pd.DataFrame:
    """
    High-level feature pipeline.

    - Optionally add physics-informed core features (create_physical_features)
    - Optionally add temporal multi-scale features (add_temporal_features)

    Use FeatureConfig to switch parts on/off for sensitivity studies.
    """
    if config is None:
        config = FeatureConfig()

    if not inplace:
        df = df.copy()

    # 1) Physics-informed base features
    if config.add_physical_core:
        # If physics_config is provided, use it; otherwise use default
        if physics_config is None:
            from .config import PhysicsFeatureConfig
            physics_config = PhysicsFeatureConfig(use_core=True, use_extended=False, use_residuals=False, use_temporal_on_physics=False)
        # Only add physics features if they haven't been added yet
        # Check by looking for core physics features
        if not any(col in df.columns for col in ["HPC_Eff_Proxy", "EGT_Drift", "Fan_HPC_Ratio", "Effizienz_HPC_Proxy"]):
            df = create_physical_features(df, physics_config, unit_col=unit_col, cycle_col=cycle_col)

    # 2) Temporal multi-scale features
    if config.add_temporal_features:
        df = add_temporal_features(
            df,
            unit_col=unit_col,
            cycle_col=cycle_col,
            config=config.temporal,
            inplace=True,  # we already control copy behavior above
        )

    # Final validation: check for any remaining NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        print(f"Warning: Found {nan_count} NaN values in {len(nan_cols)} columns after feature engineering.")
        print(f"  Columns with NaN: {nan_cols[:10]}{'...' if len(nan_cols) > 10 else ''}")
        print("  Filling remaining NaN with 0.")
        df = df.fillna(0)

    return df


def compute_per_engine_baseline(
    df: pd.DataFrame,
    cols: List[str],
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    n_baseline_cycles: int = 20,
) -> pd.DataFrame:
    """
    Compute per-engine baseline values for the given columns.
    Baseline = mean over the first n_baseline_cycles per engine.
    
    This function must not depend on RUL or any label column.
    
    Args:
        df: DataFrame with unit_col, cycle_col, and cols
        cols: List of column names to compute baselines for
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        n_baseline_cycles: Number of initial cycles to use for baseline
        
    Returns:
        DataFrame indexed by unit_col with one column per feature
    """
    # Group by unit and filter to first n_baseline_cycles
    baseline_data = []
    for unit_id, df_unit in df.groupby(unit_col):
        df_unit_sorted = df_unit.sort_values(cycle_col)
        df_unit_baseline = df_unit_sorted[df_unit_sorted[cycle_col] <= n_baseline_cycles]
        
        if len(df_unit_baseline) == 0:
            # If no baseline cycles, use mean of 0 (will be handled by fillna later)
            baseline_row = {unit_col: unit_id}
            for col in cols:
                baseline_row[col] = 0.0
        else:
            baseline_row = {unit_col: unit_id}
            for col in cols:
                if col in df_unit_baseline.columns:
                    baseline_row[col] = df_unit_baseline[col].mean()
                else:
                    baseline_row[col] = 0.0
        baseline_data.append(baseline_row)
    
    baseline_df = pd.DataFrame(baseline_data)
    baseline_df = baseline_df.set_index(unit_col)
    return baseline_df


def compute_per_condition_baseline(
    df: pd.DataFrame,
    cols: List[str],
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    cond_col: str = "ConditionID",
    n_baseline_cycles: int = 30,
) -> pd.DataFrame:
    """
    Compute per-condition baseline values for the given columns.
    Baseline = mean over the first n_baseline_cycles of all engines in each condition.
    
    This function must not depend on RUL or any label column.
    
    Args:
        df: DataFrame with unit_col, cycle_col, cond_col, and cols
        cols: List of column names to compute baselines for
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        cond_col: Name of condition column
        n_baseline_cycles: Number of initial cycles to use for baseline
        
    Returns:
        DataFrame indexed by cond_col with one column per feature
    """
    baseline_data = []
    for cond_id, df_cond in df.groupby(cond_col):
        # For each engine in this condition, get first n_baseline_cycles
        cond_baseline_rows = []
        for unit_id, df_unit in df_cond.groupby(unit_col):
            df_unit_sorted = df_unit.sort_values(cycle_col)
            df_unit_baseline = df_unit_sorted[df_unit_sorted[cycle_col] <= n_baseline_cycles]
            if len(df_unit_baseline) > 0:
                cond_baseline_rows.append(df_unit_baseline)
        
        if len(cond_baseline_rows) == 0:
            baseline_row = {cond_col: cond_id}
            for col in cols:
                baseline_row[col] = 0.0
        else:
            # Concatenate all early cycles from all engines in this condition
            all_baseline = pd.concat(cond_baseline_rows, ignore_index=True)
            baseline_row = {cond_col: cond_id}
            for col in cols:
                if col in all_baseline.columns:
                    baseline_row[col] = all_baseline[col].mean()
                else:
                    baseline_row[col] = 0.0
        baseline_data.append(baseline_row)
    
    baseline_df = pd.DataFrame(baseline_data)
    baseline_df = baseline_df.set_index(cond_col)
    return baseline_df


def add_residual_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    cond_col: str = "ConditionID",
    baseline_len: int = 30,
    mode: str = "per_engine",
    suffix: str = "_res",
    include_original: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add residual features: current value - baseline.
    
    For mode="per_engine": baseline = mean of first baseline_len cycles per engine.
    For mode="per_condition": baseline = mean of first baseline_len cycles per condition.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names to compute residuals for
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        cond_col: Name of condition column (required for mode="per_condition")
        baseline_len: Number of initial cycles to use for baseline
        mode: "per_engine" or "per_condition"
        suffix: Suffix to append to residual feature names
        include_original: If True, keep original features. If False, drop them.
        
    Returns:
        Tuple of (df_with_residuals, updated_feature_cols_list)
    """
    df_new = df.copy()
    df_new = df_new.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    
    # Filter feature_cols to only those that exist in the dataframe
    available_cols = [col for col in feature_cols if col in df_new.columns]
    if len(available_cols) == 0:
        print("Warning: No available feature columns for residual computation.")
        return df_new, feature_cols
    
    # Compute baselines
    if mode == "per_engine":
        baseline_df = compute_per_engine_baseline(
            df_new,
            cols=available_cols,
            unit_col=unit_col,
            cycle_col=cycle_col,
            n_baseline_cycles=baseline_len,
        )
        merge_key = unit_col
    elif mode == "per_condition":
        if cond_col not in df_new.columns:
            raise ValueError(f"Condition column '{cond_col}' not found for per_condition mode.")
        baseline_df = compute_per_condition_baseline(
            df_new,
            cols=available_cols,
            unit_col=unit_col,
            cycle_col=cycle_col,
            cond_col=cond_col,
            n_baseline_cycles=baseline_len,
        )
        merge_key = cond_col
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'per_engine' or 'per_condition'.")
    
    # Merge baselines back
    df_new = df_new.merge(
        baseline_df.reset_index(),
        on=merge_key,
        how='left',
        suffixes=('', '_Base')
    )
    
    # Compute residuals
    new_feature_cols = []
    for col in available_cols:
        baseline_col = f"{col}_Base"
        if baseline_col in df_new.columns:
            residual_col = f"{col}{suffix}"
            df_new[residual_col] = df_new[col] - df_new[baseline_col].fillna(0)
            new_feature_cols.append(residual_col)
            # Cleanup baseline column
            df_new = df_new.drop(columns=[baseline_col])
    
    # Update feature column list
    if include_original:
        updated_feature_cols = feature_cols + new_feature_cols
    else:
        # Remove original columns from list and add residuals
        updated_feature_cols = [col for col in feature_cols if col not in available_cols] + new_feature_cols
    
    return df_new, updated_feature_cols


def add_core_physical_features(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
) -> pd.DataFrame:
    """
    Add core physics-informed features:
    - HPC_Eff_Proxy    ~ Sensor12 / Sensor7
    - EGT_Drift        ~ Sensor17 - EGT_baseline_per_engine
    - Fan_HPC_Ratio    ~ Sensor2 / Sensor3
    
    Uses numerically safe divisions (avoid division by zero).
    Uses compute_per_engine_baseline to compute EGT_baseline from sensor 17.
    
    Args:
        df: DataFrame with sensor columns
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        
    Returns:
        DataFrame with added columns: HPC_Eff_Proxy, EGT_Drift, Fan_HPC_Ratio
    """
    df_new = df.copy()
    
    # Ensure sorted by unit and cycle
    df_new = df_new.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    
    # --- 1. HPC Efficiency Proxy (Total Pressure / Total Temperature) ---
    # Sensor 12: Total pressure at High Pressure Turbine (T4)
    # Sensor 7: Total temperature at High Pressure Compressor (T2)
    # Handle division by zero: use epsilon for safety
    epsilon = 1e-8
    if 'Sensor12' in df_new.columns and 'Sensor7' in df_new.columns:
        df_new['HPC_Eff_Proxy'] = (
            df_new['Sensor12'] / (df_new['Sensor7'] + epsilon)
        ).replace([np.inf, -np.inf], np.nan)
        # Forward fill within each unit, then fill remaining NaN with 0
        df_new['HPC_Eff_Proxy'] = (
            df_new.groupby(unit_col)['HPC_Eff_Proxy'].ffill().fillna(0)
        )
    else:
        df_new['HPC_Eff_Proxy'] = 0.0
    
    # --- 2. Exhaust Gas Temperature Drift (EGT Drift) ---
    # Sensor 17: Exhaust Gas Temperature (EGT)
    if 'Sensor17' in df_new.columns:
        # Compute baseline per engine (first 20 cycles)
        baseline_df = compute_per_engine_baseline(
            df_new,
            cols=['Sensor17'],
            unit_col=unit_col,
            cycle_col=cycle_col,
            n_baseline_cycles=20,
        )
        
        # Merge baseline back
        df_new = df_new.reset_index(drop=True)
        df_new = df_new.merge(
            baseline_df.reset_index(),
            on=unit_col,
            how='left',
            suffixes=('', '_Base')
        )
        
        # Calculate drift
        baseline_col = 'Sensor17_Base'
        if baseline_col in df_new.columns:
            df_new['EGT_Drift'] = df_new['Sensor17'] - df_new[baseline_col].fillna(0)
            df_new = df_new.drop(columns=[baseline_col])
        else:
            df_new['EGT_Drift'] = 0.0
    else:
        df_new['EGT_Drift'] = 0.0
    
    # --- 3. Fan-HPC Degradation Ratio ---
    # Sensor 2: Fan Speed
    # Sensor 3: HPC Speed
    if 'Sensor2' in df_new.columns and 'Sensor3' in df_new.columns:
        df_new['Fan_HPC_Ratio'] = (
            df_new['Sensor2'] / (df_new['Sensor3'] + epsilon)
        ).replace([np.inf, -np.inf], np.nan)
        # Forward fill within each unit, then fill remaining NaN with 0
        df_new['Fan_HPC_Ratio'] = (
            df_new.groupby(unit_col)['Fan_HPC_Ratio'].ffill().fillna(0)
        )
    else:
        df_new['Fan_HPC_Ratio'] = 0.0
    
    # Also add legacy names for backward compatibility
    if 'HPC_Eff_Proxy' in df_new.columns:
        df_new['Effizienz_HPC_Proxy'] = df_new['HPC_Eff_Proxy']
    
    return df_new


def add_extended_physical_features(
    df: pd.DataFrame,
    physics_config: PhysicsFeatureConfig,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
) -> pd.DataFrame:
    """
    Add a small, interpretable set of additional physics-based features.
    
    Examples:
    - simple temperature ratios (if use_temp_ratios)
    - simple pressure ratios (if use_press_ratios)
    - optional corrected-speed proxies (if use_corrected_speeds)
    
    Args:
        df: DataFrame with sensor columns
        physics_config: PhysicsFeatureConfig controlling which features to add
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        
    Returns:
        DataFrame with additional extended physics features
    """
    df_new = df.copy()
    epsilon = 1e-8
    
    # Temperature ratios (if enabled)
    if physics_config.use_temp_ratios:
        # Simple temperature ratio: Sensor7 / Sensor17 (compressor / exhaust)
        if 'Sensor7' in df_new.columns and 'Sensor17' in df_new.columns:
            df_new['Temp_Ratio_Comp_Exh'] = (
                df_new['Sensor7'] / (df_new['Sensor17'] + epsilon)
            ).replace([np.inf, -np.inf], np.nan)
            df_new['Temp_Ratio_Comp_Exh'] = (
                df_new.groupby(unit_col)['Temp_Ratio_Comp_Exh'].ffill().fillna(0)
            )
    
    # Pressure ratios (if enabled)
    if physics_config.use_press_ratios:
        # Simple pressure ratio: Sensor12 / Sensor13 (if available)
        if 'Sensor12' in df_new.columns and 'Sensor13' in df_new.columns:
            df_new['Press_Ratio_HPT'] = (
                df_new['Sensor12'] / (df_new['Sensor13'] + epsilon)
            ).replace([np.inf, -np.inf], np.nan)
            df_new['Press_Ratio_HPT'] = (
                df_new.groupby(unit_col)['Press_Ratio_HPT'].ffill().fillna(0)
            )
    
    # Corrected speeds (if enabled)
    if physics_config.use_corrected_speeds:
        # Corrected speed proxy: Sensor2 / sqrt(Sensor7) (Fan / sqrt(T2))
        if 'Sensor2' in df_new.columns and 'Sensor7' in df_new.columns:
            df_new['Corrected_Fan_Speed'] = (
                df_new['Sensor2'] / np.sqrt(df_new['Sensor7'] + epsilon)
            ).replace([np.inf, -np.inf], np.nan)
            df_new['Corrected_Fan_Speed'] = (
                df_new.groupby(unit_col)['Corrected_Fan_Speed'].ffill().fillna(0)
            )
    
    return df_new


def add_residual_physical_features(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    n_baseline_cycles: int = 20,
) -> pd.DataFrame:
    """
    Add residual features: current value - per-engine baseline
    for selected physical features (e.g. EGT_Drift base-sensor, HPC_Eff_Proxy inputs, etc.).
    
    Args:
        df: DataFrame with physics features
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        n_baseline_cycles: Number of initial cycles to use for baseline
        
    Returns:
        DataFrame with additional residual features
    """
    df_new = df.copy()
    
    # Ensure sorted
    df_new = df_new.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    
    # Compute baselines for key sensors used in physics features
    sensor_cols_for_residuals = ['Sensor17', 'Sensor12', 'Sensor7', 'Sensor2', 'Sensor3']
    available_sensors = [col for col in sensor_cols_for_residuals if col in df_new.columns]
    
    if len(available_sensors) > 0:
        baseline_df = compute_per_engine_baseline(
            df_new,
            cols=available_sensors,
            unit_col=unit_col,
            cycle_col=cycle_col,
            n_baseline_cycles=n_baseline_cycles,
        )
        
        # Merge baselines
        df_new = df_new.merge(
            baseline_df.reset_index(),
            on=unit_col,
            how='left',
            suffixes=('', '_Base')
        )
        
        # Compute residuals
        for col in available_sensors:
            baseline_col = f"{col}_Base"
            if baseline_col in df_new.columns:
                residual_col = f"{col}_Residual"
                df_new[residual_col] = df_new[col] - df_new[baseline_col].fillna(0)
                # Cleanup baseline column
                df_new = df_new.drop(columns=[baseline_col])
    
    return df_new


def add_temporal_physics_features(
    df: pd.DataFrame,
    physics_cols: List[str],
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    windows: List[int] = [5, 20],
) -> pd.DataFrame:
    """
    For each physics feature in physics_cols, compute per-engine rolling statistics
    (mean, std) and a simple slope over the specified windows.
    
    Args:
        df: DataFrame with physics features
        physics_cols: List of physics feature column names
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        windows: List of window sizes for rolling statistics
        
    Returns:
        DataFrame with additional temporal physics features
    """
    df_new = df.copy()
    
    # Ensure sorted
    df_new = df_new.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    
    # Group by unit
    g = df_new.groupby(unit_col, group_keys=False)
    
    # Collect new features
    new_features = {}
    
    for col in physics_cols:
        if col not in df_new.columns:
            continue
        
        series_by_unit = g[col]
        
        for w in windows:
            window_name = f"{w}"
            
            # Rolling mean
            new_name = f"{col}_rollmean_{window_name}"
            new_features[new_name] = series_by_unit.transform(
                lambda x, w=w: x.rolling(window=w, min_periods=1).mean()
            )
            
            # Rolling std
            new_name = f"{col}_rollstd_{window_name}"
            rolled_std = series_by_unit.transform(
                lambda x, w=w: x.rolling(window=w, min_periods=1).std()
            )
            new_features[new_name] = rolled_std.fillna(0.0)
            
            # Approximate slope: (x_t - x_{t-w}) / w
            new_name = f"{col}_slope_{window_name}"
            shifted = series_by_unit.shift(w)
            slope = (df_new[col] - shifted) / float(w)
            new_features[new_name] = slope.fillna(0.0)
    
    # Add all new features at once
    if len(new_features) > 0:
        new_features_df = pd.DataFrame(new_features, index=df_new.index)
        df_new = pd.concat([df_new, new_features_df], axis=1)
        
        # Ensure no NaN values
        nan_cols = new_features_df.columns[new_features_df.isna().any()].tolist()
        if len(nan_cols) > 0:
            df_new[nan_cols] = df_new[nan_cols].fillna(0)
    
    return df_new


def create_physical_features(
    df: pd.DataFrame,
    physics_config: Optional[PhysicsFeatureConfig] = None,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
) -> pd.DataFrame:
    """
    Apply physics feature pipeline according to physics_config.
    
    This is the main entry point for physics feature creation.
    
    Supports both old and new signatures:
    - Old: create_physical_features(df)  # Uses default config
    - New: create_physical_features(df, physics_config, unit_col=..., cycle_col=...)
    
    Args:
        df: DataFrame with sensor data
        physics_config: PhysicsFeatureConfig controlling which features to create.
                       If None, uses default config (core features only) for backward compatibility.
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        
    Returns:
        DataFrame with physics features added
    """
    # Backward compatibility: if physics_config is None, use default
    if physics_config is None:
        from .config import PhysicsFeatureConfig
        physics_config = PhysicsFeatureConfig(use_core=True, use_extended=False, use_residuals=False, use_temporal_on_physics=False)
    
    df_new = df.copy()
    
    if physics_config.use_core:
        df_new = add_core_physical_features(df_new, unit_col=unit_col, cycle_col=cycle_col)
    
    if physics_config.use_extended:
        df_new = add_extended_physical_features(df_new, physics_config, unit_col=unit_col, cycle_col=cycle_col)
    
    if physics_config.use_residuals or physics_config.residual.enabled:
        # Use the new general residual feature function
        # First, identify which columns to compute residuals for
        # We'll compute residuals for all sensor and physics feature columns
        # (excluding meta columns like UnitNumber, TimeInCycles, RUL, etc.)
        exclude_cols = {unit_col, cycle_col, "RUL", "RUL_raw", "MaxTime", "ConditionID"}
        candidate_cols = [
            col for col in df_new.columns
            if col not in exclude_cols
            and not col.endswith("_res")  # Avoid double-residuals
            and not col.endswith("_Base")  # Avoid baseline columns
        ]
        
        # Use residual config if available, otherwise use defaults
        residual_config = physics_config.residual if hasattr(physics_config, 'residual') else ResidualFeatureConfig()
        
        df_new, _ = add_residual_features(
            df=df_new,
            feature_cols=candidate_cols,
            unit_col=unit_col,
            cycle_col=cycle_col,
            cond_col="ConditionID",
            baseline_len=residual_config.baseline_len,
            mode=residual_config.mode,
            suffix="_res",
            include_original=residual_config.include_original,
        )
    
    if physics_config.use_temporal_on_physics:
        # Identify physics columns
        physics_cols = []
        for col in df_new.columns:
            if col in physics_config.core_features:
                physics_cols.append(col)
            elif any(keyword in col for keyword in ["Residual", "Ratio", "Drift", "Proxy", "Temp_Ratio", "Press_Ratio", "Corrected"]):
                physics_cols.append(col)
        
        if len(physics_cols) > 0:
            df_new = add_temporal_physics_features(df_new, physics_cols, unit_col=unit_col, cycle_col=cycle_col)
    
    return df_new


# ===================================================================
# Feature Groups & Ablation Support
# ===================================================================

BASE_SETTING_COLS = ["Setting1", "Setting2", "Setting3"]
BASE_SENSOR_COLS = [f"Sensor{i}" for i in range(1, 22)]
PHYSICS_COLS = ["Effizienz_HPC_Proxy", "EGT_Drift", "Fan_HPC_Ratio"]
CONDITION_COLS = ["ConditionID"]  # optional

# Continuous condition vector (for digital-twin / phys-informed models)
# Version 2: original set used by phys_v2 experiments (backwards compatible)
CONDITION_FEATURE_COLS_V2: List[str] = [
    # Operating settings (ambient / operating point)
    "Setting1",
    "Setting2",
    "Setting3",
    # Classic condition sensors (used for ConditionID)
    "Sensor1",
    "Sensor2",
    "Sensor3",
    # Physics-based proxies
    "Effizienz_HPC_Proxy",
    "Fan_HPC_Ratio",
]

# Version 3: extended set for phys_v3 experiments – adds a few slow, physics-relevant
# variables while remaining backwards compatible for older runs.
#
# Notes:
# - We reuse ambient / operating point information from Setting1/2/3.
# - We add EGT_Drift as an additional slow proxy for thermal state.
# - We keep the list small to avoid exploding feature dimensionality.
CONDITION_FEATURE_COLS_V3: List[str] = CONDITION_FEATURE_COLS_V2 + [
    "EGT_Drift",
]

# Default alias used by older helper functions that do not take a version.
CONDITION_FEATURE_COLS: List[str] = CONDITION_FEATURE_COLS_V2


def get_feature_columns(df: pd.DataFrame, cfg: FeatureGroupsConfig) -> List[str]:
    """
    Construct feature column list based on FeatureGroupsConfig.
    
    Args:
        df: DataFrame with all potential columns
        cfg: FeatureGroupsConfig specifying which groups to use
        
    Returns:
        List of feature column names
    """
    cols: List[str] = []

    if cfg.use_settings:
        cols.extend([c for c in BASE_SETTING_COLS if c in df.columns])
    if cfg.use_sensors:
        cols.extend([c for c in BASE_SENSOR_COLS if c in df.columns])
    if cfg.use_physics_core:
        cols.extend([c for c in PHYSICS_COLS if c in df.columns])
    if cfg.use_condition_id and "ConditionID" in df.columns:
        cols.extend(CONDITION_COLS)

    # Temporal-window features will be added as new columns later,
    # and then picked up by a simple prefix-based filter:
    if cfg.use_temporal_windows:
        # We create columns like "tw_Sensor2_w5_mean", etc.
        # Here we just assume they exist and add them by prefix
        cols.extend(
            [c for c in df.columns
             if c.startswith("tw_") and not c.startswith("tw_meta_")]
        )

    # Final safety filter: drop any RUL-related names
    forbidden = {"RUL", "RUL_raw", "rul", "rul_raw"}
    cols = [c for c in cols if c not in forbidden and not c.lower().startswith("rul")]
    return sorted(set(cols))


def build_condition_features(
    df: pd.DataFrame,
    *,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    version: int = 2,
) -> pd.DataFrame:
    """
    Build a continuous condition feature vector per row.

    Uses (version 2, default):
        - Operating settings: Setting1/2/3
        - Condition sensors:  Sensor1/2/3
        - Physics proxies:    Effizienz_HPC_Proxy, Fan_HPC_Ratio

    Version 3 (phys_v3) extends this with one additional slow-changing proxy:
        - EGT_Drift

    Adds new columns with prefix "Cond_":
        Cond_Setting1, Cond_Setting2, Cond_Setting3,
        Cond_Sensor1, Cond_Sensor2, Cond_Sensor3,
        Cond_Effizienz_HPC_Proxy, Cond_Fan_HPC_Ratio

    These columns together form the continuous condition vector.

    Args:
        df: Input dataframe (physics + sensors already added)
        unit_col: Engine/unit column name
        cycle_col: Cycle column name (unused, kept for API symmetry)
        version: Condition vector version (2 = original phys_v2, 3 = extended phys_v3)
    """
    df_new = df.copy()

    if version >= 3:
        cond_cols = CONDITION_FEATURE_COLS_V3
    else:
        cond_cols = CONDITION_FEATURE_COLS_V2

    for col in cond_cols:
        if col not in df_new.columns:
            # silently skip missing columns; pipeline is robust across datasets
            continue
        cond_col = f"Cond_{col}"
        if cond_col not in df_new.columns:
            df_new[cond_col] = df_new[col]

    return df_new


def get_condition_vector_cols(
    df: pd.DataFrame,
    version: int = 2,
) -> List[str]:
    """
    Return the list of continuous condition feature columns present in df.
    
    By convention these are the "Cond_*" copies of the selected CONDITION_FEATURE_COLS_V*.
    
    Args:
        df: DataFrame with potential Cond_* columns
        version: Condition vector version (2 = phys_v2, 3 = phys_v3)
    """
    if version >= 3:
        bases = CONDITION_FEATURE_COLS_V3
    else:
        bases = CONDITION_FEATURE_COLS_V2

    cols: List[str] = []
    for base in bases:
        cond_col = f"Cond_{base}"
        if cond_col in df.columns:
            cols.append(cond_col)
    return cols


def add_temporal_window_features(
    df: pd.DataFrame,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    temporal_cfg: Optional[TemporalWindowConfig] = None,
) -> pd.DataFrame:
    """
    Add temporal-window features per unit using TemporalWindowConfig.
    
    Creates features with prefix "tw_" for temporal window statistics.
    
    Args:
        df: DataFrame sorted by unit and cycle
        unit_col: Name of unit/engine column
        cycle_col: Name of cycle/time column
        temporal_cfg: TemporalWindowConfig specifying windows and stats
        
    Returns:
        DataFrame with additional temporal window features
    """
    if temporal_cfg is None:
        return df

    # Choose a reasonable set of base series for temporal stats
    # Use sensors and physics features
    base_cols = []
    for col in df.columns:
        if col.startswith("Sensor") or col in PHYSICS_COLS:
            if col not in {unit_col, cycle_col, "RUL", "MaxTime"}:
                base_cols.append(col)

    if len(base_cols) == 0:
        print("Warning: No base columns found for temporal window features.")
        return df

    df = df.sort_values([unit_col, cycle_col]).copy()
    group = df.groupby(unit_col, group_keys=False)

    # Collect all new features in a dictionary to avoid DataFrame fragmentation
    new_features = {}
    
    for col in base_cols:
        if col not in df.columns:
            continue
        series_by_unit = group[col]

        # Short + long windows
        all_windows = list(temporal_cfg.short_windows) + list(temporal_cfg.long_windows)
        for w in all_windows:
            win = int(w)
            if temporal_cfg.include_rolling_stats:
                new_features[f"tw_{col}_w{win}_mean"] = series_by_unit.transform(
                    lambda x, w=win: x.rolling(window=w, min_periods=1).mean()
                )
                new_features[f"tw_{col}_w{win}_std"] = series_by_unit.transform(
                    lambda x, w=win: x.rolling(window=w, min_periods=1).std().fillna(0.0)
                )
            if temporal_cfg.include_deltas:
                new_features[f"tw_{col}_w{win}_delta"] = series_by_unit.transform(
                    lambda x, w=win: x.diff().rolling(window=w, min_periods=1).mean().fillna(0.0)
                )

    # Add all new features at once using pd.concat to avoid fragmentation
    if len(new_features) > 0:
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        
        # Final check: ensure no NaN values remain
        nan_cols = new_features_df.columns[new_features_df.isna().any()].tolist()
        if len(nan_cols) > 0:
            df[nan_cols] = df[nan_cols].fillna(0)

    return df


def create_twin_features(
    df: pd.DataFrame,
    *,
    unit_col: str = "UnitNumber",
    cycle_col: str = "TimeInCycles",
    sensor_cols: Optional[List[str]] = None,
    condition_cols: Optional[List[str]] = None,
    baseline_len: int = 30,
    condition_vector_version: int = 2,
) -> Tuple[pd.DataFrame, HealthyTwinRegressor]:
    """
    Fit a global HealthyTwinRegressor on early cycles and
    add twin prediction + residual columns.

    Returns:
        df_with_features, twin_model
    """
    df_new = df.copy()

    if condition_cols is None:
        # Use continuous condition vector if available; fall back to raw columns.
        cond_vec_cols = get_condition_vector_cols(df_new, version=condition_vector_version)
        if len(cond_vec_cols) == 0:
            # Fall back to raw base columns for the selected version
            if condition_vector_version >= 3:
                base_cols = CONDITION_FEATURE_COLS_V3
            else:
                base_cols = CONDITION_FEATURE_COLS_V2
            cond_vec_cols = [c for c in base_cols if c in df_new.columns]
        condition_cols = cond_vec_cols

    if sensor_cols is None:
        # Default: all original sensor columns (Sensor1..Sensor21)
        sensor_cols = [col for col in df_new.columns if col.startswith("Sensor")]

    if len(condition_cols) == 0 or len(sensor_cols) == 0:
        # Nothing to do – return original df and an unfitted twin
        twin = HealthyTwinRegressor()
        return df_new, twin

    twin = HealthyTwinRegressor()
    twin.fit(
        df_new,
        unit_col=unit_col,
        cycle_col=cycle_col,
        condition_cols=condition_cols,
        sensor_cols=sensor_cols,
        baseline_len=baseline_len,
    )

    df_new = twin.add_twin_and_residuals(df_new)
    return df_new, twin


def group_feature_columns(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group feature column names into semantically meaningful blocks for encoder/decoder use.

    This helper is intentionally prefix / pattern based so that it works across
    both legacy feature names (roll_mean / trend / delta) and the newer
    digital-twin residual block:

    - raw:      everything that is *not* clearly MS_ / temporal / residual / Cond_ / Twin_
    - ms:       multi-scale / temporal features (rolling stats, trends, deltas)
                 e.g. tw_*, *_roll_*, *_trend_*, *_delta_*
    - residual: residual features produced by either
                 - HealthyTwinRegressor (Resid_*)
                 - generic residual block (*_res, *_Residual)
    - twin:     Healthy twin predictions (Twin_*)
    - cond:     continuous condition vector (Cond_*)

    The exact grouping contract is:

    - Future encoder / decoder models can safely treat `ms` as "multi-scale context",
      `residual` as "degradation signal", `cond` as "operating condition input",
      and `raw` as the remaining base signals.
    """
    raw: List[str] = []
    ms: List[str] = []
    residual: List[str] = []
    cond: List[str] = []
    twin: List[str] = []

    for col in feature_cols:
        # Continuous condition vector
        if col.startswith("Cond_"):
            cond.append(col)
            continue

        # Twin predictions (digital twin healthy predictions)
        if col.startswith("Twin_"):
            twin.append(col)
            continue

        # Residuals from twin or baseline-based residual blocks
        if (
            col.startswith("Resid_")
            or col.startswith("RES_")
            or col.endswith("_res")
            or col.endswith("_Residual")
        ):
            residual.append(col)
            continue

        # Multi-scale / temporal features.
        # We treat both the dedicated temporal-window block (tw_*) and the
        # older roll/trend/delta style features as MS_ for grouping.
        if (
            col.startswith("tw_")
            or "_roll_" in col
            or "_rollmean_" in col
            or "_rollstd_" in col
            or "_trend_" in col
            or "_delta_" in col
        ):
            ms.append(col)
            continue

        # Everything else is considered "raw" encoder input
        raw.append(col)

    return {
        "raw": raw,
        "ms": ms,
        "residual": residual,
        "cond": cond,
        "twin": twin,
    }