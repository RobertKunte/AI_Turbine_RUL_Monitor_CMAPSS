"""
Failure Case Library - Systematic worst-case analysis for CMAPSS experiments.

Builds a reproducible library of failure patterns and worst-case engines.
Supports World Model v3 with tf_cross decoder and quantile predictions.
HI is UNSUPERVISED (no HI_true required).
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass
class FailureCaseRecord:
    """Per-unit failure case record."""
    unit_id: int
    condition_id: Optional[int]

    # Last-window metrics
    rmse_last: float
    mae_last: float
    bias_last: float
    r2_last: float
    nasa_last_mean: float
    nasa_last_sum: float

    # EOL predictions
    true_last_rul: float
    pred_last_rul: float
    abs_err_last: float

    # Quantile metrics (optional)
    q10_pred: Optional[float] = None
    q50_pred: Optional[float] = None
    q90_pred: Optional[float] = None
    q10_coverage: Optional[float] = None
    q50_coverage: Optional[float] = None
    q90_coverage: Optional[float] = None

    # Failure labels
    labels: List[str] = field(default_factory=list)


@dataclass
class FailureCaseLibrary:
    """Complete failure case library for an experiment."""
    experiment_name: str
    dataset: str
    model_type: str
    decoder_type: str

    # Configuration
    config: Dict[str, Any]

    # Top-K selections
    top_k_overall: List[int]  # unit_ids
    top_k_per_condition: Dict[int, List[int]]  # condition_id -> unit_ids

    # All cases
    all_cases: List[FailureCaseRecord]

    # Metadata
    num_test_engines: int
    num_conditions: int
    ranking_metric: str


def _load_experiment_metrics(experiment_dir: Path) -> Tuple[List, Dict, Optional[Dict]]:
    """
    Load metrics from experiment directory.

    Returns:
        (eol_metrics: List[EngineEOLMetrics], config: Dict, condition_map: Dict[int, int])

    Sources (in priority order):
    1. Load from existing eol_metrics.json if exists
    2. Run inference if needed (reuse run_inference_for_experiment)
    """
    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "summary.json"
    eol_metrics_path = experiment_dir / "eol_metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {experiment_dir}")

    # Load config
    with open(summary_path) as f:
        config = json.load(f)

    # Try loading existing metrics
    if eol_metrics_path.exists():
        print(f"[FailureCaseLibrary] Loading metrics from {eol_metrics_path.name}")
        with open(eol_metrics_path) as f:
            metrics_data = json.load(f)
        eol_metrics = _parse_eol_metrics_json(metrics_data, config)
        condition_map = None  # Will load separately if needed

        # If parsing failed (old format without y_true_eol/y_pred_eol), run inference
        if not eol_metrics:
            print(f"[FailureCaseLibrary] Old format detected, running inference to regenerate metrics...")
            from src.analysis.inference import run_inference_for_experiment

            result = run_inference_for_experiment(
                experiment_dir=str(experiment_dir),
                return_hi_trajectories=False,
            )

            if isinstance(result, tuple) and len(result) >= 2:
                eol_metrics = result[0]
                condition_map = result[2] if len(result) > 2 else None
            else:
                eol_metrics = result
                condition_map = None
    else:
        print(f"[FailureCaseLibrary] Running inference to generate metrics...")
        from src.analysis.inference import run_inference_for_experiment

        # Run inference
        result = run_inference_for_experiment(
            experiment_dir=str(experiment_dir),
            return_hi_trajectories=False,  # Not needed for EOL analysis
        )

        # Handle tuple unpacking based on return signature
        if isinstance(result, tuple) and len(result) >= 2:
            eol_metrics = result[0]
            # Try to get condition map if available
            condition_map = result[2] if len(result) > 2 else None
        else:
            eol_metrics = result
            condition_map = None

    # Load condition mapping from test data if not available
    if condition_map is None:
        condition_map = _load_condition_ids_from_data(experiment_dir, config)

    return eol_metrics, config, condition_map


def _parse_eol_metrics_json(metrics_data: Dict, config: Dict) -> List:
    """Parse eol_metrics.json into EngineEOLMetrics-like objects."""
    from src.eval.eol_eval import EngineEOLMetrics

    errors = metrics_data.get("errors", [])
    y_true = metrics_data.get("y_true_eol", [])
    y_pred = metrics_data.get("y_pred_eol", [])
    nasa_scores = metrics_data.get("nasa_scores", [])

    # Fallback: if y_true_eol or y_pred_eol are missing, reconstruct from errors
    if not y_true or not y_pred:
        if errors and nasa_scores:
            print("[FailureCaseLibrary] Warning: y_true_eol/y_pred_eol missing, reconstructing from errors")
            # We have errors but not y_true/y_pred - this is an old format
            # We can't reconstruct the actual values without more info, so we need to run inference
            return []  # Signal that we need to run inference
        else:
            print("[FailureCaseLibrary] Error: Insufficient data in eol_metrics.json")
            return []

    # Try to get quantiles if available
    q10_list = metrics_data.get("q10_eol", [None] * len(errors))
    q50_list = metrics_data.get("q50_eol", [None] * len(errors))
    q90_list = metrics_data.get("q90_eol", [None] * len(errors))

    eol_metrics = []
    for i, (err, true_rul, pred_rul, nasa) in enumerate(zip(errors, y_true, y_pred, nasa_scores)):
        metric = EngineEOLMetrics(
            unit_id=i + 1,  # 1-based indexing for CMAPSS
            true_rul=float(true_rul),
            pred_rul=float(pred_rul),
            error=float(err),
            nasa=float(nasa),
            q10=float(q10_list[i]) if i < len(q10_list) and q10_list[i] is not None else None,
            q50=float(q50_list[i]) if i < len(q50_list) and q50_list[i] is not None else None,
            q90=float(q90_list[i]) if i < len(q90_list) and q90_list[i] is not None else None,
        )
        eol_metrics.append(metric)

    return eol_metrics


def _load_condition_ids_from_data(experiment_dir: Path, config: Dict) -> Dict[int, int]:
    """
    Load condition ID mapping from test data.

    Returns:
        {unit_id: condition_id}
    """
    dataset = config.get("dataset", "FD004")

    # Try to load test data to get condition IDs
    try:
        from src.data_loading import load_cmapps_subset

        df_train, df_test, y_test_true = load_cmapps_subset(
            fd_id=dataset,  # Correct parameter name
        )

        # Extract condition IDs per unit
        condition_map = {}
        if "ConditionID" in df_test.columns:
            for unit_id in df_test["UnitNumber"].unique():
                unit_df = df_test[df_test["UnitNumber"] == unit_id]
                # Use first condition ID for the unit (should be consistent)
                condition_map[int(unit_id)] = int(unit_df["ConditionID"].iloc[0])

        return condition_map
    except Exception as e:
        print(f"[FailureCaseLibrary] Warning: Could not load condition IDs: {e}")
        return {}


def _compute_quantile_coverage(eol_metrics: List) -> Dict[int, Dict[str, float]]:
    """
    Compute empirical quantile coverage per unit.

    NOTE: Currently only EOL (last window) available.
    This is a single-point check, not true coverage over multiple windows.

    Returns:
        {unit_id: {"q10_coverage": x, "q50_coverage": y, "q90_coverage": z}}
    """
    coverage_dict = {}

    for metric in eol_metrics:
        unit_id = metric.unit_id

        if metric.q10 is None or metric.q50 is None or metric.q90 is None:
            # No quantile predictions available
            coverage_dict[unit_id] = {
                "q10_coverage": None,
                "q50_coverage": None,
                "q90_coverage": None,
            }
            continue

        true_rul = metric.true_rul

        # Single-point coverage check (EOL only)
        # Ideally would compute over multiple windows, but only EOL is available
        # Check if true RUL falls within predicted intervals
        q10_cov = 1.0 if true_rul >= metric.q10 else 0.0
        q50_cov = 1.0 if abs(true_rul - metric.q50) <= abs(true_rul - metric.q10) else 0.0
        q90_cov = 1.0 if true_rul <= metric.q90 else 0.0

        coverage_dict[unit_id] = {
            "q10_coverage": q10_cov,
            "q50_coverage": q50_cov,
            "q90_coverage": q90_cov,
        }

    return coverage_dict


def _assign_failure_labels(
    case: FailureCaseRecord,
    bias_threshold_over: float,
    bias_threshold_under: float,
    early_life_range: Tuple[int, int],
    mid_life_range: Tuple[int, int],
    late_life_range: Tuple[int, int],
    coverage_deviation_threshold: float,
) -> List[str]:
    """
    Assign failure mode labels to a case.

    Labels:
    - OVER: bias_last > bias_threshold_over (over-prediction, late detection)
    - UNDER: bias_last < bias_threshold_under (under-prediction, false alarm)
    - EARLY_LIFE_FAIL: large error when true RUL in early_life_range
    - MID_LIFE_FAIL: large error when true RUL in mid_life_range
    - LATE_LIFE_FAIL: large error when true RUL in late_life_range
    - MIS_CALIBRATED_*: quantile coverage deviates > threshold
    """
    labels = []

    # Bias-based labels
    if case.bias_last > bias_threshold_over:
        labels.append("OVER")
    elif case.bias_last < bias_threshold_under:
        labels.append("UNDER")

    # RUL phase-based labels
    true_rul = case.true_last_rul
    abs_err = case.abs_err_last

    # Define "large error" threshold (> 20 cycles or > 30% of true RUL)
    error_threshold = max(20.0, 0.3 * true_rul)

    if abs_err > error_threshold:
        if early_life_range[0] <= true_rul <= early_life_range[1]:
            labels.append("EARLY_LIFE_FAIL")
        elif mid_life_range[0] <= true_rul <= mid_life_range[1]:
            labels.append("MID_LIFE_FAIL")
        elif late_life_range[0] <= true_rul <= late_life_range[1]:
            labels.append("LATE_LIFE_FAIL")

    # Calibration labels (if quantiles available)
    if case.q90_coverage is not None:
        if abs(case.q90_coverage - 0.90) > coverage_deviation_threshold:
            labels.append("MIS_CALIBRATED_Q90")
        if case.q50_coverage is not None and abs(case.q50_coverage - 0.50) > coverage_deviation_threshold:
            labels.append("MIS_CALIBRATED_Q50")
        if case.q10_coverage is not None and abs(case.q10_coverage - 0.10) > coverage_deviation_threshold:
            labels.append("MIS_CALIBRATED_Q10")

    return labels


def _select_top_k_cases(
    all_cases: List[FailureCaseRecord],
    K: int,
    ranking_metric: str,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Select top-K worst cases overall and per condition.

    Args:
        all_cases: All test engine cases
        K: Number of worst cases to select
        ranking_metric: Metric to rank by (e.g., "nasa_last_sum", "abs_err_last")

    Returns:
        (top_k_overall, top_k_per_condition)
    """
    # Overall top-K
    sorted_cases = sorted(
        all_cases,
        key=lambda c: getattr(c, ranking_metric),
        reverse=True,  # Worst = highest error/NASA score
    )
    top_k_overall = [c.unit_id for c in sorted_cases[:K]]

    # Per-condition top-K
    top_k_per_condition = {}
    cases_by_condition = {}

    for case in all_cases:
        if case.condition_id is not None:
            if case.condition_id not in cases_by_condition:
                cases_by_condition[case.condition_id] = []
            cases_by_condition[case.condition_id].append(case)

    for cond_id, cond_cases in cases_by_condition.items():
        sorted_cond = sorted(
            cond_cases,
            key=lambda c: getattr(c, ranking_metric),
            reverse=True,
        )
        # Take min(K, num_engines_in_condition)
        top_k_per_condition[cond_id] = [
            c.unit_id for c in sorted_cond[:min(K, len(sorted_cond))]
        ]

    return top_k_overall, top_k_per_condition


def build_failure_case_library(
    experiment_dir: Path,
    K: int = 10,
    ranking_metric: str = "nasa_last_sum",
    bias_threshold_over: float = 10.0,
    bias_threshold_under: float = -10.0,
    early_life_range: Tuple[int, int] = (60, 125),
    mid_life_range: Tuple[int, int] = (20, 60),
    late_life_range: Tuple[int, int] = (0, 20),
    coverage_deviation_threshold: float = 0.15,
) -> FailureCaseLibrary:
    """
    Build failure case library for an experiment.

    Args:
        experiment_dir: Path to experiment results directory
        K: Number of top-K worst cases to select
        ranking_metric: Metric to rank by ("nasa_last_sum", "abs_err_last", etc.)
        bias_threshold_over: Threshold for OVER label (late detection)
        bias_threshold_under: Threshold for UNDER label (false alarm)
        early_life_range: RUL range for early life failures (cycles)
        mid_life_range: RUL range for mid life failures (cycles)
        late_life_range: RUL range for late life failures (cycles)
        coverage_deviation_threshold: Threshold for calibration deviation

    Returns:
        FailureCaseLibrary object with all cases and top-K selections
    """
    experiment_dir = Path(experiment_dir)
    experiment_name = experiment_dir.name
    dataset = experiment_dir.parent.name

    print(f"[FailureCaseLibrary] Building library for {experiment_name}")

    # 1. Load metrics
    eol_metrics, config, condition_map = _load_experiment_metrics(experiment_dir)
    print(f"[FailureCaseLibrary] Loaded {len(eol_metrics)} test engines")

    # 2. Compute quantile coverage (if available)
    coverage_dict = _compute_quantile_coverage(eol_metrics)

    # 3. Build FailureCaseRecord for each engine
    all_cases = []
    for metric in eol_metrics:
        unit_id = metric.unit_id
        cond_id = condition_map.get(unit_id) if condition_map else None
        coverage = coverage_dict.get(unit_id, {})

        case = FailureCaseRecord(
            unit_id=unit_id,
            condition_id=cond_id,
            # Metrics (using error as proxy for RMSE/MAE when individual not available)
            rmse_last=abs(metric.error),
            mae_last=abs(metric.error),
            bias_last=metric.error,
            r2_last=0.0,  # Would need variance computation; not available from eol_metrics
            nasa_last_mean=metric.nasa,
            nasa_last_sum=metric.nasa,
            # EOL values
            true_last_rul=metric.true_rul,
            pred_last_rul=metric.pred_rul,
            abs_err_last=abs(metric.error),
            # Quantiles (if available)
            q10_pred=metric.q10,
            q50_pred=metric.q50,
            q90_pred=metric.q90,
            q10_coverage=coverage.get("q10_coverage"),
            q50_coverage=coverage.get("q50_coverage"),
            q90_coverage=coverage.get("q90_coverage"),
        )

        # Assign labels
        case.labels = _assign_failure_labels(
            case,
            bias_threshold_over,
            bias_threshold_under,
            early_life_range,
            mid_life_range,
            late_life_range,
            coverage_deviation_threshold,
        )

        all_cases.append(case)

    # 4. Select top-K cases
    top_k_overall, top_k_per_condition = _select_top_k_cases(
        all_cases, K, ranking_metric
    )

    print(f"[FailureCaseLibrary] Selected top-{K} worst cases")
    print(f"  Overall: {len(top_k_overall)} units")
    print(f"  Per-condition: {len(top_k_per_condition)} conditions")

    # 5. Build library object
    library = FailureCaseLibrary(
        experiment_name=experiment_name,
        dataset=dataset,
        model_type=config.get("encoder_type", "unknown"),
        decoder_type=config.get("world_model_config", {}).get("decoder_type", "unknown"),
        config={
            "K": K,
            "ranking_metric": ranking_metric,
            "bias_threshold_over": bias_threshold_over,
            "bias_threshold_under": bias_threshold_under,
            "early_life_range": early_life_range,
            "mid_life_range": mid_life_range,
            "late_life_range": late_life_range,
            "coverage_deviation_threshold": coverage_deviation_threshold,
        },
        top_k_overall=top_k_overall,
        top_k_per_condition=top_k_per_condition,
        all_cases=all_cases,
        num_test_engines=len(all_cases),
        num_conditions=len(top_k_per_condition),
        ranking_metric=ranking_metric,
    )

    return library


def save_failure_case_library(
    library: FailureCaseLibrary,
    experiment_dir: Path,
    save_plots: bool = True,
):
    """
    Save failure case library to disk.

    Creates:
        - failure_cases/summary.json
        - failure_cases/cases.csv
        - failure_cases/case_<unit_id>.json (for selected top-K)
        - failure_cases/plots/ (optional)
    """
    experiment_dir = Path(experiment_dir)
    failure_cases_dir = experiment_dir / "failure_cases"
    failure_cases_dir.mkdir(exist_ok=True)

    # 1. Save summary.json
    top_k_case_records = [
        asdict(c) for c in library.all_cases
        if c.unit_id in library.top_k_overall
    ]

    summary = {
        "experiment_name": library.experiment_name,
        "dataset": library.dataset,
        "model_type": library.model_type,
        "decoder_type": library.decoder_type,
        "num_test_engines": library.num_test_engines,
        "num_conditions": library.num_conditions,
        "ranking_metric": library.ranking_metric,
        "config": library.config,
        "top_k_overall": library.top_k_overall,
        "top_k_per_condition": {str(k): v for k, v in library.top_k_per_condition.items()},
        "top_k_cases": top_k_case_records,
    }

    summary_path = failure_cases_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 2. Save cases.csv
    df = pd.DataFrame([asdict(c) for c in library.all_cases])
    # Convert lists to strings for CSV (only if DataFrame is not empty)
    if not df.empty and 'labels' in df.columns:
        df['labels'] = df['labels'].apply(lambda x: ','.join(x) if x else '')
    df.to_csv(failure_cases_dir / "cases.csv", index=False)

    # 3. Save individual case files for selected top-K
    selected_unit_ids = set(library.top_k_overall)
    for cond_cases in library.top_k_per_condition.values():
        selected_unit_ids.update(cond_cases)

    for case in library.all_cases:
        if case.unit_id in selected_unit_ids:
            case_file = failure_cases_dir / f"case_{case.unit_id}.json"
            with open(case_file, "w") as f:
                json.dump(asdict(case), f, indent=2)

    # 4. Optionally save plots
    if save_plots:
        plots_dir = failure_cases_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        _save_failure_case_plots(library, experiment_dir, plots_dir, selected_unit_ids)

    print(f"[FailureCaseLibrary] Saved to {failure_cases_dir}")
    print(f"  - summary.json: {len(selected_unit_ids)} top-K cases")
    print(f"  - cases.csv: {len(library.all_cases)} total test engines")
    print(f"  - {len(selected_unit_ids)} individual case JSON files")
    if save_plots:
        print(f"  - {len(selected_unit_ids)} plots saved to plots/")


def _save_failure_case_plots(
    library: FailureCaseLibrary,
    experiment_dir: Path,
    plots_dir: Path,
    selected_unit_ids: set,
):
    """
    Save lightweight RUL plots for selected failure cases.

    Plots show:
    - True RUL (if available)
    - Predicted RUL (EOL value)
    - Quantile bands (if available)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[FailureCaseLibrary] Warning: matplotlib not available, skipping plots")
        return

    print(f"[FailureCaseLibrary] Generating plots for {len(selected_unit_ids)} cases...")

    for case in library.all_cases:
        if case.unit_id not in selected_unit_ids:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot single EOL point (since we don't have full trajectories)
        ax.scatter([0], [case.true_last_rul], color='black', s=100, marker='o',
                   label='True RUL (EOL)', zorder=5)
        ax.scatter([0], [case.pred_last_rul], color='red', s=100, marker='x',
                   label='Predicted RUL (EOL)', zorder=5)

        # Add quantile bands if available
        if case.q10_pred is not None and case.q90_pred is not None:
            ax.errorbar([0], [case.pred_last_rul],
                       yerr=[[case.pred_last_rul - case.q10_pred],
                             [case.q90_pred - case.pred_last_rul]],
                       fmt='none', color='blue', alpha=0.3, linewidth=8,
                       label='Q10-Q90 interval')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (EOL only)', fontsize=10)
        ax.set_ylabel('RUL (cycles)', fontsize=10)
        ax.set_title(f'Unit {case.unit_id} - Error: {case.bias_last:.1f} cycles\n'
                    f'Labels: {", ".join(case.labels) if case.labels else "None"}',
                    fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Save
        plot_path = plots_dir / f"case_{case.unit_id}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)

    print(f"[FailureCaseLibrary] Plots saved to {plots_dir}")
