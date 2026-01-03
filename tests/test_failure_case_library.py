"""
Unit tests for failure case library.
"""

import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd
from src.analysis.failure_case_library import (
    build_failure_case_library,
    save_failure_case_library,
    FailureCaseRecord,
    _assign_failure_labels,
    _select_top_k_cases,
)
from src.eval.eol_eval import EngineEOLMetrics


def test_assign_failure_labels_over():
    """Test OVER label assignment for late detection."""
    case = FailureCaseRecord(
        unit_id=1,
        condition_id=0,
        rmse_last=15.0,
        mae_last=15.0,
        bias_last=15.0,  # Over-prediction
        r2_last=0.8,
        nasa_last_mean=20.0,
        nasa_last_sum=20.0,
        true_last_rul=100.0,
        pred_last_rul=115.0,
        abs_err_last=15.0,
    )

    labels = _assign_failure_labels(
        case,
        bias_threshold_over=10.0,
        bias_threshold_under=-10.0,
        early_life_range=(60, 125),
        mid_life_range=(20, 60),
        late_life_range=(0, 20),
        coverage_deviation_threshold=0.15,
    )

    assert "OVER" in labels


def test_assign_failure_labels_under():
    """Test UNDER label assignment for under-prediction."""
    case = FailureCaseRecord(
        unit_id=2,
        condition_id=0,
        rmse_last=20.0,
        mae_last=20.0,
        bias_last=-20.0,  # Under-prediction
        r2_last=0.7,
        nasa_last_mean=30.0,
        nasa_last_sum=30.0,
        true_last_rul=80.0,
        pred_last_rul=60.0,
        abs_err_last=20.0,
    )

    labels = _assign_failure_labels(
        case,
        bias_threshold_over=10.0,
        bias_threshold_under=-10.0,
        early_life_range=(60, 125),
        mid_life_range=(20, 60),
        late_life_range=(0, 20),
        coverage_deviation_threshold=0.15,
    )

    assert "UNDER" in labels


def test_assign_failure_labels_early_life():
    """Test EARLY_LIFE_FAIL label for large errors in early life."""
    case = FailureCaseRecord(
        unit_id=3,
        condition_id=0,
        rmse_last=40.0,
        mae_last=40.0,
        bias_last=40.0,
        r2_last=0.5,
        nasa_last_mean=50.0,
        nasa_last_sum=50.0,
        true_last_rul=100.0,  # Early life
        pred_last_rul=140.0,
        abs_err_last=40.0,  # Large error
    )

    labels = _assign_failure_labels(
        case,
        bias_threshold_over=10.0,
        bias_threshold_under=-10.0,
        early_life_range=(60, 125),
        mid_life_range=(20, 60),
        late_life_range=(0, 20),
        coverage_deviation_threshold=0.15,
    )

    assert "EARLY_LIFE_FAIL" in labels


def test_select_top_k_overall():
    """Test top-K selection overall."""
    cases = [
        FailureCaseRecord(
            unit_id=i,
            condition_id=0,
            rmse_last=float(i),
            mae_last=float(i),
            bias_last=float(i),
            r2_last=0.8,
            nasa_last_mean=float(i * 10),
            nasa_last_sum=float(i * 10),
            true_last_rul=100.0,
            pred_last_rul=100.0,
            abs_err_last=float(i),
        )
        for i in range(1, 11)
    ]

    top_k_overall, _ = _select_top_k_cases(cases, K=3, ranking_metric="nasa_last_sum")

    # Should select units with highest NASA scores (10, 9, 8)
    assert len(top_k_overall) == 3
    assert set(top_k_overall) == {10, 9, 8}


def test_select_top_k_per_condition():
    """Test top-K selection per condition."""
    cases = []
    for cond in [0, 1]:
        for i in range(1, 6):
            cases.append(
                FailureCaseRecord(
                    unit_id=cond * 10 + i,
                    condition_id=cond,
                    rmse_last=float(i),
                    mae_last=float(i),
                    bias_last=float(i),
                    r2_last=0.8,
                    nasa_last_mean=float(i * 10),
                    nasa_last_sum=float(i * 10),
                    true_last_rul=100.0,
                    pred_last_rul=100.0,
                    abs_err_last=float(i),
                )
            )

    _, top_k_per_cond = _select_top_k_cases(cases, K=2, ranking_metric="nasa_last_sum")

    assert len(top_k_per_cond) == 2
    assert len(top_k_per_cond[0]) == 2
    assert len(top_k_per_cond[1]) == 2

    # Condition 0: should select 5, 4 (highest NASA)
    assert set(top_k_per_cond[0]) == {5, 4}
    # Condition 1: should select 15, 14
    assert set(top_k_per_cond[1]) == {15, 14}


def test_build_library_with_fake_data():
    """Test library building with minimal fake data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "FD004" / "test_exp"
        exp_dir.mkdir(parents=True)

        # Create minimal summary.json
        summary = {
            "encoder_type": "world_model_universal_v3",
            "world_model_config": {"decoder_type": "tf_cross"},
            "dataset": "FD004",
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        # Create fake eol_metrics.json
        eol_metrics = {
            "errors": [10.0, -20.0, 5.0],
            "y_true_eol": [100.0, 80.0, 50.0],
            "y_pred_eol": [110.0, 60.0, 55.0],
            "nasa_scores": [15.0, 30.0, 8.0],
        }
        with open(exp_dir / "eol_metrics.json", "w") as f:
            json.dump(eol_metrics, f)

        # Build library
        library = build_failure_case_library(exp_dir, K=2)

        # Assertions
        assert library.num_test_engines == 3
        assert len(library.top_k_overall) == 2
        assert len(library.all_cases) == 3
        assert library.experiment_name == "test_exp"
        assert library.dataset == "FD004"

        # Top-2 worst by NASA should be units 2 and 1 (NASA scores 30, 15)
        assert set(library.top_k_overall) == {2, 1}


def test_save_library():
    """Test saving failure case library to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "FD004" / "test_exp"
        exp_dir.mkdir(parents=True)

        # Create minimal data
        summary = {
            "encoder_type": "world_model_universal_v3",
            "world_model_config": {"decoder_type": "lstm"},
            "dataset": "FD004",
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        eol_metrics = {
            "errors": [10.0, -5.0],
            "y_true_eol": [100.0, 80.0],
            "y_pred_eol": [110.0, 75.0],
            "nasa_scores": [15.0, 8.0],
        }
        with open(exp_dir / "eol_metrics.json", "w") as f:
            json.dump(eol_metrics, f)

        # Build and save
        library = build_failure_case_library(exp_dir, K=1)
        save_failure_case_library(library, exp_dir, save_plots=False)

        # Check files exist
        failure_dir = exp_dir / "failure_cases"
        assert failure_dir.exists()
        assert (failure_dir / "summary.json").exists()
        assert (failure_dir / "cases.csv").exists()

        # Check top-K case files
        assert (failure_dir / "case_1.json").exists()

        # Check summary content
        with open(failure_dir / "summary.json") as f:
            saved_summary = json.load(f)

        assert saved_summary["num_test_engines"] == 2
        assert saved_summary["ranking_metric"] == "nasa_last_sum"
        assert len(saved_summary["top_k_overall"]) == 1

        # Check cases.csv
        df = pd.read_csv(failure_dir / "cases.csv")
        assert len(df) == 2
        assert "unit_id" in df.columns
        assert "bias_last" in df.columns
        assert "labels" in df.columns


def test_deterministic_ranking():
    """Test that top-K selection is deterministic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "FD004" / "test_exp"
        exp_dir.mkdir(parents=True)

        summary = {
            "encoder_type": "world_model_universal_v3",
            "world_model_config": {"decoder_type": "lstm"},
            "dataset": "FD004",
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        eol_metrics = {
            "errors": [10.0, -20.0, 5.0, 15.0],
            "y_true_eol": [100.0, 80.0, 50.0, 120.0],
            "y_pred_eol": [110.0, 60.0, 55.0, 135.0],
            "nasa_scores": [15.0, 30.0, 8.0, 22.0],
        }
        with open(exp_dir / "eol_metrics.json", "w") as f:
            json.dump(eol_metrics, f)

        # Build twice
        lib1 = build_failure_case_library(exp_dir, K=2)
        lib2 = build_failure_case_library(exp_dir, K=2)

        # Should be identical
        assert lib1.top_k_overall == lib2.top_k_overall
        assert len(lib1.all_cases) == len(lib2.all_cases)

        # Top-2 should be units 2 and 4 (NASA scores 30 and 22)
        assert set(lib1.top_k_overall) == {2, 4}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
