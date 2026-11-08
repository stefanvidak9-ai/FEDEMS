import math

import pytest

from src.baselines.evaluate import evaluate_forecasts, read_daily_csv


def test_evaluates_buildings_and_pooled_validation_and_test_metrics():
    rows = [
        {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": "2", "lag1_kwh": "1", "lag7_kwh": "0", "split": "validation"},
        {"building_id": "A", "target_date": "2024-01-02", "actual_kwh": "0", "lag1_kwh": "0", "lag7_kwh": "0", "split": "validation"},
        {"building_id": "B", "target_date": "2024-01-01", "actual_kwh": "4", "lag1_kwh": "5", "lag7_kwh": "0", "split": "validation"},
        {"building_id": "A", "target_date": "2024-02-01", "actual_kwh": "3", "lag1_kwh": "2", "lag7_kwh": "3", "split": "test"},
    ]

    results = evaluate_forecasts(rows)
    persistence = next(row for row in results if row["split"] == "validation" and row["baseline"] == "persistence" and row["scope"] == "aggregate")
    seasonal = next(row for row in results if row["split"] == "validation" and row["baseline"] == "weekly_seasonal_naive" and row["scope"] == "aggregate")
    building_a = next(row for row in results if row["split"] == "validation" and row["baseline"] == "persistence" and row["building_id"] == "A")

    assert persistence["n"] == 3
    assert persistence["mae_kwh"] == pytest.approx(2 / 3)
    assert persistence["rmse_kwh"] == pytest.approx(math.sqrt(2 / 3))
    assert persistence["r2"] == pytest.approx(0.75)
    assert seasonal["mae_kwh"] == pytest.approx(2)
    assert seasonal["rmse_kwh"] == pytest.approx(math.sqrt(20 / 3))
    assert building_a["n"] == 2
    assert building_a["mae_kwh"] == pytest.approx(0.5)
    assert any(row["split"] == "test" for row in results)


def test_missing_targets_and_either_lag_are_skipped_for_both_baselines():
    rows = [
        {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": "0", "lag1_kwh": "0", "lag7_kwh": "0", "split": "test"},
        {"building_id": "A", "target_date": "2024-01-02", "actual_kwh": "5", "lag1_kwh": "", "lag7_kwh": "3", "split": "test"},
        {"building_id": "A", "target_date": "2024-01-03", "actual_kwh": "", "lag1_kwh": "4", "lag7_kwh": "4", "split": "test"},
        {"building_id": "A", "target_date": "2024-01-04", "actual_kwh": "5", "lag1_kwh": "4", "lag7_kwh": "3", "split": "test"},
    ]

    results = evaluate_forecasts(rows)
    persistence = next(row for row in results if row["baseline"] == "persistence" and row["scope"] == "aggregate")
    seasonal = next(row for row in results if row["baseline"] == "weekly_seasonal_naive" and row["scope"] == "aggregate")

    assert persistence["n"] == 2
    assert persistence["mae_kwh"] == 0.5
    assert seasonal["n"] == persistence["n"]
    assert seasonal["mae_kwh"] == 1
    assert seasonal["r2"] == pytest.approx(0.68)


def test_constant_target_r2_is_not_reported():
    results = evaluate_forecasts(
        [
            {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": "0", "lag1_kwh": "0", "lag7_kwh": "0", "split": "test"},
            {"building_id": "A", "target_date": "2024-01-02", "actual_kwh": "0", "lag1_kwh": "1", "lag7_kwh": "1", "split": "test"},
        ]
    )
    aggregate = next(row for row in results if row["baseline"] == "persistence" and row["scope"] == "aggregate")
    assert aggregate["r2"] is None


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="invalid split"):
        evaluate_forecasts(
            [{"building_id": "A", "target_date": "2024-01-01", "actual_kwh": "1", "lag1_kwh": "1", "lag7_kwh": "1", "split": "train"}]
        )


def test_duplicate_building_target_date_is_rejected():
    row = {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": "1", "lag1_kwh": "1", "lag7_kwh": "1", "split": "test"}
    with pytest.raises(ValueError, match="Duplicate forecast target"):
        evaluate_forecasts([row, row])


def test_example_csv_is_explicitly_marked_synthetic():
    rows = read_daily_csv("tests/data/synthetic_example.csv")
    assert rows
    assert {row["data_status"] for row in rows} == {"synthetic_example"}