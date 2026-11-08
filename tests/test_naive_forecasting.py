from datetime import date, timedelta

import pytest

from src.baselines.naive_forecasting import (
    derive_lags,
    persistence_forecast,
    weekly_seasonal_naive_forecast,
)


def test_forecast_rules_use_the_requested_lag_only():
    assert persistence_forecast(12.5) == 12.5
    assert weekly_seasonal_naive_forecast(8) == 8


def test_zero_demand_is_a_valid_forecast():
    assert persistence_forecast(0) == 0
    assert weekly_seasonal_naive_forecast(0) == 0


@pytest.mark.parametrize("forecast", [persistence_forecast, weekly_seasonal_naive_forecast])
def test_missing_lag_returns_no_forecast(forecast):
    assert forecast(None) is None
    assert forecast(float("nan")) is None


def test_derive_lags_aligns_calendar_dates_per_building_without_target_leakage():
    start = date(2024, 1, 1)
    observations = []
    for offset in range(9):
        observations.append(
            {
                "building_id": "A",
                "target_date": (start + timedelta(days=offset)).isoformat(),
                "actual_kwh": offset + 1,
            }
        )
    observations.extend(
        [
            {"building_id": "B", "target_date": start.isoformat(), "actual_kwh": 100},
            {"building_id": "B", "target_date": (start + timedelta(days=7)).isoformat(), "actual_kwh": 200},
        ]
    )

    derived = derive_lags(observations)
    target_a = next(row for row in derived if row["building_id"] == "A" and row["target_date"] == "2024-01-08")
    target_b = next(row for row in derived if row["building_id"] == "B" and row["target_date"] == "2024-01-08")

    assert target_a["lag1_kwh"] == 7
    assert target_a["lag7_kwh"] == 1
    assert target_a["actual_kwh"] == 8
    assert target_b["lag1_kwh"] is None
    assert target_b["lag7_kwh"] == 100


def test_derive_lags_keeps_missing_dates_and_missing_values_missing():
    derived = derive_lags(
        [
            {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": 0},
            {"building_id": "A", "target_date": "2024-01-02", "actual_kwh": ""},
            {"building_id": "A", "target_date": "2024-01-08", "actual_kwh": 4},
        ]
    )

    assert derived[1]["lag1_kwh"] == 0
    assert derived[2]["lag1_kwh"] is None
    assert derived[2]["lag7_kwh"] == 0


def test_duplicate_building_date_is_rejected():
    with pytest.raises(ValueError, match="Duplicate observation"):
        derive_lags(
            [
                {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": 1},
                {"building_id": "A", "target_date": "2024-01-01", "actual_kwh": 2},
            ]
        )