"""Simple lag-only forecasts for next-day space-heating energy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any


def _optional_energy(value: Any) -> float | None:
    if value is None or value == "":
        return None
    energy = float(value)
    if not math.isfinite(energy):
        return None
    return energy


def persistence_forecast(lag1_kwh: float | None) -> float | None:
    """Forecast target-day heat using only the preceding day's actual heat."""
    return _optional_energy(lag1_kwh)


def weekly_seasonal_naive_forecast(lag7_kwh: float | None) -> float | None:
    """Forecast target-day heat using only the actual from seven days earlier."""
    return _optional_energy(lag7_kwh)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def derive_lags(
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add same-building lag1/lag7 values using exact prior calendar dates.

    Each input row must contain ``building_id``, ``target_date``, and
    ``actual_kwh``. Other fields, including ``split``, are preserved.
    """
    prepared: list[dict[str, Any]] = []
    values_by_building: dict[str, dict[date, float | None]] = {}

    for observation in observations:
        if not {"building_id", "target_date", "actual_kwh"}.issubset(observation):
            raise ValueError(
                "Each observation must contain building_id, target_date, and actual_kwh"
            )
        building_id = str(observation["building_id"])
        if not building_id:
            raise ValueError("building_id cannot be empty")
        target_date = _as_date(observation["target_date"])
        building_values = values_by_building.setdefault(building_id, {})
        if target_date in building_values:
            raise ValueError(f"Duplicate observation for building {building_id} on {target_date}")
        building_values[target_date] = _optional_energy(observation["actual_kwh"])
        prepared.append({**observation, "building_id": building_id, "target_date": target_date.isoformat()})

    for row in prepared:
        target_date = _as_date(row["target_date"])
        building_values = values_by_building[row["building_id"]]
        row["lag1_kwh"] = building_values.get(target_date - timedelta(days=1))
        row["lag7_kwh"] = building_values.get(target_date - timedelta(days=7))

    return prepared