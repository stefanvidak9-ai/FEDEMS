"""Evaluate simple baselines from a daily CSV with precomputed lag columns."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np

from .naive_forecasting import persistence_forecast, weekly_seasonal_naive_forecast

REQUIRED_COLUMNS = {
    "building_id",
    "target_date",
    "actual_kwh",
    "lag1_kwh",
    "lag7_kwh",
    "split",
}
BASELINES = (
    ("persistence", "lag1_kwh", persistence_forecast),
    ("weekly_seasonal_naive", "lag7_kwh", weekly_seasonal_naive_forecast),
)
RESULT_COLUMNS = (
    "split",
    "baseline",
    "scope",
    "building_id",
    "n",
    "mae_kwh",
    "rmse_kwh",
    "r2",
)


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    count = len(actual)
    if not count:
        return {"n": 0, "mae_kwh": None, "rmse_kwh": None, "r2": None}

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = predicted_values - actual_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    r2 = None
    total_sum_squares = float(np.sum(np.square(actual_values - np.mean(actual_values))))
    if count >= 2 and total_sum_squares > 0:
        residual_sum_squares = float(np.sum(np.square(errors)))
        r2 = 1.0 - residual_sum_squares / total_sum_squares
    return {"n": count, "mae_kwh": mae, "rmse_kwh": rmse, "r2": r2}


def evaluate_forecasts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-building and pooled metrics, grouped by split and baseline.

    A row is scored for both baselines only when its target and both required
    lags are present. The returned ``n`` reports their shared sample size.
    """
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_targets: set[tuple[str, date]] = set()
    for row_number, row in enumerate(rows, start=2):
        missing_columns = REQUIRED_COLUMNS.difference(row)
        if missing_columns:
            raise ValueError(
                f"Row {row_number} is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        split = str(row["split"]).strip().lower()
        if split not in {"validation", "test"}:
            raise ValueError(f"Row {row_number} has invalid split {row['split']!r}; expected validation or test")
        building_id = str(row["building_id"]).strip()
        if not building_id:
            raise ValueError(f"Row {row_number} has an empty building_id")
        target_date = date.fromisoformat(str(row["target_date"]).strip())
        target_key = (building_id, target_date)
        if target_key in seen_targets:
            raise ValueError(f"Duplicate forecast target for building {building_id} on {target_date}")
        seen_targets.add(target_key)
        rows_by_split[split].append(row)

    results: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_rows = rows_by_split.get(split, [])
        if not split_rows:
            continue
        building_ids = sorted({str(row["building_id"]) for row in split_rows})
        eligible_rows = []
        for row in split_rows:
            actual = _optional_number(row["actual_kwh"])
            lag1 = _optional_number(row["lag1_kwh"])
            lag7 = _optional_number(row["lag7_kwh"])
            if actual is not None and lag1 is not None and lag7 is not None:
                eligible_rows.append((row, actual, lag1, lag7))

        for baseline, lag_column, forecast in BASELINES:
            actual_by_building: dict[str, list[float]] = {building_id: [] for building_id in building_ids}
            predicted_by_building: dict[str, list[float]] = {building_id: [] for building_id in building_ids}
            lag_index = 2 if lag_column == "lag1_kwh" else 3
            for row, actual, lag1, lag7 in eligible_rows:
                lag_value = lag1 if lag_index == 2 else lag7
                prediction = forecast(lag_value)
                building_id = str(row["building_id"])
                actual_by_building[building_id].append(actual)
                predicted_by_building[building_id].append(prediction)

            pooled_actual: list[float] = []
            pooled_predicted: list[float] = []
            for building_id in building_ids:
                building_actual = actual_by_building[building_id]
                building_predicted = predicted_by_building[building_id]
                pooled_actual.extend(building_actual)
                pooled_predicted.extend(building_predicted)
                results.append(
                    {
                        "split": split,
                        "baseline": baseline,
                        "scope": "building",
                        "building_id": building_id,
                        **_metrics(building_actual, building_predicted),
                    }
                )
            results.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "scope": "aggregate",
                    "building_id": "",
                    **_metrics(pooled_actual, pooled_predicted),
                }
            )
    return results


def read_daily_csv(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the header of a daily forecast CSV."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty or has no header")
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")
        return list(reader)


def _write_results(results: list[dict[str, Any]], output: TextIO) -> None:
    writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
    writer.writeheader()
    writer.writerows(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Daily CSV with actuals, lag values, and split")
    args = parser.parse_args(argv)

    try:
        rows = read_daily_csv(args.csv_path)
        if any(row.get("data_status", "").strip().lower() == "synthetic_example" for row in rows):
            print("SYNTHETIC EXAMPLE ONLY - NOT RESEARCH RESULTS", file=sys.stderr)
        _write_results(evaluate_forecasts(rows), sys.stdout)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())