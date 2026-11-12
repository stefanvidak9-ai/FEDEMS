"""Shared regression metrics for heating forecasts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def regression_metrics(actual_kwh: Any, predicted_kwh: Any) -> dict[str, float | int]:
    """Calculate pooled MAE and RMSE from original-scale kWh/day values."""
    actual = np.asarray(actual_kwh, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_kwh, dtype=np.float64).reshape(-1)
    if actual.size == 0:
        raise ValueError("Cannot calculate regression metrics for zero observations")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have matching shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite")

    errors = predicted - actual
    return {
        "n": int(actual.size),
        "mae_kwh": float(np.mean(np.abs(errors))),
        "rmse_kwh": float(math.sqrt(np.mean(np.square(errors)))),
    }
