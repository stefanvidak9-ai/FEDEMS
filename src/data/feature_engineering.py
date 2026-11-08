"""Feature definitions and strict 24-hour daily aggregation helpers."""
import numpy as np
import pandas as pd

FEATURES = [
    "HtSpace_kWh_day", "Tout_mean_C", "SolGlob_kWh_m2_day", "WindSpd_mean_m_s"
]
TARGET = "HtSpace_kWh_day"
WINDOW_DAYS = 7


def complete_wh_day_to_kwh(hourly_wh):
    """Sum Wh and convert to kWh iff all 24 hourly values are finite."""
    values = np.asarray(hourly_wh, dtype=np.float64)
    if values.shape != (24,) or not np.isfinite(values).all():
        return float("nan")
    return float(values.sum() / 1000.0)


def complete_mean_day(hourly_values):
    """Compute daily mean iff all 24 hourly values are finite."""
    values = np.asarray(hourly_values, dtype=np.float64)
    if values.shape != (24,) or not np.isfinite(values).all():
        return float("nan")
    return float(values.mean())


def is_consecutive_days(dates):
    index = pd.DatetimeIndex(dates)
    if len(index) < 2:
        return True
    return bool((index[1:] - index[:-1] == pd.Timedelta(days=1)).all())
