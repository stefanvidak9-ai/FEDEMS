"""Build seven-day feature windows for common next-day prediction dates."""
import numpy as np
import pandas as pd

from src.data.feature_engineering import FEATURES, TARGET, WINDOW_DAYS


def build_windows(daily: pd.DataFrame, target_dates):
    required = ["date", *FEATURES]
    absent = set(required) - set(daily.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any() or not df["date"].is_unique:
        raise ValueError("Daily dates must be unique and valid")
    df = df.set_index("date").sort_index()
    X, y = [], []
    for date in pd.to_datetime(target_dates):
        history_dates = pd.date_range(end=date-pd.Timedelta(days=1), periods=WINDOW_DAYS, freq="D")
        history = df.reindex(history_dates)[FEATURES].to_numpy(dtype=np.float64)
        if date not in df.index:
            raise ValueError(f"Missing target date: {date}")
        target = float(df.loc[date, TARGET])
        if not np.isfinite(history).all() or not np.isfinite(target):
            raise ValueError(f"Invalid seven-day sample for {date}")
        X.append(history.T)
        y.append(target)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
