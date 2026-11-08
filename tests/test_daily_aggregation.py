"""Person A: tests for complete 24-hour daily aggregation rules."""
import unittest
import numpy as np

from src.data.feature_engineering import complete_wh_day_to_kwh, complete_mean_day, is_consecutive_days


class TestDailyAggregation(unittest.TestCase):
    def test_wh_to_kwh(self):
        self.assertAlmostEqual(complete_wh_day_to_kwh([1000.0] * 24), 24.0)

    def test_mean_temperature(self):
        self.assertAlmostEqual(complete_mean_day([10.0] * 24), 10.0)

    def test_missing_hour_invalidates_day(self):
        hourly = [1000.0] * 24
        hourly[3] = float("nan")
        self.assertTrue(np.isnan(complete_wh_day_to_kwh(hourly)))
        self.assertTrue(np.isnan(complete_wh_day_to_kwh([1000.0] * 23)))

    def test_consecutive_days(self):
        self.assertTrue(is_consecutive_days(["2022-01-01", "2022-01-02"]))
        self.assertFalse(is_consecutive_days(["2022-01-01", "2022-01-03"]))


if __name__ == "__main__":
    unittest.main()
