"""Person A: smoke tests for verified CNN-ready artifacts."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd

from src.data.create_windows import build_windows
from src.data.feature_engineering import FEATURES

ROOT = Path(__file__).resolve().parents[1]


class TestDataProcessing(unittest.TestCase):
    def test_all_17_buildings_and_splits(self):
        for bid in range(6479, 6496):
            for split, n in (("train.npz", 203), ("validation.npz", 59), ("test.npz", 45)):
                path = ROOT / "Dataset/processed" / f"building_{bid}" / split
                with self.subTest(building=bid, split=split), np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data["X"].shape, (n, 4, 7))
                    self.assertEqual(data["y"].shape, (n,))
                    self.assertTrue(np.isfinite(data["X"]).all())
                    self.assertTrue(np.isfinite(data["y"]).all())

    def test_scaler_and_date_splits(self):
        with (ROOT / "Dataset/processed/global_scaler.json").open() as f:
            scaler = json.load(f)
        self.assertEqual(scaler["features"], FEATURES)
        split = pd.read_csv(ROOT / "docs/data_reports/step21_temporal_split.csv", parse_dates=["date"])
        self.assertEqual(split["split"].value_counts().to_dict(), {"train": 203, "val": 59, "test": 45})
        self.assertTrue(split["date"].is_unique)
        self.assertLess(split.loc[split["split"] == "train", "date"].max(),
                        split.loc[split["split"] == "val", "date"].min())

    def test_window_order_and_future_target(self):
        dates = pd.date_range("2022-01-01", periods=9, freq="D")
        df = pd.DataFrame({"date": dates})
        for j, feature in enumerate(FEATURES):
            df[feature] = np.arange(9) + 100 * j
        x, y = build_windows(df, [dates[7]])
        self.assertEqual(x.shape, (1, 4, 7))
        np.testing.assert_allclose(x[0, 0], np.arange(7))
        self.assertEqual(y[0], 7)


if __name__ == "__main__":
    unittest.main()
