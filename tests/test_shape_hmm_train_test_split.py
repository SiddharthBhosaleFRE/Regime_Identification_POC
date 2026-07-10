import os
import sys
import unittest

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import run_shape_hmm_oneoff as runner


class ShapeHmmTrainTestSplitTests(unittest.TestCase):
    def test_pct_split_uses_final_rows_as_test_set(self):
        features = _feature_frame(10)

        train, test, meta = runner.split_features(features, test_size_pct=20.0)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertEqual(train.index[-1], pd.Timestamp("2026-01-08"))
        self.assertEqual(test.index[0], pd.Timestamp("2026-01-09"))
        self.assertEqual(meta["split_mode"], "test_size_pct")

    def test_split_date_uses_strictly_prior_rows_for_training(self):
        features = _feature_frame(10)

        train, test, meta = runner.split_features(
            features,
            test_size_pct=20.0,
            split_date="2026-01-06",
        )

        self.assertEqual(train.index[-1], pd.Timestamp("2026-01-05"))
        self.assertEqual(test.index[0], pd.Timestamp("2026-01-06"))
        self.assertEqual(meta["split_mode"], "split_date")
        self.assertEqual(meta["split_date"], "2026-01-06")

    def test_zero_pct_returns_no_split(self):
        features = _feature_frame(10)

        train, test, meta = runner.split_features(features, test_size_pct=0.0)

        self.assertEqual(len(train), 10)
        self.assertIsNone(test)
        self.assertEqual(meta["split_mode"], "none")

    def test_split_rejects_invalid_pct_and_empty_windows(self):
        features = _feature_frame(10)

        with self.assertRaises(SystemExit):
            runner.split_features(features, test_size_pct=-1.0)
        with self.assertRaises(SystemExit):
            runner.split_features(features, test_size_pct=100.0)
        with self.assertRaises(SystemExit):
            runner.split_features(features, split_date="2026-01-01")
        with self.assertRaises(SystemExit):
            runner.split_features(features, split_date="2026-01-20")


def _feature_frame(n_rows):
    return pd.DataFrame(
        {
            "level": range(n_rows),
            "slope": range(10, 10 + n_rows),
        },
        index=pd.date_range("2026-01-01", periods=n_rows),
    )


if __name__ == "__main__":
    unittest.main()
