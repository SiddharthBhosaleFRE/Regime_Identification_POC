import importlib
import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DIR = os.path.join(ROOT, "sofr_hmm_v3")


def load_v3_features_module():
    old_features = sys.modules.pop("features", None)
    old_config = sys.modules.pop("config", None)
    sys.path.insert(0, V3_DIR)
    try:
        module = importlib.import_module("features")
    finally:
        sys.path.remove(V3_DIR)
        sys.modules.pop("features", None)
        sys.modules.pop("config", None)
        if old_features is not None:
            sys.modules["features"] = old_features
        if old_config is not None:
            sys.modules["config"] = old_config
    return module


class V3ShapeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feat = load_v3_features_module()

    def test_shape_features_drop_change_pc1_and_slope_short(self):
        levels = self._sample_levels()

        features, meta = self.feat.build_shape_features(levels)

        self.assertEqual(
            features.columns.tolist(),
            [
                "level",
                "term_slope",
                "butterfly",
                "level_abs_daily_move_90d_mean",
            ],
        )
        self.assertNotIn("change_pc1", features.columns)
        self.assertNotIn("slope_short", features.columns)
        self.assertNotIn("level_vol", features.columns)
        self.assertNotIn("pca_change1", meta)
        self.assertNotIn("pca_level1", meta)
        self.assertEqual(meta["movement_window_days"], 90)
        self.assertEqual(
            meta["anchor_cols"],
            {"3m": "d94", "2y": "d732", "5y": "d1828", "10y": "d3654"},
        )
        self.assertEqual(meta["feature_set"], "v3")

    def test_shape_butterfly_uses_5y_between_3m_and_10y(self):
        levels = self._sample_levels()

        features, _ = self.feat.build_shape_features(levels)

        expected = levels["d1828"] - 0.5 * (levels["d94"] + levels["d3654"])
        np.testing.assert_allclose(
            features["butterfly"].to_numpy(),
            expected.loc[features.index].to_numpy(),
        )

    def test_shape_level_abs_daily_move_uses_90_day_mean(self):
        levels = self._sample_levels()

        features, _ = self.feat.build_shape_features(levels)

        expected = levels.mean(axis=1).diff().abs().rolling(
            90,
            min_periods=90,
        ).mean()
        np.testing.assert_allclose(
            features["level_abs_daily_move_90d_mean"].to_numpy(),
            expected.loc[features.index].to_numpy(),
        )

    def test_slope_2y10y_feature_set_replaces_term_slope(self):
        levels = self._sample_levels()

        features, meta = self.feat.build_shape_features(
            levels,
            feature_set="slope_2y10y",
        )

        self.assertEqual(
            features.columns.tolist(),
            [
                "level",
                "slope_2y10y",
                "butterfly",
                "level_abs_daily_move_90d_mean",
            ],
        )
        self.assertEqual(meta["feature_set"], "slope_2y10y")
        expected = levels["d3654"] - levels["d732"]
        np.testing.assert_allclose(
            features["slope_2y10y"].to_numpy(),
            expected.loc[features.index].to_numpy(),
        )

    def _sample_levels(self):
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        trend = np.linspace(0.0, 1.0, len(index))
        return pd.DataFrame(
            {
                "d94": 4.0 + 0.02 * trend,
                "d732": 4.2 + 0.05 * trend,
                "d1828": 4.35 + 0.01 * trend,
                "d3654": 4.5 - 0.03 * trend,
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
