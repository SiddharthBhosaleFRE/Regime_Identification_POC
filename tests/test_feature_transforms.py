import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
sys.path.insert(0, HMM_DIR)

import features as feat


class FeatureTransformTests(unittest.TestCase):
    def test_standard_transform_matches_zero_mean_unit_variance_scaling(self):
        features = pd.DataFrame({
            "level": [1.0, 2.0, 3.0, 4.0],
            "slope": [10.0, 20.0, 30.0, 40.0],
        })

        X, transformer, diagnostics = feat.transform_for_hmm(
            features, transform="standard"
        )

        self.assertEqual(X.shape, (4, 2))
        self.assertIsNotNone(transformer)
        np.testing.assert_allclose(X.mean(axis=0), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(X.std(axis=0), np.ones(2), atol=1e-12)
        self.assertEqual(set(diagnostics["transform"]), {"standard"})

    def test_standard_fit_apply_uses_train_statistics_for_test_data(self):
        train = pd.DataFrame({
            "level": [1.0, 2.0, 3.0, 4.0],
            "slope": [10.0, 20.0, 30.0, 40.0],
        })
        test = pd.DataFrame({
            "level": [5.0, 6.0],
            "slope": [50.0, 60.0],
        })

        transformer = feat.fit_hmm_transform(train, transform="standard")
        X_train = feat.apply_hmm_transform(train, transformer, transform="standard")
        X_test = feat.apply_hmm_transform(test, transformer, transform="standard")

        np.testing.assert_allclose(X_train.mean(axis=0), np.zeros(2), atol=1e-12)
        self.assertGreater(float(X_test.mean()), 0.0)

    def test_uniform_platykurtic_transform_is_bounded_and_light_tailed(self):
        features = pd.DataFrame({
            "level": np.linspace(-5.0, 5.0, 101),
            "slope": np.linspace(10.0, -10.0, 101),
        })

        X, transformer, diagnostics = feat.transform_for_hmm(
            features,
            transform="uniform_platykurtic",
            n_quantiles=101,
            random_seed=7,
        )

        self.assertEqual(X.shape, (101, 2))
        self.assertIsNotNone(transformer)
        self.assertLessEqual(float(X.max()), np.sqrt(3.0) + 1e-12)
        self.assertGreaterEqual(float(X.min()), -np.sqrt(3.0) - 1e-12)
        self.assertLess(kurtosis(X[:, 0], fisher=True, bias=True), 0.0)
        self.assertEqual(
            set(diagnostics["transform"]),
            {"uniform_platykurtic"},
        )

    def test_feature_distribution_diagnostics_reports_expected_columns(self):
        X = np.array([
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ])

        diagnostics = feat.feature_distribution_diagnostics(
            X,
            ["a", "b"],
            transform="standard",
        )

        self.assertEqual(
            diagnostics.columns.tolist(),
            [
                "transform",
                "feature",
                "mean",
                "std",
                "skewness",
                "excess_kurtosis",
                "min",
                "max",
            ],
        )
        self.assertEqual(diagnostics["feature"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
