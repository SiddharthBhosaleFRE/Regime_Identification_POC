import os
import sys
import unittest
import inspect

import numpy as np
from hmmlearn import hmm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
sys.path.insert(0, HMM_DIR)

import hmm_fit
import evaluate as ev
import config
import run as run_module


class DummyModel:
    def __init__(self):
        self.n_components = 2
        self.covariance_type = "diag"
        self.transmat_ = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.covars_ = np.array([[1.0, 4.0], [9.0, 16.0]])


class CalibrationTests(unittest.TestCase):
    def test_select_k_accepts_transition_floor_override(self):
        signature = inspect.signature(hmm_fit.select_k)

        self.assertIn("min_offdiag_prob", signature.parameters)

    def test_select_k_accepts_training_priors(self):
        signature = inspect.signature(hmm_fit.select_k)

        self.assertIn("transmat_prior", signature.parameters)
        self.assertIn("min_covar", signature.parameters)

    def test_blend_transition_matrix_moves_rows_toward_uniform(self):
        matrix = np.array([[0.98, 0.02], [0.10, 0.90]])

        blended = hmm_fit.blend_transition_matrix(matrix, 0.25)

        np.testing.assert_allclose(blended.sum(axis=1), np.ones(2))
        self.assertGreater(blended[0, 1], matrix[0, 1])
        self.assertLess(blended[0, 0], matrix[0, 0])
        self.assertTrue((blended >= 0).all())

    def test_calibrated_model_scales_diag_covariances_without_mutating_source(self):
        model = DummyModel()
        original_covars = model.covars_.copy()
        original_transmat = model.transmat_.copy()

        calibrated = hmm_fit.calibrated_model(model, covariance_scale=2.5,
                                              transition_blend=0.10)

        np.testing.assert_allclose(model.covars_, original_covars)
        np.testing.assert_allclose(model.transmat_, original_transmat)
        np.testing.assert_allclose(calibrated.covars_, original_covars * 2.5)
        np.testing.assert_allclose(calibrated.transmat_.sum(axis=1), np.ones(2))
        self.assertGreater(calibrated.transmat_[0, 1], original_transmat[0, 1])

    def test_calibrated_model_handles_hmmlearn_diag_covariance_shape(self):
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
        model.n_features = 2
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.95, 0.05], [0.10, 0.90]])
        model.means_ = np.array([[0.0, 0.0], [2.0, 2.0]])
        model.covars_ = np.array([[1.0, 4.0], [9.0, 16.0]])

        calibrated = hmm_fit.calibrated_model(model, covariance_scale=2.0)

        np.testing.assert_allclose(calibrated._covars_,
                                   np.array([[2.0, 8.0], [18.0, 32.0]]))

    def test_decode_supports_tied_covariance_model(self):
        import pandas as pd

        model = hmm.GaussianHMM(n_components=2, covariance_type="tied")
        model.n_features = 2
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.90, 0.10], [0.10, 0.90]])
        model.means_ = np.array([[0.0, 0.0], [3.0, 3.0]])
        model.covars_ = np.array([[1.0, 0.0], [0.0, 1.0]])
        X = np.array([[0.1, 0.1], [2.9, 2.9]])
        index = pd.date_range("2026-01-01", periods=2)

        decoded = hmm_fit.decode(model, X, index)

        self.assertEqual(len(decoded), 2)
        self.assertIn("filtered_p1", decoded.columns)
        self.assertIn("filtered_p2", decoded.columns)

    def test_calibrated_model_handles_hmmlearn_tied_covariance_shape(self):
        model = hmm.GaussianHMM(n_components=2, covariance_type="tied")
        model.n_features = 2
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.95, 0.05], [0.10, 0.90]])
        model.means_ = np.array([[0.0, 0.0], [2.0, 2.0]])
        model.covars_ = np.array([[1.0, 0.25], [0.25, 4.0]])

        calibrated = hmm_fit.calibrated_model(model, covariance_scale=2.0)

        np.testing.assert_allclose(
            calibrated._covars_,
            np.array([[2.0, 0.5], [0.5, 8.0]]),
        )

    def test_select_calibration_prefers_smallest_candidate_that_meets_targets(self):
        base_regimes = np.array([0, 0, 1, 1])
        candidates = [
            {
                "covariance_scale": 1.0,
                "transition_blend": 0.0,
                "regimes": _regime_frame(base_regimes, [0.99, 0.99, 0.99, 0.99]),
            },
            {
                "covariance_scale": 2.0,
                "transition_blend": 0.0,
                "regimes": _regime_frame(base_regimes, [0.98, 0.80, 0.82, 0.98]),
            },
            {
                "covariance_scale": 4.0,
                "transition_blend": 0.0,
                "regimes": _regime_frame(base_regimes, [0.70, 0.70, 0.70, 0.70]),
            },
        ]

        selected, summary = hmm_fit.select_calibration_candidate(
            candidates,
            base_states=base_regimes,
            transition_window=1,
            target_transition_max_prob=0.90,
            target_pct_days_max_prob_gt_99=95.0,
            max_allowed_churn_pct=10.0,
        )

        self.assertEqual(selected["covariance_scale"], 2.0)
        self.assertTrue(summary["meets_targets"])
        self.assertLessEqual(summary["worst_transition_min_max_prob"], 0.90)

    def test_select_calibration_uses_behavioral_churn_when_features_are_provided(self):
        base_states = np.array([0, 0, 1, 1])
        features = np.array([[0.0], [0.1], [10.0], [10.1]])
        candidates = [
            {
                "covariance_scale": 2.0,
                "transition_blend": 0.0,
                "regimes": _regime_frame(
                    np.array([1, 1, 0, 0]),
                    [0.80, 0.82, 0.83, 0.84],
                ),
            },
        ]

        selected, summary = hmm_fit.select_calibration_candidate(
            candidates,
            base_states=base_states,
            transition_window=1,
            target_transition_max_prob=0.90,
            target_pct_days_max_prob_gt_99=95.0,
            max_allowed_churn_pct=10.0,
            behavior_features=features,
        )

        self.assertIs(selected, candidates[0])
        self.assertTrue(summary["meets_targets"])
        self.assertEqual(summary["viterbi_churn_pct"], 100.0)
        self.assertEqual(summary["behavioral_viterbi_churn_pct"], 0.0)

    def test_apply_posterior_temperature_preserves_rows_and_state_order(self):
        regimes = _regime_frame(np.array([0, 1]), [0.99, 0.80])

        calibrated = hmm_fit.apply_posterior_temperature(regimes, temperature=2.0)

        np.testing.assert_allclose(
            calibrated[["filtered_p1", "filtered_p2"]].sum(axis=1),
            np.ones(2),
        )
        self.assertGreater(calibrated.iloc[0]["filtered_p1"],
                           calibrated.iloc[0]["filtered_p2"])
        self.assertLess(calibrated.iloc[0]["filtered_p1"], 0.99)
        self.assertEqual(calibrated.iloc[0]["viterbi_state"], 0)

    def test_calibration_metrics_reports_feature_aligned_behavioral_churn(self):
        base_states = np.array([0, 0, 1, 1])
        candidate_states = np.array([1, 1, 0, 0])
        features = np.array([
            [0.0, 0.0],
            [0.2, 0.1],
            [9.9, 10.0],
            [10.1, 9.8],
        ])
        regimes = _regime_frame(candidate_states, [0.99, 0.98, 0.97, 0.96])

        metrics = hmm_fit.calibration_metrics(
            regimes,
            base_states=base_states,
            transition_window=1,
            behavior_features=features,
        )

        self.assertEqual(metrics["viterbi_churn_pct"], 100.0)
        self.assertEqual(metrics["behavioral_viterbi_churn_pct"], 0.0)
        self.assertEqual(metrics["state_alignment"], "S1->S2; S2->S1")

    def test_shape_default_calibration_preserves_model_and_viterbi_states(self):
        model = DummyModel()
        regimes = _regime_frame(np.array([0, 0, 1, 1]), [0.99, 0.98, 0.97, 0.96])

        calibrated_model, calibrated_regimes, summary = (
            run_module._apply_shape_reporting_calibration(model, regimes)
        )

        self.assertIs(calibrated_model, model)
        self.assertEqual(config.SHAPE_POSTERIOR_TEMPERATURE, 2.0)
        np.testing.assert_array_equal(
            calibrated_regimes["viterbi_state"].to_numpy(),
            regimes["viterbi_state"].to_numpy(),
        )
        self.assertEqual(summary.iloc[0]["posterior_temperature"], 2.0)
        self.assertLess(
            calibrated_regimes[["filtered_p1", "filtered_p2"]].max(axis=1).iloc[0],
            regimes[["filtered_p1", "filtered_p2"]].max(axis=1).iloc[0],
        )

    def test_transition_window_diagnostics_reports_each_state_change(self):
        regimes = _regime_frame(
            np.array([0, 0, 1, 1, 0]),
            [0.99, 0.70, 0.82, 0.98, 0.75],
        )

        diagnostics = ev.transition_window_diagnostics(regimes, K=2, window=1)

        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(diagnostics.iloc[0]["from_state"], 1)
        self.assertEqual(diagnostics.iloc[0]["to_state"], 2)
        self.assertAlmostEqual(diagnostics.iloc[0]["min_max_filtered_prob"], 0.70)
        self.assertEqual(diagnostics.iloc[1]["from_state"], 2)
        self.assertEqual(diagnostics.iloc[1]["to_state"], 1)

    def test_regime_feature_moments_reports_mean_and_variance_by_state(self):
        import pandas as pd

        features = pd.DataFrame({
            "slope": [1.0, 3.0, 10.0, 14.0],
            "curve": [2.0, 4.0, 20.0, 24.0],
        })
        regimes = _regime_frame(np.array([0, 0, 1, 1]), [0.9, 0.8, 0.7, 0.6])

        moments = ev.regime_feature_moments(features, regimes, K=2)

        self.assertEqual(
            moments.columns.tolist(),
            ["state", "n_days", "feature", "mean", "variance"],
        )
        slope_state_1 = moments[
            (moments["state"] == 1) & (moments["feature"] == "slope")
        ].iloc[0]
        self.assertEqual(slope_state_1["n_days"], 2)
        self.assertEqual(slope_state_1["mean"], 2.0)
        self.assertEqual(slope_state_1["variance"], 1.0)


def _regime_frame(states, max_probs):
    import pandas as pd

    rows = []
    for state, max_prob in zip(states, max_probs):
        other = 1.0 - max_prob
        if state == 0:
            rows.append({"viterbi_state": state, "filtered_p1": max_prob,
                         "filtered_p2": other})
        else:
            rows.append({"viterbi_state": state, "filtered_p1": other,
                         "filtered_p2": max_prob})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
