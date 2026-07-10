import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, HMM_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import hmm_fit
import config
import run_shape_hmm_uncertainty_experiments as exp


class UncertaintyExperimentTests(unittest.TestCase):
    def test_experiment_row_contains_required_metric_columns(self):
        regimes = _regime_frame(np.array([0, 0, 1, 1]), [0.99, 0.80, 0.82, 0.98])

        row = exp.build_result_row(
            stage="unit",
            covariance_type="diag",
            best_k=2,
            min_offdiag_prob=1e-4,
            covariance_scale=2.0,
            transition_blend=0.0,
            posterior_temperature=1.0,
            regimes=regimes,
            base_states=np.array([0, 0, 1, 1]),
            behavior_features=np.array([[0.0], [0.1], [1.0], [1.1]]),
            transition_window=1,
        )

        for col in exp.RESULT_COLUMNS:
            self.assertIn(col, row)
        self.assertEqual(row["posterior_temperature"], 1.0)
        self.assertEqual(row["covariance_scale"], 2.0)
        self.assertLessEqual(row["worst_transition_min_max_prob"], 0.90)
        self.assertIn("behavioral_viterbi_churn_pct", row)

    def test_covariance_scale_sweep_uses_configured_temperature_default(self):
        rows = exp.covariance_scale_rows(
            regimes_by_scale={
                1.0: _regime_frame(np.array([0, 0]), [0.99, 0.98]),
                2.0: _regime_frame(np.array([0, 0]), [0.95, 0.90]),
            },
            base_states=np.array([0, 0]),
            behavior_features=np.array([[0.0], [0.1]]),
            covariance_type="diag",
            best_k=2,
            min_offdiag_prob=1e-4,
            transition_window=1,
        )

        self.assertEqual([r["covariance_scale"] for r in rows], [1.0, 2.0])
        self.assertEqual({r["transition_blend"] for r in rows}, {0.0})
        self.assertEqual(
            {r["posterior_temperature"] for r in rows},
            {config.SHAPE_POSTERIOR_TEMPERATURE},
        )

    def test_empty_stage_row_uses_configured_temperature_default(self):
        row = exp.empty_stage_row("baseline")

        self.assertEqual(
            row["posterior_temperature"],
            config.SHAPE_POSTERIOR_TEMPERATURE,
        )

    def test_experiment_churn_threshold_is_one_third_of_days(self):
        self.assertEqual(config.MAX_ALLOWED_VITERBI_CHURN_PCT, 33.0)

    def test_covariance_type_sweep_records_all_requested_types(self):
        rows = [
            exp.empty_stage_row("covariance_type", covariance_type=t)
            for t in ["diag", "tied", "full"]
        ]

        self.assertEqual([r["covariance_type"] for r in rows],
                         ["diag", "tied", "full"])

    def test_transition_blend_preserves_transition_row_sums(self):
        matrix = np.array([[0.99, 0.01], [0.02, 0.98]])

        blended = hmm_fit.blend_transition_matrix(matrix, 0.10)

        np.testing.assert_allclose(blended.sum(axis=1), np.ones(2))
        self.assertTrue((blended >= 0).all())

    def test_best_row_prefers_more_uncertainty_with_acceptable_churn(self):
        rows = [
            {
                "meets_targets": False,
                "worst_transition_min_max_prob": 0.70,
                "pct_days_max_prob_gt_99": 50.0,
                "n_genuinely_uncertain_days": 100,
                "viterbi_churn_pct": 96.0,
                "behavioral_viterbi_churn_pct": 96.0,
                "covariance_scale": 10.0,
                "transition_blend": 0.0,
            },
            {
                "meets_targets": True,
                "worst_transition_min_max_prob": 0.85,
                "pct_days_max_prob_gt_99": 90.0,
                "n_genuinely_uncertain_days": 20,
                "viterbi_churn_pct": 0.8,
                "behavioral_viterbi_churn_pct": 0.8,
                "covariance_scale": 1.0,
                "transition_blend": 0.0,
            },
            {
                "meets_targets": True,
                "worst_transition_min_max_prob": 0.80,
                "pct_days_max_prob_gt_99": 75.0,
                "n_genuinely_uncertain_days": 80,
                "viterbi_churn_pct": 5.0,
                "behavioral_viterbi_churn_pct": 5.0,
                "covariance_scale": 2.0,
                "transition_blend": 0.0,
            },
        ]

        selected = exp.select_best_row(rows)

        self.assertEqual(selected["viterbi_churn_pct"], 5.0)
        self.assertEqual(selected["pct_days_max_prob_gt_99"], 75.0)

    def test_best_row_uses_behavioral_churn_for_acceptability(self):
        rows = [
            {
                "meets_targets": True,
                "worst_transition_min_max_prob": 0.75,
                "pct_days_max_prob_gt_99": 30.0,
                "n_genuinely_uncertain_days": 120,
                "viterbi_churn_pct": 100.0,
                "behavioral_viterbi_churn_pct": 0.0,
                "covariance_scale": 4.0,
                "transition_blend": 0.0,
            },
            {
                "meets_targets": True,
                "worst_transition_min_max_prob": 0.85,
                "pct_days_max_prob_gt_99": 80.0,
                "n_genuinely_uncertain_days": 20,
                "viterbi_churn_pct": 0.0,
                "behavioral_viterbi_churn_pct": 0.0,
                "covariance_scale": 1.0,
                "transition_blend": 0.0,
            },
        ]

        selected = exp.select_best_row(rows)

        self.assertEqual(selected["covariance_scale"], 4.0)
        self.assertEqual(selected["viterbi_churn_pct"], 100.0)
        self.assertEqual(selected["behavioral_viterbi_churn_pct"], 0.0)


def _regime_frame(states, max_probs):
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
