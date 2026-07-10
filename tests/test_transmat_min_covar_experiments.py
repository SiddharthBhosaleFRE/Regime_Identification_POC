import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, HMM_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import run_transmat_and_min_covar_experiments as exp


class TransmatAndMinCovarExperimentTests(unittest.TestCase):
    def test_experiment_configs_are_one_at_a_time_and_deduplicate_baseline(self):
        configs = exp.experiment_configs()

        self.assertEqual(len(configs), 10)
        self.assertEqual(configs[0]["stage"], "baseline")
        self.assertEqual(configs[0]["transmat_prior"], 1.0)
        self.assertEqual(configs[0]["min_covar"], 0.001)
        self.assertEqual(
            {(c["transmat_prior"], c["min_covar"]) for c in configs},
            {
                (1.0, 0.001),
                (2.0, 0.001),
                (2.5, 0.001),
                (3.0, 0.001),
                (4.0, 0.001),
                (5.0, 0.001),
                (10.0, 0.001),
                (1.0, 0.01),
                (1.0, 0.05),
                (1.0, 0.1),
            },
        )

    def test_experiment_folder_names_are_stable(self):
        configs = exp.experiment_configs()

        self.assertEqual(
            [c["folder"] for c in configs],
            [
                "baseline",
                "transmat_prior_2_0",
                "transmat_prior_2_5",
                "transmat_prior_3_0",
                "transmat_prior_4_0",
                "transmat_prior_5_0",
                "transmat_prior_10_0",
                "min_covar_0_01",
                "min_covar_0_05",
                "min_covar_0_1",
            ],
        )

    def test_result_row_records_training_parameters_and_metrics(self):
        row = exp.build_result_row(
            config_row={
                "stage": "transmat_prior",
                "folder": "transmat_prior_5_0",
                "transmat_prior": 5.0,
                "min_covar": 0.001,
            },
            covariance_type="diag",
            best_k=5,
            min_offdiag_prob=1e-4,
            posterior_temperature=2.0,
            metrics={
                "pct_days_max_prob_gt_99": 40.0,
                "n_genuinely_uncertain_days": 900,
                "mean_entropy": 0.2,
                "n_transitions": 8,
                "worst_transition_min_max_prob": 0.8,
                "best_transition_max_entropy": 0.5,
                "viterbi_churn_pct": 12.5,
                "behavioral_viterbi_churn_pct": 6.25,
                "state_alignment": "S1->S1",
            },
            log_likelihood=-123.0,
            bic=456.0,
        )

        for column in exp.RESULT_COLUMNS:
            self.assertIn(column, row)
        self.assertEqual(row["transmat_prior"], 5.0)
        self.assertEqual(row["min_covar"], 0.001)
        self.assertEqual(row["folder"], "transmat_prior_5_0")
        self.assertEqual(row["viterbi_churn_pct"], 12.5)
        self.assertEqual(row["behavioral_viterbi_churn_pct"], 6.25)


if __name__ == "__main__":
    unittest.main()
