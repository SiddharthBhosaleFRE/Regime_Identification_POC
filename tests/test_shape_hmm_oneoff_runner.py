import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import run_shape_hmm_oneoff as runner


class ShapeHmmOneoffRunnerTests(unittest.TestCase):
    def test_parse_args_exposes_key_hyperparameters(self):
        args = runner.parse_args([
            "--n-em-restarts", "10",
            "--transmat-prior", "5.0",
            "--covariance-scale", "10.0",
            "--transition-blend", "0.05",
            "--posterior-temperature", "3.0",
            "--min-covar", "0.01",
            "--min-offdiag-prob", "0.001",
            "--feature-transform", "uniform_platykurtic",
            "--quantile-n-quantiles", "200",
            "--quantile-random-seed", "123",
        ])

        self.assertEqual(args.n_em_restarts, 10)
        self.assertEqual(args.transmat_prior, 5.0)
        self.assertEqual(args.covariance_scale, 10.0)
        self.assertEqual(args.transition_blend, 0.05)
        self.assertEqual(args.posterior_temperature, 3.0)
        self.assertEqual(args.min_covar, 0.01)
        self.assertEqual(args.min_offdiag_prob, 0.001)
        self.assertEqual(args.feature_transform, "uniform_platykurtic")
        self.assertEqual(args.quantile_n_quantiles, 200)
        self.assertEqual(args.quantile_random_seed, 123)

    def test_parse_args_exposes_train_test_split_options(self):
        args = runner.parse_args([
            "--test-size-pct", "25.0",
            "--split-date", "2025-01-01",
        ])

        self.assertEqual(args.test_size_pct, 25.0)
        self.assertEqual(args.split_date, "2025-01-01")

    def test_no_train_test_split_is_default(self):
        args = runner.parse_args([])

        self.assertEqual(args.test_size_pct, 0.0)

    def test_default_out_dir_is_one_off_experiments_folder(self):
        args = runner.parse_args([])

        self.assertEqual(
            args.out_dir,
            os.path.join("outputs", "one_off_experiments"),
        )

    def test_platykurtic_default_output_root_uses_platykurtic_subfolder(self):
        args = runner.parse_args([
            "--feature-transform", "uniform_platykurtic",
        ])

        self.assertEqual(
            runner.output_root_dir(args),
            os.path.join("outputs", "one_off_experiments", "platykurtic"),
        )

    def test_standard_default_output_root_stays_one_off_experiments_folder(self):
        args = runner.parse_args([])

        self.assertEqual(
            runner.output_root_dir(args),
            os.path.join("outputs", "one_off_experiments"),
        )

    def test_platykurtic_title_mentions_platykurtic(self):
        args = runner.parse_args([
            "--feature-transform", "uniform_platykurtic",
        ])

        self.assertIn("platykurtic", runner.experiment_title(args).lower())

    def test_default_output_folder_name_includes_main_hyperparameters(self):
        args = runner.parse_args([
            "--n-em-restarts", "10",
            "--transmat-prior", "10.0",
            "--covariance-scale", "5.0",
        ])

        folder = runner.default_output_folder(args)

        self.assertEqual(
            folder,
            "restarts_10_transmat_prior_10_covariance_scale_5",
        )

    def test_nonstandard_transform_is_included_in_default_output_folder(self):
        args = runner.parse_args([
            "--n-em-restarts", "10",
            "--transmat-prior", "20.0",
            "--covariance-scale", "5.0",
            "--feature-transform", "uniform_platykurtic",
        ])

        folder = runner.default_output_folder(args)

        self.assertEqual(
            folder,
            "restarts_10_transmat_prior_20_covariance_scale_5_uniform_platykurtic",
        )

    def test_custom_output_folder_overrides_default_name(self):
        args = runner.parse_args([
            "--output-folder", "my_shape_run",
            "--n-em-restarts", "3",
        ])

        self.assertEqual(runner.output_folder_name(args), "my_shape_run")


if __name__ == "__main__":
    unittest.main()
