import importlib
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")


def load_v3_runner_module():
    module_names = [
        "run_v3_hmm_oneoff",
        "config",
        "evaluate",
        "features",
        "hmm_fit",
    ]
    old_modules = {
        name: sys.modules.pop(name, None)
        for name in module_names
    }
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        module = importlib.import_module("run_v3_hmm_oneoff")
    finally:
        sys.path.remove(SCRIPTS_DIR)
        for name in module_names:
            sys.modules.pop(name, None)
            if old_modules[name] is not None:
                sys.modules[name] = old_modules[name]
    return module


class V3HmmOneoffRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_v3_runner_module()

    def test_runner_imports_v3_hmm_modules(self):
        self.assertTrue(self.runner.HMM_DIR.endswith("sofr_hmm_v3"))
        self.assertIn("sofr_hmm_v3", self.runner.feat.__file__)
        self.assertIn("sofr_hmm_v3", self.runner.config.__file__)

    def test_default_out_dir_is_v3_one_off_experiments_folder(self):
        args = self.runner.parse_args([])

        self.assertEqual(
            args.out_dir,
            os.path.join("outputs", "one_off_experiments", "v3"),
        )

    def test_platykurtic_default_output_root_uses_v3_platykurtic_subfolder(self):
        args = self.runner.parse_args([
            "--feature-transform",
            "uniform_platykurtic",
        ])

        self.assertEqual(
            self.runner.output_root_dir(args),
            os.path.join("outputs", "one_off_experiments", "v3", "platykurtic"),
        )

    def test_experiment_title_mentions_v3(self):
        args = self.runner.parse_args([])

        self.assertIn("v3", self.runner.experiment_title(args).lower())

    def test_default_output_folder_name_matches_existing_oneoff_pattern(self):
        args = self.runner.parse_args([
            "--n-em-restarts",
            "10",
            "--transmat-prior",
            "10.0",
            "--covariance-scale",
            "5.0",
        ])

        self.assertEqual(
            self.runner.default_output_folder(args),
            "restarts_10_transmat_prior_10_covariance_scale_5",
        )

    def test_parse_args_exposes_shape_feature_set(self):
        args = self.runner.parse_args([
            "--shape-feature-set",
            "slope_2y10y",
        ])

        self.assertEqual(args.shape_feature_set, "slope_2y10y")

    def test_variant_output_folder_includes_shape_feature_set(self):
        args = self.runner.parse_args([
            "--shape-feature-set",
            "slope_2y10y",
            "--n-em-restarts",
            "10",
            "--transmat-prior",
            "10.0",
            "--covariance-scale",
            "5.0",
        ])

        self.assertEqual(
            self.runner.default_output_folder(args),
            "slope_2y10y_restarts_10_transmat_prior_10_covariance_scale_5",
        )

    def test_variant_title_mentions_shape_feature_set(self):
        args = self.runner.parse_args([
            "--shape-feature-set",
            "slope_2y10y",
        ])

        self.assertIn("slope_2y10y", self.runner.experiment_title(args))


if __name__ == "__main__":
    unittest.main()
