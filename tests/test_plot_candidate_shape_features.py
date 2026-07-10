import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import plot_candidate_shape_features as plotter


class PlotCandidateShapeFeaturesTests(unittest.TestCase):
    def test_parse_args_defaults_to_outputs_plot(self):
        args = plotter.parse_args([])

        self.assertEqual(args.data_path, "Target_Zeros.csv")
        self.assertEqual(
            args.output_png,
            os.path.join("outputs", "plots", "candidate_features.png"),
        )
        self.assertIsNone(args.output_csv)
        self.assertFalse(args.allow_overwrite)

    def test_build_candidate_features_uses_expected_formulas(self):
        levels = self._sample_levels()

        features = plotter.build_candidate_features(levels)

        expected_slope = levels["d3654"] - levels["d732"]
        expected_level_delta = levels.mean(axis=1).diff(5)
        expected_short_delta = (levels["d732"] - levels["d94"]).diff(5)
        self.assertEqual(
            features.columns.tolist(),
            [
                "slope_2y10y",
                "weekly_delta_level",
                "weekly_delta_slope_short",
            ],
        )
        np.testing.assert_allclose(
            features["slope_2y10y"].to_numpy(),
            expected_slope.loc[features.index].to_numpy(),
        )
        np.testing.assert_allclose(
            features["weekly_delta_level"].to_numpy(),
            expected_level_delta.loc[features.index].to_numpy(),
        )
        np.testing.assert_allclose(
            features["weekly_delta_slope_short"].to_numpy(),
            expected_short_delta.loc[features.index].to_numpy(),
        )

    def test_run_refuses_existing_plot_without_allow_overwrite(self):
        tmpdir = os.path.join(ROOT, "test_candidate_feature_plot_tmp")
        output_png = os.path.join(tmpdir, "candidate_features.png")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with open(output_png, "w", encoding="utf-8") as handle:
                handle.write("already here\n")
            args = plotter.parse_args(["--output-png", output_png])

            with self.assertRaises(SystemExit):
                plotter.run(args)
        finally:
            if os.path.exists(output_png):
                os.remove(output_png)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

    def test_run_can_write_plot_and_optional_csv(self):
        tmpdir = os.path.join(ROOT, "test_candidate_feature_plot_tmp")
        output_png = os.path.join(tmpdir, "candidate_features.png")
        output_csv = os.path.join(tmpdir, "candidate_features.csv")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with mock.patch.object(plotter.feat, "load_and_pivot") as load_mock:
                load_mock.return_value = self._sample_levels()
                args = plotter.parse_args([
                    "--output-png",
                    output_png,
                    "--output-csv",
                    output_csv,
                ])

                result = plotter.run(args)

            exported = pd.read_csv(output_csv)
        finally:
            for path in (output_png, output_csv):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(result["output_png"], output_png)
        self.assertEqual(result["output_csv"], output_csv)
        self.assertTrue(result["n_rows"] > 0)
        self.assertEqual(
            exported.columns.tolist(),
            [
                "date",
                "slope_2y10y",
                "weekly_delta_level",
                "weekly_delta_slope_short",
            ],
        )

    def _sample_levels(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "d94": np.linspace(4.0, 4.2, len(index)),
                "d732": np.linspace(4.3, 4.1, len(index)),
                "d3654": np.linspace(4.8, 4.5, len(index)),
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
