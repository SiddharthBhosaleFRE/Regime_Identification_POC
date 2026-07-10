import os
import sys
import unittest
from unittest import mock

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import export_shape_features as exporter


class ExportShapeFeaturesTests(unittest.TestCase):
    def test_parse_args_defaults_to_root_shape_features_csv(self):
        args = exporter.parse_args([])

        self.assertEqual(args.data_path, "Target_Zeros.csv")
        self.assertEqual(args.output_csv, "shape_features.csv")
        self.assertFalse(args.allow_overwrite)

    def test_export_writes_expected_feature_columns(self):
        tmpdir = os.path.join(ROOT, "test_export_shape_features_tmp")
        output_csv = os.path.join(tmpdir, "shape_features.csv")
        os.makedirs(tmpdir, exist_ok=True)
        features = pd.DataFrame(
            {
                "level": [1.0],
                "term_slope": [2.0],
                "butterfly": [3.0],
                "level_abs_daily_move_90d_mean": [4.0],
            },
            index=pd.to_datetime(["2020-01-02"]),
        )
        try:
            with mock.patch.object(exporter.feat, "load_and_pivot") as load_mock:
                with mock.patch.object(exporter.feat, "build_shape_features") as build_mock:
                    load_mock.return_value = pd.DataFrame()
                    build_mock.return_value = (features, {})
                    args = exporter.parse_args([
                        "--data-path",
                        "Target_Zeros.csv",
                        "--output-csv",
                        output_csv,
                    ])

                    written = exporter.export_shape_features(args)

            exported = pd.read_csv(written)
        finally:
            if os.path.exists(output_csv):
                os.remove(output_csv)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(written, output_csv)
        self.assertEqual(
            exported.columns.tolist(),
            [
                "date",
                "level",
                "term_slope",
                "butterfly",
                "level_abs_daily_move_90d_mean",
            ],
        )

    def test_exporter_imports_v3_feature_builder(self):
        self.assertIn("sofr_hmm_v3", exporter.feat.__file__)

    def test_export_refuses_existing_csv_without_allow_overwrite(self):
        tmpdir = os.path.join(ROOT, "test_export_shape_features_tmp")
        output_csv = os.path.join(tmpdir, "shape_features.csv")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with open(output_csv, "w", encoding="utf-8") as handle:
                handle.write("already here\n")
            args = exporter.parse_args(["--output-csv", output_csv])

            with self.assertRaises(SystemExit):
                exporter.export_shape_features(args)
        finally:
            if os.path.exists(output_csv):
                os.remove(output_csv)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)


if __name__ == "__main__":
    unittest.main()
