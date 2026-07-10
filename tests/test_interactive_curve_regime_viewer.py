import os
import sys
import unittest

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import render_interactive_curve_regime_viewer as viewer


class InteractiveCurveRegimeViewerTests(unittest.TestCase):
    def test_parse_args_exposes_viewer_inputs(self):
        args = viewer.parse_args([
            "--curve-data", "Target_Zeros.csv",
            "--regime-assignments", "outputs/run/regime_assignments.csv",
            "--observed-rates", "Feature_Set.csv",
            "--shape-features", "shape_features.csv",
            "--output-html", "outputs/run/interactive_curve_regime_viewer.html",
            "--title", "My Viewer",
            "--allow-overwrite",
        ])

        self.assertEqual(args.curve_data, "Target_Zeros.csv")
        self.assertEqual(
            args.regime_assignments,
            "outputs/run/regime_assignments.csv",
        )
        self.assertEqual(args.observed_rates, "Feature_Set.csv")
        self.assertEqual(args.shape_features, "shape_features.csv")
        self.assertEqual(
            args.output_html,
            "outputs/run/interactive_curve_regime_viewer.html",
        )
        self.assertEqual(args.title, "My Viewer")
        self.assertTrue(args.allow_overwrite)

    def test_default_output_html_uses_assignment_folder(self):
        output = viewer.default_output_html(
            os.path.join("outputs", "run", "regime_assignments.csv")
        )

        self.assertEqual(
            output,
            os.path.join("outputs", "run", "interactive_curve_regime_viewer.html"),
        )

    def test_title_appends_experiment_folder(self):
        regime_assignments = os.path.join(
            "outputs",
            "one_off_experiments",
            "v3",
            "restarts_10_transmat_prior_10_covariance_scale_5",
            "regime_assignments.csv",
        )

        title = viewer.title_with_experiment_label(
            "SOFR yield curve with v3 shape HMM regimes",
            regime_assignments,
        )

        self.assertEqual(
            title,
            (
                "SOFR yield curve with v3 shape HMM regimes "
                "(restarts_10_transmat_prior_10_covariance_scale_5)"
            ),
        )

    def test_title_does_not_append_experiment_label_twice(self):
        regime_assignments = os.path.join(
            "outputs",
            "restarts_10_transmat_prior_10_covariance_scale_5",
            "regime_assignments.csv",
        )
        title = (
            "SOFR yield curve with v3 shape HMM regimes "
            "(restarts_10_transmat_prior_10_covariance_scale_5)"
        )

        self.assertEqual(
            viewer.title_with_experiment_label(title, regime_assignments),
            title,
        )

    def test_align_curve_and_regime_dates_inner_joins_and_sorts(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0, 3.1, 3.2],
                "d16": [3.3, 3.4, 3.5],
            },
            index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [2, 1],
                "filtered_p2": [0.7, 0.8],
                "filtered_p3": [0.3, 0.2],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-01"]),
        )

        aligned_curves, aligned_regimes = viewer.align_curve_and_regimes(
            curves, regimes
        )

        self.assertEqual(
            list(aligned_curves.index.strftime("%Y-%m-%d")),
            ["2020-01-01", "2020-01-02"],
        )
        self.assertEqual(aligned_regimes["viterbi_state"].tolist(), [1, 2])

    def test_load_observed_rates_pivots_feature_set_terms(self):
        tmpdir = os.path.join(ROOT, "test_interactive_curve_regime_viewer_tmp")
        input_csv = os.path.join(tmpdir, "Feature_Set.csv")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            pd.DataFrame(
                {
                    "MV1_DATE": [
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-03",
                    ],
                    "TERM": ["ON", "1W", "ON", "1W"],
                    "Mid_Market_Rate": [1.0, 1.1, 1.2, 1.3],
                }
            ).to_csv(input_csv, index=False)

            observed = viewer.load_observed_rates(input_csv)
        finally:
            if os.path.exists(input_csv):
                os.remove(input_csv)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(list(observed.columns), ["ON", "1W"])
        self.assertEqual(
            list(observed.index.strftime("%Y-%m-%d")),
            ["2020-01-02", "2020-01-03"],
        )
        self.assertEqual(observed.loc[pd.Timestamp("2020-01-03"), "1W"], 1.3)

    def test_load_shape_features_keeps_expected_feature_columns(self):
        tmpdir = os.path.join(ROOT, "test_interactive_curve_regime_viewer_tmp")
        input_csv = os.path.join(tmpdir, "shape_features.csv")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            pd.DataFrame(
                {
                    "date": ["2020-01-02"],
                    "level": [1.0],
                    "term_slope": [3.0],
                    "butterfly": [4.0],
                    "level_abs_daily_move_90d_mean": [6.0],
                }
            ).to_csv(input_csv, index=False)

            shape_features = viewer.load_shape_features(input_csv)
        finally:
            if os.path.exists(input_csv):
                os.remove(input_csv)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(shape_features.columns.tolist(), viewer.SHAPE_FEATURE_COLUMNS)
        self.assertEqual(shape_features.index[0], pd.Timestamp("2020-01-02"))

    def test_load_shape_features_accepts_variant_feature_columns(self):
        tmpdir = os.path.join(ROOT, "test_interactive_curve_regime_viewer_tmp")
        input_csv = os.path.join(tmpdir, "shape_features.csv")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            pd.DataFrame(
                {
                    "date": ["2020-01-02"],
                    "level": [1.0],
                    "slope_2y10y": [2.0],
                    "butterfly": [3.0],
                    "level_abs_daily_move_90d_mean": [4.0],
                }
            ).to_csv(input_csv, index=False)

            shape_features = viewer.load_shape_features(input_csv)
        finally:
            if os.path.exists(input_csv):
                os.remove(input_csv)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(
            shape_features.columns.tolist(),
            [
                "level",
                "slope_2y10y",
                "butterfly",
                "level_abs_daily_move_90d_mean",
            ],
        )

    def test_downsample_observed_rates_preserves_pattern_points(self):
        observed_rates = pd.DataFrame(
            {
                "ON": range(10),
                "1W": range(10, 20),
            },
            index=pd.date_range("2020-01-01", periods=10),
        )

        sampled = viewer.downsample_observed_rates(observed_rates, max_points=4)

        self.assertEqual(len(sampled), 4)
        self.assertEqual(sampled.index[0], observed_rates.index[0])
        self.assertEqual(sampled.index[-1], observed_rates.index[-1])

    def test_weekly_play_frame_names_step_forward_by_calendar_week(self):
        dates = pd.DatetimeIndex([
            "2020-01-01",
            "2020-01-02",
            "2020-01-06",
            "2020-01-08",
            "2020-01-10",
            "2020-01-15",
            "2020-01-16",
        ])

        frame_names = viewer.weekly_play_frame_names(dates)

        self.assertEqual(
            frame_names,
            ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-16"],
        )

    def test_slider_and_frames_use_weekly_dates(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0 + idx for idx in range(16)],
                "d16": [3.3 + idx for idx in range(16)],
            },
            index=pd.date_range("2020-01-01", periods=16),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [idx % 2 for idx in range(16)],
                "filtered_p1": [0.8 if idx % 2 == 0 else 0.2 for idx in range(16)],
                "filtered_p2": [0.2 if idx % 2 == 0 else 0.8 for idx in range(16)],
            },
            index=curves.index,
        )

        fig = viewer.build_figure(
            curves,
            regimes,
            title="SOFR Test Viewer",
            observed_rates=None,
        )
        play_button = fig.layout.updatemenus[0].buttons[0]
        slider_labels = [
            step.label for step in fig.layout.sliders[0].steps
        ]

        self.assertEqual(
            slider_labels,
            ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-16"],
        )
        self.assertEqual(
            [frame.name for frame in fig.frames],
            slider_labels,
        )
        self.assertIsNone(play_button.args[0])
        self.assertTrue(play_button.args[1]["fromcurrent"])

    def test_write_html_contains_expected_date_and_regime_text(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0, 3.1],
                "d16": [3.3, 3.4],
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [0, 1],
                "filtered_p1": [0.95, 0.2],
                "filtered_p2": [0.05, 0.8],
            },
            index=curves.index,
        )

        tmpdir = os.path.join(ROOT, "test_interactive_curve_regime_viewer_tmp")
        output_html = os.path.join(tmpdir, "viewer.html")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            viewer.write_viewer_html(
                curves,
                regimes,
                output_html,
                title="SOFR Test Viewer",
                observed_rates=None,
            )

            with open(output_html, encoding="utf-8") as handle:
                html = handle.read()
        finally:
            if os.path.exists(output_html):
                os.remove(output_html)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertIn("SOFR Test Viewer", html)
        self.assertIn("2020-01-02 | Regime S2", html)
        self.assertIn("Max filtered probability: 80.0%", html)

    def test_figure_uses_fixed_y_axis_and_roomier_layout(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0, 3.1],
                "d16": [3.3, 3.4],
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [0, 1],
                "filtered_p1": [0.95, 0.2],
                "filtered_p2": [0.05, 0.8],
            },
            index=curves.index,
        )

        fig = viewer.build_figure(
            curves,
            regimes,
            title="SOFR Test Viewer",
            observed_rates=None,
        )

        self.assertEqual(tuple(fig.layout.yaxis.range), (0, 6))
        self.assertGreaterEqual(fig.layout.height, 900)
        self.assertGreaterEqual(fig.layout.margin.t, 120)
        self.assertGreaterEqual(fig.layout.margin.b, 190)

    def test_figure_adds_filtered_probability_row_with_slider_cursor(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0 + idx for idx in range(16)],
                "d16": [3.3 + idx for idx in range(16)],
            },
            index=pd.date_range("2020-01-01", periods=16),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [idx % 2 for idx in range(16)],
                "filtered_p1": [0.8 if idx % 2 == 0 else 0.2 for idx in range(16)],
                "filtered_p2": [0.2 if idx % 2 == 0 else 0.8 for idx in range(16)],
            },
            index=curves.index,
        )

        fig = viewer.build_figure(
            curves,
            regimes,
            title="SOFR Test Viewer",
            observed_rates=None,
        )

        self.assertEqual(tuple(fig.layout.yaxis3.range), (-0.05, 1.05))
        self.assertFalse(fig.layout.xaxis3.showticklabels)
        self.assertEqual(tuple(fig.layout.xaxis3.range), (-0.5, 3.5))
        self.assertEqual(tuple(fig.layout.xaxis7.range), (-0.5, 3.5))
        self.assertEqual(fig.layout.xaxis3.domain, fig.layout.xaxis.domain)
        self.assertEqual(fig.layout.xaxis7.domain, fig.layout.xaxis.domain)
        self.assertEqual(fig.data[3].name, "Selected filtered date")
        self.assertIn("S1: 80.0%", fig.data[4].text[0])
        self.assertEqual(fig.data[4].y, (0.70,))
        self.assertEqual(fig.data[4].textposition, "middle right")
        self.assertEqual(fig.frames[1].traces, (0, 3, 4, 6))
        self.assertEqual(fig.frames[1].data[1].x, (1, 1))
        self.assertIn("S2: 80.0%", fig.frames[1].data[2].text[0])
        self.assertEqual(fig.frames[1].data[2].y, (0.70,))
        self.assertEqual(fig.frames[1].data[2].textposition, "middle right")

    def test_figure_adds_shape_feature_grid_with_slider_dots(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0 + idx for idx in range(16)],
                "d16": [3.3 + idx for idx in range(16)],
            },
            index=pd.date_range("2020-01-01", periods=16),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [idx % 2 for idx in range(16)],
                "filtered_p1": [0.8 if idx % 2 == 0 else 0.2 for idx in range(16)],
                "filtered_p2": [0.2 if idx % 2 == 0 else 0.8 for idx in range(16)],
            },
            index=curves.index,
        )
        shape_features = pd.DataFrame(
            {
                feature: [float(idx) for idx in range(16)]
                for feature in viewer.SHAPE_FEATURE_COLUMNS
            },
            index=curves.index,
        )

        fig = viewer.build_figure(
            curves,
            regimes,
            title="SOFR Test Viewer",
            observed_rates=None,
            shape_features=shape_features,
        )

        self.assertEqual(
            [trace.name for trace in fig.data[7:11]],
            viewer.SHAPE_FEATURE_COLUMNS,
        )
        self.assertEqual(
            [trace.name for trace in fig.data[11:15]],
            [f"Selected {feature}" for feature in viewer.SHAPE_FEATURE_COLUMNS],
        )
        self.assertEqual(fig.frames[1].traces, tuple([0, 3, 4, 6] + list(range(11, 15))))
        self.assertEqual(fig.frames[1].data[4].y, (7.0,))

    def test_figure_adds_static_observed_rate_dropdown(self):
        curves = pd.DataFrame(
            {
                "d9": [3.0, 3.1],
                "d16": [3.3, 3.4],
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        regimes = pd.DataFrame(
            {
                "viterbi_state": [0, 1],
                "filtered_p1": [0.95, 0.2],
                "filtered_p2": [0.05, 0.8],
            },
            index=curves.index,
        )
        observed_rates = pd.DataFrame(
            {
                f"T{i}": [float(i) + day for day in range(300)]
                for i in range(1, 20)
            },
            index=pd.date_range("2020-01-01", periods=300),
        )

        fig = viewer.build_figure(
            curves,
            regimes,
            title="SOFR Test Viewer",
            observed_rates=observed_rates,
        )

        dropdown = fig.layout.updatemenus[1]
        play_button = fig.layout.updatemenus[0].buttons[0]

        self.assertEqual(len(dropdown.buttons), 19)
        self.assertEqual(dropdown.buttons[0].label, "T1")
        self.assertEqual(dropdown.buttons[-1].label, "T19")
        self.assertEqual(
            len(fig.layout.sliders[0].steps),
            len(viewer.weekly_play_frame_names(pd.DatetimeIndex(curves.index))),
        )
        self.assertIsNone(play_button.args[0])
        self.assertTrue(play_button.args[1]["fromcurrent"])
        self.assertLessEqual(
            len(fig.data[7].x),
            viewer.DEFAULT_OBSERVED_RATE_MAX_POINTS,
        )
        self.assertEqual(
            pd.Timestamp(fig.layout.xaxis2.range[0]),
            observed_rates.index.min(),
        )
        self.assertEqual(
            pd.Timestamp(fig.layout.xaxis2.range[1]),
            observed_rates.index.max(),
        )
        self.assertEqual(fig.layout.sliders[0].ticklen, 0)
        self.assertEqual(fig.layout.sliders[0].minorticklen, 0)
        self.assertEqual(fig.layout.sliders[0].tickcolor, "rgba(0,0,0,0)")
        self.assertEqual(
            fig.layout.sliders[0].steps[0].label,
            "2020-01-01",
        )
        self.assertEqual(fig.frames[0].traces, tuple([0, 3, 4, 6] + list(range(26, 45))))
        self.assertEqual(fig.frames[1].data[4].y, (2.0,))


if __name__ == "__main__":
    unittest.main()
