import argparse
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v3")
if HMM_DIR in sys.path:
    sys.path.remove(HMM_DIR)
sys.path.insert(0, HMM_DIR)
for module_name in ("config", "features"):
    sys.modules.pop(module_name, None)

import config
import features as feat


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
DEFAULT_OBSERVED_RATE_MAX_POINTS = 250
SHAPE_FEATURE_COLUMNS = [
    "level",
    "term_slope",
    "butterfly",
    "level_abs_daily_move_90d_mean",
]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT, path))


def default_output_html(regime_assignments_path: str) -> str:
    return os.path.join(
        os.path.dirname(regime_assignments_path),
        "interactive_curve_regime_viewer.html",
    )


def experiment_label_from_regime_assignments(regime_assignments_path: str) -> str:
    folder = os.path.basename(os.path.dirname(regime_assignments_path))
    if folder.startswith("shape_"):
        return folder[len("shape_"):]
    return folder


def title_with_experiment_label(title: str, regime_assignments_path: str) -> str:
    label = experiment_label_from_regime_assignments(regime_assignments_path)
    if not label:
        return title
    suffix = f" ({label})"
    if title.endswith(suffix):
        return title
    return f"{title}{suffix}"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an interactive SOFR yield curve viewer annotated with "
            "HMM regime assignments."
        )
    )
    parser.add_argument("--curve-data", default=config.DATA_PATH)
    parser.add_argument("--regime-assignments", required=True)
    parser.add_argument("--observed-rates", default="Feature_Set.csv")
    parser.add_argument("--shape-features", default="shape_features.csv")
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--title", default="SOFR yield curve with HMM regimes")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def load_regime_assignments(path: str) -> pd.DataFrame:
    regimes = pd.read_csv(path)
    if "date" not in regimes.columns:
        first_col = regimes.columns[0]
        regimes = regimes.rename(columns={first_col: "date"})
    if "viterbi_state" not in regimes.columns:
        raise ValueError("Regime assignment CSV must include viterbi_state")

    regimes["date"] = pd.to_datetime(regimes["date"], format="mixed")
    return regimes.set_index("date").sort_index()


def load_observed_rates(path: str) -> pd.DataFrame:
    rates = pd.read_csv(path)
    rates.columns = rates.columns.str.strip().str.replace("\ufeff", "")
    required = {"MV1_DATE", "TERM", "Mid_Market_Rate"}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(
            "Observed rates CSV is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    rates["MV1_DATE"] = pd.to_datetime(rates["MV1_DATE"], format="mixed")
    terms = rates["TERM"].drop_duplicates().tolist()
    wide = rates.pivot_table(
        index="MV1_DATE",
        columns="TERM",
        values="Mid_Market_Rate",
        aggfunc="first",
    ).sort_index()
    return wide.reindex(columns=terms)


def load_shape_features(path: str) -> pd.DataFrame:
    shape_features = pd.read_csv(path)
    shape_features.columns = (
        shape_features.columns.str.strip().str.replace("\ufeff", "")
    )
    if "date" not in shape_features.columns:
        first_col = shape_features.columns[0]
        shape_features = shape_features.rename(columns={first_col: "date"})

    feature_columns = [
        col for col in shape_features.columns
        if col != "date"
    ]
    if not feature_columns:
        raise ValueError("Shape feature CSV must include at least one feature column")
    if len(feature_columns) > 4:
        raise ValueError("Interactive viewer supports at most four shape features")

    shape_features["date"] = pd.to_datetime(
        shape_features["date"],
        format="mixed",
    )
    return (
        shape_features
        .set_index("date")
        .sort_index()
        .reindex(columns=feature_columns)
    )


def downsample_observed_rates(
    observed_rates: pd.DataFrame,
    max_points: int = DEFAULT_OBSERVED_RATE_MAX_POINTS,
) -> pd.DataFrame:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(observed_rates) <= max_points:
        return observed_rates.copy()

    positions = np.linspace(0, len(observed_rates) - 1, max_points, dtype=int)
    positions = np.unique(positions)
    return observed_rates.iloc[positions].copy()


def weekly_frame_indices(dates: pd.DatetimeIndex) -> list[int]:
    if len(dates) == 0:
        return []

    selected = [0]
    next_date = dates[0] + pd.Timedelta(days=7)
    for idx, date in enumerate(dates[1:], start=1):
        if date >= next_date:
            selected.append(idx)
            next_date = date + pd.Timedelta(days=7)

    if selected[-1] != len(dates) - 1:
        selected.append(len(dates) - 1)
    return selected


def weekly_play_frame_names(dates: pd.DatetimeIndex) -> list[str]:
    return [
        dates[idx].date().isoformat()
        for idx in weekly_frame_indices(dates)
    ]


def align_curve_and_regimes(
    curves: pd.DataFrame,
    regimes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = curves.copy()
    regimes = regimes.copy()
    curves.index = pd.to_datetime(curves.index)
    regimes.index = pd.to_datetime(regimes.index)

    common_dates = curves.index.intersection(regimes.index).sort_values()
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between curve data and regimes")

    return curves.loc[common_dates], regimes.loc[common_dates]


def _tenor_years(curves: pd.DataFrame) -> list[float]:
    tenors = []
    for col in curves.columns:
        label = str(col)
        if not label.startswith("d"):
            raise ValueError(f"Unexpected curve tenor column: {col}")
        tenors.append(int(label[1:]) / 365.25)
    return tenors


def _state_number(regimes: pd.DataFrame, idx: int) -> int:
    return int(regimes["viterbi_state"].iloc[idx]) + 1


def _filtered_probability(regimes: pd.DataFrame, idx: int) -> float | None:
    state = _state_number(regimes, idx)
    state_col = f"filtered_p{state}"
    if state_col in regimes.columns:
        return float(regimes[state_col].iloc[idx])

    filtered_cols = [
        col for col in regimes.columns
        if str(col).startswith("filtered_p")
    ]
    if filtered_cols:
        return float(regimes[filtered_cols].iloc[idx].max())
    return None


def _filtered_probability_columns(regimes: pd.DataFrame) -> list[str]:
    cols = [
        col for col in regimes.columns
        if str(col).startswith("filtered_p")
    ]
    return sorted(cols, key=lambda col: int(str(col).replace("filtered_p", "")))


def _filtered_probability_text(regimes: pd.DataFrame,
                               idx: int,
                               filtered_cols: list[str]) -> str:
    if not filtered_cols:
        return "No filtered probabilities"
    parts = []
    for col in filtered_cols:
        state = str(col).replace("filtered_p", "S")
        parts.append(f"{state}: {float(regimes[col].iloc[idx]):.1%}")
    return "<br>".join(parts)


def _regime_title(date: pd.Timestamp, regimes: pd.DataFrame, idx: int) -> str:
    state = _state_number(regimes, idx)
    title = f"{date.date().isoformat()} | Regime S{state}"
    prob = _filtered_probability(regimes, idx)
    if prob is not None:
        title = f"{title} | Max filtered probability: {prob:.1%}"
    return title


def _state_color(state: int) -> str:
    return PALETTE[(state - 1) % len(PALETTE)]


def build_figure(
    curves: pd.DataFrame,
    regimes: pd.DataFrame,
    title: str,
    observed_rates: pd.DataFrame | None = None,
    shape_features: pd.DataFrame | None = None,
) -> go.Figure:
    if len(curves) != len(regimes):
        raise ValueError("Curves and regimes must already be aligned")
    if len(curves) == 0:
        raise ValueError("Cannot render an empty viewer")

    tenor_years = _tenor_years(curves)
    dates = pd.DatetimeIndex(curves.index)
    states = [_state_number(regimes, idx) for idx in range(len(regimes))]
    filtered_cols = _filtered_probability_columns(regimes)
    initial_state = states[0]
    initial_title = _regime_title(dates[0], regimes, 0)
    initial_probability_text = _filtered_probability_text(regimes, 0, filtered_cols)
    timeline_hover = [
        _regime_title(date, regimes, idx)
        for idx, date in enumerate(dates)
    ]
    weekly_indices = weekly_frame_indices(dates)
    weekly_positions = list(range(len(weekly_indices)))
    weekly_dates = pd.DatetimeIndex([dates[idx] for idx in weekly_indices])
    weekly_position_by_idx = {
        idx: position for position, idx in enumerate(weekly_indices)
    }
    shape_feature_columns = (
        shape_features.columns.tolist()
        if shape_features is not None else []
    )
    shape_feature_titles = (shape_feature_columns + ["", "", "", ""])[:4]
    timeline_tick_count = min(8, len(weekly_positions))
    timeline_tick_positions = (
        np.linspace(
            0,
            len(weekly_positions) - 1,
            timeline_tick_count,
            dtype=int,
        ).tolist()
        if timeline_tick_count else []
    )
    timeline_tick_positions = sorted(set(timeline_tick_positions))
    timeline_tick_labels = [
        weekly_dates[position].date().isoformat()
        for position in timeline_tick_positions
    ]

    fig = make_subplots(
        rows=3,
        cols=4,
        row_heights=[0.56, 0.24, 0.20],
        column_widths=[0.56, 0.1467, 0.1467, 0.1466],
        vertical_spacing=0.12,
        horizontal_spacing=0.045,
        specs=[
            [{}, {"colspan": 3}, None, None],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
        ],
        subplot_titles=(
            "Yield curve",
            "Market observed interest rates",
            "Filtered regime probabilities",
            shape_feature_titles[0],
            shape_feature_titles[1],
            "",
            "Regime timeline",
            shape_feature_titles[2],
            shape_feature_titles[3],
            "",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=tenor_years,
            y=curves.iloc[0].astype(float),
            mode="lines+markers",
            line={"color": _state_color(initial_state), "width": 3},
            marker={"size": 7},
            name=f"Regime S{initial_state}",
            customdata=[
                [dates[0].date().isoformat(), f"S{initial_state}"]
                for _ in tenor_years
            ],
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Regime: %{customdata[1]}<br>"
                "Tenor: %{x:.2f} years<br>"
                "SOFR zero rate: %{y:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    for col in filtered_cols:
        state = int(str(col).replace("filtered_p", ""))
        fig.add_trace(
            go.Scatter(
                x=weekly_positions,
                y=regimes[col].iloc[weekly_indices].astype(float),
                mode="lines",
                line={"color": _state_color(state), "width": 1.5},
                name=f"Filtered S{state}",
                customdata=[
                    weekly_dates[position].date().isoformat()
                    for position in weekly_positions
                ],
                hovertemplate=(
                    "Date: %{customdata}<br>"
                    f"State: S{state}<br>"
                    "Filtered probability: %{y:.1%}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    filtered_cursor_trace_idx = 1 + len(filtered_cols)
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-0.05, 1.05],
            mode="lines",
            line={"color": "#111111", "width": 1.5},
            name="Selected filtered date",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    filtered_text_trace_idx = filtered_cursor_trace_idx + 1
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0.70],
            mode="text",
            text=[initial_probability_text],
            textposition="middle right",
            textfont={"size": 12, "color": "#111111"},
            name="Selected filtered probabilities",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    timeline_trace_idx = filtered_text_trace_idx + 1
    fig.add_trace(
        go.Scatter(
            x=weekly_positions,
            y=[states[idx] for idx in weekly_indices],
            mode="markers",
            marker={
                "color": [_state_color(states[idx]) for idx in weekly_indices],
                "size": 7,
            },
            name="Assigned regime",
            text=[timeline_hover[idx] for idx in weekly_indices],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    selected_timeline_trace_idx = timeline_trace_idx + 1
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_state],
            mode="markers",
            marker={
                "color": "#111111",
                "line": {"color": "#FFFFFF", "width": 1},
                "size": 14,
                "symbol": "diamond",
            },
            name="Selected date",
            hovertemplate=initial_title + "<extra></extra>",
        ),
        row=3,
        col=1,
    )

    feature_marker_trace_indices = []
    feature_line_trace_start = selected_timeline_trace_idx + 1
    if shape_features is not None:
        shape_features = shape_features.sort_index()
        selected_shape_features = shape_features.reindex(dates, method="nearest")
        feature_positions = {
            feature: position
            for feature, position in zip(
                shape_feature_columns,
                [(2, 2), (2, 3), (3, 2), (3, 3)],
            )
        }
        for feature in shape_feature_columns:
            row, col = feature_positions[feature]
            fig.add_trace(
                go.Scatter(
                    x=shape_features.index,
                    y=shape_features[feature].astype(float),
                    mode="lines",
                    line={"color": "#4C72B0", "width": 1.2},
                    name=feature,
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        f"Feature: {feature}<br>"
                        "Value: %{y:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        feature_marker_trace_start = feature_line_trace_start + len(
            shape_feature_columns
        )
        for feature in shape_feature_columns:
            row, col = feature_positions[feature]
            fig.add_trace(
                go.Scatter(
                    x=[dates[0]],
                    y=[float(selected_shape_features[feature].iloc[0])],
                    mode="markers",
                    marker={
                        "color": "#111111",
                        "line": {"color": "#FFFFFF", "width": 1},
                        "size": 8,
                        "symbol": "circle",
                    },
                    name=f"Selected {feature}",
                    hovertemplate=(
                        "Selected date: %{x|%Y-%m-%d}<br>"
                        f"Feature: {feature}<br>"
                        "Value: %{y:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        feature_marker_trace_indices = list(
            range(
                feature_marker_trace_start,
                feature_marker_trace_start + len(shape_feature_columns),
            )
        )
    else:
        selected_shape_features = None

    observed_line_trace_start = selected_timeline_trace_idx + 1
    if shape_features is not None:
        observed_line_trace_start += 2 * len(shape_feature_columns)
    if observed_rates is not None:
        observed_rates = observed_rates.sort_index()
        observed_line_rates = downsample_observed_rates(observed_rates)
        observed_dates = pd.DatetimeIndex(observed_line_rates.index)
        selected_observed_rates = observed_rates.reindex(dates, method="nearest")
        for idx, term in enumerate(observed_rates.columns):
            fig.add_trace(
                go.Scatter(
                    x=observed_dates,
                    y=observed_line_rates[term].astype(float),
                    mode="lines",
                    line={"color": "#4C72B0", "width": 2},
                    name=f"Observed {term}",
                    visible=(idx == 0),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        f"Maturity: {term}<br>"
                        "Observed rate: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
        observed_marker_trace_start = observed_line_trace_start + len(
            observed_rates.columns
        )
        for idx, term in enumerate(observed_rates.columns):
            fig.add_trace(
                go.Scatter(
                    x=[dates[0]],
                    y=[float(selected_observed_rates[term].iloc[0])],
                    mode="markers",
                    marker={
                        "color": "#111111",
                        "line": {"color": "#FFFFFF", "width": 1},
                        "size": 12,
                        "symbol": "circle",
                    },
                    name=f"Selected observed {term}",
                    visible=(idx == 0),
                    hovertemplate=(
                        "Selected date: %{x|%Y-%m-%d}<br>"
                        f"Maturity: {term}<br>"
                        "Observed rate: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
    else:
        selected_observed_rates = None

    frames = []
    observed_marker_trace_indices = []
    if observed_rates is not None:
        observed_marker_trace_indices = list(
            range(
                observed_marker_trace_start,
                observed_marker_trace_start + len(observed_rates.columns),
            )
        )
    for idx in weekly_indices:
        date = dates[idx]
        position = weekly_position_by_idx[idx]
        state = states[idx]
        frame_title = _regime_title(date, regimes, idx)
        probability_text = _filtered_probability_text(regimes, idx, filtered_cols)
        frame_traces = [
            0,
            filtered_cursor_trace_idx,
            filtered_text_trace_idx,
            selected_timeline_trace_idx,
        ] + feature_marker_trace_indices + observed_marker_trace_indices
        frame_data = [
            go.Scatter(
                x=tenor_years,
                y=curves.iloc[idx].astype(float),
                mode="lines+markers",
                line={"color": _state_color(state), "width": 3},
                marker={"size": 7},
                name=f"Regime S{state}",
                customdata=[
                    [date.date().isoformat(), f"S{state}"]
                    for _ in tenor_years
                ],
                hovertemplate=(
                    "Date: %{customdata[0]}<br>"
                    "Regime: %{customdata[1]}<br>"
                    "Tenor: %{x:.2f} years<br>"
                    "SOFR zero rate: %{y:.4f}<extra></extra>"
                ),
            ),
            go.Scatter(
                x=[position, position],
                y=[-0.05, 1.05],
                mode="lines",
                line={"color": "#111111", "width": 1.5},
                name="Selected filtered date",
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=[position],
                y=[0.70],
                mode="text",
                text=[probability_text],
                textposition="middle right",
                textfont={"size": 12, "color": "#111111"},
                name="Selected filtered probabilities",
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=[position],
                y=[state],
                mode="markers",
                marker={
                    "color": "#111111",
                    "line": {"color": "#FFFFFF", "width": 1},
                    "size": 14,
                    "symbol": "diamond",
                },
                name="Selected date",
                hovertemplate=frame_title + "<extra></extra>",
            ),
        ]
        if selected_shape_features is not None:
            for feature in shape_feature_columns:
                frame_data.append(
                    go.Scatter(
                        x=[date],
                        y=[float(selected_shape_features[feature].iloc[idx])],
                        mode="markers",
                        marker={
                            "color": "#111111",
                            "line": {"color": "#FFFFFF", "width": 1},
                            "size": 8,
                            "symbol": "circle",
                        },
                        name=f"Selected {feature}",
                        hovertemplate=(
                            "Selected date: %{x|%Y-%m-%d}<br>"
                            f"Feature: {feature}<br>"
                            "Value: %{y:.4f}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
        if selected_observed_rates is not None:
            for term in observed_rates.columns:
                frame_data.append(
                    go.Scatter(
                        x=[date],
                        y=[float(selected_observed_rates[term].iloc[idx])],
                        mode="markers",
                        marker={
                            "color": "#111111",
                            "line": {"color": "#FFFFFF", "width": 1},
                            "size": 12,
                            "symbol": "circle",
                        },
                        name=f"Selected observed {term}",
                        hovertemplate=(
                            "Selected date: %{x|%Y-%m-%d}<br>"
                            f"Maturity: {term}<br>"
                            "Observed rate: %{y:.4f}<extra></extra>"
                        ),
                    )
                )
        frames.append(
            go.Frame(
                name=date.date().isoformat(),
                traces=frame_traces,
                data=frame_data,
                layout=go.Layout(
                    title={"text": f"{title}<br><sup>{frame_title}</sup>"}
                ),
            )
        )
    fig.frames = frames

    slider_steps = [
        {
            "args": [
                [dates[idx].date().isoformat()],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": dates[idx].date().isoformat(),
            "method": "animate",
        }
        for idx in weekly_indices
    ]
    observed_buttons = []
    if observed_rates is not None:
        for idx, term in enumerate(observed_rates.columns):
            n_terms = len(observed_rates.columns)
            visible = [True] * observed_line_trace_start + [
                obs_idx == idx for obs_idx in range(n_terms)
            ] + [
                obs_idx == idx for obs_idx in range(n_terms)
            ]
            observed_buttons.append(
                {
                    "label": str(term),
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {"yaxis2.title.text": f"{term} observed rate"},
                    ],
                }
            )

    updatemenus = [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.04,
            "y": 0,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 120, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]
    if observed_buttons:
        updatemenus.append(
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.74,
                "y": 1.11,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
                "buttons": observed_buttons,
            }
        )

    fig.update_layout(
        title={"text": f"{title}<br><sup>{initial_title}</sup>"},
        height=1120,
        hovermode="closest",
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Date: "},
                "len": 0.92,
                "minorticklen": 0,
                "pad": {"t": 75},
                "tickcolor": "rgba(0,0,0,0)",
                "ticklen": 0,
                "steps": slider_steps,
            }
        ],
        updatemenus=updatemenus,
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 70, "r": 35, "t": 120, "b": 190},
    )
    fig.update_xaxes(title_text="Tenor (years)", row=1, col=1)
    fig.update_yaxes(title_text="SOFR zero rate", range=[0, 6], row=1, col=1)
    if observed_rates is not None:
        first_term = observed_rates.columns[0]
        fig.update_xaxes(
            title_text="Date",
            range=[
                observed_rates.index.min(),
                observed_rates.index.max(),
            ],
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text=f"{first_term} observed rate", row=1, col=2)
    fig.update_xaxes(
        range=[-0.5, len(weekly_positions) - 0.5],
        showticklabels=False,
        title_text="",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Filtered probability",
        range=[-0.05, 1.05],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Date",
        range=[-0.5, len(weekly_positions) - 0.5],
        tickmode="array",
        tickvals=timeline_tick_positions,
        ticktext=timeline_tick_labels,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="Regime",
        tickmode="array",
        tickvals=sorted(set(states)),
        ticktext=[f"S{state}" for state in sorted(set(states))],
        row=3,
        col=1,
    )
    if shape_features is not None:
        feature_x_range = [
            shape_features.index.min(),
            shape_features.index.max(),
        ]
        for col in (2, 3):
            fig.update_xaxes(
                range=feature_x_range,
                showticklabels=False,
                title_text="",
                row=2,
                col=col,
            )
            fig.update_xaxes(
                range=feature_x_range,
                title_text="",
                tickfont={"size": 9},
                row=3,
                col=col,
            )
            fig.update_yaxes(
                title_text="",
                tickfont={"size": 9},
                row=2,
                col=col,
            )
            fig.update_yaxes(
                title_text="",
                tickfont={"size": 9},
                row=3,
                col=col,
            )
    return fig


def write_viewer_html(
    curves: pd.DataFrame,
    regimes: pd.DataFrame,
    output_html: str,
    title: str,
    observed_rates: pd.DataFrame | None = None,
    shape_features: pd.DataFrame | None = None,
) -> None:
    fig = build_figure(
        curves,
        regimes,
        title,
        observed_rates=observed_rates,
        shape_features=shape_features,
    )
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def render_viewer(args: argparse.Namespace) -> str:
    curve_data = _resolve_path(args.curve_data)
    regime_assignments = _resolve_path(args.regime_assignments)
    observed_rates_path = _resolve_path(args.observed_rates)
    shape_features_path = _resolve_path(args.shape_features)
    output_html = args.output_html or default_output_html(regime_assignments)
    output_html = _resolve_path(output_html)

    if os.path.exists(output_html) and not args.allow_overwrite:
        raise SystemExit(
            f"Output HTML already exists: {output_html}. "
            "Use --output-html for a new file or --allow-overwrite."
        )

    curves = feat.load_and_pivot(curve_data)
    regimes = load_regime_assignments(regime_assignments)
    observed_rates = load_observed_rates(observed_rates_path)
    shape_features = load_shape_features(shape_features_path)
    aligned_curves, aligned_regimes = align_curve_and_regimes(curves, regimes)
    viewer_title = title_with_experiment_label(args.title, regime_assignments)
    write_viewer_html(
        aligned_curves,
        aligned_regimes,
        output_html,
        title=viewer_title,
        observed_rates=observed_rates,
        shape_features=shape_features,
    )
    return output_html


def main(argv=None) -> str:
    args = parse_args(argv)
    output_html = render_viewer(args)
    print(f"Wrote interactive viewer to: {output_html}")
    return output_html


if __name__ == "__main__":
    main()
