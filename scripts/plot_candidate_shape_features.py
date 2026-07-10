import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v3")
if HMM_DIR not in sys.path:
    sys.path.insert(0, HMM_DIR)

import config
import features as feat


DEFAULT_OUTPUT_PNG = os.path.join("outputs", "plots", "candidate_features.png")


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT, path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot candidate SOFR curve features: 2y10y slope, weekly level "
            "change, and weekly short-slope change."
        )
    )
    parser.add_argument("--data-path", default=config.DATA_PATH)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_candidate_features(levels: pd.DataFrame) -> pd.DataFrame:
    c3m = feat._nearest_grid_col(config.TENOR_3M)
    c2y = feat._nearest_grid_col(config.TENOR_2Y)
    c10y = feat._nearest_grid_col(config.TENOR_10Y)

    level = levels.mean(axis=1)
    slope_short = levels[c2y] - levels[c3m]
    out = pd.DataFrame(
        {
            "slope_2y10y": levels[c10y] - levels[c2y],
            "weekly_delta_level": level.diff(5),
            "weekly_delta_slope_short": slope_short.diff(5),
        },
        index=levels.index,
    )
    return out.dropna()


def _ensure_writable(path: str, allow_overwrite: bool) -> None:
    if os.path.exists(path) and not allow_overwrite:
        raise SystemExit(
            f"Output already exists: {path}. Use --allow-overwrite to replace it."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_candidate_features(candidate_features: pd.DataFrame, output_png: str) -> str:
    labels = {
        "slope_2y10y": "2y10y slope (10y - 2y)",
        "weekly_delta_level": "Weekly change in curve level",
        "weekly_delta_slope_short": "Weekly change in short slope",
    }
    colors = {
        "slope_2y10y": "#4C72B0",
        "weekly_delta_level": "#DD8452",
        "weekly_delta_slope_short": "#55A868",
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, column in zip(axes, candidate_features.columns):
        ax.plot(
            candidate_features.index,
            candidate_features[column],
            color=colors[column],
            lw=1.3,
        )
        ax.axhline(0.0, color="#333333", lw=0.7, alpha=0.55)
        ax.set_ylabel(column)
        ax.set_title(labels[column], fontsize=10)
        ax.grid(axis="y", lw=0.4, alpha=0.45)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Candidate SOFR Shape Features", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def run(args: argparse.Namespace) -> dict:
    data_path = _resolve_path(args.data_path)
    output_png = _resolve_path(args.output_png)
    output_csv = _resolve_path(args.output_csv) if args.output_csv else None

    _ensure_writable(output_png, args.allow_overwrite)
    if output_csv is not None:
        _ensure_writable(output_csv, args.allow_overwrite)

    levels = feat.load_and_pivot(data_path)
    candidate_features = build_candidate_features(levels)
    plot_candidate_features(candidate_features, output_png)
    if output_csv is not None:
        candidate_features.to_csv(output_csv, index_label="date")

    return {
        "output_png": output_png,
        "output_csv": output_csv,
        "n_rows": len(candidate_features),
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    result = run(args)
    print(f"Wrote plot to: {result['output_png']}")
    if result["output_csv"]:
        print(f"Wrote data to: {result['output_csv']}")
    print(f"Rows plotted: {result['n_rows']}")
    return result


if __name__ == "__main__":
    main()
