import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v3")
if HMM_DIR in sys.path:
    sys.path.remove(HMM_DIR)
sys.path.insert(0, HMM_DIR)
for module_name in ("config", "features"):
    sys.modules.pop(module_name, None)

import config
import features as feat


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

DEFAULT_OUTPUT_CSV = "shape_features.csv"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT, path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SOFR shape feature time series to CSV."
    )
    parser.add_argument("--data-path", default=config.DATA_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def export_shape_features(args: argparse.Namespace) -> str:
    data_path = _resolve_path(args.data_path)
    output_csv = _resolve_path(args.output_csv)

    if os.path.exists(output_csv) and not args.allow_overwrite:
        raise SystemExit(
            f"Output CSV already exists: {output_csv}. "
            "Use --allow-overwrite to replace it."
        )

    levels = feat.load_and_pivot(data_path)
    shape_features, _ = feat.build_shape_features(levels)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    shape_features.to_csv(output_csv, index_label="date")
    return output_csv


def main(argv=None) -> str:
    args = parse_args(argv)
    output_csv = export_shape_features(args)
    print(f"Wrote shape features to: {output_csv}")
    return output_csv


if __name__ == "__main__":
    main()
