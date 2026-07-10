import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v3")
if HMM_DIR in sys.path:
    sys.path.remove(HMM_DIR)
sys.path.insert(0, HMM_DIR)
for module_name in ("config", "evaluate", "features", "hmm_fit"):
    sys.modules.pop(module_name, None)

import config
import evaluate as ev
import features as feat
import hmm_fit as hf


DEFAULT_OUT_DIR = os.path.join("outputs", "one_off_experiments", "v3")


def _number_token(value: float | int) -> str:
    return f"{value:g}".replace("-", "neg_").replace(".", "_")


def default_output_folder(args: argparse.Namespace) -> str:
    folder = (
        f"restarts_{args.n_em_restarts}"
        f"_transmat_prior_{_number_token(args.transmat_prior)}"
        f"_covariance_scale_{_number_token(args.covariance_scale)}"
    )
    if args.shape_feature_set != "v3":
        folder = f"{args.shape_feature_set}_{folder}"
    if args.feature_transform != "standard":
        folder = f"{folder}_{args.feature_transform}"
    return folder


def output_folder_name(args: argparse.Namespace) -> str:
    return args.output_folder or default_output_folder(args)


def output_root_dir(args: argparse.Namespace) -> str:
    if (
        args.feature_transform == "uniform_platykurtic"
        and args.out_dir == DEFAULT_OUT_DIR
    ):
        return os.path.join(args.out_dir, "platykurtic")
    return args.out_dir


def experiment_title(args: argparse.Namespace) -> str:
    model_name = "v3 shape model"
    if args.shape_feature_set != "v3":
        model_name = f"v3 {args.shape_feature_set} shape model"
    if args.feature_transform == "uniform_platykurtic":
        model_name = f"platykurtic {model_name}"
    return (
        f"{model_name} - "
        f"restarts {args.n_em_restarts}, "
        f"transmat prior {args.transmat_prior:g}, "
        f"covariance scale {args.covariance_scale:g}"
    )


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT, path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one configurable SOFR v3 shape HMM experiment."
    )
    parser.add_argument("--data-path", default=config.DATA_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--shape-feature-set", default="v3",
                        choices=["v3", "slope_2y10y"])

    parser.add_argument("--covariance-type", default=config.SHAPE_COVARIANCE_TYPE,
                        choices=["diag", "full", "tied"])
    parser.add_argument("--k-values", type=int, nargs="+", default=config.K_VALUES)
    parser.add_argument("--n-em-restarts", type=int, default=config.N_EM_RESTARTS)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER)
    parser.add_argument("--random-seed", type=int, default=config.RANDOM_SEED)

    parser.add_argument("--transmat-prior", type=float, default=1.0)
    parser.add_argument("--min-covar", type=float, default=0.001)
    parser.add_argument("--min-offdiag-prob", type=float,
                        default=config.MIN_OFFDIAG_PROB)

    parser.add_argument("--covariance-scale", type=float, default=1.0)
    parser.add_argument("--transition-blend", type=float, default=0.0)
    parser.add_argument("--posterior-temperature", type=float,
                        default=config.SHAPE_POSTERIOR_TEMPERATURE)
    parser.add_argument("--transition-window-days", type=int,
                        default=config.TRANSITION_WINDOW_DAYS)
    parser.add_argument("--feature-transform", default="standard",
                        choices=["standard", "uniform_platykurtic"])
    parser.add_argument("--quantile-n-quantiles", type=int, default=None)
    parser.add_argument("--quantile-random-seed", type=int, default=None)
    parser.add_argument("--test-size-pct", type=float, default=0.0)
    parser.add_argument("--split-date", default=None)
    return parser.parse_args(argv)


def _apply_runtime_config(args: argparse.Namespace) -> None:
    config.DATA_PATH = _resolve_path(args.data_path)
    config.K_VALUES = args.k_values
    config.N_EM_RESTARTS = args.n_em_restarts
    config.MAX_ITER = args.max_iter
    config.RANDOM_SEED = args.random_seed


def _transition_summary(transition_diag: pd.DataFrame) -> dict:
    if transition_diag.empty:
        return {
            "n_transitions": 0,
            "worst_transition_min_max_prob": float("nan"),
            "best_transition_max_entropy": float("nan"),
        }
    return {
        "n_transitions": int(len(transition_diag)),
        "worst_transition_min_max_prob": (
            float(transition_diag["min_max_filtered_prob"].max())
        ),
        "best_transition_max_entropy": (
            float(transition_diag["max_entropy"].min())
        ),
    }


def split_features(features_df: pd.DataFrame,
                   test_size_pct: float = 20.0,
                   split_date: str = None):
    if split_date:
        split_ts = pd.Timestamp(split_date)
        train = features_df[features_df.index < split_ts]
        test = features_df[features_df.index >= split_ts]
        meta = {
            "split_mode": "split_date",
            "split_date": split_ts.date().isoformat(),
            "test_size_pct": test_size_pct,
        }
    else:
        if test_size_pct < 0 or test_size_pct >= 100:
            raise SystemExit("test_size_pct must be >= 0 and < 100")
        if test_size_pct == 0:
            return features_df, None, {
                "split_mode": "none",
                "split_date": "",
                "test_size_pct": test_size_pct,
            }
        n_test = int(math.ceil(len(features_df) * test_size_pct / 100.0))
        train = features_df.iloc[:-n_test]
        test = features_df.iloc[-n_test:]
        meta = {
            "split_mode": "test_size_pct",
            "split_date": "",
            "test_size_pct": test_size_pct,
        }

    if len(train) == 0 or test is None or len(test) == 0:
        raise SystemExit("Train/test split produced an empty train or test window")
    return train, test, meta


def _date_value(index, attr):
    if len(index) == 0:
        return ""
    return getattr(index, attr)().date().isoformat()


def _window_config(train_features: pd.DataFrame,
                   test_features: pd.DataFrame | None,
                   split_meta: dict) -> dict:
    config_row = {
        **split_meta,
        "train_start": _date_value(train_features.index, "min"),
        "train_end": _date_value(train_features.index, "max"),
        "test_start": "",
        "test_end": "",
        "n_train_days": int(len(train_features)),
        "n_test_days": 0,
    }
    if test_features is not None:
        config_row.update({
            "test_start": _date_value(test_features.index, "min"),
            "test_end": _date_value(test_features.index, "max"),
            "n_test_days": int(len(test_features)),
        })
    return config_row


def _output_filename(prefix: str | None, suffix: str) -> str:
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


def _write_regime_outputs(stage_dir: str,
                          prefix: str | None,
                          model,
                          regimes: pd.DataFrame,
                          features_df: pd.DataFrame,
                          feature_diagnostics: pd.DataFrame,
                          best_k: int,
                          transition_window: int,
                          title: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = ev.regime_stats(features_df, regimes, best_k)
    moments = ev.regime_feature_moments(features_df, regimes, best_k)
    durations = hf.regime_duration_summary(regimes, best_k)
    diagnostics = ev.filtered_prob_diagnostics(regimes, best_k)
    transition_diag = ev.transition_window_diagnostics(
        regimes, best_k, window=transition_window
    )

    regimes.to_csv(os.path.join(
        stage_dir, _output_filename(prefix, "regime_assignments.csv")
    ))
    stats.to_csv(os.path.join(
        stage_dir, _output_filename(prefix, "regime_feature_means.csv")
    ))
    moments.to_csv(
        os.path.join(stage_dir, _output_filename(
            prefix, "regime_feature_mean_variances.csv"
        )),
        index=False,
    )
    durations.to_csv(
        os.path.join(stage_dir, _output_filename(prefix, "regime_durations.csv")),
        index=False,
    )
    diagnostics.to_csv(
        os.path.join(stage_dir, _output_filename(prefix, "filtered_diagnostics.csv")),
        index=False,
    )
    transition_diag.to_csv(
        os.path.join(stage_dir, _output_filename(prefix, "transition_diagnostics.csv")),
        index=False,
    )
    feature_diagnostics.to_csv(
        os.path.join(
            stage_dir,
            _output_filename(prefix, "feature_distribution_diagnostics.csv"),
        ),
        index=False,
    )
    ev.plot_regime_timeline(
        regimes,
        features_df,
        best_k,
        title=title,
        save_path=os.path.join(stage_dir, _output_filename(
            prefix, "regime_timeline.png"
        )),
    )
    ev.plot_feature_distributions(
        features_df,
        regimes,
        best_k,
        title=title,
        save_path=os.path.join(stage_dir, _output_filename(
            prefix, "feature_dists.png"
        )),
    )
    return diagnostics, transition_diag


def run_shape_hmm(args: argparse.Namespace) -> dict:
    _apply_runtime_config(args)
    out_dir = _resolve_path(output_root_dir(args))
    stage_dir = os.path.join(out_dir, output_folder_name(args))
    if os.path.exists(stage_dir) and not args.allow_overwrite:
        raise SystemExit(
            f"Output folder already exists: {stage_dir}. "
            "Use --output-folder for a new run or --allow-overwrite."
        )
    os.makedirs(stage_dir, exist_ok=args.allow_overwrite)

    levels = feat.load_and_pivot(config.DATA_PATH)
    shape_features_df, _ = feat.build_shape_features(
        levels,
        feature_set=args.shape_feature_set,
    )
    shape_features_df.to_csv(
        os.path.join(stage_dir, "features.csv"),
        index_label="date",
    )
    train_features_df, test_features_df, split_meta = split_features(
        shape_features_df,
        test_size_pct=args.test_size_pct,
        split_date=args.split_date,
    )
    quantile_random_seed = (
        args.random_seed if args.quantile_random_seed is None
        else args.quantile_random_seed
    )
    feature_transformer = feat.fit_hmm_transform(
        train_features_df,
        transform=args.feature_transform,
        n_quantiles=args.quantile_n_quantiles,
        random_seed=quantile_random_seed,
    )
    X_train = feat.apply_hmm_transform(
        train_features_df,
        feature_transformer,
        transform=args.feature_transform,
    )
    train_feature_diagnostics = feat.feature_distribution_diagnostics(
        X_train,
        train_features_df.columns.tolist(),
        args.feature_transform,
    )
    X_test = None
    test_feature_diagnostics = None
    if test_features_df is not None:
        X_test = feat.apply_hmm_transform(
            test_features_df,
            feature_transformer,
            transform=args.feature_transform,
        )
        test_feature_diagnostics = feat.feature_distribution_diagnostics(
            X_test,
            test_features_df.columns.tolist(),
            args.feature_transform,
        )
    quantile_n_quantiles = getattr(
        feature_transformer, "n_quantiles_", args.quantile_n_quantiles
    )

    label = output_folder_name(args).replace("_", "-")
    results = hf.select_k(
        X_train,
        args.covariance_type,
        label=label,
        min_offdiag_prob=args.min_offdiag_prob,
        transmat_prior=args.transmat_prior,
        min_covar=args.min_covar,
    )
    best_k = results["best_k"]
    fitted_model = results[best_k]["model"]
    report_model = hf.calibrated_model(
        fitted_model,
        covariance_scale=args.covariance_scale,
        transition_blend=args.transition_blend,
    )
    train_regimes = hf.decode(
        report_model, X_train, index=train_features_df.index
    )
    train_regimes = hf.apply_posterior_temperature(
        train_regimes, temperature=args.posterior_temperature
    )
    test_regimes = None
    if X_test is not None:
        test_regimes = hf.decode(
            report_model, X_test, index=test_features_df.index
        )
        test_regimes = hf.apply_posterior_temperature(
            test_regimes, temperature=args.posterior_temperature
        )

    tmat = ev.transition_table(report_model)

    title = experiment_title(args)
    output_prefix = None if test_features_df is None else "train"
    train_diagnostics, train_transition_diag = _write_regime_outputs(
        stage_dir,
        output_prefix,
        report_model,
        train_regimes,
        train_features_df,
        train_feature_diagnostics,
        best_k,
        args.transition_window_days,
        title if output_prefix is None else f"{title} - train",
    )
    test_diagnostics = None
    test_transition_diag = None
    if test_regimes is not None:
        test_diagnostics, test_transition_diag = _write_regime_outputs(
            stage_dir,
            "test",
            report_model,
            test_regimes,
            test_features_df,
            test_feature_diagnostics,
            best_k,
            args.transition_window_days,
            f"{title} - test",
        )

    model_config = {
        "data_path": config.DATA_PATH,
        "feature_version": (
            "v3" if args.shape_feature_set == "v3"
            else f"v3_{args.shape_feature_set}"
        ),
        "shape_feature_set": args.shape_feature_set,
        "feature_columns": " ".join(train_features_df.columns.tolist()),
        "covariance_type": args.covariance_type,
        "k_values": " ".join(str(k) for k in args.k_values),
        "best_k": best_k,
        "n_em_restarts": args.n_em_restarts,
        "max_iter": args.max_iter,
        "random_seed": args.random_seed,
        "min_offdiag_prob": args.min_offdiag_prob,
        "transmat_prior": args.transmat_prior,
        "min_covar": args.min_covar,
        "covariance_scale": args.covariance_scale,
        "transition_blend": args.transition_blend,
        "posterior_temperature": args.posterior_temperature,
        "transition_window_days": args.transition_window_days,
        "feature_transform": args.feature_transform,
        "quantile_n_quantiles": quantile_n_quantiles,
        "quantile_random_seed": quantile_random_seed,
        "train_log_likelihood": results[best_k]["log_likelihood"],
        "train_bic": results[best_k]["bic"],
        "test_log_likelihood": (
            float(fitted_model.score(X_test)) if X_test is not None else ""
        ),
        "log_likelihood": results[best_k]["log_likelihood"],
        "bic": results[best_k]["bic"],
        **_window_config(train_features_df, test_features_df, split_meta),
        **_transition_summary(train_transition_diag),
        **train_diagnostics.iloc[0].to_dict(),
    }

    tmat.to_csv(os.path.join(stage_dir, "transition_matrix.csv"))
    pd.DataFrame([model_config]).to_csv(
        os.path.join(stage_dir, "model_config.csv"), index=False
    )

    ev.plot_bic(results, title=title,
                save_path=os.path.join(stage_dir, "bic_plot.png"))

    model_config["output_dir"] = stage_dir
    return model_config


def main(argv=None) -> dict:
    args = parse_args(argv)
    result = run_shape_hmm(args)
    print(f"Wrote outputs to: {result['output_dir']}")
    print(pd.DataFrame([result]).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
