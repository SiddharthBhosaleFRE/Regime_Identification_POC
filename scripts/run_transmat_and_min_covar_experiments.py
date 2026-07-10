import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
if HMM_DIR not in sys.path:
    sys.path.insert(0, HMM_DIR)

import config
import evaluate as ev
import features as feat
import hmm_fit as hf


OUT_DIR = os.path.join(ROOT, "outputs", "transmat_and_min_covar_experiments")
TRANSMAT_PRIOR_GRID = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
MIN_COVAR_GRID = [0.001, 0.01, 0.05, 0.1]

RESULT_COLUMNS = [
    "stage",
    "folder",
    "covariance_type",
    "best_k",
    "min_offdiag_prob",
    "transmat_prior",
    "min_covar",
    "posterior_temperature",
    "pct_days_max_prob_gt_99",
    "n_genuinely_uncertain_days",
    "mean_entropy",
    "n_transitions",
    "worst_transition_min_max_prob",
    "best_transition_max_entropy",
    "viterbi_churn_pct",
    "behavioral_viterbi_churn_pct",
    "state_alignment",
    "log_likelihood",
    "bic",
]


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def experiment_configs() -> list[dict]:
    baseline = {"transmat_prior": 1.0, "min_covar": 0.001}
    configs = [{
        "stage": "baseline",
        "folder": "baseline",
        **baseline,
    }]

    seen = {(baseline["transmat_prior"], baseline["min_covar"])}
    for transmat_prior in TRANSMAT_PRIOR_GRID:
        key = (transmat_prior, baseline["min_covar"])
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "stage": "transmat_prior",
            "folder": f"transmat_prior_{_number_token(transmat_prior)}",
            "transmat_prior": transmat_prior,
            "min_covar": baseline["min_covar"],
        })

    for min_covar in MIN_COVAR_GRID:
        key = (baseline["transmat_prior"], min_covar)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "stage": "min_covar",
            "folder": f"min_covar_{_number_token(min_covar)}",
            "transmat_prior": baseline["transmat_prior"],
            "min_covar": min_covar,
        })

    return configs


def build_result_row(config_row: dict,
                     covariance_type: str,
                     best_k: int,
                     min_offdiag_prob: float,
                     posterior_temperature: float,
                     metrics: dict,
                     log_likelihood: float,
                     bic: float) -> dict:
    row = {
        "stage": config_row["stage"],
        "folder": config_row["folder"],
        "covariance_type": covariance_type,
        "best_k": best_k,
        "min_offdiag_prob": min_offdiag_prob,
        "transmat_prior": config_row["transmat_prior"],
        "min_covar": config_row["min_covar"],
        "posterior_temperature": posterior_temperature,
        "log_likelihood": log_likelihood,
        "bic": bic,
        **metrics,
    }
    return {column: row[column] for column in RESULT_COLUMNS}


def _write_model_outputs(stage_dir: str, results: dict, model, regimes: pd.DataFrame,
                         features_df: pd.DataFrame, best_k: int,
                         config_row: dict, result_row: dict) -> None:
    os.makedirs(stage_dir, exist_ok=True)

    stats = ev.regime_stats(features_df, regimes, best_k)
    moments = ev.regime_feature_moments(features_df, regimes, best_k)
    tmat = ev.transition_table(model)
    durations = hf.regime_duration_summary(regimes, best_k)
    diagnostics = ev.filtered_prob_diagnostics(regimes, best_k)
    transition_diag = ev.transition_window_diagnostics(
        regimes, best_k, window=config.TRANSITION_WINDOW_DAYS
    )

    ev.plot_bic(
        results,
        title=f"shape model - {config_row['folder']}",
        save_path=os.path.join(stage_dir, "bic_plot.png"),
    )
    ev.plot_regime_timeline(
        regimes,
        features_df,
        best_k,
        title=f"Shape model - {config_row['folder']}",
        save_path=os.path.join(stage_dir, "regime_timeline.png"),
    )
    ev.plot_feature_distributions(
        features_df,
        regimes,
        best_k,
        title=f"Shape model - {config_row['folder']}",
        save_path=os.path.join(stage_dir, "feature_dists.png"),
    )

    regimes.to_csv(os.path.join(stage_dir, "regime_assignments.csv"))
    stats.to_csv(os.path.join(stage_dir, "regime_feature_means.csv"))
    moments.to_csv(
        os.path.join(stage_dir, "regime_feature_mean_variances.csv"),
        index=False,
    )
    tmat.to_csv(os.path.join(stage_dir, "transition_matrix.csv"))
    durations.to_csv(os.path.join(stage_dir, "regime_durations.csv"), index=False)
    diagnostics.to_csv(os.path.join(stage_dir, "filtered_diagnostics.csv"), index=False)
    transition_diag.to_csv(
        os.path.join(stage_dir, "transition_diagnostics.csv"),
        index=False,
    )
    pd.DataFrame([result_row]).to_csv(
        os.path.join(stage_dir, "model_config.csv"),
        index=False,
    )


def run_experiments() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    levels = feat.load_and_pivot(config.DATA_PATH)
    shape_features_df, _ = feat.build_shape_features(levels)
    X_shape, _ = feat.scale_for_hmm(shape_features_df)

    rows = []
    base_states = None
    for config_row in experiment_configs():
        print(
            f"\n[experiment] {config_row['folder']} "
            f"transmat_prior={config_row['transmat_prior']:g} "
            f"min_covar={config_row['min_covar']:g}"
        )
        results = hf.select_k(
            X_shape,
            config.SHAPE_COVARIANCE_TYPE,
            label=config_row["folder"],
            min_offdiag_prob=config.MIN_OFFDIAG_PROB,
            transmat_prior=config_row["transmat_prior"],
            min_covar=config_row["min_covar"],
        )
        best_k = results["best_k"]
        model = results[best_k]["model"]
        regimes = hf.decode(model, X_shape, index=shape_features_df.index)
        regimes = hf.apply_posterior_temperature(
            regimes, temperature=config.SHAPE_POSTERIOR_TEMPERATURE
        )
        if base_states is None:
            base_states = regimes["viterbi_state"].to_numpy()
        metrics = hf.calibration_metrics(
            regimes,
            base_states=base_states,
            transition_window=config.TRANSITION_WINDOW_DAYS,
            behavior_features=X_shape,
        )
        result_row = build_result_row(
            config_row=config_row,
            covariance_type=config.SHAPE_COVARIANCE_TYPE,
            best_k=best_k,
            min_offdiag_prob=config.MIN_OFFDIAG_PROB,
            posterior_temperature=config.SHAPE_POSTERIOR_TEMPERATURE,
            metrics=metrics,
            log_likelihood=results[best_k]["log_likelihood"],
            bic=results[best_k]["bic"],
        )
        rows.append(result_row)
        _write_model_outputs(
            os.path.join(OUT_DIR, config_row["folder"]),
            results,
            model,
            regimes,
            shape_features_df,
            best_k,
            config_row,
            result_row,
        )

    summary = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("\n[summary.csv]")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_experiments()
