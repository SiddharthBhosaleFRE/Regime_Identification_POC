import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HMM_DIR = os.path.join(ROOT, "sofr_hmm_v2")
if HMM_DIR not in sys.path:
    sys.path.insert(0, HMM_DIR)

import config
import features as feat
import hmm_fit as hf
import evaluate as ev


OUT_DIR = os.path.join(ROOT, "outputs", "uncertainty_experiments")
TRANSITION_FLOOR_GRID = [1e-4, 1e-3, 1e-2, 2e-2, 5e-2]
COVARIANCE_TYPE_GRID = ["diag", "tied", "full"]

RESULT_COLUMNS = [
    "stage",
    "selected",
    "meets_targets",
    "covariance_type",
    "best_k",
    "min_offdiag_prob",
    "covariance_scale",
    "transition_blend",
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


def empty_stage_row(stage: str, **overrides) -> dict:
    row = {col: np.nan for col in RESULT_COLUMNS}
    row.update({
        "stage": stage,
        "selected": False,
        "meets_targets": False,
        "posterior_temperature": config.SHAPE_POSTERIOR_TEMPERATURE,
    })
    row.update(overrides)
    return row


def build_result_row(stage: str,
                     covariance_type: str,
                     best_k: int,
                     min_offdiag_prob: float,
                     covariance_scale: float,
                     transition_blend: float,
                     posterior_temperature: float,
                     regimes: pd.DataFrame,
                     base_states: np.ndarray,
                     behavior_features: np.ndarray,
                     transition_window: int,
                     log_likelihood: float = np.nan,
                     bic: float = np.nan,
                     selected: bool = False) -> dict:
    metrics = hf.calibration_metrics(
        regimes,
        base_states,
        transition_window,
        behavior_features=behavior_features,
    )
    meets_targets = (
        metrics["worst_transition_min_max_prob"] <= config.TARGET_MAX_PROB_AT_TRANSITION
        and metrics["pct_days_max_prob_gt_99"] <= config.TARGET_PCT_DAYS_MAX_PROB_GT_99
        and metrics["behavioral_viterbi_churn_pct"] <= config.MAX_ALLOWED_VITERBI_CHURN_PCT
    )
    return empty_stage_row(
        stage,
        selected=selected,
        meets_targets=meets_targets,
        covariance_type=covariance_type,
        best_k=best_k,
        min_offdiag_prob=min_offdiag_prob,
        covariance_scale=covariance_scale,
        transition_blend=transition_blend,
        posterior_temperature=posterior_temperature,
        log_likelihood=log_likelihood,
        bic=bic,
        **metrics,
    )


def covariance_scale_rows(regimes_by_scale: dict[float, pd.DataFrame],
                          base_states: np.ndarray,
                          behavior_features: np.ndarray,
                          covariance_type: str,
                          best_k: int,
                          min_offdiag_prob: float,
                          transition_window: int) -> list[dict]:
    rows = []
    for scale, regimes in regimes_by_scale.items():
        rows.append(build_result_row(
            stage="covariance_scale",
            covariance_type=covariance_type,
            best_k=best_k,
            min_offdiag_prob=min_offdiag_prob,
            covariance_scale=scale,
            transition_blend=0.0,
            posterior_temperature=config.SHAPE_POSTERIOR_TEMPERATURE,
            regimes=regimes,
            base_states=base_states,
            behavior_features=behavior_features,
            transition_window=transition_window,
        ))
    return rows


def load_shape_data():
    levels = feat.load_and_pivot(config.DATA_PATH)
    shape_features_df, _ = feat.build_shape_features(levels)
    X_shape, _ = feat.scale_for_hmm(shape_features_df)
    return shape_features_df, X_shape


def fit_shape_model(X, covariance_type: str, min_offdiag_prob: float, label: str):
    results = hf.select_k(
        X,
        covariance_type,
        label=label,
        min_offdiag_prob=min_offdiag_prob,
    )
    best_k = results["best_k"]
    return results[best_k]["model"], best_k, results[best_k]


def decode_candidate(model, X, index,
                     covariance_scale: float = 1.0,
                     transition_blend: float = 0.0,
                     posterior_temperature: float = config.SHAPE_POSTERIOR_TEMPERATURE):
    candidate_model = hf.calibrated_model(
        model,
        covariance_scale=covariance_scale,
        transition_blend=transition_blend,
    )
    regimes = hf.decode(candidate_model, X, index=index)
    return hf.apply_posterior_temperature(regimes, posterior_temperature)


def select_best_row(rows: list[dict]) -> dict:
    def churn_for_selection(row):
        return row.get("behavioral_viterbi_churn_pct", row["viterbi_churn_pct"])

    rows_to_rank = [
        row for row in rows
        if churn_for_selection(row) <= config.MAX_ALLOWED_VITERBI_CHURN_PCT
    ] or rows

    def uncertainty_rank(row):
        return (
            row["pct_days_max_prob_gt_99"],
            -row["n_genuinely_uncertain_days"],
            row["worst_transition_min_max_prob"],
            churn_for_selection(row),
        )

    passing = [row for row in rows_to_rank if row["meets_targets"]]
    return min(passing or rows_to_rank, key=uncertainty_rank)


def write_stage(filename: str, rows: list[dict]) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    df.to_csv(os.path.join(OUT_DIR, filename), index=False)
    selected = df[df["selected"]]
    print(f"\n[{filename}]")
    print((selected if not selected.empty else df).to_string(index=False))
    return df


def write_feature_moments(stage: str, features_df: pd.DataFrame,
                          regimes: pd.DataFrame, best_k: int) -> pd.DataFrame:
    stage_dir = os.path.join(OUT_DIR, stage)
    os.makedirs(stage_dir, exist_ok=True)
    moments = ev.regime_feature_moments(features_df, regimes, best_k)
    moments.to_csv(
        os.path.join(stage_dir, "regime_feature_mean_variances.csv"),
        index=False,
    )
    return moments


def mark_selected(rows: list[dict], selected_row: dict) -> list[dict]:
    for row in rows:
        row["selected"] = row is selected_row
    return rows


def run_experiments():
    shape_features_df, X_shape = load_shape_data()
    min_floor = config.MIN_OFFDIAG_PROB

    base_model, base_k, base_info = fit_shape_model(
        X_shape, config.SHAPE_COVARIANCE_TYPE, min_floor, "shape-baseline"
    )
    base_regimes = decode_candidate(base_model, X_shape, shape_features_df.index)
    base_states = base_regimes["viterbi_state"].to_numpy()

    baseline_rows = [build_result_row(
        stage="baseline",
        covariance_type=config.SHAPE_COVARIANCE_TYPE,
        best_k=base_k,
        min_offdiag_prob=min_floor,
        covariance_scale=1.0,
        transition_blend=0.0,
        posterior_temperature=config.SHAPE_POSTERIOR_TEMPERATURE,
        regimes=base_regimes,
        base_states=base_states,
        behavior_features=X_shape,
        transition_window=config.TRANSITION_WINDOW_DAYS,
        log_likelihood=base_info["log_likelihood"],
        bic=base_info["bic"],
        selected=True,
    )]
    write_stage("00_baseline.csv", baseline_rows)

    regimes_by_scale = {
        scale: decode_candidate(base_model, X_shape, shape_features_df.index,
                                covariance_scale=scale)
        for scale in config.SHAPE_COVARIANCE_SCALE_GRID
    }
    scale_rows = covariance_scale_rows(
        regimes_by_scale,
        base_states=base_states,
        behavior_features=X_shape,
        covariance_type=config.SHAPE_COVARIANCE_TYPE,
        best_k=base_k,
        min_offdiag_prob=min_floor,
        transition_window=config.TRANSITION_WINDOW_DAYS,
    )
    best_scale_row = select_best_row(scale_rows)
    write_stage("01_covariance_scale.csv", mark_selected(scale_rows, best_scale_row))
    best_scale = best_scale_row["covariance_scale"]
    write_feature_moments(
        "covariance_scale",
        shape_features_df,
        regimes_by_scale[best_scale],
        base_k,
    )

    type_rows = []
    type_models = {}
    type_regimes = {}
    for cov_type in COVARIANCE_TYPE_GRID:
        model, best_k, info = fit_shape_model(
            X_shape, cov_type, min_floor, f"shape-covariance-{cov_type}"
        )
        type_models[cov_type] = (model, best_k, info)
        regimes = decode_candidate(model, X_shape, shape_features_df.index,
                                   covariance_scale=best_scale)
        type_regimes[cov_type] = regimes
        type_rows.append(build_result_row(
            stage="covariance_type",
            covariance_type=cov_type,
            best_k=best_k,
            min_offdiag_prob=min_floor,
            covariance_scale=best_scale,
            transition_blend=0.0,
            posterior_temperature=config.SHAPE_POSTERIOR_TEMPERATURE,
            regimes=regimes,
            base_states=base_states,
            behavior_features=X_shape,
            transition_window=config.TRANSITION_WINDOW_DAYS,
            log_likelihood=info["log_likelihood"],
            bic=info["bic"],
        ))
    best_type_row = select_best_row(type_rows)
    write_stage("02_covariance_type.csv", mark_selected(type_rows, best_type_row))
    best_cov_type = best_type_row["covariance_type"]
    write_feature_moments(
        "covariance_type",
        shape_features_df,
        type_regimes[best_cov_type],
        int(best_type_row["best_k"]),
    )

    transition_rows = []
    transition_regimes = {}
    for floor in TRANSITION_FLOOR_GRID:
        model, best_k, info = fit_shape_model(
            X_shape, best_cov_type, floor, f"shape-transition-floor-{floor:g}"
        )
        for blend in config.SHAPE_TRANSITION_BLEND_GRID:
            regimes = decode_candidate(
                model,
                X_shape,
                shape_features_df.index,
                covariance_scale=best_scale,
                transition_blend=blend,
            )
            row = build_result_row(
                stage="transition_stickiness",
                covariance_type=best_cov_type,
                best_k=best_k,
                min_offdiag_prob=floor,
                covariance_scale=best_scale,
                transition_blend=blend,
                posterior_temperature=config.SHAPE_POSTERIOR_TEMPERATURE,
                regimes=regimes,
                base_states=base_states,
                behavior_features=X_shape,
                transition_window=config.TRANSITION_WINDOW_DAYS,
                log_likelihood=info["log_likelihood"],
                bic=info["bic"],
            )
            transition_rows.append(row)
            transition_regimes[id(row)] = regimes
    best_transition_row = select_best_row(transition_rows)
    write_stage("03_transition_stickiness.csv",
                mark_selected(transition_rows, best_transition_row))
    write_feature_moments(
        "transition_stickiness",
        shape_features_df,
        transition_regimes[id(best_transition_row)],
        int(best_transition_row["best_k"]),
    )

    return {
        "baseline": baseline_rows[0],
        "covariance_scale": best_scale_row,
        "covariance_type": best_type_row,
        "transition_stickiness": best_transition_row,
    }


if __name__ == "__main__":
    run_experiments()
