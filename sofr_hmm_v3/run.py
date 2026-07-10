# run.py - v3: runs LEVEL model (v1, kept for comparison) and revised SHAPE
# model side by side. All outputs prefixed level_* / shape_*.

import matplotlib
matplotlib.use("Agg")

import pandas as pd

import config
import features as feat
import hmm_fit as hf
import evaluate as ev


def _apply_shape_reporting_calibration(best_model, base_regimes):
    base_states = base_regimes["viterbi_state"].to_numpy()
    regimes = hf.apply_posterior_temperature(
        base_regimes, temperature=config.SHAPE_POSTERIOR_TEMPERATURE
    )
    metrics = hf.calibration_metrics(
        regimes,
        base_states=base_states,
        transition_window=config.TRANSITION_WINDOW_DAYS,
    )
    summary = {
        "meets_targets": (
            metrics["worst_transition_min_max_prob"] <= config.TARGET_MAX_PROB_AT_TRANSITION
            and metrics["pct_days_max_prob_gt_99"] <= config.TARGET_PCT_DAYS_MAX_PROB_GT_99
            and metrics["viterbi_churn_pct"] <= config.MAX_ALLOWED_VITERBI_CHURN_PCT
        ),
        "covariance_scale": 1.0,
        "transition_blend": 0.0,
        "posterior_temperature": config.SHAPE_POSTERIOR_TEMPERATURE,
        **metrics,
    }
    summary_df = pd.DataFrame([summary])
    return best_model, regimes, summary_df


def run_one_model(label, features_df, X, covariance_type, calibrate_posteriors=False):
    print(f"\n{'='*60}\n{label.upper()} MODEL\n{'='*60}")

    results = hf.select_k(X, covariance_type, label=label)
    best_k = results["best_k"]
    best_model = results[best_k]["model"]

    ev.plot_bic(results, title=f"{label} model", save_path=f"{label}_bic_plot.png")

    regimes = hf.decode(best_model, X, index=features_df.index)
    calibration_summary = None
    if calibrate_posteriors:
        best_model, regimes, calibration_summary = (
            _apply_shape_reporting_calibration(best_model, regimes)
        )
        print("\n--- shape: posterior reporting calibration ---")
        print(calibration_summary.to_string(index=False))

    print(f"\n--- {label}: transition matrix (K={best_k}) ---")
    tmat = ev.transition_table(best_model)
    print(tmat.to_string())

    print(f"\n--- {label}: regime durations ---")
    dur = hf.regime_duration_summary(regimes, best_k)
    print(dur.to_string(index=False))

    print(f"\n--- {label}: regime-conditional feature means ---")
    stats = ev.regime_stats(features_df, regimes, best_k)
    print(stats.to_string())
    moments = ev.regime_feature_moments(features_df, regimes, best_k)

    print(f"\n--- {label}: occupancy ---")
    counts = regimes["viterbi_state"].value_counts().sort_index()
    for k, c in counts.items():
        print(f"  State {k+1}: {c:>5} days  ({c/len(regimes)*100:.1f}%)")

    print(f"\n--- {label}: filtered probability diagnostics ---")
    diag = ev.filtered_prob_diagnostics(regimes, best_k)
    print(diag.to_string(index=False))
    transition_diag = None
    if calibrate_posteriors:
        print(f"\n--- {label}: transition-window diagnostics ---")
        transition_diag = ev.transition_window_diagnostics(
            regimes, best_k, window=config.TRANSITION_WINDOW_DAYS
        )
        print(transition_diag.to_string(index=False))

    ev.plot_regime_timeline(regimes, features_df, best_k,
                            title=f"{label.capitalize()} model — filtered probabilities",
                            save_path=f"{label}_regime_timeline.png")
    ev.plot_feature_distributions(features_df, regimes, best_k,
                                  title=f"{label.capitalize()} model — feature distributions",
                                  save_path=f"{label}_feature_dists.png")

    regimes.to_csv(f"{label}_regime_assignments.csv")
    stats.to_csv(f"{label}_regime_feature_means.csv")
    moments.to_csv(f"{label}_regime_feature_mean_variances.csv", index=False)
    tmat.to_csv(f"{label}_transition_matrix.csv")
    dur.to_csv(f"{label}_regime_durations.csv", index=False)
    diag.to_csv(f"{label}_filtered_diagnostics.csv", index=False)
    if calibration_summary is not None:
        calibration_summary.to_csv(f"{label}_calibration_summary.csv", index=False)
    if transition_diag is not None:
        transition_diag.to_csv(f"{label}_transition_diagnostics.csv", index=False)

    print(f"\n  Saved: {label}_*.csv, {label}_*.png")
    return regimes, best_model, results, diag


def main():
    print("=" * 60)
    print("SOFR Regime Identification - v3")
    print("Level model (v1, kept) vs. revised shape model")
    print("=" * 60)

    # Load raw curve once, share between both feature builders
    print("\n[1/3] Loading and splining curve data...")
    levels = feat.load_and_pivot(config.DATA_PATH)

    print("\n[2/3] Building feature sets...")
    level_features_df, level_meta = feat.build_level_features(levels)
    X_level, level_scaler = feat.scale_for_hmm(level_features_df)

    shape_features_df, shape_meta = feat.build_shape_features(levels)
    X_shape, shape_scaler = feat.scale_for_hmm(shape_features_df)

    print("\n[3/3] Fitting both models...")
    level_regimes, level_model, level_results, level_diag = run_one_model(
        "level", level_features_df, X_level, config.LEVEL_COVARIANCE_TYPE
    )
    shape_regimes, shape_model, shape_results, shape_diag = run_one_model(
        "shape", shape_features_df, X_shape, config.SHAPE_COVARIANCE_TYPE,
        calibrate_posteriors=config.SHAPE_CALIBRATE_POSTERIORS
    )

    # ----------------------------------------------------------------
    # SIDE-BY-SIDE COMPARISON
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LEVEL vs SHAPE — direct comparison")
    print("=" * 60)

    comparison = pd.DataFrame({
        "level_model": level_diag.iloc[0],
        "shape_model": shape_diag.iloc[0],
    })
    print("\nFiltered probability behaviour:")
    print(comparison.to_string())

    print(f"\nBest K — level: {level_results['best_k']}   shape: {shape_results['best_k']}")

    # Cross-tab: how often does each level-state coincide with each shape-state?
    merged = level_regimes[["viterbi_state"]].rename(columns={"viterbi_state": "level_state"})
    merged = merged.join(
        shape_regimes[["viterbi_state"]].rename(columns={"viterbi_state": "shape_state"})
    )
    crosstab = pd.crosstab(merged["level_state"] + 1, merged["shape_state"] + 1)
    crosstab.index.name = "level_state"
    crosstab.columns.name = "shape_state"
    print("\nCross-tab — level state vs shape state (day counts):")
    print(crosstab.to_string())
    crosstab.to_csv("level_vs_shape_crosstab.csv")

    print("\nSaved: level_vs_shape_crosstab.csv")
    print("\nDone. Compare level_*.png against shape_*.png for the visual story.")

    return {
        "level": (level_regimes, level_model, level_results),
        "shape": (shape_regimes, shape_model, shape_results),
        "comparison": comparison,
        "crosstab": crosstab,
    }


if __name__ == "__main__":
    out = main()
