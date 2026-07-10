# V3 One-Off HMM And Viewer Usage

This guide explains how to run v3 SOFR shape HMM one-off experiments and render the interactive curve-regime viewer.

Run commands from the project root.

Use Python 3.13 for HMM work:

```powershell
py -3.13 --version
```

## Run A Standard V3 One-Off

Basic command:

```powershell
py -3.13 scripts\run_v3_hmm_oneoff.py --n-em-restarts 10 --transmat-prior 10 --covariance-scale 8
```

This uses the current v3 default feature set:

```text
level
term_slope
butterfly
level_abs_daily_move_90d_mean
```

Default output folder pattern:

```text
outputs\one_off_experiments\v3\restarts_{N}_transmat_prior_{P}_covariance_scale_{S}
```

For the example above:

```text
outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8
```

The script refuses to overwrite an existing folder unless `--allow-overwrite` is passed.

## Use An Explicit Output Folder

Use `--output-folder` when a setting is not included in the default folder name, such as `posterior_temperature`.

Example:

```powershell
py -3.13 scripts\run_v3_hmm_oneoff.py --n-em-restarts 10 --transmat-prior 10 --covariance-scale 8 --posterior-temperature 2.5 --output-folder restarts_10_transmat_prior_10_covariance_scale_8_posterior_temperature_2_5
```

This prevents accidental collisions with the default covariance-scale-8 folder.

## Important Runner Options

Common fitting/model-selection options:

```text
--k-values
--n-em-restarts
--max-iter
--random-seed
--covariance-type
--transmat-prior
--min-covar
--min-offdiag-prob
```

Common posterior/reporting calibration options:

```text
--covariance-scale
--transition-blend
--posterior-temperature
--transition-window-days
```

Feature options:

```text
--shape-feature-set v3
--shape-feature-set slope_2y10y
```

The default is:

```text
--shape-feature-set v3
```

The `slope_2y10y` variant replaces `term_slope` with `slope_2y10y`.

## Fixed-State Runs

To force exactly 4 regimes:

```powershell
py -3.13 scripts\run_v3_hmm_oneoff.py --k-values 4 --n-em-restarts 10 --transmat-prior 10 --covariance-scale 8 --posterior-temperature 2.0 --output-folder k4_restarts_10_transmat_prior_10_covariance_scale_8_posterior_temperature_2
```

To force exactly 6 regimes:

```powershell
py -3.13 scripts\run_v3_hmm_oneoff.py --k-values 6 --n-em-restarts 10 --transmat-prior 10 --covariance-scale 8 --posterior-temperature 2.0 --output-folder k6_restarts_10_transmat_prior_10_covariance_scale_8_posterior_temperature_2
```

## Output Files

Each one-off output folder contains files like:

```text
model_config.csv
regime_assignments.csv
regime_feature_means.csv
regime_feature_mean_variances.csv
regime_durations.csv
filtered_diagnostics.csv
transition_diagnostics.csv
transition_matrix.csv
features.csv
bic_plot.png
regime_timeline.png
feature_dists.png
```

The most important files are:

- `model_config.csv`: records settings and headline diagnostics.
- `regime_assignments.csv`: Viterbi states and posterior probabilities by date.
- `features.csv`: exact feature matrix used for the run.
- `transition_diagnostics.csv`: uncertainty around state transitions.

## Render A Viewer

Use the run's `regime_assignments.csv` and run-local `features.csv`.

One-line command example:

```powershell
py -3.13 scripts\render_interactive_curve_regime_viewer.py --curve-data Target_Zeros.csv --regime-assignments "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\regime_assignments.csv" --observed-rates Feature_Set.csv --shape-features "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\features.csv" --title "SOFR yield curve with v3 shape HMM regimes"
```

Default viewer output:

```text
outputs\one_off_experiments\v3\...\interactive_curve_regime_viewer.html
```

If the viewer already exists, either choose a new output path:

```powershell
py -3.13 scripts\render_interactive_curve_regime_viewer.py --curve-data Target_Zeros.csv --regime-assignments "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\regime_assignments.csv" --observed-rates Feature_Set.csv --shape-features "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\features.csv" --output-html "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\viewer_retry.html" --title "SOFR yield curve with v3 shape HMM regimes"
```

or explicitly overwrite:

```powershell
py -3.13 scripts\render_interactive_curve_regime_viewer.py --curve-data Target_Zeros.csv --regime-assignments "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\regime_assignments.csv" --observed-rates Feature_Set.csv --shape-features "outputs\one_off_experiments\v3\restarts_10_transmat_prior_10_covariance_scale_8\features.csv" --title "SOFR yield curve with v3 shape HMM regimes" --allow-overwrite
```

## PowerShell Line Continuation

PowerShell uses a backtick for line continuation. The backtick must be the final character on the line.

This works:

```powershell
py -3.13 scripts\run_v3_hmm_oneoff.py `
  --n-em-restarts 10 `
  --transmat-prior 10 `
  --covariance-scale 8
```

This often fails if there is any invisible whitespace after the backtick. If copying from chat causes argument errors, use the one-line command instead.

## Recommended Current Working Settings

Current practical default for v3 comparison work:

```text
n_em_restarts = 10
transmat_prior = 10
covariance_scale = 8
posterior_temperature = 2.0
feature_transform = standard
shape_feature_set = v3
```

Use `n_em_restarts=20` for a slower final confirmation run.

Treat these as experimental:

```text
posterior_temperature = 2.5
posterior_temperature = 3.0
fixed K=6
shape_feature_set = slope_2y10y
uniform_platykurtic feature transform
transition_blend > 0
```

## Regenerate Root V3 Feature CSV

The root `shape_features.csv` can be regenerated from the v3 feature builder:

```powershell
py -3.13 scripts\export_shape_features.py --data-path Target_Zeros.csv --output-csv shape_features.csv --allow-overwrite
```

Expected columns:

```text
date
level
term_slope
butterfly
level_abs_daily_move_90d_mean
```
