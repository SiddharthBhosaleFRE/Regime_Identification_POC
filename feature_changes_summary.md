# SOFR HMM Feature Changes Summary

Important:

- HMM feature matrices are standardized before fitting, so raw units do not directly determine feature weight.
- The HMM does not learn linear feature weights. Feature influence has been estimated with between-state separation and between/within-state signal proxies.

## Feature Change Table

| Feature or feature set | First baseline definition | Later/current definition | Status | Reason for change | Observed effect |
|---|---|---|---|---|---|
| `level_pc1`, `level_pc2`, `level_pc3` | PCA scores from standardized full zero curve levels | Still available in `build_level_features()` for level-model comparison | Tested baseline / kept for comparison | `level_pc1` captured about 96% of variance, making regimes mostly rate-level buckets | Useful baseline, but too dominated by broad rate cycle |
| `change_pc1`, `change_pc2` | PCA scores from daily curve changes in level model | v2 shape used only `change_pc1`; v3 removed it from shape | Tested then removed from v3 shape | Short-horizon change features were noisy and less interpretable | v3 became cleaner and lower-dimensional |
| `level_vol` | 21-day rolling std of `level_pc1.diff()` | Replaced by `level_abs_daily_move_90d_mean` | Tested then replaced in v3 | PCA-specific, weak state driver, less interpretable | Prior baseline signal share only about 1.4% by between/within proxy |
| `level` | Mean rate across curve, added in v2 shape | Still current v3 feature | Tested and kept | Captures rate environment directly | Strongest state driver; risk is level-dominated regimes |
| `slope_short` | `2y - 3m` | Removed from v3 shape | Tested then removed | Redundant front-end/policy-sensitive slope signal | Lower v3 dimensionality; less slope duplication |
| `term_slope` | `10y - 3m` | Still current v3 default slope | Tested and kept | Broad curve steepness / inversion signal | Strong state driver; correlated with `butterfly` |
| `butterfly` | `5y - 0.5 * (3m + 10y)` | Still current v3 curvature feature | Tested and kept | Interpretable belly curvature measure | Useful but correlated with slope under diagonal covariance |
| `level_abs_daily_move_90d_mean` | Not in v1/v2 | `level.diff().abs().rolling(90).mean()` | Tested and promoted to v3 default | Smoother, direct quarterly movement intensity | Improved BIC and transitions versus `level_vol`; signal share about 7.5% |
| `slope_2y10y` | Not baseline | `10y - 2y` | Tested only | Reduce front-end policy-rate dominance and slope/curvature redundancy | Reduced slope/butterfly correlation, but worsened BIC and restored level dominance |
| `weekly_delta_level` | Not baseline | `level.diff(5)` candidate | Explored in candidate plots | Capture short-horizon rate-cycle direction | User/plan notes raw short-horizon deltas looked noisy |
| `weekly_delta_slope_short` | Not baseline | `slope_short.diff(5)` in `scripts/plot_candidate_shape_features.py` | Explored in candidate plots | Early short-slope momentum candidate | Not current HMM input |

## PCA Level/Change Features Versus Explicit Curve-Shape Features

The level-model PCA features are still present in code as a side-by-side comparison path. The shape-model direction moved away from PCA because `level_pc1` mostly measured the broad 2020-2026 rate cycle. v2 introduced explicit curve-shape features so `level` became one HMM feature rather than the dominant PCA axis.

## Removal Of `slope_short` And `change_pc1`

v3 removed `slope_short` and `change_pc1` from the shape feature set.

Reasoning:

- `slope_short = 2y - 3m` overlapped with front-end policy effects already present in `term_slope`.
- `change_pc1` was a noisy daily-change feature and less directly interpretable.
- Removing both reduced the shape model from six features to four.

Current v3 shape columns:

```text
level
term_slope
butterfly
level_abs_daily_move_90d_mean
```

## Replacement Of `level_vol` With `level_abs_daily_move_90d_mean`

The key v3 replacement is:

```text
level_abs_daily_move_90d_mean = level.diff().abs().rolling(90).mean()
```

This replaced PCA-derived `level_vol`.

The replacement is more interpretable:

- `level_vol`: 21-day rolling volatility of PCA level-factor changes.
- `level_abs_daily_move_90d_mean`: 90-day average absolute daily move in direct curve level.

The plan records that this change improved BIC from `3793.3` to `3067.3`, reduced Viterbi transitions from `7` to `4`, and increased the movement feature's between/within signal share to about `7.5%`.

## `term_slope` Versus `slope_2y10y`

`term_slope = 10y - 3m` remains the current default.

`slope_2y10y = 10y - 2y` was tested to reduce 3m/front-end dominance. It reduced correlation with `butterfly`, but the plan records worse BIC and stronger level dominance.

## `butterfly` And Slope/Curvature Redundancy

`butterfly = 5y - 0.5 * (3m + 10y)` remains the current curvature feature.

It is economically interpretable, but it is correlated with `term_slope`. Because the shape HMM uses diagonal covariance, correlated features can effectively double-count similar curve-shape movement.

Candidate future fixes:

- residualize `butterfly` against the selected slope feature,
- test `fly_2y5y10y = 5y - 0.5 * (2y + 10y)`,
- compare long-end slope variants before changing curvature.

## Candidate Momentum And Movement-Intensity Features

Candidate features mentioned in the plan or plotting script:

- `weekly_delta_level`
- `weekly_delta_term_slope`
- `weekly_delta_slope_short`
- `term_slope_abs_daily_move_90d_mean`
- `butterfly_abs_daily_move_90d_mean`

The short-horizon weekly deltas are not current defaults because raw/smoothed short-horizon deltas looked noisy. The next more promising direction is to compare 90-day movement-intensity variants for `term_slope` and `butterfly` against the current level movement-intensity feature.
