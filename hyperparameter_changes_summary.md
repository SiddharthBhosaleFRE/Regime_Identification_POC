# SOFR HMM Hyperparameter Changes Summary

## Hyperparameter Table (everything tested so far)

| Hyperparameter | First baseline | Later/current values | Fitting or reporting? | Reason for change | Observed effect |
|---|---:|---:|---|---|---|
| `K_VALUES` / state count | Earliest available configs use `[2,3,4,5]`; v1 exact CSV not present | v2/v3 default `[2,3,4,5]`; one-offs fixed `K=4` and `K=6` | Fitting/model selection | Compare interpretability and density fit across state counts | v3 `K=4`: BIC `4429.2`, 4 transitions. v3 `K=6`: BIC `1255.2`, 5 transitions. Default grid picks `K=5` in many runs. |
| `N_EM_RESTARTS` | v2/v3 config default `20` | one-offs commonly `10`; fast v2 run used `5` | Fitting | Faster iteration while retaining enough restart search | v2 fast `5` restarts produced usable high-uncertainty run; v3 one-offs mostly use `10`. `20` remains config default. |
| `covariance_type` | v1 level model kept as `full`; v2 comments say shape switched to `diag` as a fix | shape default `diag`; experiments tested `diag`, `tied`, `full` | Fitting | Avoid overfit/overconfident shape HMM and improve regime stability | uncertainty sweep selected `diag`; `tied` and `full` had high churn in `outputs/uncertainty_experiments/02_covariance_type.csv`. |
| `transmat_prior` | one-off/default runner value `1.0` | tested `2, 2.5, 3, 4, 5, 10, 20`; current v3 one-offs use `10` | Fitting | Regularize transition matrix away from pathological/extreme transitions | v2 summary: `1.0` had 6 transitions; `10.0` had 8 transitions and much higher churn. In v3 with revised features, `10` has been stable and interpretable. |
| `min_covar` | `0.001` | tested `0.01`, `0.05`, `0.1` | Fitting | Check whether emission variance floor could soften overconfidence | v2 summary: `0.01` had no material change; `0.05`/`0.1` changed regime map heavily. Not a preferred lever. |
| `min_offdiag_prob` | absent or not recorded for v1; v2/v3 default `1e-4` | uncertainty sweep tested `1e-4`, `0.001`, `0.01`, `0.02`, `0.05` | Post-fit transition regularization affecting decoding/reporting | Prevent absorbing transition rows; code comment says v1 had `a_kk=1.0` absorbing state | `0.05` plus `transition_blend=0.10` made posteriors very soft: `4.4% > 0.99`, 1506 uncertain days. |
| `covariance_scale` | reporting default `1.0` | tested `2,4,6,8,10`; v3 current comparison center is `8` | Reporting/posterior calibration | Soften posterior probabilities without refitting HMM | v2 uncertainty sweep selected `10`; v3 `scale=8,temp=2` gave BIC `3067.3`, 4 transitions, mean entropy `0.293`. |
| `transition_blend` | `0.0` | tested `0.025`, `0.05`, `0.10` | Reporting/posterior calibration | Blend transition rows toward uniform to reduce posterior certainty | With `covariance_scale=10`, `blend=0.10`, `min_offdiag=0.05`: very soft posterior, but this is not current preferred setup. |
| `posterior_temperature` | v1 effectively uncalibrated; v2/v3 default `2.0` | tested `2.5`, `3.0`; grid also includes `1.0,1.5,2.0,3.0,4.0` | Reporting/posterior calibration | Soften filtered/smoothed probabilities while preserving Viterbi path | v3 `scale=8`: temp `2.5` had mean entropy `0.410`; temp `3.0` had `0.5173` and only `0.1% > 95`, likely too soft. |
| `transition_window_days` | not present in v1 evidence | v2/v3 default `3` | Evaluation/reporting | Measure uncertainty around regime transitions | Used for `worst_transition_min_max_prob`; does not change model. |
| `feature_transform` | standard scaling | `standard`; tested `uniform_platykurtic` | Fitting transform | Reduce heavy tails / distribution shape effects | v2 platykurtic runs exist; not promoted as current default. |
| train/test split | none | one-off runner supports `test_size_pct` and `split_date`; smoke test used 20% | Fitting/evaluation protocol | Enable holdout scoring | `outputs/smoke_train_test_split/model_config.csv` records `test_size_pct=20`, `K=2`, `uniform_platykurtic`; this is a smoke artifact, not current model choice. |
| feature set/version | v2 six-feature shape set with `level_vol`; v3 four-feature set | v3 replaces `level_vol` with `level_abs_daily_move_90d_mean`; tested `slope_2y10y` | Feature definition, affects fitting | Improve interpretability and reduce weak/noisy feature effects | v3 movement feature improved BIC and transitions versus prior `level_vol`; `slope_2y10y` was not promoted. |

## The three most effective hyperparameter changes: transmat_prior, covariance_scale, and posterior_temperature

### `transmat_prior`

`transmat_prior` is a fitting-time transition-matrix prior passed into `hmmlearn.GaussianHMM`.

Mathematically, each transition row is estimated like a probability vector. The prior acts like pseudo-counts added to transition probabilities during EM. A larger value pulls transition probabilities away from extreme maximum-likelihood estimates. In this project, that matters because unconstrained or weakly regularized HMM fits can produce transition rows with near-zero off-diagonal probabilities or near-absorbing states.

Intuitively:

- Low `transmat_prior` lets the data drive transition probabilities more aggressively.
- Higher `transmat_prior` discourages extreme transition probabilities.
- It can prevent pathological transition matrices, but it can also change the fitted state map if pushed too far.

Why it was changed:

- The first baseline evidence showed an absorbing-state problem.
- v2 introduced transition regularization work to avoid brittle transition matrices.
- v3 one-offs moved toward `transmat_prior=10` because, with the revised v3 feature set, it gave stable and interpretable regime paths.

Observed project behavior:

- In v2 transition-prior experiments, increasing `transmat_prior` from `1.0` to `10.0` increased uncertainty but also caused large Viterbi churn versus the v2 baseline.
- In current v3 one-offs, `transmat_prior=10` is less disruptive and has been a useful working setting.
- It changes the fitted HMM, so BIC/log likelihood and Viterbi assignments can change.

## Posterior Calibration

`covariance_scale` and `posterior_temperature` are reporting/posterior calibration parameters. They do not change the fitted HMM likelihood/BIC in the same way as fitting knobs. Current evidence says `posterior_temperature=3.0` is too soft; `2.0` is the best working default, with `2.5` as an exploratory middle ground.

### `covariance_scale`

`covariance_scale` is a reporting-time calibration parameter. It is applied after fitting by copying the fitted HMM and multiplying the emission covariance matrices by the scale.

Mathematically, for each state emission distribution:

```text
original:   x | state k ~ Normal(mean_k, covariance_k)
calibrated: x | state k ~ Normal(mean_k, covariance_scale * covariance_k)
```

The state means do not move. The fitted model used for model selection is not retrained. The calibrated copy simply treats each state's emission distribution as wider when computing posterior probabilities.

Intuitively:

- Larger `covariance_scale` makes each regime less narrowly confident about which observations belong to it.
- This softens filtered probabilities, especially near boundaries between regimes.
- The Viterbi path may remain the same, but posterior certainty becomes less binary.

Why it was changed:

- The original and early v2 shape HMMs were too confident: most days had max filtered probability above `0.99`.
- The goal was not necessarily to change the regime timeline, but to make posterior probabilities better reflect uncertainty.
- Covariance scaling gave a clean way to soften posteriors without changing the trained state means or retraining the model.

Observed project behavior:

- In v2 uncertainty experiments, higher covariance scales produced many more uncertain days.
- `covariance_scale=10` was selected in the v2 uncertainty sweep, but that was for the v2 feature set and calibration target.
- In current v3 runs, `covariance_scale=8` with `posterior_temperature=2.0` is a stronger working candidate than pushing all the way to very soft settings.
- It is reporting calibration, not a fitting hyperparameter. It should not be interpreted as improving BIC.

### `posterior_temperature`

`posterior_temperature` is another reporting-time probability calibration parameter. It is applied to already-decoded filtered/smoothed probabilities.

Mathematically, the probability vector for a day is softened by raising probabilities to `1 / temperature` and renormalizing:

```text
p_calibrated(k) = p(k)^(1 / T) / sum_j p(j)^(1 / T)
```

where `T` is `posterior_temperature`.

When `T = 1`, probabilities are unchanged. When `T > 1`, large probabilities are pulled down and small probabilities are lifted up. The state order is preserved for each day, so the most likely state remains the most likely state unless there are numerical ties.

Intuitively:

- Temperature is like turning down overconfidence in the probability display.
- It does not move regime boundaries by itself.
- Too much temperature makes the model indecisive everywhere, which is not useful for regime interpretation.

Why it was changed:

- Early shape-model probabilities were effectively hard classifications.
- `posterior_temperature=2.0` softened filtered probabilities while preserving Viterbi assignments.
- Later one-offs tested `2.5` and `3.0` to see whether more uncertainty was useful.

Observed project behavior:

- `posterior_temperature=2.0` is the current default in v2/v3 config and remains the best working default.
- At v3 `covariance_scale=8`, `posterior_temperature=2.5` made posteriors much softer: mean entropy rose to about `0.410`.
- At v3 `covariance_scale=8`, `posterior_temperature=3.0` was likely too much: only about `0.1%` of days had max filtered probability above `95%`.
- It is reporting calibration. It does not refit the HMM and does not change BIC/log likelihood.

## State-Count Experiments

The current v3 exact-state tests are:

- `outputs/one_off_experiments/v3/k4_restarts_10_transmat_prior_10_covariance_scale_8_posterior_temperature_2`
- `outputs/one_off_experiments/v3/k6_restarts_10_transmat_prior_10_covariance_scale_8_posterior_temperature_2`

`K=6` has much better BIC than `K=4`, but the practical question is whether the sixth regime is interpretable rather than just a finer in-sample partition.

## EM Restarts

Config default remains `20`. Most current v3 one-offs use `10` for faster iteration.
