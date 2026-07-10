# hmm_fit.py  —  v2: fit, select K, decode, with transition-matrix flooring
# to prevent absorbing states (v1's S1, a_11=1.0 problem)

import copy
import itertools

import numpy as np
import pandas as pd
from hmmlearn import hmm

import config


# -----------------------------------------------------------------------
# 1. FIT ONE MODEL
# -----------------------------------------------------------------------

def fit_hmm(X: np.ndarray, K: int, covariance_type: str, seed: int = 0,
            transmat_prior: float = 1.0, min_covar: float = 0.001):
    model = hmm.GaussianHMM(
        n_components=K,
        covariance_type=covariance_type,
        min_covar=min_covar,
        transmat_prior=transmat_prior,
        n_iter=config.MAX_ITER,
        tol=1e-4,
        random_state=seed,
        init_params="stmc",
        params="stmc",
    )
    try:
        model.fit(X)
        return model
    except Exception as e:
        print(f"  [warn] K={K} seed={seed} failed: {e}")
        return None


# -----------------------------------------------------------------------
# 2. FIT WITH MULTIPLE RESTARTS
# -----------------------------------------------------------------------

def fit_best(X: np.ndarray, K: int, covariance_type: str,
             transmat_prior: float = 1.0, min_covar: float = 0.001):
    best_model, best_ll = None, -np.inf
    for seed in range(config.N_EM_RESTARTS):
        model = fit_hmm(
            X,
            K,
            covariance_type,
            seed=config.RANDOM_SEED + seed,
            transmat_prior=transmat_prior,
            min_covar=min_covar,
        )
        if model is None:
            continue
        ll = model.score(X)
        if ll > best_ll:
            best_ll, best_model = ll, model
    if best_model is None:
        raise RuntimeError(f"All restarts failed for K={K}")
    print(f"  K={K}  best log-likelihood: {best_ll:.2f}")
    return best_model, best_ll


# -----------------------------------------------------------------------
# 3. TRANSITION MATRIX FLOOR  — fixes v1's absorbing-state problem
# -----------------------------------------------------------------------

def floor_transition_matrix(model: hmm.GaussianHMM, min_offdiag: float = config.MIN_OFFDIAG_PROB):
    """
    v1 produced a state with a_kk = 1.0000 (fully absorbing). Once entered,
    the filtered probability for that state can never decrease — it's stuck
    at 1.0 regardless of future data. This is a real defect for any live use.

    Fix: floor every off-diagonal transition probability at min_offdiag,
    then renormalise each row to sum to 1. This guarantees every state has
    a (small but nonzero) escape probability, without materially changing
    a well-estimated transition matrix — it only bites when EM has driven
    a probability all the way to numerical zero.
    """
    A = model.transmat_.copy()
    K = A.shape[0]
    n_floored = 0

    for i in range(K):
        off_diag = [j for j in range(K) if j != i]
        for j in off_diag:
            if A[i, j] < min_offdiag:
                A[i, j] = min_offdiag
                n_floored += 1
        A[i] = A[i] / A[i].sum()   # renormalise row

    if n_floored:
        print(f"  [floor] {n_floored} transition probabilities floored "
              f"at {min_offdiag:.0e} (was at/near 0)")

    model.transmat_ = A
    return model


def blend_transition_matrix(A: np.ndarray, blend_weight: float) -> np.ndarray:
    """Blend transition rows toward uniform probabilities and renormalise."""
    if blend_weight < 0 or blend_weight > 1:
        raise ValueError("blend_weight must be between 0 and 1")

    K = A.shape[0]
    uniform = np.full_like(A, 1.0 / K, dtype=float)
    blended = (1.0 - blend_weight) * A + blend_weight * uniform
    blended = np.clip(blended, 0.0, None)
    row_sums = blended.sum(axis=1, keepdims=True)
    return np.divide(blended, row_sums, out=np.zeros_like(blended),
                     where=row_sums > 0)


def calibrated_model(model: hmm.GaussianHMM,
                     covariance_scale: float = 1.0,
                     transition_blend: float = 0.0):
    """
    Return a copied HMM with softer emissions/transitions for posterior use.

    The fitted model is left untouched; calibration affects decoded/reported
    probabilities and optionally Viterbi paths computed from this copy.
    """
    if covariance_scale <= 0:
        raise ValueError("covariance_scale must be positive")

    calibrated = copy.deepcopy(model)
    if hasattr(calibrated, "_covars_"):
        calibrated._covars_ = calibrated._covars_ * covariance_scale
    else:
        calibrated.covars_ = calibrated.covars_ * covariance_scale
    calibrated.transmat_ = blend_transition_matrix(calibrated.transmat_,
                                                   transition_blend)
    return calibrated


def apply_posterior_temperature(regimes: pd.DataFrame,
                                temperature: float = 1.0) -> pd.DataFrame:
    """Soften posterior probability columns while preserving their ranking."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    out = regimes.copy()
    for prefix in ("filtered_p", "smoothed_p"):
        cols = [c for c in out.columns if c.startswith(prefix)]
        if not cols:
            continue
        vals = out[cols].clip(1e-15, 1).to_numpy(dtype=float)
        vals = vals ** (1.0 / temperature)
        vals = vals / vals.sum(axis=1, keepdims=True)
        out.loc[:, cols] = vals
    return out


def state_feature_alignment(candidate_states: np.ndarray,
                            base_states: np.ndarray,
                            behavior_features: np.ndarray | None) -> dict:
    candidate_states = np.asarray(candidate_states)
    base_states = np.asarray(base_states)
    raw_mapping = {int(state): int(state) for state in np.unique(candidate_states)}

    if behavior_features is None:
        return raw_mapping

    features = np.asarray(behavior_features, dtype=float)
    if len(features) != len(candidate_states) or len(features) != len(base_states):
        raise ValueError("behavior_features must align with candidate and base states")

    candidate_labels = sorted(int(s) for s in np.unique(candidate_states))
    base_labels = sorted(int(s) for s in np.unique(base_states))
    candidate_centroids = {
        state: features[candidate_states == state].mean(axis=0)
        for state in candidate_labels
    }
    base_centroids = {
        state: features[base_states == state].mean(axis=0)
        for state in base_labels
    }

    distances = np.array([
        [
            np.linalg.norm(candidate_centroids[candidate] - base_centroids[base])
            for base in base_labels
        ]
        for candidate in candidate_labels
    ])

    if len(candidate_labels) <= len(base_labels):
        best_perm = None
        best_cost = np.inf
        for perm in itertools.permutations(range(len(base_labels)), len(candidate_labels)):
            cost = sum(distances[i, j] for i, j in enumerate(perm))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        return {
            candidate: base_labels[best_perm[i]]
            for i, candidate in enumerate(candidate_labels)
        }

    nearest = distances.argmin(axis=1)
    return {
        candidate: base_labels[int(nearest[i])]
        for i, candidate in enumerate(candidate_labels)
    }


def _alignment_label(mapping: dict) -> str:
    return "; ".join(
        f"S{candidate + 1}->S{base + 1}"
        for candidate, base in sorted(mapping.items())
    )


def calibration_metrics(regimes: pd.DataFrame,
                        base_states: np.ndarray,
                        transition_window: int,
                        behavior_features: np.ndarray | None = None) -> dict:
    fp_cols = [c for c in regimes.columns if c.startswith("filtered_p")]
    max_fp = regimes[fp_cols].max(axis=1)
    probs = regimes[fp_cols].clip(1e-15, 1)
    entropy = -(probs * np.log(probs)).sum(axis=1)

    states = regimes["viterbi_state"].to_numpy()
    change_idx = np.flatnonzero(states[1:] != states[:-1]) + 1
    transition_mins = []
    transition_entropies = []
    for idx in change_idx:
        lo = max(0, idx - transition_window)
        hi = min(len(regimes), idx + transition_window + 1)
        transition_mins.append(float(max_fp.iloc[lo:hi].min()))
        transition_entropies.append(float(entropy.iloc[lo:hi].max()))

    churn = float((states != base_states).mean() * 100)
    alignment = state_feature_alignment(states, base_states, behavior_features)
    aligned_states = np.array([alignment[int(state)] for state in states])
    behavioral_churn = float((aligned_states != base_states).mean() * 100)
    return {
        "pct_days_max_prob_gt_99": float((max_fp > 0.99).mean() * 100),
        "n_genuinely_uncertain_days": int((max_fp < 0.99).sum()),
        "mean_entropy": float(entropy.mean()),
        "n_transitions": int(len(change_idx)),
        "worst_transition_min_max_prob": (
            max(transition_mins) if transition_mins else float(max_fp.max())
        ),
        "best_transition_max_entropy": (
            min(transition_entropies) if transition_entropies else float(entropy.max())
        ),
        "viterbi_churn_pct": churn,
        "behavioral_viterbi_churn_pct": behavioral_churn,
        "state_alignment": _alignment_label(alignment),
    }


def select_calibration_candidate(candidates: list[dict],
                                 base_states: np.ndarray,
                                 transition_window: int,
                                 target_transition_max_prob: float,
                                 target_pct_days_max_prob_gt_99: float,
                                 max_allowed_churn_pct: float,
                                 behavior_features: np.ndarray | None = None) -> tuple[dict, dict]:
    scored = []
    for candidate in candidates:
        metrics = calibration_metrics(candidate["regimes"], base_states,
                                      transition_window, behavior_features)
        meets_targets = (
            metrics["worst_transition_min_max_prob"] <= target_transition_max_prob
            and metrics["pct_days_max_prob_gt_99"] <= target_pct_days_max_prob_gt_99
            and metrics["behavioral_viterbi_churn_pct"] <= max_allowed_churn_pct
        )
        scored.append((candidate, metrics, meets_targets))
        if meets_targets:
            summary = {
                "meets_targets": True,
                "covariance_scale": candidate["covariance_scale"],
                "transition_blend": candidate["transition_blend"],
                "posterior_temperature": candidate.get("posterior_temperature", 1.0),
                **metrics,
            }
            return candidate, summary

    def penalty(item):
        _, metrics, _ = item
        return (
            max(0.0, metrics["worst_transition_min_max_prob"] -
                target_transition_max_prob),
            max(0.0, metrics["pct_days_max_prob_gt_99"] -
                target_pct_days_max_prob_gt_99),
            max(0.0, metrics["behavioral_viterbi_churn_pct"] -
                max_allowed_churn_pct),
        )

    candidate, metrics, _ = min(scored, key=penalty)
    summary = {
        "meets_targets": False,
        "covariance_scale": candidate["covariance_scale"],
        "transition_blend": candidate["transition_blend"],
        "posterior_temperature": candidate.get("posterior_temperature", 1.0),
        **metrics,
    }
    return candidate, summary


# -----------------------------------------------------------------------
# 4. BIC
# -----------------------------------------------------------------------

def n_params(K: int, D: int, covariance_type: str) -> int:
    n_trans = K * (K - 1)
    n_init  = K - 1
    n_means = K * D
    if covariance_type == "full":
        n_covs = K * D * (D + 1) // 2
    elif covariance_type == "diag":
        n_covs = K * D
    elif covariance_type == "tied":
        n_covs = D * (D + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return n_trans + n_init + n_means + n_covs


def bic(model: hmm.GaussianHMM, X: np.ndarray, covariance_type: str) -> float:
    T, D = X.shape
    K = model.n_components
    ll = model.score(X)
    return -2 * ll + n_params(K, D, covariance_type) * np.log(T)


def select_k(X: np.ndarray, covariance_type: str, label: str = "",
             min_offdiag_prob: float = config.MIN_OFFDIAG_PROB,
             transmat_prior: float = 1.0, min_covar: float = 0.001):
    results = {}
    print(f"  transmat_prior={transmat_prior:g}  min_covar={min_covar:g}")
    print(f"\n[model selection — {label}]  covariance_type={covariance_type}")
    for K in config.K_VALUES:
        model, ll = fit_best(
            X,
            K,
            covariance_type,
            transmat_prior=transmat_prior,
            min_covar=min_covar,
        )
        model = floor_transition_matrix(model, min_offdiag=min_offdiag_prob)
        b = bic(model, X, covariance_type)
        print(f"  K={K}  BIC: {b:.2f}")
        results[K] = {"model": model, "log_likelihood": ll, "bic": b}

    best_k = min(results, key=lambda k: results[k]["bic"])
    print(f"  -> BIC-preferred K = {best_k}")
    results["best_k"] = best_k
    results["covariance_type"] = covariance_type
    results["min_offdiag_prob"] = min_offdiag_prob
    results["transmat_prior"] = transmat_prior
    results["min_covar"] = min_covar
    return results


# -----------------------------------------------------------------------
# 5. DECODE REGIMES
# -----------------------------------------------------------------------

def decode(model: hmm.GaussianHMM, X: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    K = model.n_components
    _, states = model.decode(X, algorithm="viterbi")
    smoothed = model.predict_proba(X)
    filtered = _forward_filtered(model, X)

    out = pd.DataFrame({"viterbi_state": states}, index=index)
    for k in range(K):
        out[f"filtered_p{k+1}"] = filtered[:, k]
        out[f"smoothed_p{k+1}"] = smoothed[:, k]
    return out


def _forward_filtered(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Causal forward pass with scaling (Rabiner Section V.A)."""
    from scipy.stats import multivariate_normal

    T, _ = X.shape
    K = model.n_components
    A = model.transmat_
    pi = model.startprob_

    log_b = np.zeros((T, K))
    for k in range(K):
        cov = model.covars_[k]
        if model.covariance_type == "diag" and cov.ndim == 1:
            cov = np.diag(cov)
        log_b[:, k] = multivariate_normal.logpdf(X, mean=model.means_[k], cov=cov)

    alpha = np.zeros((T, K))
    alpha[0] = pi * np.exp(log_b[0] - log_b[0].max())
    c = alpha[0].sum()
    alpha[0] = alpha[0] / c if c > 0 else alpha[0]

    for t in range(1, T):
        raw = (alpha[t-1] @ A) * np.exp(log_b[t] - log_b[t].max())
        c = raw.sum()
        alpha[t] = raw / c if c > 0 else raw

    return alpha


# -----------------------------------------------------------------------
# 6. REGIME DURATION SUMMARY
# -----------------------------------------------------------------------

def regime_duration_summary(regimes: pd.DataFrame, K: int) -> pd.DataFrame:
    states = regimes["viterbi_state"].values
    rows = []
    for k in range(K):
        runs, in_run, length = [], False, 0
        for s in states:
            if s == k:
                in_run, length = True, length + 1
            else:
                if in_run:
                    runs.append(length)
                in_run, length = False, 0
        if in_run:
            runs.append(length)
        avg_dur = np.mean(runs) if runs else 0
        flag = "too short" if avg_dur < config.MIN_AVG_DURATION else "ok"
        rows.append({"state": k + 1, "n_episodes": len(runs),
                      "avg_duration_days": round(avg_dur, 1),
                      "min_days": min(runs) if runs else 0,
                      "max_days": max(runs) if runs else 0,
                      "flag": flag})
    return pd.DataFrame(rows)
