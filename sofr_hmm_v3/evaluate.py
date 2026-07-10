# evaluate.py  —  v2: same as before, label-parameterised for two models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm


def regime_stats(features: pd.DataFrame, regimes: pd.DataFrame, K: int) -> pd.DataFrame:
    aligned = features.copy()
    aligned["state"] = regimes["viterbi_state"].values
    stats = aligned.groupby("state")[features.columns.tolist()].mean()
    stats.index = [f"State {k+1}" for k in stats.index]
    return stats.round(4)


def regime_feature_moments(features: pd.DataFrame, regimes: pd.DataFrame,
                           K: int) -> pd.DataFrame:
    aligned = features.copy()
    aligned["state"] = regimes["viterbi_state"].values

    rows = []
    for k in range(K):
        group = aligned[aligned["state"] == k]
        for feature in features.columns:
            values = group[feature]
            rows.append({
                "state": k + 1,
                "n_days": int(len(group)),
                "feature": feature,
                "mean": round(float(values.mean()), 4) if len(group) else np.nan,
                "variance": round(float(values.var(ddof=0)), 4) if len(group) else np.nan,
            })
    return pd.DataFrame(rows)


def transition_table(model: hmm.GaussianHMM) -> pd.DataFrame:
    K = model.n_components
    A = model.transmat_
    df = pd.DataFrame(
        A,
        index  =[f"From S{k+1}" for k in range(K)],
        columns=[f"To S{k+1}"   for k in range(K)],
    ).round(4)
    df["implied_duration_days"] = [
        round(1 / (1 - A[k, k]), 1) if A[k, k] < 1 else np.inf
        for k in range(K)
    ]
    return df


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def plot_regime_timeline(regimes, features, K, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax = axes[0]
    for k in range(K):
        ax.plot(regimes.index, regimes[f"filtered_p{k+1}"],
                label=f"State {k+1}", color=PALETTE[k % len(PALETTE)],
                lw=1.2, alpha=0.85)
    ax.set_ylabel("Filtered probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    ax2 = axes[1]
    states, dates = regimes["viterbi_state"].values, regimes.index
    for k in range(K):
        ax2.fill_between(dates, 0, 1, where=(states == k),
                         color=PALETTE[k % len(PALETTE)], alpha=0.7,
                         label=f"S{k+1}")
    ax2.set_ylabel("Viterbi state")
    ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=8, ncol=K)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_distributions(features, regimes, K, title="", save_path=None):
    cols = features.columns.tolist()
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    states = regimes["viterbi_state"].values
    for i, col in enumerate(cols):
        ax = axes[i]
        for k in range(K):
            ax.hist(features[col].values[states == k], bins=40, density=True,
                    alpha=0.45, color=PALETTE[k % len(PALETTE)],
                    label=f"S{k+1}")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("value")
    axes[0].set_ylabel("density")
    plt.suptitle(title, y=1.02, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_bic(results: dict, title="", save_path=None):
    ks = sorted([k for k in results if isinstance(k, int)])
    bics = [results[k]["bic"] for k in ks]
    lls = [results[k]["log_likelihood"] for k in ks]
    best = results["best_k"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axes[0]
    ax.plot(ks, bics, "o-", color="#4C72B0", lw=2, ms=7)
    ax.axvline(best, color="#C44E52", lw=1.2, ls="--", label=f"BIC-best K={best}")
    ax.set_xlabel("Number of states K")
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title(f"{title} — BIC")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    ax2 = axes[1]
    ax2.plot(ks, lls, "s-", color="#55A868", lw=2, ms=7)
    ax2.set_xlabel("Number of states K")
    ax2.set_ylabel("Log-likelihood")
    ax2.set_title(f"{title} — LL elbow")
    ax2.set_xticks(ks)
    ax2.grid(axis="y", lw=0.4, alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def filtered_prob_diagnostics(regimes: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Quantifies how 'binary' the filtered probabilities are.
    v1 had 98.7% of days at max-prob > 0.99 — essentially a hard classifier
    rather than a probabilistic one. This table lets v2 be compared directly.
    """
    fp_cols = [f"filtered_p{k+1}" for k in range(K)]
    max_fp = regimes[fp_cols].max(axis=1)
    entropy = -(regimes[fp_cols].clip(1e-15, 1) *
                np.log(regimes[fp_cols].clip(1e-15, 1))).sum(axis=1)

    n = len(regimes)
    return pd.DataFrame([{
        "pct_days_max_prob_gt_99": round((max_fp > 0.99).mean() * 100, 1),
        "pct_days_max_prob_gt_95": round((max_fp > 0.95).mean() * 100, 1),
        "n_genuinely_uncertain_days": int((max_fp < 0.99).sum()),
        "mean_entropy": round(entropy.mean(), 4),
        "max_possible_entropy": round(np.log(K), 4),
    }])


def transition_window_diagnostics(regimes: pd.DataFrame, K: int,
                                  window: int = 3) -> pd.DataFrame:
    fp_cols = [f"filtered_p{k+1}" for k in range(K)]
    max_fp = regimes[fp_cols].max(axis=1)
    probs = regimes[fp_cols].clip(1e-15, 1)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    states = regimes["viterbi_state"].to_numpy()
    index = regimes.index

    rows = []
    for idx in np.flatnonzero(states[1:] != states[:-1]) + 1:
        lo = max(0, idx - window)
        hi = min(len(regimes), idx + window + 1)
        rows.append({
            "transition_date": index[idx],
            "from_state": int(states[idx - 1] + 1),
            "to_state": int(states[idx] + 1),
            "window_start": index[lo],
            "window_end": index[hi - 1],
            "min_max_filtered_prob": round(float(max_fp.iloc[lo:hi].min()), 6),
            "max_entropy": round(float(entropy.iloc[lo:hi].max()), 6),
        })
    return pd.DataFrame(rows)
