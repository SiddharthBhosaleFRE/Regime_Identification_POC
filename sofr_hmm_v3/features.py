# features.py - v3: load+spline (unchanged from v1) -> two feature sets
#
#   build_level_features() - v1's PCA-on-levels approach, kept for comparison
#   build_shape_features() - v3 shape set: level, term slope, original
#                            5y butterfly, and 90d level movement intensity

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

import config

GRID_ARR    = np.array(config.FIXED_GRID_DAYS, dtype=float)
GRID_LABELS = [f"d{d}" for d in config.FIXED_GRID_DAYS]


# -----------------------------------------------------------------------
# 1. LOAD AND SPLINE ONTO FIXED GRID  (unchanged from v1 — this part was
#    already correct: maturity = AnchorDate - MV1_DATE, Pchip spline onto
#    the grid shared with Ridge V2)
# -----------------------------------------------------------------------

def load_and_pivot(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df["MV1_DATE"]   = pd.to_datetime(df["MV1_DATE"],   format="mixed")
    df["AnchorDate"] = pd.to_datetime(df["AnchorDate"], format="mixed")
    df["days_to_anc"] = (df["AnchorDate"] - df["MV1_DATE"]).dt.days
    df = df.dropna(subset=["days_to_anc", "SOFRZeroRate"])

    rows = []
    skipped = {"short": 0, "range": 0, "nan": 0}

    for date, grp in df.groupby("MV1_DATE"):
        grp = grp.sort_values("days_to_anc").drop_duplicates("days_to_anc")
        if len(grp) < config.MIN_KNOTS:
            skipped["short"] += 1
            continue
        x = grp["days_to_anc"].to_numpy(float)
        y = grp["SOFRZeroRate"].to_numpy(float)
        if x.min() > GRID_ARR.min() or x.max() < GRID_ARR.max():
            skipped["range"] += 1
            continue
        vals = PchipInterpolator(x, y)(GRID_ARR)
        if np.any(np.isnan(vals)):
            skipped["nan"] += 1
            continue
        rows.append([date, *vals.tolist()])

    wide = (
        pd.DataFrame(rows, columns=["date"] + GRID_LABELS)
          .set_index("date")
          .sort_index()
    )

    print(f"[data]  date range : {wide.index.min().date()} -> {wide.index.max().date()}")
    print(f"[data]  grid points: {len(GRID_LABELS)}  "
          f"({GRID_ARR[0]:.0f}d -> {GRID_ARR[-1]:.0f}d)")
    print(f"[data]  obs        : {len(wide):,} trading days")
    for k, v in skipped.items():
        if v:
            print(f"[data]  dropped    : {v} dates ({k})")

    return wide


# -----------------------------------------------------------------------
# 2A. LEVEL FEATURE SET  (v1 — PCA on levels, kept for side-by-side comparison)
# -----------------------------------------------------------------------

def build_level_features(levels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    v1 approach. Known issue: level_pc1 captures ~96% of variance because
    all 50 tenors moved together through the 2020-2023 hiking cycle. This
    makes the resulting HMM a rate-LEVEL classifier, not a curve-shape model.
    Kept here so its output can be compared directly against build_shape_features().
    """
    scaler_lvl = StandardScaler()
    X_lvl = scaler_lvl.fit_transform(levels.values)
    pca_lvl = PCA(n_components=config.N_LEVEL_FACTORS)
    scores_lvl = pca_lvl.fit_transform(X_lvl)
    var_lvl = pca_lvl.explained_variance_ratio_

    print(f"[level]  PCA factors: {', '.join(f'{v*100:.1f}%' for v in var_lvl)}  "
          f"(cumulative {var_lvl.sum()*100:.1f}%)")

    lvl_df = pd.DataFrame(
        scores_lvl, index=levels.index,
        columns=[f"level_pc{i+1}" for i in range(config.N_LEVEL_FACTORS)],
    )

    changes = levels.diff().dropna()
    scaler_chg = StandardScaler()
    X_chg = scaler_chg.fit_transform(changes.values)
    pca_chg = PCA(n_components=config.N_CHANGE_FACTORS)
    scores_chg = pca_chg.fit_transform(X_chg)

    chg_df = pd.DataFrame(
        scores_chg, index=changes.index,
        columns=[f"change_pc{i+1}" for i in range(config.N_CHANGE_FACTORS)],
    )

    lvl_pc1_chg = lvl_df["level_pc1"].diff().dropna()
    vol = (
        lvl_pc1_chg.rolling(config.VOL_WINDOW, min_periods=config.VOL_WINDOW)
                   .std().rename("level_vol")
    )

    features = pd.concat([lvl_df, chg_df, vol], axis=1).dropna()
    print(f"[level]  feature shape: {features.shape}  "
          f"columns: {list(features.columns)}")

    meta = {"pca_level": pca_lvl, "pca_change": pca_chg,
            "scaler_level": scaler_lvl, "scaler_change": scaler_chg,
            "level_var_explained": var_lvl}
    return features, meta


# -----------------------------------------------------------------------
# 2B. SHAPE FEATURE SET  (v3 - explicit spreads, level is 1 of 4 features)
# -----------------------------------------------------------------------

def _nearest_grid_col(days: int) -> str:
    """Map a target tenor (days) to the nearest fixed-grid column label."""
    idx = np.abs(GRID_ARR - days).argmin()
    return GRID_LABELS[idx]


def build_shape_features(levels: pd.DataFrame,
                         feature_set: str = "v3") -> tuple[pd.DataFrame, dict]:
    """
    v3 shape set. Explicit curve-shape features built from spreads between
    anchor tenors, rather than letting PCA on raw levels be dominated by
    the 2020-2023 parallel rate shift. 'level' here is one feature among
    four, not 96% of the variance.

    Features:
        level        — mean rate across the curve (overall rate environment)
        term_slope   — 10y - 3m  (full curve steepness)
        butterfly    — 5y - 0.5*(3m + 10y)   (curvature / hump)
        level_abs_daily_move_90d_mean
                     — 90d average absolute daily move in curve level
    """
    c3m  = _nearest_grid_col(config.TENOR_3M)
    c2y  = _nearest_grid_col(config.TENOR_2Y)
    c5y  = _nearest_grid_col(config.TENOR_5Y)
    c10y = _nearest_grid_col(config.TENOR_10Y)
    print(
        f"[shape]  anchor tenors -> 3m:{c3m}  2y:{c2y}  "
        f"5y:{c5y}  10y:{c10y}"
    )

    level = levels.mean(axis=1)
    level_abs_daily_move = (
        level.diff().abs()
             .rolling(90, min_periods=90)
             .mean()
             .rename("level_abs_daily_move_90d_mean")
    )

    slope_col = "term_slope"
    slope = levels[c10y] - levels[c3m]
    if feature_set == "slope_2y10y":
        slope_col = "slope_2y10y"
        slope = levels[c10y] - levels[c2y]
    elif feature_set != "v3":
        raise ValueError(f"Unsupported v3 shape feature set: {feature_set}")

    shape = pd.DataFrame({
        "level"     : level,
        slope_col   : slope,
        "butterfly" : levels[c5y] - 0.5 * (levels[c3m] + levels[c10y]),
    }, index=levels.index)

    features = pd.concat([shape, level_abs_daily_move], axis=1).dropna()

    print(f"[shape]  feature shape: {features.shape}  "
          f"columns: {list(features.columns)}")
    print(f"[shape]  feature std (raw, pre-scaling): "
          f"{', '.join(f'{c}={features[c].std():.3f}' for c in features.columns)}")

    meta = {
        "feature_set": feature_set,
        "anchor_cols": {"3m": c3m, "2y": c2y, "5y": c5y, "10y": c10y},
        "movement_window_days": 90,
    }
    return features, meta


# -----------------------------------------------------------------------
# 3. FINAL SCALING FOR HMM
# -----------------------------------------------------------------------

def scale_for_hmm(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return X, scaler


def feature_distribution_diagnostics(X: np.ndarray,
                                     feature_names: list[str],
                                     transform: str) -> pd.DataFrame:
    rows = []
    for idx, feature in enumerate(feature_names):
        values = X[:, idx]
        rows.append({
            "transform": transform,
            "feature": feature,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "skewness": float(skew(values, bias=True)),
            "excess_kurtosis": float(kurtosis(values, fisher=True, bias=True)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def transform_for_hmm(features: pd.DataFrame,
                      transform: str = "standard",
                      n_quantiles: int = None,
                      random_seed: int = config.RANDOM_SEED):
    feature_names = features.columns.tolist()
    transformer = fit_hmm_transform(
        features,
        transform=transform,
        n_quantiles=n_quantiles,
        random_seed=random_seed,
    )
    X = apply_hmm_transform(features, transformer, transform=transform)
    diagnostics = feature_distribution_diagnostics(X, feature_names, transform)
    return X, transformer, diagnostics


def fit_hmm_transform(features: pd.DataFrame,
                      transform: str = "standard",
                      n_quantiles: int = None,
                      random_seed: int = config.RANDOM_SEED):
    if transform == "standard":
        return StandardScaler().fit(features.values)

    if transform == "uniform_platykurtic":
        if n_quantiles is None:
            n_quantiles = min(1000, len(features))
        n_quantiles = min(int(n_quantiles), len(features))
        return QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            random_state=random_seed,
        ).fit(features.values)

    raise ValueError(f"Unsupported HMM feature transform: {transform}")


def apply_hmm_transform(features: pd.DataFrame, transformer,
                        transform: str = "standard") -> np.ndarray:
    if transform == "standard":
        return transformer.transform(features.values)

    if transform == "uniform_platykurtic":
        U = transformer.transform(features.values)
        return np.sqrt(12.0) * (U - 0.5)

    raise ValueError(f"Unsupported HMM feature transform: {transform}")


# -----------------------------------------------------------------------
# CONVENIENCE WRAPPERS
# -----------------------------------------------------------------------

def build_level_matrix(path: str = config.DATA_PATH):
    levels = load_and_pivot(path)
    features, meta = build_level_features(levels)
    X, scaler = scale_for_hmm(features)
    meta["final_scaler"] = scaler
    return features, X, meta


def build_shape_matrix(path: str = config.DATA_PATH, levels: pd.DataFrame = None):
    if levels is None:
        levels = load_and_pivot(path)
    features, meta = build_shape_features(levels)
    X, scaler = scale_for_hmm(features)
    meta["final_scaler"] = scaler
    return features, X, meta
