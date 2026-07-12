# config.py - v3: two parallel models, level regime + revised shape regime

import os


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Target_Zeros.csv")

# ── Fixed tenor grid (days) — same grid as Ridge V2 and v1 ────────────
FIXED_GRID_DAYS = [
    9, 16, 23, 33, 63, 94, 124, 155, 186,
    277, 367, 551, 732, 1098,
    1461, 1828, 2192, 2557, 2922,
    3654, 3837, 4020, 4203, 4386, 4569, 4752, 4935,
    5118, 5301, 5484, 5667, 5850, 6033, 6216, 6399,
    6582, 6765, 6948, 7131,
    7307, 7672, 8037, 8402, 8767,
    9132, 9497, 9862, 10227, 10592, 10959,
]
MIN_KNOTS = 4

# ── Anchor tenors (days) used to build explicit spread features ───────
# Chosen from the fixed grid: 3m, 5y, 10y.
# term_slope = rate(10yr) - rate(3m)
# butterfly = rate(5yr) - 0.5 * (rate(3m) + rate(10yr))
TENOR_3M  = 94
TENOR_2Y  = 732
TENOR_5Y  = 1828
TENOR_10Y = 3654

# ── Rolling volatility ────────────────────────────────────────────────
VOL_WINDOW = 21

# ── PCA — used only by the LEVEL model ────────────────────────────────
N_LEVEL_FACTORS  = 3
N_CHANGE_FACTORS = 2

# ── Model selection ─────────────────────────────────────────────────────
# Set False to skip the level model entirely and only fit/report the shape
# model. Level code path is kept intact for easy re-enabling.
RUN_LEVEL_MODEL = False

# ── HMM — shared settings, applied to both models ─────────────────────
K_VALUES = [2, 3, 4, 5]
N_EM_RESTARTS = 20
MAX_ITER = 500
RANDOM_SEED = 42

# Covariance type per model — level model keeps "full" (v1 baseline,
# kept for comparison); shape model uses "diag" (v2 fix — see notes).
LEVEL_COVARIANCE_TYPE = "full"
SHAPE_COVARIANCE_TYPE = "diag"

# Shape posterior calibration. The fitted HMM parameters still determine the
# regime model; calibration only softens posterior inference for reporting.
SHAPE_CALIBRATE_POSTERIORS = True
SHAPE_POSTERIOR_TEMPERATURE = 2.0
SHAPE_COVARIANCE_SCALE_GRID = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
SHAPE_TRANSITION_BLEND_GRID = [0.0, 0.025, 0.05, 0.10]
SHAPE_POSTERIOR_TEMPERATURE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]

# ── Evaluation ─────────────────────────────────────────────────────────
MIN_AVG_DURATION = 20

# Transition-focused uncertainty targets for calibrated shape posteriors.
TRANSITION_WINDOW_DAYS = 3
TARGET_MAX_PROB_AT_TRANSITION = 0.90
TARGET_PCT_DAYS_MAX_PROB_GT_99 = 95.0
MAX_ALLOWED_VITERBI_CHURN_PCT = 33.0

# ── Transition matrix regularisation ──────────────────────────────────
# Floor applied to off-diagonal transition probabilities after fitting,
# to prevent a_kk = 1.0 absorbing states (v1's S1 problem). Renormalised
# after the floor is applied. Set to 0 to disable.
MIN_OFFDIAG_PROB = 1e-4
