import numpy as np
import pandas as pd
import statsmodels

from statsmodels.discrete.truncated_model import HurdleCountModel
from kanly.distributional_models import PoissonHurdle


# ============================================================
# Helpers
# ============================================================

def sample_zero_truncated_poisson(rng, rate):
    """Draw Poisson(rate) conditional on Y > 0."""
    y = rng.poisson(rate)

    zero = y == 0
    while np.any(zero):
        y[zero] = rng.poisson(rate[zero])
        zero = y == 0

    return y


# ============================================================
# Simulate data
# ============================================================

rng = np.random.default_rng(12345)

n = 20_000

x1 = rng.normal(size=n)
x2 = rng.normal(size=n)

X = np.column_stack([
    np.ones(n),
    x1,
    x2,
])

names = ["const", "x1", "x2"]
k = X.shape[1]


# Positive-count equation:
#
#   lambda_positive = exp(X beta)
#
# Conditional on being positive:
#
#   Y | Y > 0 ~ Zero-Truncated Poisson(lambda_positive)

beta_true = np.array([
    0.20,
    0.40,
    -0.25,
])

lambda_positive = np.exp(X @ beta_true)


# Hurdle equation:
#
#   lambda_hurdle = exp(X gamma)
#
# with censored-Poisson / cloglog hurdle:
#
#   P(Y = 0) = exp(-lambda_hurdle)
#   P(Y > 0) = 1 - exp(-lambda_hurdle)

gamma_true = np.array([
    -0.30,
    0.35,
    0.20,
])

lambda_hurdle = np.exp(X @ gamma_true)

p_positive = -np.expm1(-lambda_hurdle)

cross_hurdle = rng.random(n) < p_positive


# Generate y

y = np.zeros(n, dtype=int)

y[cross_hurdle] = sample_zero_truncated_poisson(
    rng,
    lambda_positive[cross_hurdle],
)


print("=" * 70)
print("DATA")
print("=" * 70)

print(f"N:                   {n:,}")
print(f"Fraction zero:       {np.mean(y == 0):.4f}")
print(f"Fraction positive:   {np.mean(y > 0):.4f}")
print(f"Mean y:              {np.mean(y):.4f}")


# ============================================================
# Fit statsmodels
# ============================================================

sm_model = HurdleCountModel(
    endog=y,
    exog=X,
    dist="poisson",
    zerodist="poisson",
)

sm_fit = sm_model.fit(
    method="bfgs",
    maxiter=500,
    disp=False,
)


# ============================================================
# Fit kanly
# ============================================================

kanly_model = PoissonHurdle(
    endog=y,
    exog=X,
    exog_infl=X,

    # Important:
    # use Poisson hurdle rather than kanly's default logit hurdle
    # so this matches statsmodels.
    zero_model="poisson",

    exog_names=names,
    exog_infl_names=names,
)

kanly_fit = kanly_model.fit(
    cov_type="NONROBUST",
)


# ============================================================
# Convert parameters to NumPy
#
# IMPORTANT:
# Do this BEFORE slicing so pandas index alignment cannot
# interfere with the comparison.
# ============================================================

sm_params = np.asarray(sm_fit.params)
sm_bse = np.asarray(sm_fit.bse)

kanly_params = np.asarray(kanly_fit.params)
kanly_bse = np.asarray(kanly_fit.bse)


# ============================================================
# Parameter ordering
# ============================================================
#
# statsmodels:
#
#     [hurdle gamma, positive beta]
#
# kanly:
#
#     [positive beta, hurdle gamma]
#

sm_gamma = sm_params[:k]
sm_beta = sm_params[k:]

sm_gamma_se = sm_bse[:k]
sm_beta_se = sm_bse[k:]


kanly_beta = kanly_params[:k]
kanly_gamma = kanly_params[k:]

kanly_beta_se = kanly_bse[:k]
kanly_gamma_se = kanly_bse[k:]


# ============================================================
# Positive component
# ============================================================

positive_comparison = pd.DataFrame(
    {
        "true": beta_true,
        "statsmodels": sm_beta,
        "kanly": kanly_beta,
        "sm - kanly": sm_beta - kanly_beta,
        "sm_se": sm_beta_se,
        "kanly_se": kanly_beta_se,
    },
    index=names,
)


print()
print("=" * 70)
print("POSITIVE COUNT COMPONENT")
print("log(lambda_positive) = X beta")
print("=" * 70)

print(positive_comparison)


# ============================================================
# Hurdle component
# ============================================================

hurdle_comparison = pd.DataFrame(
    {
        "true": gamma_true,
        "statsmodels": sm_gamma,
        "kanly": kanly_gamma,
        "sm - kanly": sm_gamma - kanly_gamma,
        "sm_se": sm_gamma_se,
        "kanly_se": kanly_gamma_se,
    },
    index=names,
)


print()
print("=" * 70)
print("POISSON HURDLE COMPONENT")
print("log(lambda_hurdle) = X gamma")
print("=" * 70)

print(hurdle_comparison)


# ============================================================
# Log likelihood
# ============================================================

print()
print("=" * 70)
print("LOG LIKELIHOOD")
print("=" * 70)

print(f"statsmodels: {sm_fit.llf:.12f}")
print(f"kanly:       {kanly_fit.llf:.12f}")
print(f"difference:  {sm_fit.llf - kanly_fit.llf:.12e}")


# ============================================================
# Predictions
# ============================================================
#
# statsmodels              kanly
# ------------------------------------------------------------
# mean                     mean
# mean-main                underlying_mean
# mean-nonzero             positive_mean
# prob-zero                zero_probability
# prob-main                positive_probability
#

predictions = {
    "overall mean": (
        np.asarray(sm_fit.predict(which="mean")),
        np.asarray(kanly_fit.predict(which="mean")),
    ),

    "underlying Poisson mean": (
        np.asarray(sm_fit.predict(which="mean-main")),
        np.asarray(kanly_fit.predict(which="underlying_mean")),
    ),

    "mean conditional Y>0": (
        np.asarray(sm_fit.predict(which="mean-nonzero")),
        np.asarray(kanly_fit.predict(which="positive_mean")),
    ),

    "P(Y=0)": (
        np.asarray(sm_fit.predict(which="prob-zero")),
        np.asarray(kanly_fit.predict(which="zero_probability")),
    ),

    "P(Y>0)": (
        np.asarray(sm_fit.predict(which="prob-main")),
        np.asarray(kanly_fit.predict(which="positive_probability")),
    ),
}


prediction_rows = []

for quantity, (sm_pred, kanly_pred) in predictions.items():

    diff = sm_pred - kanly_pred

    prediction_rows.append({
        "quantity": quantity,
        "max abs diff": np.max(np.abs(diff)),
        "mean abs diff": np.mean(np.abs(diff)),
        "RMSE": np.sqrt(np.mean(diff ** 2)),
    })


prediction_comparison = pd.DataFrame(
    prediction_rows
).set_index("quantity")


print()
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)

print(prediction_comparison)


# ============================================================
# Covariance matrices
# ============================================================

sm_cov = np.asarray(sm_fit.cov_params())
kanly_cov = np.asarray(kanly_fit.cov_params())


# statsmodels covariance ordering:
#
#     gamma, beta
#
# Reorder to:
#
#     beta, gamma
#
# so it matches kanly.

reorder = np.r_[
    np.arange(k, 2 * k),
    np.arange(0, k),
]

sm_cov_reordered = sm_cov[np.ix_(reorder, reorder)]

cov_diff = sm_cov_reordered - kanly_cov


print()
print("=" * 70)
print("COVARIANCE")
print("=" * 70)

print(
    f"Max absolute covariance difference: "
    f"{np.max(np.abs(cov_diff)):.12e}"
)

print(
    f"Frobenius norm covariance difference: "
    f"{np.linalg.norm(cov_diff):.12e}"
)


# ============================================================
# Standard-error comparison
# ============================================================

sm_bse_reordered = np.r_[
    sm_beta_se,
    sm_gamma_se,
]

se_diff = sm_bse_reordered - kanly_bse


print()
print("=" * 70)
print("STANDARD ERRORS")
print("=" * 70)

print(
    f"Maximum absolute SE difference: "
    f"{np.max(np.abs(se_diff)):.12e}"
)


# ============================================================
# Overall parameter comparison
# ============================================================

sm_params_reordered = np.r_[
    sm_beta,
    sm_gamma,
]

param_diff = sm_params_reordered - kanly_params


print()
print("=" * 70)
print("OVERALL AGREEMENT")
print("=" * 70)

print(
    f"Maximum absolute coefficient difference: "
    f"{np.max(np.abs(param_diff)):.12e}"
)

print(
    f"RMSE coefficients: "
    f"{np.sqrt(np.mean(param_diff ** 2)):.12e}"
)

print(
    f"Maximum absolute prediction difference: "
    f"{max(np.max(np.abs(a - b)) for a, b in predictions.values()):.12e}"
)


# ============================================================
# Full parameter table
# ============================================================

full_names = (
    [f"positive_{name}" for name in names]
    + [f"hurdle_{name}" for name in names]
)

true_params = np.r_[
    beta_true,
    gamma_true,
]

full_comparison = pd.DataFrame(
    {
        "true": true_params,
        "statsmodels": sm_params_reordered,
        "kanly": kanly_params,
        "difference": param_diff,
        "statsmodels_se": sm_bse_reordered,
        "kanly_se": kanly_bse,
    },
    index=full_names,
)


print()
print("=" * 70)
print("FULL PARAMETER COMPARISON")
print("=" * 70)

print(full_comparison)


# ============================================================
# Versions / parameter names
# ============================================================

print()
print("=" * 70)
print("VERSIONS")
print("=" * 70)

print("statsmodels version:", statsmodels.__version__)
print("kanly parameter names:", kanly_fit.param_names)
