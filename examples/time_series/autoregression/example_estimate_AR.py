from kanly.api import simulate_sarima, estimate_ar, AUTOREG, SARIMAX
import numpy as np
import pandas as pd

# simulate data
n = 150
mu_unconditional = 2
ar = [.4, -.15]
mu = mu_unconditional * (1 - sum(ar))
sigma2 = 3.0
y = mu + simulate_sarima(n=n, seed=0, ar=ar, sigma2=sigma2, burnin=500)
true_params = np.hstack([mu, ar, sigma2])


# ----------------------------- #
# Approximate or Conditional AR #
# ----------------------------- #

print("Approx or Conditional MLE:\n")

# Estimate AR using Conditional Sum of Squares (CSS)
res_css = estimate_ar(y, lags=2, method='css')

# Estimate AR using Yule-Walker
res_yw = estimate_ar(y, lags=2, method='yw')

# Estimate AR using Burg
res_burg = estimate_ar(y, lags=2, method='burg')

# Estimate AR using AUTOREG interface
res_autoreg = AUTOREG(y, lags=2, cov_type='nonrobust')

# Estimate AR using conditional MLE interface
# (equivalent numerically to CSS)
res_mle_conditional = estimate_ar(y, lags=2, method='mle_conditional', compute_cov=True)

df = pd.DataFrame(
    np.array([res_yw['params'], res_burg['params'],
              res_css['params'],
              res_mle_conditional['params'],
              np.hstack([res_autoreg.params, res_autoreg.scale_mle]),
              true_params
              ]).T,
    columns=['yule-walker', 'burg', 'css', 'mle-conditional', 'autoreg', '*truth*'],
    index=res_yw['param_names'],
)
print(df.round(4))
print()
print(f'AUTOREG log-likelihood function: {res_autoreg.llf:.3f}')
print(f'Conditional MLE log-likelihood function function: {res_mle_conditional["llf"]:.3f}')

print('\n\nVariance-Covariance from CSS and Conditional MLE:')
print("\nCSS:\n", res_autoreg.cov_params())

print('\n\nVariance-Covariance from CSS and Conditional MLE:')
print("\nConditional MLE:\n", res_mle_conditional['cov_params'])

"""
Approx or Conditional MLE:

           yule-walker    burg     css  mle-conditional  autoreg  *truth*
Intercept       1.1703  1.1647  1.1566           1.1566   1.1566     1.50
L1              0.4596  0.4593  0.4593           0.4593   0.4593     0.40
L2             -0.2398 -0.2358 -0.2398          -0.2398  -0.2398    -0.15
sigma2          3.5953  3.6203  3.6200           3.6200   3.6200     3.00

AUTOREG log-likelihood function: -305.202
Conditional MLE log-likelihood function function: -305.202

Variance-Covariance from CSS and Conditional MLE:

CSS:
            Intercept   L1[<y>]   L2[<y>]
Intercept   0.042228 -0.005937 -0.005970
L1[<y>]    -0.005937  0.006366 -0.002361
L2[<y>]    -0.005970 -0.002361  0.006335


Variance-Covariance from CSS and Conditional MLE:

Conditional MLE:
            Intercept        L1        L2  sigma2
Intercept   0.042215 -0.005931 -0.005969     0.0
L1         -0.005931  0.006365 -0.002362     0.0
L2         -0.005969 -0.002362  0.006336     0.0
sigma2      0.000000  0.000000  0.000000     0.0
"""


# ----------------------------- #
# Exact Likelihood MLE          #
# ----------------------------- #

print("\n\nExact Maximum Likelihood:\n")

# Estimate AR using *exact* MLE interface
res_mle_exact = estimate_ar(y, lags=2, method='mle_exact')

# Estimate AR using SARIMAX interface
res_sarimax_exact = SARIMAX(y, order=(2, 0, 0), trend='c')

df = pd.DataFrame(
    np.array([res_mle_exact['params'],
              res_sarimax_exact.params,
              true_params]).T,
    columns=['mle-exact', 'sarimax', '*truth*'],
    index=res_yw['param_names'],
)
print(df.round(4))
print("\nNote that the intercept parametrization is different,\n\tbut it is the same likelihood.")
print()
print(f'Exact Max Likelihood Exact log-likelihood function: {res_mle_exact["llf"]:.3f}')
print(f'SARIMAX log-likelihood function function: {res_sarimax_exact.llf:.3f}')

"""
Exact Maximum Likelihood:

           mle-exact  sarimax  *truth*
Intercept     1.1691   1.5004     1.50
L1            0.4610   0.4588     0.40
L2           -0.2402  -0.2379    -0.15
sigma2        3.5912   3.5912     3.00

Note that the intercept parametrization is different,
	but it is the same likelihood.

Exact Max Likelihood Exact log-likelihood function: -308.859
SARIMAX log-likelihood function function: -308.860
"""

# # TIMING
# from kanly.api import timer, clear_timers
# from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_SM
# from kanly.time_series.sarimax.sarimax_internal import sarimax_internal
#
# n = 20_000
# mu_unconditional = 2
# ar = [.4, -.15]
# mu = mu_unconditional * (1 - sum(ar))
# sigma2 = 3.0
# y = mu + simulate_sarima(n=n, seed=0, ar=ar, sigma2=sigma2, burnin=500)
# true_params = np.hstack([mu, ar, sigma2])
#
# T = 20
#
# timer('autoreg')
# for _ in range(T):
#     AUTOREG(y, lags=2)
# timer('autoreg')
#
# timer('sarimax_int')
# for _ in range(T):
#     sarimax_internal(y, order=(2, 0, 0))
# timer('sarimax_int')
#
# timer('sarimax')
# for _ in range(T):
#     f0=SARIMAX(y, order=(2, 0, 0))
# timer('sarimax')
#
# timer('sarimax_sm')
# for _ in range(T):
#     f = SARIMAX_SM(y, order=(2, 0, 0), trend='c').fit(disp=False)
# timer('sarimax_sm')
#
# timer('mle')
# for _ in range(T):
#     estimate_ar(y, lags=2, method='mle_exact')
# timer('mle')
#
# timer('mle_c')
# for _ in range(T):
#     estimate_ar(y, lags=2, method='mle_conditional')
# timer('mle_c')
#
# timer('burg')
# for _ in range(T):
#     estimate_ar(y, lags=2, method='burg')
# timer('burg')
#
# print(f.summary())
# print(f0.summary())
