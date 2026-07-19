import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from kanly.api import qr, lm

n = 150_000
np.random.seed(0)
df = pd.DataFrame()
df['x'] = np.abs(np.random.randn(n))
df['geo'] = np.random.randint(0, 5, n)
df['z'] = np.abs(np.random.randn(n))
df['y'] = np.exp(-2.2 + .3 * df.x - .2 * df.z ** 2 + .15 * df.x * df.z
                 + .15 * np.random.randn(n) + df.geo * .005) ** 2

taus = .1, .9
# formula = 'y ~ x + z + I(x**2) + C(geo)'
formula = 'y ~ poly(x,3)'
fit = qr(formula, df, taus, debug=False,
         cov_type='bootstrap(100)',
         xtol=1e-8,
         line_search=True, dense_threshold_mb=None)
print(fit)

fit_mean = lm(formula, df)

plt.scatter(df.x, df.y, alpha=.5, c='grey')
x_temp = np.linspace(0, 5, 100)
for t, f in fit.items():
    plt.plot(x_temp, f.predict(data=pd.DataFrame({'x': x_temp})),
             label=f'quantile={t}')
plt.plot(x_temp, fit_mean.predict(data=pd.DataFrame({'x': x_temp})), label='mean')
plt.legend(loc='best')
plt.show()

"""
═══════════════════════════════════════════════════════════════════════════
Quantile Regression Results
═══════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:               Feb 10, 2025    No. Obs.                  150000
Time:                   14:49:48    Df Residuals:             149996
Model Elapsed:            0.04 s    Df Model:                      3
Fit Elapsed:              0.21 s    Converged:                  True
Cov Elapsed:             12.08 s    Iterations:                    9
Quantile:                    0.9    Error:                  6.00e-07
Pseudo-rsquared:          0.5715    Cost:                 1.2272e+02
Method:                     IRLS    True Cost:            1.2272e+02
Covariance Type:       BOOTSTRAP    Line Search:                True

═══════════════════════════════════════════════════════════════════════════
                coef          std err       t   p>|t|   [0.025,      0.975]
───────────────────────────────────────────────────────────────────────────
Intercept    0.01506  ****  0.0001103  136.51  <0.001    0.01485    0.01528
x            0.01581  ****  0.0006503   24.32  <0.001    0.01454    0.01709
I(x**2)    -0.003398  ****  0.0008691   -3.91  <0.001  -0.005102  -0.001695
I(x**3)     0.004757  ****  0.0002978   15.97  <0.001   0.004173   0.005341
═══════════════════════════════════════════════════════════════════════════

formula:  y ~ poly(x,3)

Converged: relative change in cost 4.13e-09 less than `f_tol` 1.00e-08
Used t distribution with 149996 df at test level 0.0500.
Did 500 Bayesian bootstrap repetitions, alpha=1.000.

                                                         [kanly v=0.0.860]
"""

