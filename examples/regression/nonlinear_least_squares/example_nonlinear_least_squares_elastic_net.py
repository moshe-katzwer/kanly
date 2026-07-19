import numpy as np
import pandas as pd

from kanly.api import nlls_en

n = 25_000
n_g = 30
n_z2 = 10

np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randn(n) ** 2,
    'g': np.random.randint(0, n_g, n),
})

Z = np.random.randn(n, 6)
Z = Z.dot(np.random.randn(6, n_z2))

for j in range(n_z2):
    df[f'z{j}'] = Z[:, j]

df['y'] = -2 + np.exp(1.2 * df.x) + .3 * df.x * df.w + 0.6 * df.x ** 2 + np.random.randn(n) * .3 + .54 * np.sqrt(
    df.g + 5)

alpha = {f'q{j}': .02 for j in range(n_z2)}
alpha.update({f'C(g)[{j}]': .1 for j in range(n_g)})
#print(alpha)

alpha = {a: 100000 * i for a, i in alpha.items()}

fit = nlls_en(
    '[y] ~ {a} + [poly(x,3,-1)] + np.exp({zeta}*[x]) * [w]**(1+{beta}) + np.exp(-1+{phi}*[x]+{psi}*[z0])'
    + '+' + ' + '.join(f'{{q{j}}}*[z{j}]' for j in range(n_z2))
    + '+ [C(g,-1)]'
    + " $ [w]"
    ,
    df,
    #positive={'a': True},
    alpha=alpha,
    ftol=1e-8,
    xtol=1e-6,
    gtol=1e-4,
    debug=True,
    active_set=True,
    #positive=True,
    shrink_factor=.33,
    num_shrinkage=4,
    #prompt_for_more_iters=True, max_iter=10,
    max_iter=1000,
    selection='cyclic',
    # one_dim_search_cadence=None,
    # one_dim_search_init_value=.01,
    # one_dim_search_multiplier=5,
    seed=0,
    #jac_method='analytic',
)
print(fit)
print(fit.objective)

"""
══════════════════════════════════════════════════════════════════════════
Nonlinear Least Squares Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:                  May 16, 2026    R-squared:                   0.9600
Time:                      16:25:16    Adj. R-squared:              0.9599
Weights:                          w    Model Time:                   0.01s
Nobs:                         25000    Fit Time:                    10.71s
Df Residuals:                 24953    Cov Time:                     0.00s
Df Model:                        47    Iterations:                     424
Cost:                    4.1568e-01    Converged:                     True
Scale:                   8.3696e-01    Status:                           1
LLF:                    -4.9128e+04    Covariance Type:       NOT COMPUTED
Penalty:                 4.7086e-06    Active Constraints:               0
Objective:               4.1568e-01    Method:                          CD
Optimality:                4.24e-04                                       

════════════════════
                coef
────────────────────
a             -0.162
beta         -0.8475
phi            1.329
psi        -0.006328
q0        -1.559e-05
q1         1.774e-05
q2         1.459e-05
q3        -5.848e-06
q4         4.131e-05
q5        -1.498e-05
q6         1.047e-05
q7        -1.131e-05
q8         2.085e-05
q9         2.868e-05
zeta           0.984
_A[x]         0.5468
_A[x**2]      0.4082
_A[x**3]    -0.05051
C(g)[1]   -2.976e-06
C(g)[10]  -4.566e-07
C(g)[11]  -7.484e-07
C(g)[12]  -2.776e-07
C(g)[13]  -3.693e-07
C(g)[14]   2.915e-07
C(g)[15]   1.111e-07
C(g)[16]   5.425e-07
C(g)[17]   4.693e-07
C(g)[18]   1.309e-06
C(g)[19]   9.512e-07
C(g)[2]   -3.329e-06
C(g)[20]   1.207e-06
C(g)[21]   1.646e-06
C(g)[22]   1.712e-06
C(g)[23]   2.041e-06
C(g)[24]   2.021e-06
C(g)[25]   2.224e-06
C(g)[26]   2.134e-06
C(g)[27]   2.582e-06
C(g)[28]   2.493e-06
C(g)[29]   2.656e-06
C(g)[3]   -3.208e-06
C(g)[4]   -2.271e-06
C(g)[5]   -1.905e-06
C(g)[6]   -1.833e-06
C(g)[7]   -1.364e-06
C(g)[8]   -1.168e-06
C(g)[9]   -1.023e-06
════════════════════

formula:  [y] ~ {a} + [poly(x,3,-1)] + np.exp({zeta}*[x]) *
          [w]**(1+{beta}) +
          np.exp(-1+{phi}*[x]+{psi}*[z0])+{q0}*[z0] +
          {q1}*[z1] + {q2}*[z2] + {q3}*[z3] + {q4}*[z4] +
          {q5}*[z5] + {q6}*[z6] + {q7}*[z7] + {q8}*[z8] +
          {q9}*[z9]+ [C(g,-1)] $ [w]


message: converged, relative change in objective_function < ftol: (-9.75040892470247e-09 < 1e-08)

                                                       [kanly v=0.0.1020]
"""