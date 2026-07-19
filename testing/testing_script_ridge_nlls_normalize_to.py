from kanly.bayes_wip import BayesianNonlinearLeastSquaresModel
import pandas as pd
from kanly.api import minimize

import numpy as np
import pandas as pd

mean_x = -2.5
std_x = .15

n = 1000
np.random.seed(0)
x = np.random.rand(n)
y = 1.2 * np.random.randn(n) + 3 * x - 2.5
df = pd.DataFrame({
    'x': x, 'y': y
})

bmodel = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
    '[y]~{Intercept}+{x}*[x]',
    df,
    priors={'x': f'normal({mean_x},{std_x})'},
)

fit_no_prior = bmodel.nlls_model.fit()

fit_mcmc = bmodel.amha([0., 0, 1], debug=True, n_burnin=25_000, n_samples=350_000, n_chains=12,
                       max_processes=12, max_subchain_draws=30_000, thinning=1)

for x_scale in ['jac', None]:

    # convert from prior to ridge penalty
    penalty_coef = fit_mcmc['__sigma2'] / std_x ** 2

    nlls_fit = bmodel.nlls_model.fit(l2_penalties={'x': penalty_coef},
                                     x_scale=x_scale,
                                     regularize_to_values={'x': mean_x},
                                     scale_l2_penalties=False,
                                     jac_method='analytic',
                                     xtol=1e-12, gtol=1e-12, ftol=0)


    # sigma2 = 1.0
    # for i in range(10):
    #     penalty_coef = sigma2 / std_x ** 2
    #
    #     nlls_fit = bmodel.nlls_model.fit(l2_penalties={'x': penalty_coef},
    #                                      regularize_to_values={'x': mean_x},
    #                                      scale_l2_penalties=False)
    #
    #     sigma2 = nlls_fit.scale
    #     print(nlls_fit.params, nlls_fit.scale)



    def objective(params):
        return (np.sum((df.y - params[0] - params[1] * df.x) ** 2) + penalty_coef * (params[1] - mean_x) ** 2) / 2


    fit_minimization = minimize(objective, [0, 1], xtol=1e-12, gtol=1e-12, ftol=0)

    results = pd.DataFrame(
        [fit_no_prior.params, nlls_fit.params, fit_mcmc.mean_params,
         pd.Series(fit_minimization.x, index=nlls_fit.params.index)],
        index=['NLLS', 'NLLS Ridge', 'MCMC', 'Nonlinear Optimization']
    ).T

    print()
    print(results)
