"""Experimental cross-validation sketch for penalized linear models.

This file is kept as WIP/reference code rather than a stable API entry point.
"""

from kanly.regression.linear_models.penalized.model import *
import numpy as np
import pandas as pd
from kanly.api import elastic_net
from kanly.regression.linear_models.penalized.sparse_elastic_net_internal import _get_rsquared
from kanly.regression.linear_models.linear_model_2_quadratic_form import _linear_model_components_2_quadratic_form_and_likelihood
from kanly import tqdm
import time

DEFAULT_EN_CV_L1_RANGE = [0.0, .1, .5, .7, .9, .95, .99, 1.0]


def cross_validate(self, kfolds=5, seed=0,
                   fit_intercept=DEFAULT_EN_FIT_INTERCEPT,
                   normalize=DEFAULT_EN_NORMALIZE, max_iter=DEFAULT_EN_MAX_ITER, tol=DEFAULT_EN_TOL,
                   positive=DEFAULT_EN_POSITIVE, debug=False, l1_range=DEFAULT_EN_CV_L1_RANGE
                   ):
    """Experimental cross-validation helper for elastic-net penalty search.

    This WIP routine partitions rows into folds, fits a grid of ``l1_ratio`` values
    for a candidate ``alpha`` exponent, caches quadratic forms and warm starts, and
    then uses scalar minimization over log10(alpha).  It is not part of the stable
    public API and may lag current solver arguments.

    Args:
        self: ``SparsePenalizedLinearModel`` instance.
        kfolds: Number of folds.
        seed: Random seed for fold assignment.
        fit_intercept: Whether to fit an intercept.
        normalize: When True, scale penalties by column standard deviation.
        max_iter: Maximum iterations for each fold fit.
        tol: Legacy tolerance argument used by this WIP helper.
        positive: Whether to constrain coefficients non-negative.
        debug: Whether to print diagnostics.
        l1_range: Candidate ``l1_ratio`` values.

    Returns:
        None; prints optimization diagnostics and fold scores."""
    rand = np.random.RandomState(seed=seed)

    blocks = rand.permutation(np.hstack([[i] * (n // kfolds) for i in range(kfolds)]
                                        + [np.arange(n % kfolds)]))

    ssr_quad_forms = dict()
    results = []

    if self.weights is not None:
        raise Exception("!")

    y = self.endog.toarray().flatten()
    num_params = self.exog.shape

    param_dict = dict()

    def test(a):
        """Evaluate cross-validated score for one log10(alpha) candidate.

        Args:
            a: Base-10 exponent, where ``alpha = 10**a``.

        Returns:
            Negative out-of-sample score, suitable for minimization."""

        alpha = 10.0 ** a

        scores_oos = []
        scores_is = []

        _t = time.time()

        for i, l1 in enumerate(l1_range):

            score_oos = 0.0
            score_is = 0.0

            for k in range(kfolds):

                x0, intercept_ = param_dict.get((l1, k), (None, None))

                # Encode the fold split as observation weights so the same
                # model object and quadratic-form machinery can be reused.
                include = (blocks != k).astype(float)
                #include *= self.nobs / sum(include)

                if k not in ssr_quad_forms.keys():
                    ssr_quad_forms[k] = _linear_model_components_2_quadratic_form_and_likelihood(
                        self.endog, self.exog, include)[1]

                fit = self.fit(fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter,
                               tol=tol, positive=positive, override_weights=include, alpha=alpha,
                               l1_ratio=l1, debug=False,
                               start_coef=x0, start_intercept=intercept_,
                               ssr_quad_form=ssr_quad_forms[k])

                if not fit.converged:
                    score_is += -np.inf
                    score_oos += -np.inf

                else:
                    resid = fit.resid
                    score_in_sample = _get_rsquared(True, include, resid, y.copy())
                    score_out_of_sample = _get_rsquared(True, 1 - include, resid, y.copy())
                    score_is += score_in_sample
                    score_oos += score_out_of_sample

                # Warm-start the next alpha candidate for this l1/fold pair.
                param_dict[(l1, k)] = fit.coef_.flatten(), fit.intercept_

            score_is /= kfolds
            score_oos /= kfolds

            scores_is.append(score_is)
            scores_oos.append(score_oos)

            results.append({'a': a, 'alpha': alpha, 'l1_ratio': l1,
                            'score_oos': score_oos, 'score_is': score_is,
                            'converged': fit.converged})

            #print("\t", results[-1])

            # import matplotlib.pyplot as plt
            # plt.scatter(y, fit.fittedvalues, alpha=.2)
            # plt.title(results[-1])
            # #plt.ylim([min(y), max(y)])
            # #plt.xlim([min(y), max(y)])
            # plt.show()

        best_l1_index = np.argmax(scores_oos)
        best_l1 = l1_range[best_l1_index]
        best_score_in = scores_is[best_l1_index]
        best_score_oos = scores_oos[best_l1_index]
        print('* ', alpha, best_l1, best_score_in, best_score_oos, f'{np.round(time.time()-_t,2)}s')

        return -best_score_oos

    a_ = 5.
    best_score = np.inf
    new_score = test(a_ - 1.)
    while new_score <= best_score:
        print('   good ', a_ - 1., new_score)
        a_ -= 1.
        best_score = new_score
        new_score = test(a_ - 1.)
    else:
        print('>> bad  ', a_ - 1., new_score)

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(test, method='bounded', bounds=(a_-1, a_+1), tol=1e-3)
    print(res)
    res_df = pd.DataFrame(results)
    print(res_df.sort_values(by=['alpha', 'l1_ratio']))

    print(res_df[res_df.a == res.x])


n = 255
p = 250
np.random.seed(0)
X = np.random.randn(n, p)
Pi = (np.random.rand(p, p) < .15).astype(float)
Pi += Pi.T
Pi += np.eye(p)
X = X.dot(Pi)

df = pd.DataFrame(X, columns=[f'x{j}' for j in range(p)])
df['y'] = X.dot(np.exp(np.random.randn(p))/20) + 50 * np.random.randn(n)

model = elastic_net('y ~ ' + '+'.join([f'x{j}' for j in range(p)]),
                    df, debug=False).model

cross_validate(model, seed=0, kfolds=5, max_iter=500)
