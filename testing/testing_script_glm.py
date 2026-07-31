import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from kanly.api import glm, GLM, lm
from kanly.regression.generalized_linear_models.sparse_glm_internal import METHOD_IRLS, METHOD_COORD_DESC

from statsmodels.genmod.families import Tweedie, Gamma, Gaussian, InverseGaussian, NegativeBinomial, Binomial, Poisson
from statsmodels.genmod.families.links import inverse_power, Link, inverse_squared

families_sm = [Tweedie, Gamma, Gaussian, InverseGaussian, NegativeBinomial, Binomial, Poisson]
upper_to_sm_family_dict = {f.__name__.upper().replace('_', ''): f for f in families_sm}
links_sm = [x for f in families_sm for x in f.links]
upper_to_sm_link_dict = {l.__name__.upper().replace('_', ''): l for l in links_sm}
print("Z ", upper_to_sm_link_dict.keys())

upper_to_sm_link_dict['INVERSE'] = inverse_power
upper_to_sm_link_dict['NEGATIVEINVERSE'] = inverse_power
upper_to_sm_link_dict['NEGATIVETWOINVERSESQUARED'] = inverse_squared

from kanly.regression.generalized_linear_models.families import (
    FAMILIES, Poisson, Gaussian, Binomial, Gamma, InverseGaussian,
    NegativeBinomial, ZeroTruncatedPoisson,
)
from statsmodels.formula.api import glm as glm_sm
from statsmodels.api import families as families_sm
from numpy.testing import assert_array_almost_equal
import warnings



warnings.filterwarnings('ignore')

families_sm_dict = {k.upper(): v for k, v in families_sm.__dict__.items()}

failed = []

# kanly_to_sm_links = {
#     'LOGIT': 'Logit',
#     'PROBIT': 'Probit',
#     'IDENTITY': 'Identity',
#     'LOG': 'Log',
#     'CLOGLOG': 'CLogLog',
#     'CAUCHY': 'Cauchy',
#     'SQRT': 'Sqrt',
#     'INVERSE': 'InversePower',
#     'INVERSE_SQUARED': 'InverseSquared',
#     'NEGATIVE_INVERSE': 'InversePower',
#     'NEGATIVE_TWO_INVERSE_SQUARED': 'InverseSquared',
# }

families_sm_dict = {k.upper(): v for k, v in families_sm.__dict__.items()}
print(families_sm_dict.keys())
print([("\t", f.__name__, {l.__name__: l for l in f.links}) for f in families_sm_dict.keys() if hasattr(f, '__name__')])




results = dict()


def sample_zero_truncated_poisson(rate):
    """Draw Poisson observations conditional on each count being positive."""
    counts = np.random.poisson(rate)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = np.random.poisson(rate[is_zero])
        is_zero = counts == 0
    return counts


def zero_truncated_poisson_mean(rate):
    """Compute the conditional mean independently of Kanly's link code."""
    rate = np.asarray(rate, dtype=float)
    small = rate < 1e-5
    safe_rate = np.where(small, 1.0, rate)
    mean = safe_rate / (-np.expm1(-safe_rate))
    rate2 = rate * rate
    mean_small = 1.0 + rate / 2.0 + rate2 / 12.0 - rate2 ** 2 / 720.0
    return np.where(small, mean_small, mean)


def fit_zero_truncated_poisson_reference(endog, exog, var_weights=None):
    """Fit a ZTP log-rate regression by direct maximum likelihood.

    Statsmodels does not provide a zero-truncated Poisson GLM family, so this
    independent SciPy optimization is the reference used by the parity sweep.
    """
    endog = np.asarray(endog, dtype=float)
    exog = np.asarray(exog, dtype=float)
    weights = (
        np.ones_like(endog)
        if var_weights is None
        else np.asarray(var_weights, dtype=float)
    )

    def objective(params):
        eta = exog @ params
        rate = np.exp(eta)
        loglike_obs = (
            endog * eta - rate - gammaln(endog + 1.0)
            - np.log(-np.expm1(-rate))
        )
        return -np.dot(weights, loglike_obs)

    def gradient(params):
        eta = exog @ params
        rate = np.exp(eta)
        score_eta = endog - zero_truncated_poisson_mean(rate)
        return -(exog.T @ (weights * score_eta))

    result = minimize(
        objective,
        np.zeros(exog.shape[1]),
        jac=gradient,
        method='BFGS',
        options={'gtol': 1e-10, 'maxiter': 2000},
    )
    # BFGS can report precision loss after reaching the optimum.  Treat a
    # negligible analytical score as convergence in that case.
    result.success = bool(
        result.success or np.max(np.abs(gradient(result.x))) < 1e-5
    )
    return result

for opt_method in [
    METHOD_IRLS,
    METHOD_COORD_DESC
]:
    for do_iv in [True, False]:
        for residual_inclusion in [True, False]:
            for do_weighted in [True, False]:
                for family in [
                    Poisson, Gaussian, Gamma,
                    InverseGaussian,
                    Binomial,
                    ZeroTruncatedPoisson,
                    # NegativeBinomial(alpha=.5),
                ]:

                    n = 150
                    np.random.seed(0)
                    df = pd.DataFrame()
                    df['z'] = 1.5 + 0.6 * np.random.randn(n)
                    df['x'] = 3 + 0.1 * np.random.randn(n) + 0.8 * df['z']
                    df['e'] = .3 * np.random.rand(n)
                    df['wtsvar'] = np.exp(np.random.rand(n))

                    if do_iv:
                        fit_iv = lm('x ~ z' + (' $ wtsvar' if do_weighted else ''), df)
                        df['x_pred'] = fit_iv.fittedvalues
                        df['x_ri'] = df.x.values - fit_iv.fittedvalues
                        print(fit_iv)

                    #f_sm_cls = families_sm_dict[family.name().replace('_', '')]
                    #links_sm = {l.__name__: l for l in f_sm_cls.links}

                    for link_cls in family.safe_links():

                        link = link_cls()
                        print("\n" * 3 + "=" * 100)
                        print((family.name(), link.name(), f'wtd={do_weighted}', f'iv={do_iv}', f'method={opt_method}'))
                        print("\n" * 3)

                        # print(">> ", links_sm)

                        #print(link.name(), kanly_to_sm_links.keys(), links_sm.keys())
                        # print('a, ', kanly_to_sm_links[link.name()])
                        # print('b, ', links_sm[kanly_to_sm_links[link.name()]])
                        # print('c, ', links_sm[kanly_to_sm_links[link.name()]]())
                        # f_sm = f_sm_cls(links_sm[kanly_to_sm_links[link.name()]]())
                        is_ztp = family.name() == 'ZERO_TRUNCATED_POISSON'
                        if not is_ztp:
                            link_sm = upper_to_sm_link_dict[link.__class__.__name__.upper().replace('_', '')]
                            if not isinstance(link_sm, Link):
                                link_sm = link_sm()
                            f_sm = upper_to_sm_family_dict[family.__name__.upper().replace('_', '')](
                                link_sm
                            )

                        if family.name() == 'BINOMIAL':
                            if link.name() == 'IDENTITY':
                                df['y'] = .5 + .05 * df.x
                                df['y'] = (np.random.rand(n) < df['y']).astype(float)
                            else:
                                df['y'] = np.exp(-5 + 1.5 * df.x + df['e'])
                                df['y'] /= (1.0 + df['y'])
                                df['y'] = (np.random.rand(n) < df['y']).astype(float)
                        elif is_ztp:
                            # The ZTP link models log(lambda), while its inverse
                            # returns E[Y | Y > 0], so simulate from lambda
                            # explicitly rather than treating the mean as a rate.
                            lin_pred = -0.5 + 0.35 * df.x + df['e']
                            df['y'] = sample_zero_truncated_poisson(np.exp(lin_pred))
                        else:

                            lin_pred = 3 + 1.5 * df.x + df['e']
                            if link.name() in ['NEGATIVE_INVERSE', 'NEGATIVE_TWO_INVERSE_SQUARED']:
                                df['y'] = link.inverse_link(-lin_pred)
                            else:
                                df['y'] = link.inverse_link(lin_pred)

                        # import matplotlib.pyplot as plt
                        # plt.scatter(df.x, df.y)
                        # plt.title((family.name(), link.name()))
                        # plt.show()

                        formula = 'y ~ x'
                        if do_iv:
                            formula += '_pred'
                            if residual_inclusion:
                                formula += ' + x_ri'

                        formula_kanly = 'y ~ x'
                        if do_iv:
                            formula_kanly += ' | z'
                        if do_weighted:
                            formula_kanly += ' $ wtsvar'

                        if is_ztp:
                            reference_predictors = ['x_pred' if do_iv else 'x']
                            if do_iv and residual_inclusion:
                                reference_predictors.append('x_ri')
                            exog_reference = np.column_stack(
                                [np.ones(n)]
                                + [df[name].values for name in reference_predictors]
                            )
                            fit_reference = fit_zero_truncated_poisson_reference(
                                df.y.values,
                                exog_reference,
                                df.wtsvar.values if do_weighted else None,
                            )
                            print(fit_reference)
                        else:
                            fit_sm = glm_sm(
                                formula,
                                df,
                                family=f_sm,
                                var_weights=df['wtsvar'] if do_weighted else None,
                            ).fit(tol=1e-12, max_iter=1000)
                            print(fit_sm.summary())

                        fit_kanly = glm(formula_kanly, df, family=family, link=link, cov_type='HC1', tol=1e-12,
                                        max_iter=5000, opt_method=opt_method,
                                        line_search_fallback=True, residual_inclusion=residual_inclusion,
                                        # start_params=results.get(
                                        #     (do_iv, do_weighted, residual_inclusion, family.name(), link.name()),
                                        #     None),
                                        pick_default_start=True,
                                        debug=False,
                                        )
                        print(fit_kanly)

                        param_kanly = (
                                fit_kanly.params
                                * (-1. if link.name() in ['NEGATIVE_INVERSE', 'NEGATIVE_TWO_INVERSE_SQUARED'] else 1.)
                                * (2.0 if link.name() == 'NEGATIVE_TWO_INVERSE_SQUARED' else 1.0)
                        )

                        reference_params = (
                            fit_reference.x if is_ztp else fit_sm.params.values
                        )
                        tbl = pd.DataFrame(
                            {'reference': reference_params,
                             'kanly': param_kanly.values},
                            index=fit_kanly.params.index
                        )
                        print(tbl)

                        try:
                            assert_array_almost_equal(
                                tbl.reference,
                                tbl.kanly,
                                decimal=5
                            )
                            assert fit_kanly.converged
                            if is_ztp:
                                assert fit_reference.success
                        except Exception as e:
                            failed.append(
                                (
                                    [(do_iv, do_weighted, residual_inclusion, opt_method, family.name(),
                                      link.name())],
                                    [tbl, ],
                                    [pd.Series({
                                        'reference llf': (
                                            -fit_reference.fun if is_ztp else fit_sm.llf
                                        ),
                                        'fit_kn llf': fit_kanly.llf,
                                    })]
                                )
                            )
                            print(failed[-1])

                            print(glm(formula_kanly, df, family=family, link=link, cov_type='HC1', tol=1e-10,
                                      max_iter=5000, opt_method=METHOD_IRLS,
                                      line_search_fallback=True, residual_inclusion=residual_inclusion,
                                      # start_params=results.get(
                                      #     (do_iv, do_weighted, residual_inclusion, family.name(), link.name()),
                                      #     None),
                                      pick_default_start=True,
                                      debug=False,
                                      ))

                        results[(do_iv, do_weighted, residual_inclusion, family.name(), link.name())] = fit_kanly.params

print("\n" * 5, "=" * 100, "\n" * 5)

for f in failed:
    print('\n\n')
    for c in f:
        print(c)

print("\n" * 5, "=" * 100, "\n" * 5)

print(f"Num Failed {len(failed)}")
for f in failed:
    print(f[0])
