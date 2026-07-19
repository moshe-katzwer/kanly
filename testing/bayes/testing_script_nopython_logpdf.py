from numpy.testing import assert_allclose
from scipy.stats import beta, cauchy, laplace, expon, t, gamma, invgamma, \
    lognorm, logistic, gennorm, chi2, multivariate_normal, halfnorm, truncnorm, \
    pareto, halfcauchy, multivariate_t, loguniform, genextreme, f, weibull_min

from kanly.stats.distributions.nopython_frozen_logpdf import *
from kanly.stats.distributions.nopython_logpdf import *

for nopython in [True, False]:
    print('nopython = ', nopython)
    combos = [
        (logistic, logpdf_logistic, nopython_logpdf_logistic, dict(loc=2, scale=.665), [-3, 1]),
        (f, logpdf_f, nopython_logpdf_f, dict(dfn=10, dfd=5, loc=1, scale=3), [1.01, 4]),
        (genextreme, logpdf_genextreme, nopython_logpdf_genextreme, dict(c=0, loc=2, scale=.5), [5, 6]),
        (genextreme, logpdf_genextreme, nopython_logpdf_genextreme, dict(c=1.1, loc=2, scale=.5), [-1, 0]),
        (genextreme, logpdf_genextreme, nopython_logpdf_genextreme, dict(c=-1.1, loc=2, scale=.5), [3, 5]),
        (truncnorm, logpdf_truncnorm, nopython_logpdf_truncnorm, dict(a=1, b=10, loc=2, scale=.5), [5, 6]),
        (chi2, logpdf_chi2, nopython_logpdf_chi2, dict(df=3,loc=2, scale=.5), [2.1, 6]),
        (expon, logpdf_expon, nopython_logpdf_expon, dict(loc=2, scale=.5), [2.1, 6]),
        (invgamma, logpdf_invgamma, nopython_logpdf_invgamma, dict(a=7, loc=2, scale=.5), [2.1, 6]),
        (cauchy, logpdf_cauchy, nopython_logpdf_cauchy, dict(loc=2, scale=.5), [3, 6]),
        (halfcauchy, logpdf_halfcauchy, logpdf_halfcauchy, dict(loc=-1, scale=.5), [-.99, 4]),
        (laplace, logpdf_laplace, nopython_logpdf_laplace, dict(loc=2, scale=.5), [3, 6]),
        (gamma, logpdf_gamma, nopython_logpdf_gamma, dict(a=5, loc=2, scale=.5), [3, 6]),
        (norm, logpdf_norm, nopython_logpdf_norm, dict(loc=2, scale=.5), [-1, 2]),
        (halfnorm, logpdf_halfnorm, nopython_logpdf_halfnorm, dict(loc=.5, scale=.665), [.5, 8]),
        (beta, logpdf_beta, nopython_logpdf_beta, dict(a=5, b=1, loc=2, scale=2), [2.001, 2.5]),
        (lognorm, logpdf_lognorm, nopython_logpdf_lognorm, dict(s=4, loc=.5, scale=.665), [2.1, 5]),
        (t, logpdf_t, nopython_logpdf_t, dict(df=2, loc=2, scale=.665), [-3, 1]),
        (pareto, logpdf_pareto, nopython_logpdf_pareto, dict(b=3, loc=2, scale=.665), [3.4, 4]),
        (weibull_min, logpdf_weibull_min, nopython_logpdf_weibull_min, dict(c=3, loc=2, scale=.665), [3.4, 4]),
    ]
    for func_scipy, func2, func_nopython, kwargs, xbnd in combos:
        xrng = np.linspace(*xbnd, 5)

        if nopython:
            func2 = func_nopython

        rv1 = func_scipy(**kwargs).logpdf(xrng)
        rv2 = func2(xrng, **kwargs)

        print('\t', func_scipy.name, end='')
        assert_allclose(rv1, rv2, rtol=1e-4, atol=.0001)
        print(' ... passed')
        # print(rv1[:5], rv2[:5])

for nopython in [True, False]:
    print('nopython = ', nopython)
    for func_scipy, func2, kwargs, xbnd in [
        (f, get_frozen_logpdf_f, dict(dfn=10, dfd=5, loc=1, scale=3), [1.01, 4]),
        (genextreme, get_frozen_logpdf_genextreme, dict(c=1.1, loc=2, scale=.5), [-1, 0]),
        (loguniform, get_frozen_logpdf_loguniform, dict(a=1, b=2, loc=3, scale=2.), [5, 7]),
        (halfcauchy, get_frozen_logpdf_halfcauchy, dict(loc=-1, scale=.5), [-.99, 2]),
        (norm, get_frozen_logpdf_norm, dict(loc=2, scale=.5), [-1, 2]),
        (halfnorm, get_frozen_logpdf_halfnorm, dict(loc=1, scale=.5), [1.01, 2]),
        (multivariate_normal, get_frozen_logpdf_multivariate_normal,
         dict(mean=np.ones(2), cov=[[1, .2], [.2, 3]]), None),
        (chi2, get_frozen_logpdf_chi2, dict(df=3., loc=.2, scale=4.), [1, 4]),
        (invgamma, get_frozen_logpdf_invgamma, dict(a=2, loc=.5, scale=.665), [.51, 4]),
        (beta, get_frozen_logpdf_beta, dict(a=5, b=1, loc=.4, scale=2), [.41, 2.39]),
        (cauchy, get_frozen_logpdf_cauchy, dict(loc=1.2, scale=1.678), [.41, 2.39]),
        (laplace, get_frozen_logpdf_laplace, dict(loc=1.6, scale=1.678), [.41, 2.39]),
        (expon, get_frozen_logpdf_expon, dict(loc=2, scale=2.86), [3, 10]),
        (t, get_frozen_logpdf_t, dict(df=2, loc=2, scale=.665), [-3, 1]),
        (multivariate_t, get_frozen_logpdf_multivariate_t,
         dict(df=3, loc=np.ones(2), shape=[[1, .2], [.2, 3]]), None),
        (gamma, get_frozen_logpdf_gamma, dict(a=2, loc=2, scale=.665), [2.1, 5]),
        (lognorm, get_frozen_logpdf_lognorm, dict(s=4, loc=.5, scale=.665), [2.1, 5]),
        (logistic, get_frozen_logpdf_logistic, dict(loc=.5, scale=.665), [-4, 4]),
        (gennorm, get_frozen_logpdf_gennorm, dict(beta=1.5, loc=.5, scale=.665), [-4, 4]),
        (truncnorm, get_frozen_logpdf_truncnorm, dict(a=-1, b=3, loc=1, scale=.2), [.9, 1.3]),
        (pareto, get_frozen_logpdf_pareto, dict(b=2.0, loc=1, scale=1.5), [2.5, 5]),
        (weibull_min, get_frozen_logpdf_weibull_min, dict(c=3, loc=2, scale=.665), [3.4, 4]),
    ]:

        rv1 = func_scipy(**kwargs).logpdf
        rv2 = func2(**kwargs, nopython=nopython)

        if func_scipy is multivariate_normal or func_scipy is multivariate_t:
            xrng = np.array([-5, 3.])
        else:
            xrng = np.linspace(*xbnd, 100)

        print(f'{nopython=}, {func2.__name__=}')  # , rv1(xrng)[:4], rv2(xrng)[:4])
        try:
            assert_allclose(rv1(xrng), rv2(xrng), rtol=1e-4, atol=.001)
        except Exception as e:
            print(rv1(xrng)[:10], rv2(xrng)[:10])
            raise e
