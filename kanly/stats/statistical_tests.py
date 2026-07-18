from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import spmatrix
from scipy.stats import f as F_dist, chi2 as chi2_dist, norm as norm_dist

from kanly.utils.linalg_utils import get_matrix_inverse_internal


class StatisticalTests(object):
    """Collection of inference helpers for asymptotic and simulation tests.

    The methods work with dense arrays or scipy sparse matrices, converting
    inputs to arrays before evaluating CLT simulations, delta-method tests,
    Fieller intervals, and Wald tests.
    """

    @staticmethod
    def sparse_to_dense(*args):
        """Convert sparse matrix inputs to dense arrays.

        Args:
            *args: Arrays, sparse matrices, or array-like objects to convert.

        Returns:
            List of dense ``numpy.ndarray`` objects in the same order as the
            inputs.
        """
        ret = []
        for x in list(args):
            if isinstance(x, spmatrix):
                ret.append(x.toarray())
            else:
                ret.append(np.array(x))
        return ret

    @staticmethod
    def simulate_from_clt(func, mean, cov, n_trials=100_000, seed=0):
        """Simulate a transformed estimator from a multivariate CLT.

        Args:
            func: Callable applied to each simulated parameter draw.
            mean: Mean vector of the asymptotic normal distribution.
            cov: Covariance matrix of the asymptotic normal distribution.
            n_trials: Number of Monte Carlo draws.
            seed: Random seed for reproducible simulation.

        Returns:
            Array containing ``func`` evaluated at each simulated draw.
        """
        rand = np.random.RandomState(seed)
        mean, cov = StatisticalTests.sparse_to_dense(mean, cov)
        samples = rand.multivariate_normal(mean=mean, cov=cov, size=n_trials)
        return StatisticalTests._apply_func_to_samples(func, samples)

    @staticmethod
    def _apply_func_to_samples(func, samples):
        """Apply a scalar-valued callable to each row of sampled parameters.

        Args:
            func: Callable accepting one sampled parameter vector.
            samples: Two-dimensional array of sampled parameter vectors.

        Returns:
            One-dimensional array of scalar function values.
        """
        return np.array([float(func(r)) for r in samples])

    @staticmethod
    def delta_method_test(func, mean, cov, null_hypothesis=0, step=.002, test_level=.05):
        """Run a two-sided delta-method z-test for a transformed estimate.

        Args:
            func: Scalar-valued differentiable function of the parameter vector.
            mean: Estimated parameter vector.
            cov: Estimated covariance matrix for ``mean``.
            null_hypothesis: Null value for ``func(mean)``.
            step: Symmetric finite-difference step for the numerical gradient.
            test_level: Significance level used for the confidence interval.

        Returns:
            Dictionary with estimate, confidence interval, p-value, z-statistic,
            and standard error.
        """

        mean, cov = StatisticalTests.sparse_to_dense(mean, cov)
        estimate = float(func(mean))
        f_prime = np.zeros(len(mean))

        identity = np.eye(cov.shape[0])
        for i, x in enumerate(mean):
            # Central differences approximate the gradient needed by the delta
            # method variance f'(mean)' cov f'(mean).
            f_prime[i] = (func(mean + identity[i] * step) - func(mean - identity[i] * step)) / (2.0 * step)

        var = float(np.dot(f_prime, cov).dot(f_prime))
        se = float(np.sqrt(var))

        crit_value = float(norm_dist.ppf(1.0 - test_level / 2.0))

        test_stat = float((estimate - null_hypothesis) / se)

        return {
            "estimate": estimate,
            "ci_lo": estimate - crit_value * se,
            "ci_hi": estimate + crit_value * se,
            "pvalue": 2.0 * norm_dist.sf(abs(test_stat)),
            "test_stat": test_stat,
            "std_err": se,
        }

    @staticmethod
    def clt_simulation_test(func, mean, cov, null_hypothesis=0, n_trials=100000, test_level=.05, tail='two',
                            seed=0, include_samples=False):
        """Run a Monte Carlo CLT test for a transformed estimator.

        Args:
            func: Scalar-valued function of the parameter vector.
            mean: Estimated parameter vector.
            cov: Estimated covariance matrix for ``mean``.
            null_hypothesis: Null value for ``func(mean)``.
            n_trials: Number of simulated CLT draws.
            test_level: Significance level used for the confidence interval.
            tail: ``'one'`` for one-sided p-values or ``'two'`` for two-sided.
            seed: Random seed for reproducible simulation.
            include_samples: Whether to include simulated values in the result.

        Returns:
            Dictionary with estimate, confidence interval, p-value, standard
            error, and optionally the simulated samples.
        """
        mean, cov = StatisticalTests.sparse_to_dense(mean, cov)
        param = func(mean)
        samples = StatisticalTests.simulate_from_clt(func, mean, cov, n_trials=n_trials, seed=seed)
        return StatisticalTests._test_from_samples(param, samples, null_hypothesis=null_hypothesis,
                                                   tail=tail, test_level=test_level, include_samples=include_samples)

    @staticmethod
    def _test_from_samples(estimate, samples, null_hypothesis=0, test_level=.05, tail='two', include_samples=False):
        """Build a hypothesis-test summary from simulated sample values.

        Args:
            estimate: Point estimate being tested.
            samples: Simulated sampling distribution for the estimate.
            null_hypothesis: Null value for the estimate.
            test_level: Significance level used for the empirical interval.
            tail: ``'one'`` for one-sided p-values or ``'two'`` for two-sided.
            include_samples: Whether to include ``samples`` in the result.

        Returns:
            Dictionary with estimate, empirical confidence interval, p-value,
            standard error, and optional samples.
        """
        # Appeals to CLT asymptotics
        # for p-value, see: http://qed.econ.queensu.ca/working_papers/papers/qed_wp_1127.pdf

        if tail not in ['one', 'two']:
            raise Exception("`tail` must be in ['one', 'two']!")

        ci_lo = np.quantile(samples, test_level / 2)
        ci_hi = np.quantile(samples, 1.0 - test_level / 2)

        if np.var(samples) < 1e-10:
            # Degenerate simulated distributions cannot support the empirical
            # tail-count p-value used below.
            p = float(abs(estimate - np.mean(samples)) < 1e-10)

        else:
            p = min(np.mean(samples <= null_hypothesis), np.mean(samples >= null_hypothesis))
            if tail == 'two':
                p *= 2.0

        std_err = np.std(samples)

        result = {"estimate": estimate, "ci_lo": ci_lo, "ci_hi": ci_hi, "pvalue": p, 'std_err': std_err,
                  'test_stat': None}  # (param - null_hypothesis)/std_err}
        if include_samples:
            result.update({'samples': samples})

        return result

    @staticmethod
    def ratio_fieller_test(params, cov_params, top, bottom, null_hypothesis=0, test_level=.05,
                           top_constant=0, bottom_constant=0):
        """Compute a Fieller confidence interval for a linear ratio.

        Args:
            params: Estimated parameter vector.
            cov_params: Covariance matrix for ``params``.
            top: Linear coefficients for the numerator.
            bottom: Linear coefficients for the denominator.
            null_hypothesis: Null ratio value for the p-value test.
            test_level: Significance level used for the confidence interval.
            top_constant: Constant added to the numerator.
            bottom_constant: Constant added to the denominator.

        Returns:
            Dictionary with ratio estimate, Fieller confidence interval,
            p-value, test statistic, standard error placeholder, and sometimes
            test level.
        """

        params, cov_params = StatisticalTests.sparse_to_dense(params, cov_params)
        params = np.array(params).reshape((-1, 1))

        top = np.array(top).reshape((-1, 1))
        bottom = np.array(bottom).reshape((-1, 1))

        estimate = ((top.T.dot(params) + top_constant) / (bottom.T.dot(params) + bottom_constant)).item()

        t2 = norm_dist.ppf(1.0 - test_level / 2) ** 2.0

        top_beta = np.dot(top.T, params)
        bot_beta = np.dot(bottom.T, params)

        top_c = top_constant
        bot_c = bottom_constant

        aL = (bot_c ** 2 + bot_beta ** 2 + 2 * bot_c * bot_beta)
        bL = -2.0 * (top_c * bot_c + top_c * bot_beta + bot_c * top_beta + bot_beta * top_beta)
        cL = top_c ** 2 + top_beta ** 2 + 2 * top_c * top_beta

        aR = t2 * np.dot(bottom.T, cov_params).dot(bottom)
        bR = -2 * t2 * np.dot(top.T, cov_params).dot(bottom)
        cR = t2 * np.dot(top.T, cov_params).dot(top)

        _a = aL - aR
        _b = bL - bR
        _c = cL - cR

        disc = (_b ** 2 - 4.0 * _a * _c).item()

        coef = (top - null_hypothesis * bottom)
        test_stat = ((coef.T.dot(params) - (bottom_constant * null_hypothesis - top_constant))
                          / np.sqrt(coef.T.dot(cov_params).dot(coef))).item()
        p_value = 2.0 * norm_dist.sf(abs(test_stat))

        if disc < 0:
            # A negative discriminant gives an unbounded Fieller interval.
            return {
                "estimate": estimate,
                "ci_lo": -np.inf,
                "ci_hi": np.inf,
                "pvalue": p_value,
                "test_stat": test_stat,
                "std_err": None,
                "test_level": test_level,
            }
        else:
            ci_lo = ((-_b - np.sqrt(_b ** 2 - 4 * _a * _c)) / (2 * _a)).item()
            ci_hi = ((-_b + np.sqrt(_b ** 2 - 4 * _a * _c)) / (2 * _a)).item()

            return {
                "estimate": estimate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "pvalue": p_value,
                "test_stat": test_stat,
                "std_err": None
            }

    @staticmethod
    def wald_test(params, cov_params, degrees_freedom, r_matrix=None, q=None, use_f=False):
        """Run a Wald test for linear restrictions on parameters.

        Args:
            params: Estimated parameter vector.
            cov_params: Covariance matrix for ``params``.
            degrees_freedom: Denominator degrees of freedom for the F test.
            r_matrix: Restriction matrix ``R``. Defaults to the identity.
            q: Restriction target vector. Defaults to zeros.
            use_f: If True, use an F distribution; otherwise use chi-square.

        Returns:
            Dictionary with ``test_stat`` and ``pvalue``. Returns NaNs when the
            restricted covariance matrix cannot be inverted.
        """

        if r_matrix is None:
            r_matrix = np.eye(len(params))
        if q is None:
            q = np.zeros(len(r_matrix))

        params, cov_params, r_matrix, q = StatisticalTests.sparse_to_dense(params, cov_params, r_matrix, q)

        num_restrictions = q.shape[0]

        Rbq = r_matrix.dot(params) - q
        Rbq = Rbq.reshape((-1, 1))

        cov_p = r_matrix.dot(cov_params).dot(r_matrix.transpose())
        try:
            inv_cov_p = get_matrix_inverse_internal(cov_p)
        except:
            # Singular restricted covariance matrices make the quadratic form
            # undefined, so report an explicit failed test result.
            return {"test_stat": np.nan, "pvalue": np.nan}

        stat = Rbq.transpose().dot(inv_cov_p).dot(Rbq)

        if use_f:
            stat = stat / num_restrictions
            p_value = F_dist.sf(stat, num_restrictions, degrees_freedom)
        else:
            p_value = chi2_dist.sf(stat, df=num_restrictions)

        if isinstance(stat, np.ndarray):
            stat = stat.ravel()[0]
        if isinstance(p_value, np.ndarray):
            p_value = p_value.ravel()[0]
        return {"test_stat": float(stat), "pvalue": float(p_value)}
