"""
Bayesian Linear Regression Results with conjugate NormalInverseGammaPrior
"""
from __future__ import absolute_import, print_function

import datetime

import numpy as np
import pandas as pd

from kanly.bayes.bayesian_linear_regression_conjugate_prior.normal_inverse_gamma import NormalInverseGamma
from kanly.regression.regression_results_base import TestingResultsBase
import scipy.linalg
from pandas import DataFrame, Series
from scipy.linalg import pinv
from scipy.stats import t as tdist, invgamma, multivariate_t

from kanly import __version__

DEFAULT_TEST_LEVEL = .1


class BayesianLinearRegressionResults(TestingResultsBase):
    """
    Bayesian Linear Regression Results with conjugate NormalInverseGammaPrior
    """

    def __init__(self, nobs, num_params,
                 is_weighted, prior: NormalInverseGamma, posterior: NormalInverseGamma,
                 has_intercept, has_implicit_constant,
                 endog_name=None, weights_name=None,
                 fittedvalues=None, resid=None,
                 rsquared=None,
                 fit_elasped=None,
                 model_elapsed=None,
                 param_names=None, model=None, specification_name=None):
        """Initialize Bayesian linear-regression result object.

        Args:
            nobs: Number of observations.
            num_params: Number of model parameters.
            is_weighted: Whether weighted fitting was used.
            prior: Prior distribution object.
            posterior: Posterior distribution object.
            has_intercept: Whether model has explicit intercept.
            has_implicit_constant: Whether regressors imply an intercept.
            endog_name: Optional outcome label.
            weights_name: Optional weights label.
            fittedvalues: Optional fitted values.
            resid: Optional residual vector.
            rsquared: Optional R-squared value.
            fit_elasped: Fit elapsed seconds.
            model_elapsed: Model-build elapsed seconds.
            param_names: Optional ordered parameter names.
            model: Optional originating model object.
            specification_name: Optional fit label.
        """

        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

        self.is_weighted = is_weighted

        self.num_params = num_params
        self.nobs = nobs
        self.method = 'BLM'

        if param_names is None:
            param_names = [f'<x{j}>' for j in range(self.num_params-1)] + ['__sigma2']
        self.param_names = param_names
        if endog_name is None:
            endog_name = '<y>'
        self.endog_name = endog_name
        if is_weighted:
            if weights_name is None:
                weights_name = '<wts>'
        else:
            weights_name = None
        self.weights_name = weights_name

        self.fit_elapsed = fit_elasped
        self.model_elapsed = model_elapsed

        self.rsquared = rsquared
        self.fittedvalues = fittedvalues
        self.resid = resid

        self.model = model
        self.specification_name = specification_name

        self.has_intercept = has_intercept
        self.has_implicit_constant = has_implicit_constant

        self._set_prior_posterior(prior, posterior)


    def _set_prior_posterior(self, prior: NormalInverseGamma, posterior: NormalInverseGamma):
        """Store conjugate prior/posterior and derive parameter moments/distributions.

        Args:
            prior: Prior distribution object.
            posterior: Posterior distribution object.
        """
        self.prior: NormalInverseGamma = prior
        self.posterior: NormalInverseGamma = posterior

        self.scale_beta = pinv(self.posterior.Lambda)
        b, a = self.posterior.b, self.posterior.a
        t_df = 2 * a
        self.cov_params_beta = (b / a) * self.scale_beta * t_df / (t_df - 2)
        bse_beta_marginal = np.sqrt(np.diag(self.cov_params_beta))

        mean_sigma2 = b / (a - 1)
        var_sigma2 = b ** 2 / ((a - 1) ** 2 * (a - 2))

        self._params = np.hstack([self.posterior.mu, [mean_sigma2]])
        self.mean_params = Series(self._params, index=self.param_names)
        self.map_params = Series(np.hstack([self.posterior.mu, self.posterior.b / (self.posterior.a + 1)]),
                                 index=self.param_names)
        self.params = self.mean_params

        self._bse = np.hstack([bse_beta_marginal, var_sigma2 ** .5])
        self.bse = Series(self._bse, index=self.param_names)
        self._cov_params = scipy.linalg.block_diag(self.cov_params_beta, [[var_sigma2]])
        self.cov_params = pd.DataFrame(
            self._cov_params,
            index=self.param_names, columns=self.param_names
        )
        self.corr_params = self._cov_params / self.bse.values.reshape((1, -1)) / self.bse.values.reshape((-1, 1))

        # Marginal posterior distributions for scalar summaries and intervals.
        self.posterior_marginal_rv = {
            **{'__sigma2': invgamma(a=a, scale=b),
               'beta': multivariate_t(loc=self.posterior.mu, shape=(b / a) * self.scale_beta, df=t_df)
               },
            **{
                k: tdist(loc=self.posterior.mu[j], scale=np.sqrt((b / a) * self.scale_beta[j, j]), df=t_df)
                for j, k in enumerate(self.param_names[:-1])
            }
        }

        self.coef = self.params.iloc[:-1].values
        self.scale = self.params.iloc[-1]

    @staticmethod
    def get_result_type():
        """Return short result-type identifier.

        Args:
            None.
        """
        return 'BLM'

    def cov_params(self):
        """Return covariance matrix of posterior parameter means.

        Args:
            None.
        """
        return self._cov_params.copy()

    def __call__(self, i, *args, **kwargs):
        """Return posterior mean for a selected parameter.

        Args:
            i: Parameter key/index.
            *args: Unused positional args.
            **kwargs: Unused keyword args.
        """
        return self.mean_params.loc[i]

    def summary(self, test_level=DEFAULT_TEST_LEVEL):
        """Return a printable summary string with metadata and posterior intervals.

        Args:
            test_level: Tail probability used for interval reporting.
        """

        sum_df = self.summary_df(test_level=test_level)
        v = sum_df.to_string().split('\n')
        l = len(v[0])

        info = np.array([
            ['Dep. Variable: ', self.endog_name],
            ['', ''],
            ['Date:', self.date],
            ['Time:  ', self.timestamp],
            ['Model Elapsed:', '%.2f s' % self.model_elapsed],
            ['Fit Elapsed:', '%.2f s' % self.fit_elapsed],
            ['Method:', 'BLM'],
            ['Weights:', self.weights_name],
            ['Intercept:', self.has_intercept],
            ['Implicit Intercept:', self.has_implicit_constant],
            ['No. Obs.: ', self.nobs],
            ['rsquared: ', "%.4f" % self.rsquared if self.rsquared is not None else 'not computed'],
        ])
        info = Series(info[:, 1], index=info[:, 0])

        s = "\n".join(['═' * l, self.get_result_name() +
                       (('\n' + self.specification_name) if self.specification_name is not None else ''),
                       "═" * l,
                       '']
                      + info.to_string().split('\n') + ['\n'])

        s += "\n".join(["═" * l] + [v[0]] + ["─" * l] + v[1:] + ['─' * l])
        vers_str = "[kanly package, v=%s]" % __version__
        s += "\n" + (" " * max(l - len(vers_str), 0)) + vers_str + "\n"

        return s

    def predict(self, data=None, params=None, debug=False, index=None, *args, **kwargs):
        """Predict with fitted model or supplied params/data.

        Args:
            data: Optional new dataset for prediction.
            params: Optional parameter vector/series.
            debug: Debug flag forwarded to model predictor.
            index: Optional row subset.
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.
        """

        if params is None and data is None:
            return self.fittedvalues.copy()

        if params is None:
            params = self.params.copy()

        return self.model.predict(params, data=data, index=index, debug=debug, *args, **kwargs)

    def equitail_credible_interval(self, test_level=DEFAULT_TEST_LEVEL):
        """Equal-tail credible intervals from each marginal posterior distribution.

        Args:
            test_level: Tail probability for interval construction.
        """
        return pd.DataFrame(
            [[self.posterior_marginal_rv[k].ppf(test_level / 2), self.posterior_marginal_rv[k].ppf(1 - test_level / 2)]
             for k in self.param_names],
            columns=[test_level / 2, 1 - test_level / 2],
            index=self.param_names)

    def summary_df(self, test_level=DEFAULT_TEST_LEVEL):
        """Tabular posterior summary with means, std errors, and credible intervals.

        Args:
            test_level: Tail probability for credible interval columns.
        """
        t_df = 2 * self.posterior.a
        df_results = DataFrame(
            {
                'mean params': self.mean_params,
                'std': self.bse,
                f'[{test_level / 2}, ': np.hstack(
                    [tdist.ppf(test_level / 2, loc=self.mean_params[:-1], scale=self.bse[:-1], df=t_df),
                     [invgamma.ppf(test_level / 2, a=self.posterior.a, scale=self.posterior.b)]]),
                f' {1 - test_level / 2}]': np.hstack(
                    [tdist.ppf(1 - test_level / 2, loc=self.mean_params[:-1], scale=self.bse[:-1], df=t_df),
                     [invgamma.ppf(1 - test_level / 2, a=self.posterior.a, scale=self.posterior.b)]]),
                'median': np.hstack([self.posterior.mu, invgamma.ppf(0.5, a=self.posterior.a, scale=self.posterior.b)]),
                'map': self.map_params,
            },
            index=self.param_names
        )
        return df_results

    # def compute_posterior_predictive(self, data=None, mu_beta=None, var_beta=None, sigma2=None, full=False):
    #     # TODO
    #
    # @staticmethod
    # def _compute_posterior_predictive_internal(exog=None, mu_beta=None, var_beta=None, sigma2=None, full=False):
    #     # TODO weights?
    #
    #     if isspmatrix(exog):
    #         fitted = exog.dot(mu_beta.reshape((-1, 1))).toarray().flatten()
    #     else:
    #         fitted = exog.dot(mu_beta)
    #
    #     if full:
    #         if isspmatrix(exog):
    #             return fitted, exog.dot(csc_matrix(var_beta)).dot(exog.transpose()).toarray() + sigma2
    #         else:
    #             return fitted, exog.dot(var_beta).dot(exog.transpose()) + sigma2
    #     else:
    #         eigvals, eigvecs = np.linalg.eigh(var_beta)
    #         rt_var_beta = eigvecs.dot(np.diag(eigvals ** .5))
    #         if isspmatrix(exog):
    #             return fitted, exog.dot(csc_matrix(rt_var_beta)).sum(axis=1).toarray().flatten() + sigma2
    #         else:
    #             return fitted, exog.dot(rt_var_beta).sum(axis=1) + sigma2

    @staticmethod
    def get_result_name():
        """Return user-facing name of this results object.

        Args:
            None.
        """
        return "Bayesian Linear Regression (Conjugate Prior)"

    def get_header_info_array(self):
        """Return summary header metadata array.

        Args:
            None.
        """
        raise NotImplementedError

    def get_footer_info(self, *args, **kwargs):
        """Return summary footer metadata.

        Args:
            *args: Unused positional args.
            **kwargs: Unused keyword args.
        """
        raise NotImplementedError
