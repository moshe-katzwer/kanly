"""Regression-results container shared by distributional and hurdle models."""

from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from kanly.regression.regression_results_base import RegressionResultsBase


DEFAULT_TEST_LEVEL = .05


class DistributionalModelResults(RegressionResultsBase):
    """Fitted results for direct-likelihood models and separable hurdles.

    The class supplies the normal-asymptotic inference, named covariance,
    summary, likelihood, score, prediction, bootstrap, and model metadata APIs
    expected from other ``RegressionResultsBase`` subclasses.  Raw BFGS output
    and hurdle component GLM fits are retained rather than exposed as the
    public result object themselves.
    """

    def __init__(
            self, model, params, cov_params, cov_type, cov_kwds, llf,
            converged, message=None, method='BFGS-PQN',
            optimization_result=None, information=None, bread=None, meat=None,
            bootstrapped_params=None, bootstrap_string=None,
            fittedvalues=None, fit_elapsed=0.0, cov_elapsed=0.0,
            iterations=None, score_at_params=None, scale=1.0,
            positive_fit=None, hurdle_fit=None, component_cov_type=None,
            specification_name=None, test_level=DEFAULT_TEST_LEVEL,
            loglike_kwargs=None, optimizer_converged=None,
            first_order_valid=True, inference_valid=True,
            inference_issues=None, normalized_score=None,
            covariance_status=None, information_rank=None,
            information_condition=None, information_min_eigenvalue=None,
            is_quasi_likelihood=False, report_information_criteria=True):
        """Initialize a complete distributional-model regression result.

        Args:
            model: Fitted distributional or hurdle model.
            params: Combined fitted parameter vector.
            cov_params: Estimated parameter covariance matrix.
            cov_type: Public covariance estimator name.
            cov_kwds: Covariance options used during fitting.
            llf: Aggregated fitted log likelihood.
            converged: Whether estimation converged.
            message: Optimizer or component-convergence message.
            method: Estimation method label.
            optimization_result: Raw optimizer result for direct MLE models.
            information: Observed information matrix, when computed.
            bread: Inverse observed-information matrix, when computed.
            meat: Observation-score cross-product, for sandwich covariance.
            bootstrapped_params: Successful combined bootstrap draws.
            bootstrap_string: Human-readable bootstrap summary.
            fittedvalues: Optional precomputed unconditional fitted means.
            fit_elapsed: Estimation time in seconds.
            cov_elapsed: Covariance computation time in seconds.
            iterations: Iteration count or component-iteration description.
            score_at_params: Aggregated score at the fitted parameters.
            scale: Distribution or positive-component scale shown in summaries.
            positive_fit: Positive-response component result for hurdle
                models; either a GLM result or direct-likelihood result.
            hurdle_fit: Bernoulli/logit GLM result for hurdle models.
            component_cov_type: Native component covariance type for hurdles.
            specification_name: Optional summary title suffix.
            test_level: Significance level for inference.
            loglike_kwargs: Extra keywords required to evaluate likelihoods,
                such as a fitted Gamma hurdle scale.
            optimizer_converged: Raw optimizer convergence indicator.
            first_order_valid: Whether fitted scores satisfy validation.
            inference_valid: Whether covariance diagnostics permit inference.
            inference_issues: Human-readable fit or covariance problems.
            normalized_score: Maximum score magnitude divided by total weight.
            covariance_status: Short covariance diagnostic description.
            information_rank: Numerical rank of observed information.
            information_condition: Condition number of observed information.
            information_min_eigenvalue: Smallest information eigenvalue.
            is_quasi_likelihood: Whether the fit is estimating-equation based
                rather than a fully maximized likelihood.
            report_information_criteria: Whether AIC and BIC are meaningful.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        param_names = list(model.get_param_names())
        if len(params) != len(param_names):
            raise ValueError('params and model parameter names must align')

        num_intercepts = self._count_intercepts(model)
        df_model = max(len(params) - num_intercepts, 0)
        df_resid = max(model.nobs - len(params), 0)
        endog_name = model.endog_name or 'y'

        # Distributional-model likelihood inference is asymptotically normal,
        # including when finite residual df are displayed for diagnostics.
        super().__init__(
            model.nobs,
            params,
            cov_params,
            df_model,
            df_resid,
            df_resid,
            exog_names=param_names,
            endog_name=endog_name,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            test_level=test_level,
            use_t=False,
            alpha=0.0,
            l1_ratio=0.0,
            specification_name=specification_name,
        )
        self.cov_kwds = {} if cov_kwds is None else dict(cov_kwds)

        self.model = model
        self.keep_model = True
        self.model_name = model.__class__.__name__
        self.method = method
        self.converged = bool(converged)
        self.optimizer_converged = (
            self.converged
            if optimizer_converged is None
            else bool(optimizer_converged)
        )
        self.first_order_valid = bool(first_order_valid)
        self.inference_valid = bool(inference_valid)
        self.inference_issues = (
            [] if inference_issues is None else list(inference_issues)
        )
        self.normalized_score = (
            None if normalized_score is None else float(normalized_score)
        )
        self.covariance_status = covariance_status
        self.information_rank = information_rank
        self.information_condition = information_condition
        self.information_min_eigenvalue = information_min_eigenvalue
        self.is_quasi_likelihood = bool(is_quasi_likelihood)
        self.report_information_criteria = bool(report_information_criteria)
        self.message = '' if message is None else str(message)
        self.llf = float(llf)
        self.weight_total = (
            float(self.nobs)
            if model.weights is None else float(np.sum(model.weights))
        )
        self.average_loglike = self.llf / self.weight_total
        self.num_params = len(params)
        if self.report_information_criteria:
            self.aic = -2.0 * self.llf + 2.0 * self.num_params
            self.bic = -2.0 * self.llf + np.log(self.nobs) * self.num_params
        else:
            self.aic = None
            self.bic = None
        self.scale = float(scale)

        self.optimization_result = optimization_result
        self.information = information
        self.bread = bread
        self.meat = meat
        self.fit_elapsed = float(fit_elapsed or 0.0)
        self.cov_elapsed = float(cov_elapsed or 0.0)
        self.model_elapsed = float(getattr(model, 'model_elapsed', 0.0))
        self.iterations = iterations
        self.iter = iterations
        self.options = getattr(optimization_result, 'options', None)
        self.x = self._params.copy()
        self.fun = self.llf

        # Preserve the optimizer diagnostics that callers of the historical
        # raw BFGS result commonly inspected, while keeping the unmodified
        # optimizer object available at ``optimization_result``.
        optimizer_attributes = (
            'grad_projected', 'hess', 'hess_inv', 'ferr', 'xerr',
            'bounds', 'maximize', 'optimization_path', 'time_elapsed',
            'ub_binding', 'lb_binding', 'H0', 'fun_callable',
        )
        for attribute in optimizer_attributes:
            if optimization_result is not None:
                setattr(
                    self, attribute,
                    getattr(optimization_result, attribute, None),
                )
        self.optimizer_gnorm = getattr(
            optimization_result, 'gnorm', None
        )

        if score_at_params is None:
            score_at_params = model.score(
                self._params, **(loglike_kwargs or {})
            )
        self.grad = np.asarray(score_at_params, dtype=float).reshape(-1)
        self.score_at_params = self.grad.copy()
        self.gnorm = (
            float(np.max(np.abs(self.grad))) if len(self.grad) else 0.0
        )

        self.loglike_kwargs = (
            {} if loglike_kwargs is None else dict(loglike_kwargs)
        )
        self.weights_name = model.weights_name
        self.is_weighted = model.is_weighted
        self.formula = model.formula
        self.formula_design_info = model.formula_design_info
        self.from_formula = model.from_formula
        self.valid_obs_rows = np.asarray(model.valid_obs_rows).copy()
        self.null_rows_info_dict = model.null_rows_info_dict.copy()
        self.index = model.index
        self.has_intercept = bool(model.has_intercept)
        self.has_implicit_constant = bool(model.has_implicit_constant)

        self.positive_fit = positive_fit
        self.hurdle_fit = hurdle_fit
        self.zero_fit = hurdle_fit
        self.component_cov_type = component_cov_type
        self.is_hurdle = positive_fit is not None and hurdle_fit is not None
        self.is_zero_inflated = (
            hasattr(model, 'k_inflate') and not self.is_hurdle
        )
        self.nobs_zero = int(np.count_nonzero(model.endog == 0.0))
        self.nobs_positive = int(self.nobs - self.nobs_zero)
        self.zero_fraction = self.nobs_zero / self.nobs
        if self.is_hurdle:
            self.positive_scale = float(positive_fit.scale)
            self.positive_bootstrapped_params = getattr(
                positive_fit, 'bootstrapped_params', None
            )
            self.hurdle_bootstrapped_params = getattr(
                hurdle_fit, 'bootstrapped_params', None
            )

        if fittedvalues is None:
            fittedvalues = model.predict(self._params, which='mean')
        self.fittedvalues = np.asarray(fittedvalues, dtype=float).reshape(-1)
        self.endog_predicted = self.fittedvalues
        self.resid = model.endog - self.fittedvalues
        self.resid_response = self.resid

        self._set_model_specific_attributes()
        if self.is_hurdle:
            self.resid_zero = (
                (model.endog == 0.0).astype(float) - self.zero_probability
            )
            self.resid_positive = np.full(self.nobs, np.nan, dtype=float)
            positive = model.endog > 0.0
            self.resid_positive[positive] = (
                model.endog[positive] - self.positive_mean[positive]
            )

        self.bootstrap_string = bootstrap_string
        self.bootstrapped_params = None
        if bootstrapped_params is not None:
            self.set_bootstrapped_params(
                np.asarray(bootstrapped_params, dtype=float),
                cov_string=bootstrap_string,
            )
            if self.is_hurdle:
                k_positive = model.k_positive
                self.positive_bootstrapped_params = (
                    self.bootstrapped_params[:, :k_positive].copy()
                )
                self.hurdle_bootstrapped_params = (
                    self.bootstrapped_params[:, k_positive:].copy()
                )
        self.cov_string = self._get_covariance_description()
        self.standard_errors = getattr(self, 'bse', None)

    @staticmethod
    def _first_column_is_constant(exog):
        """Return whether an array starts with a constant column."""
        exog = np.asarray(exog)
        return bool(
            exog.ndim == 2 and exog.shape[1]
            and np.allclose(exog[:, 0], exog[0, 0])
        )

    @classmethod
    def _count_intercepts(cls, model):
        """Count explicit intercepts across one- and two-part designs."""
        count = int(
            bool(getattr(model, 'has_intercept', False))
            or cls._first_column_is_constant(model.exog)
        )
        if hasattr(model, 'exog_infl'):
            count += int(cls._first_column_is_constant(model.exog_infl))
        return count

    def _set_model_specific_attributes(self):
        """Attach transformed dispersion and two-part prediction metadata."""
        if 'log_alpha' in self.param_names:
            location = self.param_names.index('log_alpha')
            self.log_alpha = float(self._params[location])
            self.dispersion = float(np.exp(self.log_alpha))
            self.dispersion_name = 'alpha = exp(log_alpha)'
        elif 'alpha' in self.param_names:
            location = self.param_names.index('alpha')
            self.dispersion = float(self._params[location])
            self.dispersion_name = 'alpha'
        else:
            self.dispersion = None
            self.dispersion_name = None

        if hasattr(self.model, 'negative_binomial_p'):
            self.negative_binomial_p = self.model.negative_binomial_p
        elif hasattr(self.model, 'p'):
            self.generalized_poisson_p = self.model.p

        if self.is_hurdle:
            self.zero_probability = self.model.predict(
                self._params, which='zero_probability'
            )
            self.positive_probability = 1.0 - self.zero_probability
            self.positive_mean = self.model.predict(
                self._params, which='positive_mean'
            )
        elif self.is_zero_inflated:
            self.inflation_probability = self.model.predict(
                self._params, which='inflation_probability'
            )
            self.count_mean = self.model.predict(
                self._params, which='count_mean'
            )
            self.zero_probability = self.model.predict(
                self._params, which='zero_probability'
            )
            self.positive_probability = 1.0 - self.zero_probability

    def _get_covariance_description(self):
        """Return an estimator-specific covariance explanation."""
        if self.is_hurdle:
            if str(self.cov_type).upper() in {'SANDWICH', 'HC1'}:
                correction = (
                    ' with an HC1 finite-sample correction'
                    if str(self.cov_type).upper() == 'HC1' else ''
                )
                return (
                    'Covariance uses the full combined observation-score '
                    f'sandwich{correction}; its bread is block diagonal but '
                    'cross-component score covariance is retained.'
                )
            if str(self.cov_type).upper() == 'BOOTSTRAP':
                return (
                    self.bootstrap_string
                    or 'Covariance uses coherent full-sample Bayesian '
                    'bootstrap refits of both hurdle components.'
                )
            return (
                'Covariance is block diagonal; the positive-response and '
                f'zero-hurdle blocks each use the component '
                f'{self.component_cov_type} estimator.'
            )
        descriptions = {
            'NONROBUST': (
                'Covariance is the inverse observed information matrix.'
            ),
            'SANDWICH': (
                'Covariance uses the observation-score sandwich estimator.'
            ),
            'HC1': (
                'Covariance uses the GLM HC1 sandwich estimator.'
            ),
            'BOOTSTRAP': (
                self.bootstrap_string
                or 'Covariance uses Bayesian-bootstrap refits.'
            ),
        }
        description = descriptions.get(
            str(self.cov_type).upper(),
            f'Covariance type: {self.cov_type}.',
        )
        return description

    def loglike(self, params=None, **kwargs):
        """Evaluate the aggregated model likelihood."""
        if params is None:
            params = self._params
        evaluation_kwds = self.loglike_kwargs.copy()
        evaluation_kwds.update(kwargs)
        return self.model.loglike(np.asarray(params), **evaluation_kwds)

    def loglike_obs(self, params=None, **kwargs):
        """Evaluate unweighted observation likelihood contributions."""
        if params is None:
            params = self._params
        evaluation_kwds = self.loglike_kwargs.copy()
        evaluation_kwds.update(kwargs)
        return self.model.loglike_obs(np.asarray(params), **evaluation_kwds)

    def score(self, params=None, **kwargs):
        """Evaluate the aggregated model score."""
        if params is None:
            params = self._params
        evaluation_kwds = self.loglike_kwargs.copy()
        evaluation_kwds.update(kwargs)
        return self.model.score(np.asarray(params), **evaluation_kwds)

    def score_obs(self, params=None, **kwargs):
        """Evaluate unweighted observation score contributions."""
        if params is None:
            params = self._params
        evaluation_kwds = self.loglike_kwargs.copy()
        evaluation_kwds.update(kwargs)
        return self.model.score_obs(np.asarray(params), **evaluation_kwds)

    def predict(
            self, exog=None, params=None, exog_infl=None, which='mean',
            data=None, index=None, debug=False):
        """Predict means or probabilities from numeric designs or new data.

        With no arguments, returns the in-sample unconditional fitted mean.
        Models constructed from formulas can evaluate those formulas on new
        DataFrame or dict-like data.
        """
        if (exog is None and exog_infl is None and data is None
                and params is None
                and which == 'mean'):
            return self.fittedvalues.copy()
        if params is None:
            params = self._params
        return self.model.predict(
            np.asarray(params), exog=exog, exog_infl=exog_infl, which=which,
            data=data, index=index, debug=debug,
        )

    def summary_df(self, test_level=DEFAULT_TEST_LEVEL):
        """Return distributional-model estimates with normal ``z`` inference."""
        result = pd.DataFrame(
            {'coef': self.params}, index=self.params.index
        )
        if self.did_compute_var_covar():
            result['std err'] = self._bse
            result['z'] = self._tvalues
            result['p>|z|'] = self._pvalues
            ci_lo, ci_hi = self._conf_int(test_level)
            result['[%.3f, ' % (test_level / 2)] = ci_lo
            result['%.3f]' % (1.0 - test_level / 2)] = ci_hi
            result['stars'] = [
                '****' if p < .001
                else ('*** ' if p < .01
                      else ('**  ' if p < .05
                            else ('*   ' if p < .1 else '')))
                for p in self._pvalues
            ]
        return result.copy()

    def summary_table(self, test_level=DEFAULT_TEST_LEVEL):
        """Return a compact programmatic table with ``z`` statistics."""
        table = self.summary_df(test_level=test_level)
        if 'stars' in table:
            del table['stars']
            table.columns = [
                'param', 'bse', 'z', 'p', 'ci_lo', 'ci_hi'
            ]
        return table

    def get_result_type(self):
        """Return the short result label used by comparison utilities."""
        return 'HURDLE' if self.is_hurdle else 'DISTRIBUTIONAL'

    def get_result_name(self):
        """Return a model-specific title for printed summaries."""
        return f'{self.model_name} Results'

    def get_header_info_array(self):
        """Build distribution- and hurdle-specific summary metadata."""
        iterations = self.iterations
        if isinstance(iterations, (tuple, list)):
            iterations = ' / '.join(str(value) for value in iterations)
        header = [
            ('Model:', self.model_name),
            ('Date:', self.date),
            ('Method:', self.method),
            ('Time:', self.timestamp),
            ('Nobs:', self.nobs),
            ('Df Residuals:', self.df_resid),
            ('Df Model:', self.df_model),
            ('No. Params:', self.num_params),
            (
                'Quasi-Likelihood:' if self.is_quasi_likelihood
                else (
                    'Weighted Log-Likelihood:' if self.is_weighted
                    else 'Log-Likelihood:'
                ),
                '%.4e' % self.llf,
            ),
            (
                'Average Quasi-LL:' if self.is_quasi_likelihood
                else 'Average LL:',
                '%.4e' % self.average_loglike,
            ),
            ('Cov. Type:', self.cov_type),
            ('Weights:', (
                self.weights_name
                if self.weights_name is not None
                else ('Provided' if self.is_weighted else 'None')
            )),
            ('Converged:', self.converged),
            ('Optimizer Converged:', self.optimizer_converged),
            ('Inference Valid:', self.inference_valid),
            ('Iterations:', iterations if iterations is not None else 'N/A'),
            ('Max |score|:', '%.2e' % self.gnorm),
            ('Zero Nobs:', self.nobs_zero),
            ('Zero Fraction:', '%.4f' % self.zero_fraction),
            ('Model Time:', '%.2fs' % self.model_elapsed),
            ('Fit Time:', '%.2fs' % self.fit_elapsed),
            ('Cov Time:', '%.2fs' % self.cov_elapsed),
        ]
        if self.normalized_score is not None:
            header.append((
                'Scaled |score|:', '%.2e' % self.normalized_score
            ))
        if self.information_rank is not None:
            header.append((
                'Information Rank:',
                f'{self.information_rank}/{self.num_params}',
            ))
        if self.information_condition is not None:
            header.append((
                'Information Cond.:', '%.2e' % self.information_condition
            ))
        if self.report_information_criteria:
            header.extend([
                ('AIC:', '%.4f' % self.aic),
                ('BIC:', '%.4f' % self.bic),
            ])
        if self.dispersion is not None:
            header.append(('Dispersion:', '%.4e' % self.dispersion))
        if hasattr(self, 'generalized_poisson_p'):
            header.append(('GP p:', self.generalized_poisson_p))
        if hasattr(self, 'negative_binomial_p'):
            header.append(('NB p:', self.negative_binomial_p))
        if self.is_zero_inflated:
            header.extend([
                ('Count Params:', self.model.exog.shape[1]),
                ('Inflation Params:', self.model.k_inflate),
            ])
        if self.is_hurdle:
            header.extend([
                ('Positive Nobs:', self.nobs_positive),
                ('Positive Scale:', '%.4e' % self.positive_scale),
                ('Component Cov:', self.component_cov_type),
            ])
        return header

    def get_footer_info(self, *args, **kwargs):
        """Build likelihood, dispersion, covariance, and convergence notes."""
        del args
        lines = []
        if self.is_hurdle:
            lines.append(
                'The Bernoulli/logit zero hurdle and positive-response model '
                'were estimated separately.'
            )
            positive_description = getattr(
                self.model, '_positive_component_description', None
            )
            if positive_description is not None:
                lines.append(
                    f'Positive component: {positive_description()}.'
                )
            else:
                lines.append(
                    f'Positive family/link: '
                    f'{self.positive_fit.family.name()} / '
                    f'{self.positive_fit.link.name()}.'
                )
            lines.append(
                'Parameter order is all positive-response parameters '
                'followed by hurdle coefficients.'
            )
            if self.is_quasi_likelihood:
                lines.append(
                    'The Gamma positive component uses GLM estimating '
                    'equations with Pearson-estimated scale. This is a '
                    'quasi-likelihood fit; likelihood-based AIC and BIC are '
                    'not reported.'
                )
        else:
            lines.append(
                'Parameters were estimated by quasi-likelihood-like '
                'optimization.'
                if self.is_quasi_likelihood else
                'Parameters were estimated by maximum likelihood.'
            )

        if self.is_zero_inflated:
            lines.append(
                'Inflation coefficients use a logit for the structural-zero '
                'probability.'
            )
        if self.dispersion is not None:
            lines.append(
                f'Distribution dispersion {self.dispersion_name}: '
                f'{self.dispersion:.6g}.'
            )
        if hasattr(self, 'generalized_poisson_p'):
            lines.append(
                f'Generalized-Poisson parameterization: '
                f'p={self.generalized_poisson_p:g}.'
            )
        if hasattr(self, 'negative_binomial_p'):
            lines.append(
                'Negative-binomial-P parameterization: '
                f'p={self.negative_binomial_p:d}; alpha=exp(log_alpha).'
            )
        if self.is_weighted:
            lines.append(
                'Importance weights multiply observation likelihood and '
                'score contributions during aggregation.'
            )
            if not self.report_information_criteria:
                lines.append(
                    'Likelihood-based AIC and BIC are not reported for '
                    'importance-weighted objectives.'
                )

        if self.covariance_status:
            lines.append(f'Covariance status: {self.covariance_status}.')
        if self.inference_issues:
            lines.append(
                'Inference warning: ' + ' '.join(self.inference_issues)
            )

        if self.did_compute_var_covar():
            lines.append(
                'Used asymptotic Normal inference at test level '
                f"{kwargs.get('test_level', self.test_level):.4f}."
            )
            lines.append(self._get_covariance_description())
        else:
            lines.append('Parameter covariance was not computed.')
        if self.message:
            lines.append(f'Convergence message: {self.message}')
        return '\n'.join(lines)

    def __str__(self):
        """Return the formatted regression summary."""
        return self.summary()

    def __repr__(self):
        """Return the formatted regression summary."""
        return self.summary()


__all__ = ['DistributionalModelResults']
