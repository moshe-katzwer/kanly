"""
Conjugate prior for bayesian linear regression
"""
from __future__ import absolute_import, print_function

import numpy as np


class NormalInverseGamma(object):
    """
    Conjugate prior for bayesian linear regression
    """

    def __init__(self, mu=0, Lambda=0, a=0, b=0, p=None):
        """
        Conjugate prior for bayesian linear regression

            * sigma2 ~ InvGamma(a, b)
            * beta | sigma2 ~ Normal( mu, sigma2 * inv(lambda) )

        Consider a setting of the form

            y[i] = f(x[i]; beta) + sigma * epsilon[i]

        where epsilon ~ N(0,1)

        Callable here returns something that is the same as the log pdf
        of the implied prior, *up to a constant*.

        :param mu: mean on beta parameters
        :param lambda: inverse variance-covariance matrix on beta parameters
        :param a: shape parameter in inverse gamma prior on sigma2
        :param b: scale parameter in inverse gamma prior on sigma2

        Args:
            mu: Scalar/vector prior mean for regression coefficients.
            Lambda: Scalar/vector/matrix prior precision for coefficients.
            a: Inverse-gamma shape hyperparameter for ``sigma2``.
            b: Inverse-gamma scale hyperparameter for ``sigma2``.
            p: Number of coefficient parameters when scalars are supplied.
        """

        # Inverse-Gamma hyperparameters must be non-negative for this parameterization.
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        assert a >= 0 and b >= 0

        self.a = float(a)
        self.b = float(b)
        # applies to jeffrey's case of inverse gamma case
        self._log_pdf_sigma2 = lambda s2: -(self.a + 1) * np.log(s2) - self.b / s2

        if mu is None or Lambda is None:
            assert mu is None and Lambda is None
            assert p is not None
            flat_beta = True
        else:

            if isinstance(mu, (int, float)) and isinstance(Lambda, (int, float)):
                assert p is not None
                assert isinstance(p, int) and p > 0
            else:
                if isinstance(mu, (int, float)):
                    p = np.shape(Lambda)[0]
                else:
                    p = np.shape(mu)[0]

            if isinstance(mu, (int, float)):
                mu = np.full(p, float(mu), dtype=float)
            else:
                mu = np.array(mu).flatten()
                assert mu.shape[0] == p
            if isinstance(Lambda, (int, float)):
                Lambda = np.diag([float(Lambda)] * p)
            elif np.ndim(Lambda) == 1:
                Lambda = np.diag(Lambda)
            assert Lambda.shape == (p, p)
            flat_beta = False
            mu = np.array(mu, dtype=float)
            Lambda = np.array(Lambda, dtype=float)
            zero_diag = np.diag(Lambda) <= 0
            if np.any(zero_diag):
                Lambda[zero_diag, :] = 0
                Lambda[:, zero_diag] = 0

            # Degenerate precision matrix => effectively flat prior on beta.
            if np.all(Lambda == 0):
                flat_beta = True
                Lambda = mu = 0.0

        self.p = p
        self.mu = mu
        self.Lambda = Lambda
        self.flat_beta = flat_beta

        if self.flat_beta:
            self._log_pdf_beta_conditional_sigma2 = lambda beta, s2: 0.0
        else:
            self._log_pdf_beta_conditional_sigma2 = lambda beta, s2: (
                    -np.dot(self.mu - beta, self.Lambda).dot(self.mu - beta) / (2.0 * s2)
                    - self.p / 2 * np.log(s2)
            )

    def __call__(self, params):
        """Evaluate log-prior kernel (up to additive constants) at ``[beta..., sigma2]``.

        Args:
            params: Concatenated parameter vector ``[beta..., sigma2]``.
        """
        params = np.asarray(params)
        return self._log_pdf_sigma2(params[-1]) + self._log_pdf_beta_conditional_sigma2(params[:-1], params[-1])

    @staticmethod
    def build_normal_invgamma_from_penalties(mean_dict, penalty_dict, param_names, a=0, b=0):
        """Assumes that the last entry in `param_names` is the scale parameter.

        Args:
            mean_dict: Prior means keyed by parameter name.
            penalty_dict: Diagonal precision sparse_terms keyed by parameter name.
            param_names: Ordered full parameter names ending in scale parameter.
            a: Inverse-gamma shape hyperparameter.
            b: Inverse-gamma scale hyperparameter.
        """

        num_beta_params = len(param_names) - 1
        mu = np.array([mean_dict.get(k, 0.0) for k in param_names[:-1]])
        Lambda = np.diag([penalty_dict.get(k, 0.0) for k in param_names[:-1]])
        return NormalInverseGamma(mu, Lambda, a, b, num_beta_params)

    def copy(self):
        """Return a deep-ish copy of prior hyperparameters.

        Args:
            None.
        """
        return NormalInverseGamma(
            self.mu.copy() if isinstance(self.mu, np.ndarray) else self.mu,
            self.Lambda.copy() if isinstance(self.Lambda, np.ndarray) else self.Lambda,
            self.a, self.b, self.p
        )

    def __str__(self):
        """Format hyperparameters as a readable multiline string.

        Args:
            None.
        """
        return "\n".join([
            f'a: {self.a}',
            f'b: {self.b}',
            f'\nmu: {self.mu}',
            f'\nLambda:\n{self.Lambda}',
        ])

    def __repr__(self):
        """Return representation string.

        Args:
            None.
        """
        return str(self)
