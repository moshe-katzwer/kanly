from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import issparse

from kanly.regression.cov_types import NONROBUST
from kanly.regression.regression_results_base import RegressionResultsBase


class PlsRegressionResults(RegressionResultsBase):
    """
    Container for PLS regression results with prediction capability.

    Stores the fitted PLS model components and provides methods for making
    predictions on new data.

    Attributes
    ----------
    T : ndarray, shape (n_samples, l)
        Score matrix (latent variables) from training data
    P : ndarray, shape (n_features, l)
        X-loadings matrix
    q : ndarray, shape (l,)
        y-loadings vector
    W : ndarray, shape (n_features, l)
        Weight matrix (used for projection)
    coef : ndarray, shape (n_features,)
        Regression coefficients (excluding intercept)
    intercept : float
        Intercept term
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        Training predictor matrix (stored for reference)
    y : ndarray, shape (n_samples,)
        Training response vector (stored for reference)
    l : int
        Number of latent components used in the model

    Methods
    -------
    predict(X)
        Make predictions on new data using the fitted PLS model
    """

    def __init__(self, T, P, q, W, coef, intercept, X, y, weights, l, center, fittedvalues, resid, wssr, wsst, rsquared,
                 scale,
                 cov_params,
                 test_level=0.05,
                 specification_name=None,
                 model_elapsed=np.nan,
                 fit_elapsed=np.nan, cov_elapsed=np.nan,
                 endog_name=None, exog_names=None):

        """
        Initialize PLS regression results.

        Parameters
        ----------
        T : ndarray, shape (n_samples, l)
            Score matrix from training
        P : ndarray, shape (n_features, l)
            X-loadings matrix
        q : ndarray, shape (l,)
            y-loadings vector
        W : ndarray, shape (n_features, l)
            Weight matrix
        coef : ndarray, shape (n_features,)
            Regression coefficients (excluding intercept)
        intercept : float
            Intercept term
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            Training predictor matrix
        y : ndarray, shape (n_samples,)
            Training response vector
        l : int
            Number of latent components
        """

        if exog_names is None:
            exog_names = [f'x{j}' for j in range(len(coef))]
        exog_names = ['Intercept'] + list(exog_names)
        super().__init__(len(y), np.hstack([intercept, coef]), cov_params, np.nan, np.nan, np.nan,
                         exog_names=exog_names, endog_name=endog_name, cov_type=NONROBUST,
                         cov_kwds=None, test_level=test_level, use_t=False, alpha=0, l1_ratio=0,
                         specification_name=specification_name)

        # Store PLS components
        self.T = T
        self.P = P
        self.q = q
        self.W = W
        self.coef = coef
        self.intercept = intercept

        # derived quantities
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.wsst = wsst
        self.wssr = wssr
        self.rsquared = rsquared
        self.scale = scale

        # Store training data for reference
        self.X = X
        self.y = y
        self.weights = weights
        self.l = l
        self.center = center

        # Store metadata
        self.n_samples, self.n_features = X.shape
        self.n_components = self.l
        self.is_sparse = issparse(X)

        self.fit_elapsed = fit_elapsed
        self.cov_elapsed = cov_elapsed
        self.model_elapsed = model_elapsed

    @staticmethod
    def get_result_type():
        return 'Partial Least Squares'

    def get_result_name(self):
        return "Partial Least Squares Results"

    def get_header_info_array(self):
        return np.array([
            ['Date:', self.date],
            ['Time:  ', self.timestamp],
            ["", ""],
            ['Model Elapsed:', '%.2f s' % self.model_elapsed],
            ['Fit Elapsed:', '%.2f s' % self.fit_elapsed],
            ['Cov Elapsed:', '%.2f s' % self.cov_elapsed],
            ['No. Obs.', self.nobs],
            ['No. Features', self.n_features],
            ['No. Components', self.n_components],
            ['Covariance Type:', self.cov_type],
            [f'R-squared{("(uncentered)" if not self.center else "")}:', np.round(self.rsquared, 4)],
            ['scale:', "%.4e" % self.scale],
            ['centered:', self.center],
        ])

    def get_footer_info(self, *args, **kwargs):
        foot = ""
        if not self.center:
            foot += "Warning: data was not centered in estimation,\n\tapproach with caution!"
        return foot

    def predict(self, X=None):
        """
        Make predictions using the fitted PLS model.

        Uses the linear regression formula:
            y_pred = X @ coef + intercept

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples_new, n_features)
            New predictor data. Must have the same number of features as
            the training data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples_new,)
            Predicted response values

        Raises
        ------
        ValueError
            If X does not have the correct number of features

        Examples
        --------
        >>> # After fitting PLS model
        >>> results = PlsRegressionResults(T, P, q, W, coef, intercept, X_train, y_train, l)
        >>> y_pred = results.predict(X_test)
        """

        if X is None:
            return self.fittedvalues.copy()

        # Convert to array if needed (handles both dense and sparse)
        X = np.asarray(X) if not issparse(X) else X

        # Validate dimensions
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on "
                f"{self.n_features} features"
            )

        # Compute predictions: y_pred = X @ coef + intercept
        # This works for both sparse and dense X
        y_pred = X @ self.coef + self.intercept

        return y_pred
