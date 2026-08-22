from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import issparse

from kanly.regression.cov_types import NONROBUST
from kanly.regression.regression_results_base import RegressionResultsBase


class PlsRegressionResults(RegressionResultsBase):
    """Fitted PLS1 or PLS2 regression results.

    The class stores the latent PLS components, fitted values, residuals, and
    regression coefficients and exposes the standard kanly result interface.
    PLS1 coefficients have shape ``(n_features,)``; PLS2 coefficients have
    shape ``(n_features, n_targets)``.

    For compatibility with the former PLS2 dictionary return value, component
    fields can also be accessed by key (for example, ``fit['coef']``). New code
    should prefer attributes and methods such as ``fit.coef`` and
    ``fit.predict(X_new)``.
    """

    _COMPATIBILITY_KEYS = {
        "T",
        "U",
        "P",
        "Q",
        "q",
        "W",
        "C",
        "coef",
        "B_pls",
        "intercept",
        "fittedvalues",
        "resid",
        "X_mean",
        "Y_mean",
        "n_iter",
        "converged",
    }

    def __init__(
            self,
            T,
            P,
            q,
            W,
            coef,
            intercept,
            X,
            y,
            weights,
            l,
            center,
            fittedvalues,
            resid,
            wssr,
            wsst,
            rsquared,
            scale,
            cov_params,
            test_level=0.05,
            specification_name=None,
            model_elapsed=np.nan,
            fit_elapsed=np.nan,
            cov_elapsed=np.nan,
            endog_name=None,
            exog_names=None,
            U=None,
            C=None,
            X_mean=None,
            Y_mean=None,
            n_iter=None,
            converged=None,
            model_type=None,
    ):
        coef = np.asarray(coef, dtype=float)
        if coef.ndim not in (1, 2):
            raise ValueError(f"coef must be one- or two-dimensional, got {coef.shape}")

        self.is_pls2 = coef.ndim == 2
        self.n_targets = coef.shape[1] if self.is_pls2 else 1
        self.is_multi_response = self.n_targets > 1
        self.model_type = model_type or ("PLS2" if self.is_pls2 else "PLS1")

        if exog_names is None:
            feature_names = [f"x{j}" for j in range(coef.shape[0])]
        else:
            feature_names = list(exog_names)
        if len(feature_names) != coef.shape[0]:
            raise ValueError(
                f"Expected {coef.shape[0]} exog names, got {len(feature_names)}"
            )

        if self.is_pls2:
            intercept = np.asarray(intercept, dtype=float).reshape(-1)
            if intercept.size != self.n_targets:
                raise ValueError(
                    f"Expected {self.n_targets} intercepts, got {intercept.size}"
                )
            endog_names = self._normalise_endog_names(endog_name, self.n_targets)
            parameter_matrix = np.column_stack((intercept, coef.T))
            if self.n_targets == 1:
                parameter_names = ["Intercept"] + feature_names
            else:
                parameter_names = [
                    f"{target}:{term}"
                    for target in endog_names
                    for term in ["Intercept"] + feature_names
                ]
            params = parameter_matrix.ravel()
            display_endog_name = ", ".join(endog_names)
        else:
            intercept = float(np.asarray(intercept).squeeze())
            endog_names = ["y" if endog_name is None else str(endog_name)]
            parameter_matrix = np.hstack((intercept, coef)).reshape(1, -1)
            parameter_names = ["Intercept"] + feature_names
            params = parameter_matrix.ravel()
            display_endog_name = endog_names[0]

        super().__init__(
            len(y),
            params,
            cov_params,
            np.nan,
            np.nan,
            np.nan,
            exog_names=parameter_names,
            endog_name=display_endog_name,
            cov_type=NONROBUST if cov_params is not None else None,
            cov_kwds=None,
            test_level=test_level,
            use_t=False,
            alpha=0,
            l1_ratio=0,
            specification_name=specification_name,
        )

        # PLS components and regression parameters.
        self.T = np.asarray(T)
        self.U = None if U is None else np.asarray(U)
        self.P = np.asarray(P)
        self.q = np.asarray(q)
        self.Q = self.q
        self.W = np.asarray(W)
        self.C = None if C is None else np.asarray(C)
        self.coef = coef
        self.B_pls = self.coef
        self.intercept = intercept
        self.parameter_matrix = parameter_matrix
        self.params_by_response = parameter_matrix

        # Fit statistics. PLS2 stores one value per response.
        self.fittedvalues = np.asarray(fittedvalues)
        self.resid = np.asarray(resid)
        self.wsst = self._scalar_or_array(wsst)
        self.wssr = self._scalar_or_array(wssr)
        self.rsquared = self._scalar_or_array(rsquared)
        self.scale = self._scalar_or_array(scale)

        # Training data and metadata.
        self.X = X
        self.y = y
        self.weights = weights
        self.l = int(l)
        self.center = bool(center)
        self.feature_names = feature_names
        self.endog_names = endog_names
        self.X_mean = None if X_mean is None else np.asarray(X_mean)
        self.Y_mean = None if Y_mean is None else np.asarray(Y_mean)
        self.n_iter = None if n_iter is None else np.asarray(n_iter, dtype=int)
        self.converged = None if converged is None else np.asarray(converged, dtype=bool)

        self.n_samples, self.n_features = X.shape
        self.n_components = self.l
        self.is_sparse = issparse(X)
        self.is_sparse_y = issparse(y)

        self.fit_elapsed = fit_elapsed
        self.cov_elapsed = cov_elapsed
        self.model_elapsed = model_elapsed
        self.test_level = test_level

    @staticmethod
    def _normalise_endog_names(endog_name, n_targets):
        if endog_name is None:
            return ["y" if n_targets == 1 else f"y{j}" for j in range(n_targets)]
        if isinstance(endog_name, str):
            if n_targets == 1:
                return [endog_name]
            raise ValueError("endog_name must contain one name per PLS2 response")
        names = [str(name) for name in endog_name]
        if len(names) != n_targets:
            raise ValueError(f"Expected {n_targets} endog names, got {len(names)}")
        return names

    @staticmethod
    def _scalar_or_array(value):
        value = np.asarray(value)
        return value.item() if value.size == 1 else value

    @staticmethod
    def _format_stat(value, precision=4, scientific=False):
        value = np.asarray(value)
        if value.ndim == 0:
            if scientific:
                return f"{value.item():.4e}"
            return str(np.round(value.item(), precision))
        format_code = "e" if scientific else "f"
        formatter = {
            "float_kind": lambda item: f"{item:.{precision}{format_code}}"
        }
        return np.array2string(value, formatter=formatter, separator=", ")

    def _format_response_stat(self, value, scientific=False):
        values = np.asarray(value).reshape(-1)
        format_code = ".4e" if scientific else ".4f"
        return ", ".join(
            f"{name}={number:{format_code}}"
            for name, number in zip(self.endog_names, values)
        )

    @staticmethod
    def get_result_type():
        return "Partial Least Squares"

    def get_result_name(self):
        return f"Partial Least Squares ({self.model_type}) Results"

    def get_header_info_array(self):
        response_suffix = " (by response)" if self.is_multi_response else ""
        center_suffix = "(uncentered)" if not self.center else ""
        rsquared_value = (
            "below" if self.is_pls2 else self._format_stat(self.rsquared)
        )
        scale_value = (
            "below"
            if self.is_pls2
            else self._format_stat(self.scale, scientific=True)
        )
        if self.is_pls2:
            response_suffix = ""
            center_suffix = ""
        return np.array([
            ["Date:", self.date],
            ["Time:  ", self.timestamp],
            ["", ""],
            ["Model Elapsed:", "%.2f s" % self.model_elapsed],
            ["Fit Elapsed:", "%.2f s" % self.fit_elapsed],
            ["Cov Elapsed:", "%.2f s" % self.cov_elapsed],
            ["No. Obs.", self.nobs],
            ["No. Features", self.n_features],
            ["No. Responses", self.n_targets],
            ["No. Components", self.n_components],
            ["Covariance Type:", self.cov_type],
            [
                f"R-squared{center_suffix}{response_suffix}:",
                rsquared_value,
            ],
            [f"scale{response_suffix}:", scale_value],
            ["centered:", self.center],
        ])

    def get_footer_info(self, *args, **kwargs):
        comments = []
        if self.is_pls2:
            rsquared_label = (
                "R-squared (uncentered)" if not self.center else "R-squared"
            )
            comments.extend([
                f"{rsquared_label} by response: "
                f"{self._format_response_stat(self.rsquared)}",
                "Scale by response: "
                f"{self._format_response_stat(self.scale, scientific=True)}",
            ])
        if not self.center:
            comments.append(
                "Warning: data was not centered in estimation,\n"
                "\tapproach with caution!"
            )
        return "\n".join(comments)

    def predict(self, X=None):
        """Predict responses for dense or sparse predictor matrices."""
        if X is None:
            return self.fittedvalues.copy()

        X = X if issparse(X) else np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on "
                f"{self.n_features} features"
            )
        return np.asarray(X @ self.coef) + self.intercept

    def __getitem__(self, key):
        """Provide legacy dictionary-style access to PLS component fields."""
        if key == "predict":
            return self.predict
        if isinstance(key, str) and key in self._COMPATIBILITY_KEYS:
            return getattr(self, key)
        return super().__getitem__(key)
