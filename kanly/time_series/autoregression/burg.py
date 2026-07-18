from __future__ import absolute_import, print_function

import numpy as np


def burg(y, lags=1):
    """
    Estimate AR(p) coefficients using Burg's method.

    Burg's method estimates an autoregressive model by recursively
    minimizing both forward and backward one-step prediction errors.
    It produces reflection coefficients (partial autocorrelations)
    sequentially and enforces stationarity at every recursion step.

    Model:
        x_t = phi_1 x_{t-1} + ... + phi_p x_{t-p} + epsilon_t

    Parameters
    ----------
    y : array-like, shape (n,)
        Time series data.
    lags : int
        AR order p.

    Returns
    -------
    phi : ndarray, shape (p,)
        Estimated AR coefficients [phi_1, ..., phi_p].

    Notes
    -----
    - Internally estimates reflection coefficients (kappa_p) via
      symmetric forward/backward prediction error minimization.
    - Uses mean-centered data.
    - Enforces stationarity implicitly via |kappa_p| < 1.
    - Does NOT estimate innovation variance sigma^2.

    References
    ----------
    Burg (1975), "Maximum Entropy Spectral Analysis"
    Kay & Marple (1981), "Spectrum Analysis—A Modern Perspective"

    """

    y = np.asarray(y, dtype=float)
    n = len(y)
    const = y.mean()
    x = y - const

    f = x.copy()
    b = x.copy()

    phi = np.zeros(lags + 1)  # phi[0] unused

    for p in range(1, lags + 1):

        # ------------------------------------------------------------
        # Correct forward/backward alignment (KEY FIX)
        # ------------------------------------------------------------
        f_p = f[p:]
        b_p = b[p - 1:-1]

        # ------------------------------------------------------------
        # Reflection coefficient (NO minus sign for this convention)
        # ------------------------------------------------------------
        num = 2.0 * np.dot(f_p, b_p)
        den = np.dot(f_p, f_p) + np.dot(b_p, b_p)

        kappa_p = num / den

        # optional safety clamp (helps near-unit-root cases)
        # kappa_p = np.tanh(np.arctanh(np.clip(kappa_p, -0.999999, 0.999999)))

        # ------------------------------------------------------------
        # Update AR coefficients (Levinson recursion)
        # ------------------------------------------------------------
        phi_old = phi.copy()
        phi[p] = kappa_p

        for j in range(1, p):
            phi[j] = phi_old[j] - kappa_p * phi_old[p - j]

        # ------------------------------------------------------------
        # Update forward/backward residuals (must use old copies)
        # ------------------------------------------------------------
        f_old = f.copy()
        b_old = b.copy()

        f = f_old[1:] - kappa_p * b_old[:-1]
        b = b_old[:-1] - kappa_p * f_old[1:]

    sigma2 = np.sum(f ** 2) / (n - p)
    arparams = phi[1:]
    const *= (1 - sum(arparams))
    params = np.hstack([const, arparams, sigma2])
    param_names = ['Intercept'] + [f'L{j}' for j in range(1, lags + 1)] + ['sigma2']

    return {'params': params,
            'param_names': param_names,
            'arparams': arparams,
            'const': params[0],
            'sigma2': params[-1],
            'cov_params': None,
            }
