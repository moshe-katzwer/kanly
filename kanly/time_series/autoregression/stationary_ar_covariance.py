from __future__ import absolute_import, print_function

import numpy as np


def ar_yule_walker_autocovariances(rho, sigma2=1.0):
    """
    Solve Yule-Walker equations for gamma_0, ..., gamma_p for a stationary AR(p).

    Under stationarity, autocovariances satisfy the Yule-Walker system; the
    solution is used to build the initial covariance block in Prais-Winsten
    whitening (see :func:`ar_initial_gamma`).

    Model
    -----
    .. math::

        u_t = \\rho_1 u_{t-1} + \\cdots + \\rho_p u_{t-p} + \\varepsilon_t,
        \\quad \\varepsilon_t \\sim (0, \\sigma^2)

    Parameters
    ----------
    rho : array-like, shape (p,)
        AR coefficients ``[rho_1, ..., rho_p]``.
    sigma2 : float, default=1.0
        Innovation variance.

    Returns
    -------
    gamma : ndarray, shape (p + 1,)
        Autocovariances ``gamma[0], ..., gamma[p]`` (``gamma[0] = Var(u_t)``).
    """
    rho = np.asarray(rho, dtype=float)
    p = len(rho)

    # Unknown vector is [gamma_0, gamma_1, ..., gamma_p]
    A = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)

    # h = 0 equation:
    # gamma_0 - sum_{k=1}^p rho_k gamma_k = sigma2
    A[0, 0] = 1.0
    for k in range(1, p + 1):
        A[0, k] -= rho[k - 1]
    b[0] = sigma2

    # h = 1, ..., p equations:
    # gamma_h - sum_{k=1}^p rho_k gamma_{|h-k|} = 0
    for h in range(1, p + 1):
        A[h, h] = 1.0
        for k in range(1, p + 1):
            A[h, abs(h - k)] -= rho[k - 1]

    gamma = np.linalg.solve(A, b)

    return gamma  # gamma[0], ..., gamma[p]


def ar_initial_gamma(rho, m=None, sigma2=1.0):
    """
    Return :math:`\\Gamma_m = \\mathrm{Cov}(u_1, \\ldots, u_m)` for stationary AR(p).

    By default ``m = p``, which is the block needed for the **Prais-Winsten**
    top rows of :func:`make_ar_full_information_W` when
    ``full_information=True``. Those rows whiten the first ``p`` observations
    using the stationary initial distribution instead of dropping them.

    Parameters
    ----------
    rho : array-like, shape (p,)
        AR coefficients.
    m : int or None, default=None
        Size of the covariance matrix. If None, uses ``m = p``.
    sigma2 : float, default=1.0
        Innovation variance passed to the Yule-Walker solver.

    Returns
    -------
    Gamma : ndarray, shape (m, m)
        Toeplitz covariance of the first ``m`` observations under stationarity.
    """
    rho = np.asarray(rho, dtype=float)
    p = len(rho)

    if p < 1:
        raise ValueError("Need at least one AR coefficient.")

    if m is None:
        m = p

    if m < 0:
        raise ValueError("m must be nonnegative.")

    if m == 0:
        return np.empty((0, 0))

    gamma = ar_yule_walker_autocovariances(rho, sigma2=sigma2)

    # If m > p+1, extend autocovariances recursively.
    gammas = list(gamma)
    for h in range(p + 1, m):
        gammas.append(sum(rho[k - 1] * gammas[h - k] for k in range(1, p + 1)))

    Gamma = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            Gamma[i, j] = gammas[abs(i - j)]

    return Gamma
