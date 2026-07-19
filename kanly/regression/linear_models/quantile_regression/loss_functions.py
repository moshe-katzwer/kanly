"""Smooth surrogate loss functions for IRLS quantile regression.

The true quantile (check) loss  ρ_τ(z) = z·(τ - 𝟙{z<0})  is not
differentiable at zero, which would stall a gradient-based solver.  Each
subclass in this module replaces the kink with a smooth approximation so
that IRLS weights remain finite and well-behaved.

Subclasses implement two static methods:
  - ``loss(z, k, tau)``            — the smooth loss value for each residual
  - ``loss_derivative(z, k, tau)`` — its element-wise derivative ψ(z)

The base class then provides:
  - ``cost``    — total (summed) loss, halved for convention
  - ``weights`` — IRLS weights computed as ψ(z)/z
"""
from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod

import numpy as np

from kanly.regression.linear_models.quantile_regression.constants import DEFAULT_QR_MIN_RESID_CLIP


class QuantileRegressionLossFunction(ABC):
    """Abstract base class defining the interface for IRLS quantile loss functions.

    Concrete subclasses must implement the static methods ``loss`` and
    ``loss_derivative``.  The concrete class methods ``cost`` and ``weights``
    are provided here and delegate to those two primitives.

    All methods operate element-wise on residual arrays and support both
    tau < 0.5 (lower quantiles) and tau > 0.5 (upper quantiles) via
    asymmetric tau/(1-tau) scaling.
    """

    @classmethod
    def cost(cls, z, k, tau):
        """Compute the total surrogate loss (halved sum of element-wise losses).

        Args:
            z (ndarray): Residual vector (y - Xβ), shape (n,).
            k (float): Smoothing bandwidth; controls the width of the quadratic
                region around zero.
            tau (float): Quantile level in (0, 1).

        Returns:
            float: Sum of element-wise losses divided by 2.0.
        """
        return np.sum(cls.loss(z, k, tau)) / 2.0

    @classmethod
    def weights(cls, z, k, tau, clip=DEFAULT_QR_MIN_RESID_CLIP):
        """Compute IRLS weights as ψ(z)/z, where ψ = loss_derivative.

        The IRLS algorithm re-expresses the quantile objective_function as a sequence of
        weighted least squares problems.  The weight for each observation is
        ψ(rᵢ)/rᵢ.  Residuals smaller than ``clip`` are clamped to avoid
        division by zero while preserving sign.

        Args:
            z (ndarray): Residual vector (y - Xβ), shape (n,).
            k (float): Smoothing bandwidth for the loss function.
            tau (float): Quantile level in (0, 1).
            clip (float): Minimum absolute residual value before computing
                ψ(z)/z.  Defaults to ``DEFAULT_QR_MIN_RESID_CLIP``.

        Returns:
            ndarray: Non-negative weight array, shape (n,).
        """
        z = np.clip(np.abs(z), clip, np.inf) * np.where(z < 0, -1, 1.0)
        dl = cls.loss_derivative(z, k, tau)
        return dl / z

    @staticmethod
    @abstractmethod
    def loss(z, k, tau):
        """Compute the element-wise smooth surrogate loss.

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Smoothing bandwidth.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation loss values, shape (n,).
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loss_derivative(z, k, tau):
        """Compute the element-wise derivative of the smooth surrogate loss.

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Smoothing bandwidth.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation loss derivative ψ(z), shape (n,).
        """
        raise NotImplementedError


class Huber(QuantileRegressionLossFunction):
    """Asymmetric Huber (smooth check) loss for quantile regression.

    Inside the interval ``(-k, k)`` the loss is quadratic (z²/2k); outside it
    is linear (|z| - k/2).  The asymmetry factor ``τ`` (for positive
    residuals) and ``1-τ`` (for negative residuals) recovers the quantile
    check loss as k → 0.

    This is the default loss function because it has stable IRLS weights
    across a wide range of residual magnitudes.
    """

    @staticmethod
    def loss(z, k, tau):
        """Compute the asymmetric Huber loss.

        For |z| < k:  l(z) = z²/(2k) · τ_sign
        For |z| ≥ k:  l(z) = (|z| - k/2) · τ_sign
        where τ_sign = τ if z ≥ 0 else (1-τ).

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Half-width of the quadratic region.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation Huber loss values, shape (n,).
        """
        idx_small = np.abs(z) < k

        l = (np.abs(z) - 0.5 * k)
        l[idx_small] = z[idx_small] ** 2 / (2 * k)

        l *= np.where(z < 0, 1 - tau, tau)

        return l

    @staticmethod
    def loss_derivative(z, k, tau):
        """Compute the derivative of the asymmetric Huber loss.

        For |z| < k:  ψ(z) = z/k · τ_sign
        For |z| ≥ k:  ψ(z) = sign(z) · τ_sign
        where τ_sign = τ if z ≥ 0 else (1-τ).

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Half-width of the quadratic region.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation loss derivative, shape (n,).
        """
        idx_small = np.abs(z) < k
        idx_neg = z < 0
        l = np.where(idx_small, z / k, np.sign(z))
        l *= np.where(idx_neg, 1 - tau, tau)

        return l


class SmoothCup(QuantileRegressionLossFunction):
    """Smooth asymmetric cup loss for quantile regression.

    Extends the standard Huber loss by shifting the quadratic region so that
    its minimum aligns with the tau-quantile of the residuals.  The shift is
    computed as  z ← z + b/(2a)  where  a = 1/(4k), b = τ - 0.5.

    This centering makes the smooth region track the true check-loss kink
    more closely than the symmetric Huber does for extreme quantiles.
    """

    @staticmethod
    def loss(z, k, tau):
        """Compute the smooth cup loss.

        Applies the quantile shift  z ← z − b/(2a)  before evaluating the
        piecewise quadratic/linear form.

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Half-width of the quadratic region (before shifting).
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation smooth cup loss values, shape (n,).
        """
        a = 1 / (4 * k)
        b = tau - 0.5
        c = k / 4

        # Shift z so the quadratic region is centred at the check-loss kink.
        z = z + (-b / (2 * a))

        l = np.abs(z) * tau
        idx_small = np.abs(z) < k
        idx_neg = z < 0
        l[idx_neg] *= (1 - tau) / tau

        l[idx_small] = a * z[idx_small] ** 2 + b * z[idx_small] + c

        return l

    @staticmethod
    def loss_derivative(z, k, tau):
        """Compute the derivative of the smooth cup loss.

        Applies the same quantile shift as ``loss`` before differentiating.

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Half-width of the quadratic region.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation loss derivative, shape (n,).
        """
        a = 1 / (4 * k)
        b = tau - 0.5

        z = z + (-b / (2 * a))

        idx_neg = z < 0
        idx_small = np.abs(z) < k

        dl = np.where(z <= 0, -1, 1) * tau
        dl[idx_neg] *= (1 - tau) / tau  # comes *before* quadratic at kink
        dl[idx_small] = 2 * a * z[idx_small] + b

        return dl


class SoftL1(QuantileRegressionLossFunction):
    """Soft-L1 (pseudo-Huber / Charbonnier) loss for quantile regression.

    Unlike Huber, the soft-L1 loss is smooth everywhere — there is no hard
    linear/quadratic boundary.  The formula  k²(√(1+(z/k)²) − 1)  interpolates
    smoothly between quadratic behaviour near zero and linear behaviour in the
    tails.  The asymmetric tau/(1-tau) scaling is applied to negative residuals
    to target the tau-quantile.

    Useful when very smooth IRLS weight transitions are desired, at the cost
    of slower convergence for large residuals compared to Huber.
    """

    @staticmethod
    def loss(z, k, tau):
        """Compute the asymmetric soft-L1 loss.

        l(z) = k²(√(1+(z/k)²) − 1) · τ_sign
        where τ_sign = τ if z ≥ 0 else (1-τ).

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Scale parameter controlling the quadratic/linear trade-off.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation soft-L1 loss values, shape (n,).
        """
        l = (k ** 2 * (1 + (z / k) ** 2) ** 0.5 - k ** 2) * tau
        l[z < 0] *= (1 - tau) / tau
        return l

    @staticmethod
    def loss_derivative(z, k, tau):
        """Compute the derivative of the asymmetric soft-L1 loss.

        ψ(z) = z / √(1+(z/k)²) · τ_sign
        where τ_sign = τ if z ≥ 0 else (1-τ).

        Args:
            z (ndarray): Residual vector, shape (n,).
            k (float): Scale parameter.
            tau (float): Quantile level in (0, 1).

        Returns:
            ndarray: Per-observation loss derivative, shape (n,).
        """
        dl = ((1 + (z / k) ** 2) ** -.5) * z * tau
        dl[z < 0] *= (1 - tau) / tau
        return dl


# Mapping from normalised string name to loss class.
# String keys are lowercased and have spaces/underscores stripped before lookup.
loss_str_to_loss_deriv = {
    'huber': Huber,
    'softl1': SoftL1,
    'smoothcup': SmoothCup,
}


def get_loss_deriv(arg):
    """Resolve a loss function name or class to a ``QuantileRegressionLossFunction`` subclass.

    Accepts either a string name (case-insensitive, spaces and underscores
    stripped) or a class that is already a subclass of
    ``QuantileRegressionLossFunction``.  Valid string values are
    ``'huber'``, ``'softl1'``, and ``'smoothcup'``.

    Args:
        arg (str or type): Loss function identifier.  If a string, it is
            normalised and looked up in ``loss_str_to_loss_deriv``.  If a
            class, it is returned unchanged.

    Returns:
        type: A ``QuantileRegressionLossFunction`` subclass (not an instance).

    Raises:
        Exception: If ``arg`` is a string that does not match any registered
            loss function.
    """
    if isinstance(arg, str):
        arg = arg.lower().replace(' ', '').replace('_', '')
        if arg not in loss_str_to_loss_deriv.keys():
            raise Exception(f'loss func must be one of {str(loss_str_to_loss_deriv.keys())}, not {arg}!')
        return loss_str_to_loss_deriv[arg]
    else:
        return arg


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     funcs = [Huber, SmoothCup, SoftL1]
#     x = np.linspace(-2, 2, 1000)
#     k = 1
#     tau = .9
#     f, ax = plt.subplots(nrows=2, ncols=3)
#     for i in range(3):
#         ax[0,i].set_title(funcs[i].__name__)
#         ax[0,i].plot(x, funcs[i].loss(x, k, tau))
#         ax[1, i].plot(x, funcs[i].loss_derivative(x, k, tau))
#         ax[1, i].plot(x, (funcs[i].loss(x+1e-6, k, tau)-funcs[i].loss(x, k, tau))/1e-6, ls=':')
#         ax[0,i].axvline(0, color='k')
#         ax[1,i].axvline(0, color='k')
#     plt.show()