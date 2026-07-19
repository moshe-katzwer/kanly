from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod
import numpy as np


class RootLossFunction(ABC):
    """Interface for robust residual transformations used by NLLS.

    A root-loss function maps raw residuals ``z`` to transformed residuals
    ``root_loss`` such that the optimizer can still minimise a sum of squares.
    It also returns ``d_root_loss``, the derivative of that transformation with
    respect to ``z``, so Jacobians can be chain-rule adjusted.
    """

    @abstractmethod
    def __call__(self, z):
        """Transform residuals and return the derivative of the transformation.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(root_loss, d_root_loss)`` with arrays matching ``z``.
        """
        raise NotImplementedError


class HuberLoss(RootLossFunction):
    """Huber-style root loss that is quadratic near zero and linear in the tails."""

    def __init__(self, c=1.345):
        """Initialise the Huber cutoff.

        Args:
            c: Residual magnitude at which the loss transitions from quadratic
                to linear behaviour.
        """
        self.c = c

    def __call__(self, z):
        """Apply the Huber root-loss transformation.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(root_loss, d_root_loss)`` for robust least-squares fitting.
        """

        z = np.asarray(z)
        root_loss = np.sqrt(0.5)*np.abs(z)
        idx = np.abs(z) > self.c
        root_loss[idx] = np.sqrt(self.c) * np.sqrt(np.abs(z[idx]) - 0.5 * self.c)

        d_root_loss = np.full_like(z, np.sqrt(0.5))
        idx = np.abs(z) > self.c
        d_root_loss[idx] = np.sqrt(self.c) * 0.5 / np.sqrt(np.abs(z[idx]) - 0.5 * self.c)
        d_root_loss *= np.sign(z)

        return root_loss, d_root_loss

    def __str__(self):
        """Return a compact string representation for summaries."""
        return 'HuberLoss(c=%.4f)' % self.c


class LeastSquares(RootLossFunction):
    """Identity root-loss corresponding to ordinary least squares."""

    def __call__(self, z):
        """Return absolute residuals and their sign derivative.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(abs(z), sign(z))``.
        """
        return np.abs(z), np.sign(z)

    def __str__(self):
        """Return a compact string representation for summaries."""
        return 'LeastSquares'


class QuantileHuberLoss(RootLossFunction):
    """Asymmetric Huber-like root loss for quantile-oriented NLLS objectives."""

    def __init__(self, tau=.5, k=1e-8):
        """Initialise a quantile Huber loss.

        Args:
            tau: Target quantile in ``(0, 1)``.
            k: Minimum absolute residual used to avoid division by zero in the
                derivative near zero.
        """
        self.tau = tau
        self.k = k

    def __call__(self, z):
        """Apply the asymmetric quantile root-loss transformation.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(root_loss, d_root_loss)``.
        """
        z = np.clip(np.abs(z), a_min=self.k, a_max=np.inf) * np.where(z <= 0, -1., 1.)
        root_loss = np.sqrt(np.abs(z)) * np.sqrt(self.tau)
        idx = z < 0
        root_loss[idx] *= np.sqrt((1.0 - self.tau) / self.tau)

        d_root_loss = 0.5 / np.sqrt(np.abs(z)) * np.sqrt(self.tau) * np.where(z <= 0, -1., 1.)
        idx = z < 0
        d_root_loss[idx] *= np.sqrt((1.0 - self.tau) / self.tau)

        return root_loss, d_root_loss

    def __str__(self):
        """Return a compact string representation for summaries."""
        return 'QuantileHuberLoss(tau=%.4f, k=%.2e)' % (self.tau, self.k)


class SoftL1(RootLossFunction):
    """Placeholder for a soft-L1 root loss.

    The class exists so the public loss hierarchy names the intended robust
    option, but no implementation is currently provided.
    """
    pass


class QuantileSmooth(RootLossFunction):
    """Smooth asymmetric quantile loss with a quadratic region around zero."""

    def __init__(self, tau=.5, k=.01):
        """Initialise the smooth quantile loss.

        Args:
            tau: Target quantile in ``(0, 1)``.
            k: Width of the smoothing region around zero.
        """
        self.k = k
        self.tau = tau

    def __call__(self, z):
        """Apply the smooth quantile root-loss transformation.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(sqrt(loss), derivative_of_sqrt_loss)``.
        """
        z = np.asarray(z)
        l = np.abs(z) * self.tau
        idx = np.abs(z) < self.k
        idx_neg = z < 0
        l[idx_neg] *= (1 - self.tau) / self.tau

        a = 1 / (4 * self.k)
        b = self.tau - 0.5
        c = self.k / 4

        l[idx] = a * z[idx] ** 2 + b * z[idx] + c

        dl = np.where(z <= 0, -1, 1) * self.tau
        dl[idx_neg] *= (1 - self.tau) / self.tau
        dl[idx] = 2 * a * z[idx] + b

        return l ** .5, .5 * l ** -.5 * dl

    def __str__(self):
        """Return a compact string representation for summaries."""
        return "QuantileSmoothLoss(tau=%.4f,k=%.3e)" % (self.tau, self.k)


class QuantilePseudoHuberLoss(RootLossFunction):
    """Pseudo-Huber quantile loss with smooth tails and asymmetric weighting."""

    def __init__(self, tau=.5, k=.01):
        """Initialise the pseudo-Huber quantile loss.

        Args:
            tau: Target quantile in ``(0, 1)``.
            k: Smoothness scale; smaller values more closely approximate the
                absolute quantile loss.
        """
        self.k = k
        self.tau = tau

    def __call__(self, z):
        """Apply the pseudo-Huber quantile root-loss transformation.

        Args:
            z: Raw residual vector.

        Returns:
            Tuple ``(sqrt(loss), derivative_of_sqrt_loss)``.
        """
        z = np.asarray(z)
        idx_neg = z < 0

        l = self.k ** 2 * (np.sqrt(1 + (z / self.k) ** 2) - 1) * self.tau
        l[idx_neg] *= (1 - self.tau) / self.tau

        dl = z / (np.sqrt(1 + (z / self.k) ** 2))

        l = l ** 0.5
        dl = (0.5 / l) * dl
        return l, dl

    def __str__(self):
        """Return a compact string representation for summaries."""
        return "QuantilePseudoHuberLoss(tau=%.4f,k=%.3e)" % (self.tau, self.k)


root_loss_func_dict = {
    l().__class__.__name__.lower(): l
    for l in [HuberLoss, LeastSquares, QuantileHuberLoss, QuantileSmooth]
}

built_in_root_loss_functions = list(root_loss_func_dict.keys())


def get_root_loss_func(arg):
    """Resolve a root-loss argument to a callable loss object.

    Args:
        arg: Either a string key naming a built-in loss (case-insensitive) or
            an already-instantiated loss object/callable.

    Returns:
        Root-loss object implementing ``__call__(z) -> (root_loss, d_root_loss)``.

    Raises:
        Exception: If ``arg`` is a string and does not name a built-in root loss.
    """

    if isinstance(arg, str):
        if arg.lower() in root_loss_func_dict.keys():
            return root_loss_func_dict[arg.lower()]()
        else:
            raise Exception
    else:
        return arg
