"""M-estimator norm (influence) functions for robust linear regression.

Each subclass implements the four-method API required by the IRLS algorithm:

  - ``rho(z)``        — the loss function being minimised
  - ``psi(z)``        — its first derivative ρ′(z) (influence function)
  - ``psi_deriv(z)``  — its second derivative ρ″(z)
  - ``weights(z)``    — IRLS weights ψ(z)/z

Available norms:

  - ``HuberT``         (default) — quadratic inside, linear outside
  - ``LeastSquares``   — equivalent to OLS; no downweighting
  - ``TukeyBiweight``  — hard-redescending; zero influence beyond c
  - ``TrimmedMean``    — trimmed mean; hard-zeros outliers beyond c
  - ``AndrewWave``     — sinusoidal, smooth redescending
  - ``RamsayE``        — exponential, soft-redescending
"""
from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod

import numpy as np

# from statsmodels.robust.norms import Hampel AndrewWave, HuberT, Hampel, TukeyBiweight, TrimmedMean, LeastSquares,
# RamsayE, RobustNorm


class RobustNormFunction(ABC):
    """Abstract base class defining the four-method interface for M-estimator norms.

    Concrete subclasses implement ``rho``, ``psi``, ``psi_deriv``, and
    ``weights`` for a specific robust loss function.  The IRLS loop in
    ``rlm_internal`` calls ``weights`` to form observation weights at each
    iteration and ``rho`` to monitor the cost.
    """

    @abstractmethod
    def rho(self, z):
        """Evaluate the loss function element-wise.

        The IRLS objective_function is Σ ρ(rᵢ/σ̂) where rᵢ are residuals and σ̂ is
        the MAD-based scale estimate.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation loss values, shape (n,).
        """
        pass

    @abstractmethod
    def psi(self, z):
        """Evaluate the influence function ψ(z) = ρ′(z) element-wise.

        ψ(z) is the first derivative of the loss; it controls how strongly
        each observation influences the coefficient estimates.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values ψ(z), shape (n,).
        """
        pass

    @abstractmethod
    def weights(self, z):
        """Compute IRLS weights ψ(z)/z element-wise.

        The IRLS algorithm solves a sequence of WLS problems with weights
        wᵢ = ψ(rᵢ/σ̂)/(rᵢ/σ̂).  Large residuals receive lower weights,
        reducing the influence of outliers.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Non-negative IRLS weight array, shape (n,).
        """
        pass

    @abstractmethod
    def psi_deriv(self, z):
        """Evaluate the second derivative of the loss ρ″(z) element-wise.

        Used in the H1/H2/H3 sandwich covariance estimators to build the
        expected Hessian approximation.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        pass


class HuberT(RobustNormFunction):
    """Huber's T norm (default M-estimator, c=1.345).

    Quadratic for |z| < c (like OLS) and linear for |z| ≥ c (bounding
    influence of large residuals).  The default c=1.345 achieves approximately
    95% efficiency relative to OLS under normally distributed errors while
    providing strong protection against outliers.

    This is the recommended norm for most applications.
    """

    def __init__(self, c=1.345):
        """Initialise with the clipping threshold.

        Args:
            c (float): Transition point between quadratic and linear regions.
                Default 1.345 gives 95% efficiency under normality.
        """
        self.c = c

    def rho(self, z):
        """Evaluate the Huber loss.

        ρ(z) = z²/2           for |z| < c
        ρ(z) = c·|z| − c²/2   for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation Huber loss values, shape (n,).
        """
        l = np.abs(z)
        idx = l < self.c
        l = np.where(idx, l ** 2 / 2, self.c * l - self.c * self.c / 2)
        return l

    def psi(self, z):
        """Evaluate the Huber influence function ψ(z) = ρ′(z).

        ψ(z) = z          for |z| < c
        ψ(z) = c·sign(z)  for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values, shape (n,).
        """
        z = np.asarray(z)
        l = np.array(z)
        idx = np.abs(z) >= self.c
        l[idx] = self.c * np.sign(z[idx])
        return l

    def psi_deriv(self, z):
        """Evaluate the second derivative of the Huber loss ρ″(z).

        ρ″(z) = 1  for |z| < c
        ρ″(z) = 0  for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        z = np.asarray(z)
        l = np.ones(len(z))
        idx = np.abs(z) >= self.c
        l[idx] = 0.0
        return l

    def weights(self, z):
        """Compute Huber IRLS weights min(c/|z|, 1).

        Observations with |z| ≤ c get weight 1 (same as OLS); larger
        residuals are downweighted proportional to 1/|z|.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: IRLS weights clipped to [0, 1], shape (n,).
        """
        return np.clip(self.c / np.abs(z), a_min=0.0, a_max=1.0)


class LeastSquares(RobustNormFunction):
    """Standard least squares (OLS) norm — ρ(z) = z²/2.

    All IRLS weights are identically 1 and ``psi_deriv`` is identically 1,
    so using this norm reduces ``rlm_internal`` to a plain OLS solve.
    Useful as a baseline for comparison or for testing the IRLS loop.
    """

    def rho(self, z):
        """Evaluate the OLS loss z²/2.

        Args:
            z (ndarray): Residual vector, shape (n,).

        Returns:
            ndarray: Per-observation loss values z²/2, shape (n,).
        """
        return np.array(z) ** 2 / 2

    def psi(self, z):
        """Return z unchanged (ρ′(z) = z for the OLS loss).

        Args:
            z (ndarray): Residual vector, shape (n,).

        Returns:
            ndarray: Identity copy of z, shape (n,).
        """
        return np.array(z)

    def psi_deriv(self, z):
        """Return an array of ones (ρ″(z) = 1 for the OLS loss).

        Args:
            z (ndarray): Residual vector, shape (n,).

        Returns:
            ndarray: Array of 1.0 with shape matching z.
        """
        return np.full(np.shape(z), 1.0)

    def weights(self, z):
        """Return an array of ones — no downweighting (OLS behaviour).

        Args:
            z (ndarray): Residual vector, shape (n,).

        Returns:
            ndarray: Array of 1.0 with shape matching z.
        """
        return np.full(np.shape(z), 1.0)

# TODO
class TukeyBiweight(RobustNormFunction):
    """Tukey's bisquare (biweight) norm — hard-redescending, c=4.685.

    Inside |z| < c the loss is a smooth degree-6 polynomial; outside it is
    identically zero.  Outliers beyond c are *completely* discarded (zero
    weight), giving a high breakdown point.  The loss is non-convex, so IRLS
    may not always find the global minimum; good starting parameters help.

    Note: marked ``# TODO`` — use with care until fully validated.
    """

    def __init__(self, c=4.685):
        """Initialise with the hard-rejection threshold.

        Args:
            c (float): Radius beyond which outliers are given zero weight.
                Default 4.685 gives 95% efficiency under normality.
        """
        self.c = c

    def rho(self, z):
        """Evaluate the Tukey biweight loss.

        ρ(z) = −c²/6·(1-(1−(z/c)²)³)  for |z| < c
        ρ(z) = 0                      for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation biweight loss values, shape (n,).
        """

        const = self.c ** 2 / 6
        return np.where(np.abs(z) < self.c,
                        const * (1 - (1 - (z / self.c) ** 2) ** 3),
                        0.0)

    def psi(self, z):
        """Evaluate the Tukey biweight influence function.

        ψ(z) = z·(1−(z/c)²)²   for |z| < c
        ψ(z) = 0               for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values, shape (n,).
        """
        return np.where(np.abs(z) < self.c,
                        z * (1 - (z / self.c) ** 2) ** 2,
                        0.0)

    def psi_deriv(self, z):
        """Evaluate the second derivative of the Tukey biweight loss.

        ρ″(z) = 5(z/c)⁴ − 6(z/c)² + 1  for |z| < c
        ρ″(z) = 0                         for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        return np.where(np.abs(z) < self.c,
                        5 * (z / self.c) ** 4 - 6 * (z / self.c) ** 2 + 1,
                        0.0)

    def weights(self, z):
        """Compute Tukey biweight IRLS weights — zero beyond c.

        w(z) = (1−(z/c)²)²  for |z| < c
        w(z) = 0             for |z| ≥ c

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Non-negative IRLS weights, shape (n,).
        """
        return np.where(np.abs(z) < self.c,
                        (1 - (z / self.c) ** 2) ** 2,
                        0.0)

class TrimmedMean(RobustNormFunction):
    """Trimmed mean norm — hard-zero outlier influence, c=2.0.

    Quadratic inside (−c, c) like OLS; outside the loss is a constant c²/2
    so ψ(z)=0 and w(z)=0 for |z|≥c.  Simpler derivative structure than
    TukeyBiweight but the same hard-rejection behaviour.
    """

    def __init__(self, c=2.0):
        """Initialise with the trimming threshold.

        Args:
            c (float): Residuals with |z| ≥ c are completely discarded.
        """
        self.c = c

    def rho(self, z):
        """Evaluate the trimmed mean loss.

        ρ(z) = z²/2   for |z| < c
        ρ(z) = c²/2   for |z| ≥ c  (constant — no further penalty)

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation trimmed mean loss values, shape (n,).
        """
        return np.where(np.abs(z) < self.c,  z ** 2 / 2,          self.c ** 2 / 2)

    def psi(self, z):
        """Evaluate the trimmed mean influence function.

        ψ(z) = z  for |z| < c;  0 for |z| ≥ c.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values, shape (n,).
        """
        return np.where(np.abs(z) < self.c, z, 0.0)

    def psi_deriv(self, z):
        """Evaluate the second derivative of the trimmed mean loss.

        ρ″(z) = 1 for |z| < c;  0 for |z| ≥ c.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        return np.where(np.abs(z) < self.c, 1.0, 0.0)

    def weights(self, z):
        """Compute trimmed mean IRLS weights.

        w(z) = 1 for |z| < c (OLS weight);  0 for |z| ≥ c (hard rejection).

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Binary (0/1) IRLS weights, shape (n,).
        """
        return np.where(np.abs(z) < self.c, 1.0, 0.0)


class AndrewWave(RobustNormFunction):
    """Andrews' sine wave norm — smooth redescending, c=1.339.

    Sinusoidal inside |z| < c·π; zero outside.  Provides smooth, bounded
    influence and a continuous derivative at the rejection boundary, unlike the
    hard cutoffs of TukeyBiweight and TrimmedMean.
    """

    def __init__(self, c=1.339):
        """Initialise with the sinusoidal scale parameter.

        Args:
            c (float): Controls the period of the sine; observations with
                |z| ≥ c·π receive zero weight.  Default 1.339.
        """
        self.c = c

    def rho(self, z):
        """Evaluate the Andrews wave loss.

        ρ(z) = c^2·(1 − cos(z/c))  for |z| < c·π
        ρ(z) = 2c                  for |z| ≥ c·π

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation Andrews wave loss values, shape (n,).
        """
        return np.where(np.abs(z) < self.c * np.pi,
                        self.c ** 2 * (1 - np.cos(z / self.c)),
                        self.c * 2)

    def psi(self, z):
        """Evaluate the Andrews wave influence function.

        ψ(z) = c·sin(z/c)  for |z| < c·π
        ψ(z) = 0           for |z| ≥ c·π

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values, shape (n,).
        """
        return np.where(np.abs(z) < self.c * np.pi,
                        self.c * np.sin(z / self.c),
                        0.0)

    def psi_deriv(self, z):
        """Evaluate the second derivative of the Andrews wave loss.

        ρ″(z) = cos(z/c)  for |z| < c·π
        ρ″(z) = 0         for |z| ≥ c·π

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        return np.where(np.abs(z) < self.c * np.pi,
                        np.cos(z / self.c),
                        0.0)

    def weights(self, z):
        """Compute Andrews wave IRLS weights sin(z/c)/(z/c).

        w(z) = sin(z/c)/(z/c)  for |z| < c·π
        w(z) = 0               for |z| ≥ c·π

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Non-negative IRLS weights, shape (n,).
        """
        return np.where(np.abs(z) < self.c * np.pi,
                        np.sin(z / self.c) / (z / self.c),  # TODO z/c rather than z?
                        0.0)


class RamsayE(RobustNormFunction):
    """Ramsay's exponential norm — soft-redescending, c=0.3.

    Uses  ρ(z) ∝ 1 − e^{−c|z|}(1+c|z|).  Unlike hard-redescending norms,
    very large outliers still receive small but non-zero weight (soft
    downweighting), making this norm more forgiving in the presence of extreme
    but legitimate observations.
    """

    def __init__(self, c=0.3):
        """Initialise with the exponential decay rate.

        Args:
            c (float): Rate of exponential downweighting; larger c gives
                stronger downweighting of outliers.  Default 0.3.
        """
        self.c = c

    def rho(self, z):
        """Evaluate the Ramsay E loss.

        ρ(z) = (1 − e^{−c|z|}(1+c|z|)) / c²

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation Ramsay loss values, shape (n,).
        """
        z = np.abs(z)
        return (1 - np.exp(-self.c * z) * (1 + self.c * z)) / self.c ** 2

    def psi(self, z):
        """Evaluate the Ramsay E influence function.

        ψ(z) = z · e^{−c|z|}

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation influence values, shape (n,).
        """
        return z * np.exp(-self.c * np.abs(z))

    def psi_deriv(self, z):
        """Evaluate the second derivative of the Ramsay E loss.

        ρ″(z) = e^{−c|z|}·(1 − c|z|)

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Per-observation second derivative values, shape (n,).
        """
        z = np.abs(z)
        return np.exp(-self.c * z) * (1 - self.c * z)

    def weights(self, z):
        """Compute Ramsay E IRLS weights e^{−c|z|}.

        All residuals receive positive weight (soft redescending); very large
        outliers get exponentially small but non-zero weights.

        Args:
            z (ndarray): Scaled residual vector, shape (n,).

        Returns:
            ndarray: Positive IRLS weights in (0, 1], shape (n,).
        """
        return np.exp(-self.c * np.abs(z))


# Mapping from normalised string name to norm class.
# Keys are lowercased with spaces, dashes, and underscores stripped.
norm_str_to_class = {
    'andrewwave': AndrewWave, 'hubert': HuberT,
    # 'hampel': Hampel,
    'tukeybiweight': TukeyBiweight,
    'trimmedmean': TrimmedMean, 'leastsquares': LeastSquares,
    'ramsaye': RamsayE
}


def get_norm(norm, *args, **kwargs):
    """Resolve a norm name, class, or instance to a ``RobustNormFunction`` object.

    Three dispatch paths are supported:

    1. **String** — normalised (spaces/dashes/underscores stripped, lowercased)
       and looked up in ``norm_str_to_class``; the matching class is
       instantiated with ``*args, **kwargs``.
    2. **Instance** — an already-constructed ``RobustNormFunction`` is returned
       unchanged.
    3. **Subclass** — a ``RobustNormFunction`` subclass (not an instance) is
       instantiated with ``*args, **kwargs``.

    Args:
        norm (str, RobustNormFunction, or type): Norm identifier.  Valid
            string values: ``'hubert'``, ``'leastsquares'``, ``'tukeybiweight'``,
            ``'trimmedmean'``, ``'andrewwave'``, ``'ramsaye'`` (case-insensitive,
            spaces/dashes/underscores stripped).
        *args: Positional arguments forwarded to the norm class constructor.
        **kwargs: Keyword arguments forwarded to the norm class constructor.

    Returns:
        RobustNormFunction: An instantiated norm object.

    Raises:
        KeyError: If ``norm`` is a string that does not match any registered norm.
    """
    if isinstance(norm, str):
        for c in ' -_':
            norm = norm.replace(c, '')
        norm = norm.lower()
        return norm_str_to_class[norm](*args, **kwargs)
    elif isinstance(norm, RobustNormFunction):
        return norm
    elif issubclass(norm, RobustNormFunction):
        return norm(*args, **kwargs)
