from __future__ import absolute_import, print_function

import numpy as np


def update_bfgs_hessian_approx(hessian_cur, grad_new, grad_cur, x_new, x_cur, use_inv_hessian):
    """
    https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm#:~:text=In%20numerical%20optimization%2C%20the%20Broyden,solving%20unconstrained%20nonlinear%20optimization%20problems.

    Apply the standard BFGS update to a Hessian approximation.

    Args:
        hessian_cur: Current Hessian approximation.
        grad_new: Gradient at ``x_new``.
        grad_cur: Gradient at ``x_cur``.
        x_new: New parameter vector after an accepted step.
        x_cur: Previous parameter vector before the accepted step.
        use_inv_hessian:
            If False, apply the standard BFGS Hessian update.
            If True, treat hessian_cur as an inverse Hessian and apply
            the inverse-BFGS update.

    Returns:
        Updated Hessian approximation using the curvature pair
        ``y = grad_new - grad_cur`` and ``s = x_new - x_cur``.
    """

    y = grad_new - grad_cur
    s = x_new - x_cur

    ys = np.dot(y, s)
    if ys <= 0:
        raise ValueError(
            f"BFGS curvature condition violated: y^T s = {ys}"
        )

    if use_inv_hessian:
        # Inverse-BFGS update
        Hinv = hessian_cur

        rho = 1.0 / ys
        I = np.eye(Hinv.shape[0])

        V = I - rho * np.outer(s, y)

        return V @ Hinv @ V.T + rho * np.outer(s, s)

    else:
        # Standard BFGS Hessian update
        H = hessian_cur

        Hy = H @ s
        sHs = np.dot(s, Hy)

        if sHs <= 0:
            raise ValueError(
                f"BFGS Hessian update failed: s^T H s = {sHs}"
            )

        return H + np.outer(y, y) / ys - np.outer(Hy, Hy) / sHs


def get_gradient_function(func, onesided=True, dx=1e-6):
    """Build a finite-difference gradient callable for an objective_function.

    Args:
        func: Objective callable accepting a one-dimensional parameter vector.
        onesided: If True, use forward differences. If False, use centered
            differences.
        dx: Relative finite-difference step size. Each coordinate step is
            scaled by ``max(1, abs(x[i]))``.

    Returns:
        A callable that accepts a parameter vector and returns a gradient
        estimate with the same length.
    """

    def grad(x):
        """Estimate ``func``'s gradient at ``x``.

        Args:
            x: Point at which to estimate the gradient.

        Returns:
            One-dimensional finite-difference gradient estimate.
        """
        df_dx = np.full(len(x), 0, dtype=float)

        f0 = func(x)
        for i in range(len(x)):

            dx_i = dx * max(1.0, abs(x[i]))
            x1 = x.copy()
            x1[i] += dx_i

            if onesided:
                df_dx[i] = (func(x1) - f0) / dx_i

            else:
                x2 = x.copy()
                x2[i] -= dx_i

                df_dx[i] = (func(x1) - func(x2)) / (2 * dx_i)
        return df_dx

    return grad