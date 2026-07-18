"""Runtime composition of callables via overloaded arithmetic operators.

Unlike :class:`kanly.automatic_differentiation.function_callable.FunctionCallable`, this
module does not parse strings or emit symbolic derivatives; it only wraps Python
callables so ``f + g``, ``f * g``, etc. produce new callables.
"""
from __future__ import absolute_import, print_function

from typing import Callable

from kanly.dill_object import DillObject


class FunctionWrapper(DillObject):
    """Wrap a callable or numeric constant so ``+``, ``-``, ``*``, ``/``, ``**`` compose new callables."""

    def __init__(self, function):
        """Store ``function`` as the underlying unary callable (constants become constant lambdas).

        Parameters
        ----------
        function : callable or int or float
            If callable, invoked as ``function(x)``. If numeric, treated as ``lambda x: constant``.
        """
        if isinstance(function, Callable):
            self.function = function
        elif isinstance(function, (int, float)):
            self.function = lambda x: function
        else:
            raise Exception

    def __call__(self, x, *args, **kwargs):
        """Evaluate the wrapped function at ``x`` (extra args forwarded)."""
        return self.function(x)

    def __neg__(self):
        """Return a wrapper for ``-self(x)``."""
        return FunctionWrapper(lambda x: -self.function(x))

    def __pos__(self):
        """Identity: return ``self``."""
        return self

    def __add__(self, other):
        """Return ``self + other`` as a new :class:`FunctionWrapper` (right-hand numeric/callable coerced)."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: self.function(x) + other.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return self + FunctionWrapper(other)
        else:
            raise Exception

    def __radd__(self, other):
        """``other + self``; delegates to :meth:`__add__` (addition is commutative)."""
        return self.__add__(other)

    def __sub__(self, other):
        """Return ``self - other`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: self.function(x) - other.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return self - FunctionWrapper(other)
        else:
            raise Exception

    def __rsub__(self, other):
        """Return ``other - self`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: other.function(x) - self.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return FunctionWrapper(other) - self
        else:
            raise Exception

    def __mul__(self, other):
        """Return ``self * other`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: other.function(x) * self.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return FunctionWrapper(other) * self
        else:
            raise Exception

    def __rmul__(self, other):
        """``other * self``; delegates to :meth:`__mul__`."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Return ``other / self`` (note operand order matches historical implementation)."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: other.function(x) / self.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return FunctionWrapper(other) / self
        else:
            raise Exception

    def __rtruediv__(self, other):
        """Return ``self / other`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: self.function(x) / other.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return self / FunctionWrapper(other)
        else:
            raise Exception

    def __pow__(self, other):
        """Return ``self ** other`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: self.function(x) ** other.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return self ** FunctionWrapper(other)
        else:
            raise Exception

    def __rpow__(self, other):
        """Return ``other ** self`` as a new wrapper."""
        if isinstance(other, FunctionWrapper):
            return FunctionWrapper(
                lambda x: other.function(x) ** self.function(x)
            )
        elif isinstance(other, (Callable, int, float)):
            return FunctionWrapper(other) ** self
        else:
            raise Exception


# if __name__ == '__main__':
#
#     f1 = FunctionWrapper(lambda x: x ** 2)
#     f2 = lambda x: 1.2 * x
#
#     print('* ', f1(1), f2(1))
#
#     print((f1 + f2)(1.))
#     print((f2 + f1)(1.))
#
#     print((f1 - f2)(1.))
#     print((f2 - f1)(1.))
#
#     print((f1 * f2)(1))
#     print((f2 * f1)(1))
#     print((f1 * 2)(1))
#
#     print((f1 / f2)(1))
#     print((f2 / f1)(1))
#
#     print((f1 ** f2)(1.1))
#     print((f2 ** f1)(1.1))
