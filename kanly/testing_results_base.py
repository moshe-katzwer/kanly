from __future__ import absolute_import, print_function

import pprint
from abc import abstractmethod

import numpy as np
import pandas as pd

from kanly import __version__
from kanly.parameter_collection import ParameterCollection
from kanly.stats.statistical_tests import StatisticalTests
from kanly.utils.function_str_to_callable import _check_func_for_test


class TestingResultsBase(ParameterCollection):
    """Base class for testing"""

    def test_delta_method(self, func, null_hypothesis=0, step=.002, test_level=.05):
        func = _check_func_for_test(func, self.param_names)
        return StatisticalTests.delta_method_test(
            func, self._params, self._cov_params, null_hypothesis=null_hypothesis,
            step=step, test_level=test_level)

    def simulate_from_clt(self, func, n_trials=100000):
        func = _check_func_for_test(func, self.param_names)
        return StatisticalTests.simulate_from_clt(func, self._params, self._cov_params, n_trials=n_trials)

    def test_from_clt_simulation(self, func, null_hypothesis=0, n_trials=100_000, test_level=.05, tail='two', seed=0,
                                 include_samples=False):
        func = _check_func_for_test(func, self.param_names)
        return StatisticalTests.clt_simulation_test(func, self._params, self._cov_params,
                                                    null_hypothesis=null_hypothesis, n_trials=n_trials,
                                                    test_level=test_level, include_samples=include_samples,
                                                    tail=tail, seed=seed)

    def test_linear_combination(self, coefficients, null_hypothesis=0, step=.005, test_level=.05):
        if isinstance(coefficients, dict):
            coefficients = np.array([coefficients.get(k, 0.) for k in self.param_names])
        return self.test_delta_method(lambda x: np.dot(x, coefficients),
                                      null_hypothesis=null_hypothesis, step=step, test_level=test_level)

    def test_linear_combination_from_param_dict(self, param_dict, null_hypothesis=0, step=.005, test_level=.05):
        coefficients = np.array([param_dict[k] for k in self.param_names])
        return self.test_linear_combination(coefficients=coefficients, null_hypothesis=null_hypothesis, step=step,
                                            test_level=test_level)

    def test_ratio_fieller(self, top, bottom, null_hypothesis=0, top_constant=0, bottom_constant=0, test_level=.05):
        if isinstance(top, dict):
            top = [top.get(k, 0) for k in self.param_names]
        if isinstance(bottom, dict):
            bottom = [bottom.get(k, 0) for k in self.param_names]
        return StatisticalTests.ratio_fieller_test(self._params, self._cov_params, top, bottom,
                                                   null_hypothesis=null_hypothesis, test_level=test_level,
                                                   top_constant=top_constant, bottom_constant=bottom_constant)

    @abstractmethod
    def summary(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key):
        raise NotImplementedError

    def to_string(self, keys=None):
        self_dict = self.__dict__
        if keys is not None:
            self_dict = {k: self_dict.get(k, None) for k in keys}
        return pprint.pformat(self_dict)

    @staticmethod
    def get_version_str(width):
        return " " * max(width - len(__version__) - 11, 0) + "[kanly v=%s]\n" % __version__
