from __future__ import absolute_import, print_function

from collections.abc import Iterable

import numpy as np

from kanly.dill_object import DillObject
from kanly.utils.overwrite_parameter import overwrite_parameter_index
from kanly.utils.dict_2_array import dict_2_array


class ParameterCollection(DillObject):
    """Class mainly for sharing parameter grouping functionality across
    `DataModel` and `BayesianModel` classes.

    Also used for `RegressionResultsBase`-extending classes.
    """

    def __init__(self, param_names, parameter_groupings=None):
        self.num_params = len(param_names)
        self.param_names = param_names
        self.param_2_idx = dict(zip(param_names, range(self.num_params)))
        if parameter_groupings is None:
            parameter_groupings = dict()
        self.parameter_groupings = parameter_groupings

    # def param_dict_to_vec(self, param_dict, x=None, ignore_extra_keys=True):
    #    return dict_2_array(param_dict, self.param_names, ignore_extra_keys=ignore_extra_keys, return_array=x)

    def overwrite_parameter_index(self, x: Iterable, overwrite_vals: dict, copy: bool = True):
        return overwrite_parameter_index(x, overwrite_vals, param_2_idx=self.param_2_idx, copy=copy)

    def dict_2_array(self, param_dict, ignore_extra_keys=True, default_value=0.0):
        return dict_2_array(param_dict, self.param_names, ignore_extra_keys=ignore_extra_keys, default_value=default_value)

    def bounds_dict_2_array(self):
        bounds = self.bounds
        if bounds is None:
            bounds = dict()
        return np.array([
            bounds.get(k, (-np.inf, np.inf)) for k in self.param_names
        ]).T

    def get_frozen_function(self, function, fixed_param_dict):

        if fixed_param_dict is None:
            return function

        fixed_params_ind = np.array([
            self.param_2_idx.get(k, i)
            for i, k in enumerate(fixed_param_dict.keys())])
        fixed_params_vals = np.array(list(fixed_param_dict.values()))

        def temp_func(x):
            x = np.asarray(x)
            x[fixed_params_ind] = fixed_params_vals
            return function(x)

        return temp_func
