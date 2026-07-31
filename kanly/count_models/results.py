from __future__ import absolute_import, print_function

from kanly.regression.regression_results_base import RegressionResultsBase

DEFAULT_TEST_LEVEL = .05

class CountModelResults(RegressionResultsBase):
    
    def __init__(self, nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=None, endog_name=None,
                 cov_type=None, cov_kwds=None, test_level=DEFAULT_TEST_LEVEL, use_t=True,
                 alpha=0.0, l1_ratio=0.0, specification_name=None):

        super().__init__(
            nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=None, endog_name=None,
                 cov_type=None, cov_kwds=None, test_level=DEFAULT_TEST_LEVEL, use_t=True,
                 alpha=0.0, l1_ratio=0.0, specification_name=None
        )