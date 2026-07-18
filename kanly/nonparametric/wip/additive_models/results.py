from __future__ import absolute_import, print_function

import datetime

import matplotlib.pyplot as plt
import numpy as np

from kanly.dill_object import DillObject


class GeneralizedAdditiveModelResults(DillObject):

    def __init__(self, alpha, f, fittedvalues, resid_resid, linear_predictor, irls_weights, family, family_name, link,
                 link_name,
                 error, iter, fit_elapsed, pseudo_rsquared, iter_inner, prediction_function, interp_functions,
                 exog, endog, exog_names, endog_name, specification_name):
        self.alpha = alpha
        self.f = f
        self.num_predictors = f.shape[1]
        self.nobs = f.shape[0]
        self.fittedvalues = fittedvalues
        self.resid_resid = resid_resid
        self.linear_predictor = linear_predictor
        self.irls_weights = irls_weights
        self.family = family
        self.family_name = family_name
        self.link = link
        self.link_name = link_name
        self.error = error
        self.iter = iter
        self.fit_elapsed = fit_elapsed
        self.pseudo_rsquared = pseudo_rsquared
        self.iter_inner = iter_inner
        self.prediction_function = prediction_function
        self.interp_functions = interp_functions
        self.from_formula = False

        self.exog_names = ['<x%d>' % j for j in range(self.num_predictors)] if exog_names is None else exog_names
        self.endog_name = '<y>' if endog_name is None else endog_name

        self.exog = exog
        self.endog = endog

        self.specification_name = str(specification_name) if specification_name is not None else None
        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

    def plot(self, dpi=150, figsize=(7, 3), show=False):
        N = self.num_predictors + 1
        if N < 2:
            ncols, nrows = N, 1
        else:
            ncols = 2
            nrows = int(np.ceil(N / 2))
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, dpi=dpi, figsize=figsize)
        for j in range(self.num_predictors):
            r, c = j // 2, j % 2
            ax[r][c].scatter(self.exog[:, j], self.link.link(self.endog) - self.linear_predictor + self.f[:, j],
                             alpha=.5,
                             marker='x', c='grey')
            ax[r][c].plot(xx := np.linspace(self.exog[:, j].min(), self.exog[:, j].max(), 100),
                          self.interp_functions[j](xx), c='r')
            ax[r][c].set_xlabel(self.exog_names[j])
            ax[r][c].set_ylabel(f'f({self.exog_names[j]})')
        r, c = (j + 1) // 2, (j + 1) % 2
        ax[r][c].scatter(self.endog, self.fittedvalues, alpha=.5, marker='x', c='grey')
        ax[r][c].plot(xx := (self.endog.min(), self.endog.max()), xx, c='r')
        ax[r][c].set_xlabel(self.endog_name)
        ax[r][c].set_ylabel(f'fitted {self.endog_name}')
        plt.tight_layout()

        if show:
            plt.show()

        return fig