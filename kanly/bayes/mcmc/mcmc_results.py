from __future__ import absolute_import, print_function

import datetime
import gc
import pprint
import random
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from kanly import __version__
from kanly.bayes.mcmc.adaptive_metropolis.constants import METHOD as AMH_METHOD
from kanly.bayes.mcmc.aggregate_covariances import aggregate_covs
from kanly.bayes.mcmc.diagnostics.batched_means import \
    get_ess_batched_means_for_chains, DEFAULT_BATCHED_MEANS_SELECTION, multi_ess
from kanly.bayes.mcmc.diagnostics.gelman_rubin_split_rhat import get_rhat
from kanly.bayes.mcmc.diagnostics.geweke import geweke_approx
from kanly.dill_object import DillObject
from kanly.nonparametric.kde import kde
from kanly.time_series.auto_correlation_function import auto_correlation_function as acf
from kanly.utils.function_str_to_callable import _check_func_for_test, get_key
from kanly.utils.highest_density_interval import get_highest_density_interval
from kanly.utils.plot_confidence_intervals import plot_confidence_intervals

THRESHOLD_R_HAT = 1.01
THRESHOLD_ESS = 5_000


class MCMCResults(DillObject):

    """
    Rich post-sampling results container for MCMC chains.

        Stores all draw samples from one or more parallel chains, computes
        and caches convergence diagnostics (split R-hat, ESS via ACF and
        batched means, Geweke z-scores), provides summary tables, plotting
        utilities (trace, KDE, histogram, rank, scatter), credible intervals
        (equi-tailed and HPDI), and supports extending sampling by calling
        ``amha`` again from the last state of each chain.

        Inherits from ``DillObject`` so the full results object can be
        serialized to disk with ``dill`` for later inspection.

        Key attributes (set by ``__init__`` and ``set_summary_df``):

        - ``sample_df``: DataFrame of shape ``(n_chains * n_iterations, num_params)``
          containing all raw draw samples.
        - ``sample_info_df``: Per-draw metadata (chain index, log-posterior,
          burn-in flag, bloc/pid).
        - ``summary_df``: Per-parameter summary statistics (mean, std, MCSE,
          median, credible intervals, ESS, R-hat).
        - ``cov_params``: Posterior sample covariance matrix as a DataFrame.
        - ``map_params``: Parameter values at the highest log-posterior draw.
        - ``R_hat``: Split Gelman-Rubin R-hat per parameter.
        - ``ess``: ESS per parameter per chain from the ACF estimator.
    """
    def __init__(self, method, num_params, log_posterior, log_posterior_jacobian_adjustment, param_names,
                 chain_results, mcmc_time, n_burnin, n_chains, thinning, options, specification_name=None,
                 debug=False, model=None, fix_params=None, other_info=None, transformations=None,
                 ):

        """
        Initialize MCMCResults and compute all convergence diagnostics.

        Args:
            method: Name of the MCMC algorithm used (e.g. ``'amha'`` or ``'mala'``).
            num_params: Number of sampled (free) parameters.
            log_posterior: Callable ``log_posterior(x)`` evaluated on the unbounded/transformed space.
            log_posterior_jacobian_adjustment: Callable that returns the Jacobian log-determinant of the bounded-to-unbounded transformation, or None.
            param_names: Ordered sequence of parameter name strings; auto-generated as ``<theta_i>`` when None.
            chain_results: List of per-chain result dicts returned by the sampler, each containing ``'samples'``, ``'log_posterior'``, ``'accepteds'``, ``'bloc'``, ``'pid'``, etc.
            mcmc_time: Wall-clock time in seconds spent running the sampler.
            n_burnin: Number of leading draws per chain to discard as burn-in.
            n_chains: Number of parallel MCMC chains.
            thinning: Thinning interval applied to draws during sampling.
            options: Dict of sampler configuration options forwarded from the calling function.
            specification_name: Optional display label for the model or run.
            debug: Whether to print timing and diagnostic messages during initialization.
            model: Optional reference to the originating ``BayesianModel`` instance.
            fix_params: Optional dict mapping parameter names (or integer indices) to fixed values that were held constant during sampling.
            other_info: Dict of additional sampler state (e.g. proposal covariance, scaler) for restarting.
            transformations: Dict mapping parameter indices to ``TransformedParameter`` objects that reparameterize bounded parameters to an unbounded space.
        """
        _result_init_time = time.time()

        self.method = method
        self.log_posterior = log_posterior
        self.log_posterior_jacobian_adjustment = log_posterior_jacobian_adjustment
        self.specification_name = specification_name
        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

        self.fix_params = fix_params
        if self.fix_params is not None:
            self.fix_params = self.fix_params.copy()
            if len(self.fix_params) == 0:
                self.fix_params = None

        if param_names is None:
            param_names = [f'<theta{d}>' for d in range(num_params)]

        self.param_names = np.array(param_names)
        self.num_params = num_params

        if self.fix_params is not None:
            par_names_set = set(self.param_names)
            self.fix_params = {(k if k in par_names_set else param_names[int(k)]): v
                               for k, v in fix_params.items()}

        self.sample_df, self.sample_info_df, self.pids \
            = self.build_sample_df(chain_results, n_burnin, n_chains, param_names, debug=debug)

        self.n_chains = n_chains
        self.thinning = thinning
        self.n_iterations = len(self.sample_df) // self.n_chains
        self.set_n_burnin(n_burnin, debug=debug)

        self.map_idx, self.max_log_posterior, self.map_params = self.get_map()

        self.chain_results = chain_results
        self.acceptance_rate = np.mean([np.mean(c['accepteds']) for c in chain_results])

        self.mcmc_time = mcmc_time

        self.options = options
        self.other_info = other_info
        self.model = model

        self.transformations = transformations.copy() if transformations is not None else None
        self.transformed_model = self.transformations is not None and len(self.transformations)

        self.param_2_idx = {p: i for i, p in enumerate(self.param_names)}

        self.__version__ = __version__

        self.result_init_time = time.time() - _result_init_time

        gc.collect()

    @staticmethod
    def get_result_type():
        """
        Return the result type identifier for this object.

        Returns:
            The string ``'MCMC'``.
        """
        return 'MCMC'

    def set_n_burnin(self, n_burnin, debug=False):
        """
        Set (or update) the burn-in cutoff and recompute all summary statistics.

        Args:
            n_burnin: Number of leading draws per chain to classify as burn-in.
            debug: Whether to print timing diagnostics.
        """
        assert n_burnin * self.n_chains < len(self.sample_df)

        self.n_burnin = n_burnin
        self.n_samples = self.n_iterations - self.n_burnin

        self.sample_info_df['is_burnin'] = False
        for c in range(self.n_chains):
            self.sample_info_df.loc[
                self.sample_info_df.index[self.sample_info_df.chain == c][np.arange(self.n_iterations) < n_burnin],
                'is_burnin'] = True

        self.set_summary_df(debug=debug)

    def compute_ess_from_acf(self, maxlag=10_000, min_rho=0, debug=False):

        """
        Compute per-chain ACFs and ACF-based effective sample sizes.

                For each parameter and chain, computes the ACF of the post-burn-in
                draws up to ``maxlag``, truncating at the first lag where consecutive
                ACF pairs sum below ``min_rho`` (the initial positive-sequence
                estimator truncation rule).  ESS is then estimated as
                ``n / (1 + 2 * sum(acf[1:]))``.

        Args:
            maxlag: Maximum lag to include in the ACF summation.
            min_rho: Truncation threshold; the ACF sum is cut at the first lag pair below this value.
            debug: Whether to print timing and progress information.

        Returns:
            3-tuple ``(acfs_dict, ess, ess_time)`` where ``acfs_dict`` maps each parameter name to a per-chain dict of truncated ACF arrays, ``ess`` is a DataFrame of per-chain ESS values indexed by parameter name, and ``ess_time`` is the wall-clock time in seconds.
        """
        _t = time.time()
        if debug:
            print("Computing Auto-correlations and Effective Sample Size (ESS)...", end="")

        acfs_dict = {k: dict() for k in self.param_names}
        ch_index = {c: ~self.sample_info_df.is_burnin & (self.sample_info_df.chain == c)
                    for c in self.sample_info_df.chain.unique()}
        ess = []
        for key in tqdm(self.param_names, disable=not debug):
            for c in self.sample_info_df.chain.unique():
                idx = ch_index[c]
                x = self.sample_df[key].loc[idx].values

                acf_vals = acf(x)

                i = np.where(acf_vals[1:] + acf_vals[:-1] < min_rho)[0]
                if len(i):
                    i = min(i[0], maxlag)
                else:
                    i = maxlag
                acf_vals2 = acf_vals[:i]
                acfs_dict[key][c] = acf_vals2

                ess.append({'param': key,
                            'chain': c,
                            'ESS': int(len(x) * 1 / (1 + 2 * np.sum(acf_vals2[1:])))
                            })

        ess = pd.DataFrame(ess).pivot_table(index='param', columns='chain')
        ess = ess.loc[self.param_names]
        ess.columns = range(self.n_chains)
        ess.index = self.param_names

        ess_time = time.time() - _t

        if debug:
            print("%.2fs" % ess_time)

        return acfs_dict, ess, ess_time

    def set_summary_df(self, maxlag=10_000, min_rho=.0, debug=False):  # , geweke_frac1=.1, geweke_frac2=.5):

        """
        Build and cache the summary statistics DataFrame.

                Computes posterior covariance, means, quantiles, fraction positive,
                split Gelman-Rubin R-hat via ``get_rhat``, and ESS via
                ``compute_ess_from_acf``.  Results are stored in ``self.summary_df``,
                ``self.cov_params``, ``self.mean_params``, ``self.R_hat``,
                ``self.ess``, etc.  Fixed parameters (if any) have their R-hat and
                MCSE set to NaN.

        Args:
            maxlag: Maximum ACF lag passed to ``compute_ess_from_acf``.
            min_rho: Truncation threshold passed to ``compute_ess_from_acf``.
            debug: Whether to print timing messages.
        """
        draws = self.sample_df.loc[~self.sample_info_df.is_burnin, self.param_names]

        cov = np.cov(draws, rowvar=False)
        if np.ndim(cov) < 2:
            cov = cov.reshape((1, 1))
        self.cov_params = pd.DataFrame(cov, columns=self.param_names, index=self.param_names)
        self.mean_params = draws.mean(axis=0)

        if debug:
            _t = time.time()
            print("Computing equi-tail 90% credible intervals... ", end="")
        quantiles = np.array([
            np.quantile(draws[k], (.05, .5, .95))
            for k in self.param_names
        ]).T
        positive = np.array([
            np.count_nonzero(draws[k] > 0) / len(draws)
            for k in self.param_names
        ])
        if debug:
            print('%.2fs' % (time.time() - _t))

        draws_per_chain = len(self.sample_df) / self.n_chains
        R_hat, _, self.R_hat_time, _, _ = get_rhat(
            self._get_samples_split_by_chain(include_burnin=False),
            split=2, debug=debug)

        if self.fix_params is not None:
            for i, k in enumerate(self.param_names):
                if k in self.fix_params:
                    R_hat[i] = np.nan

        self.R_hat = pd.Series(R_hat, index=self.param_names)
        self.max_R_hat = np.nanmax(self.R_hat.max())
        self.avg_R_hat = np.nanmean(self.R_hat.mean())
        self.median_R_hat = np.nanmedian(self.R_hat.mean())

        with np.errstate(divide='ignore', invalid='ignore'):
            std_params = np.diag(self.cov_params) ** .5

        self.summary_df = pd.DataFrame(
            {
                'mean': self.mean_params,
                'std': std_params,
                'median': quantiles[1],
                '[.05, ': quantiles[0],
                '.95]': quantiles[2],
                'positive': positive,
                'R_hat': self.R_hat,
            },
            index=self.param_names
        )
        if self.fix_params is not None:
            self.summary_df['fixed'] = [
                '*' if k in self.fix_params else ''
                for k in self.param_names
            ]

        # self.geweke_df = self.geweke(frac1=geweke_frac1, frac2=geweke_frac2, debug=debug)
        # self.summary_df['max|z_Geweke|'] = np.abs(self.geweke_df).max(axis=1).round(2)

        self.set_ess(maxlag=maxlag, min_rho=min_rho, debug=debug)
        # self.set_multi_ess(debug=debug)

        if self.fix_params is not None:
            idx = [k in self.fix_params for k in self.param_names]
            self.summary_df.loc[idx, 'MCSE'] = np.nan
            self.summary_df.loc[idx, 'std'] = np.nan
            # self.geweke_df.loc[idx] = np.nan
            # self.summary_df.loc[idx, 'max|z_Geweke|'] = np.nan

        self.mean_params = self.summary_df['mean'].copy()
        self.params = self.mean_params
        self.median_params = self.summary_df['median'].copy()
        self.bse = self.summary_df['std'].copy()

    def set_ess(self, maxlag=10_000, min_rho=.0, debug=False):

        """
        Compute and store ESS and MCSE columns in ``summary_df``.

                Calls ``compute_ess_from_acf``, stores ``self.ess`` and ``self.acfs``,
                appends ``ESS``, ``MCSE``, and ``Rel MCSE`` columns to ``summary_df``,
                and computes ``min_efficiency``, ``avg_efficiency``, and
                ``max_efficiency`` as the ESS-to-sample ratio across free parameters.

        Args:
            maxlag: Maximum ACF lag.
            min_rho: ACF truncation threshold.
            debug: Whether to print timing messages.
        """
        _t = time.time()
        # if debug:
        #    print('Setting ess...', end='')

        acfs, ess, self.ess_time = self.compute_ess_from_acf(maxlag, min_rho, debug=debug)
        self.ess = ess
        self.acfs = acfs

        if self.fix_params is not None:
            for i, k in enumerate(self.param_names):
                if k in self.fix_params:
                    ess.loc[k] = 0

        self.summary_df['ESS'] = ess.sum(axis=1)
        self.summary_df['MCSE'] = self.summary_df['std'] / np.sqrt(self.summary_df['ESS'])
        self.summary_df['Rel MCSE'] = self.summary_df['MCSE'] / self.summary_df['mean']

        if self.fix_params is not None:
            not_fixed_index = [k not in self.fix_params for k in self.param_names]
        else:
            not_fixed_index = self.summary_df.index

        self.min_efficiency = self.summary_df['ESS'].loc[not_fixed_index].min() / (self.n_samples * self.n_chains)
        self.avg_efficiency = self.summary_df['ESS'].loc[not_fixed_index].mean() / (self.n_samples * self.n_chains)
        self.max_efficiency = self.summary_df['ESS'].loc[not_fixed_index].max() / (self.n_samples * self.n_chains)

        self.summary_df = self.summary_df[
            ['mean', 'std', 'MCSE', 'median', '[.05, ', '.95]', 'positive', 'Rel MCSE', 'ESS', 'R_hat',
             # 'max|z_Geweke|'
             ]
            + (['fixed'] if 'fixed' in self.summary_df.columns else [])
            ]

        # if debug:
        #    print('%.2fs' % (time.time() - _t))

    @staticmethod
    def build_sample_df(chain_results, n_burnin, n_chains, param_names, debug=False):

        """
        Concatenate per-chain raw draw arrays into tidy DataFrames.

                Stacks the ``'samples'`` arrays from all chains row-wise into
                ``sample_df``, assembles per-draw metadata (chain index, log-posterior,
                burn-in flag, bloc, pid) into ``sample_info_df``, and deletes the
                large array keys from ``chain_results`` to free memory.

        Args:
            chain_results: List of per-chain result dicts from the sampler.
            n_burnin: Number of leading draws per chain marked as burn-in.
            n_chains: Number of parallel chains.
            param_names: Ordered parameter name list used as DataFrame column names.
            debug: Whether to print timing messages.

        Returns:
            3-tuple ``(sample_df, sample_info_df, pids)`` where ``sample_df`` is a DataFrame of shape ``(n_chains * n_iterations, num_params)``, ``sample_info_df`` holds per-draw metadata, and ``pids`` is the array of unique worker process IDs.
        """
        _t = time.time()
        if debug:
            print("Joining samples from chains... ", end='')

        samples = np.vstack([r['samples'] for r in chain_results])
        blocs = np.hstack([r['bloc'] for r in chain_results])
        pids = np.hstack([r['pid'] for r in chain_results])
        gc.collect()

        for r in chain_results:
            for key in ['samples', 'bloc', 'pid']:
                del r[key]

        sample_df = pd.DataFrame(samples, columns=param_names)

        sample_info_df = pd.DataFrame()
        log_posterior_arr = np.hstack([r['log_posterior'] for r in chain_results])
        sample_info_df['log_posterior'] = log_posterior_arr
        for r in chain_results:
            del r['log_posterior']

        _total_n_draws = len(sample_df)
        sample_info_df['chain'] = np.repeat(range(n_chains), _total_n_draws // n_chains)
        sample_info_df['is_burnin'] = (np.arange(_total_n_draws) % (_total_n_draws // n_chains)) < n_burnin
        sample_info_df['bloc'] = blocs
        sample_info_df['pid'] = pids
        pids = sample_info_df['pid'].unique()

        if debug:
            print("%.2fs" % (time.time() - _t))

        return sample_df, sample_info_df, pids

    def get_map(self):
        """
        Locate the maximum a posteriori (MAP) draw across all chains and iterations.

                Scans ``sample_info_df['log_posterior']`` for its argmax and returns
                the corresponding row index, log-posterior value, and parameter vector.

        Returns:
            3-tuple ``(map_idx, max_log_posterior, map_params)`` where ``map_idx`` is the integer row index in ``sample_df``, ``max_log_posterior`` is the corresponding log-posterior value, and ``map_params`` is a Series of parameter values at that draw.
        """
        assert hasattr(self, 'sample_info_df')
        assert hasattr(self, 'sample_df')
        map_idx = np.argmax(self.sample_info_df['log_posterior'])
        max_log_posterior = self.sample_info_df['log_posterior'][map_idx]
        map_params = self.sample_df.loc[map_idx, :].copy()
        return map_idx, max_log_posterior, map_params

    def log_posterior_original(self, x):
        """
        Evaluate the log-posterior in the original (possibly bounded) parameter space.

                If the model was reparameterized (``self.transformed_model`` is True),
                first maps ``x`` to the unbounded space via ``inv_transform`` before
                calling ``self.log_posterior``.  Otherwise calls ``self.log_posterior``
                directly.

        Args:
            x: Parameter vector in the original (bounded) space.

        Returns:
            Scalar log-posterior value.
        """
        if self.transformed_model:
            return self.log_posterior(self.inv_transform(x))
        else:
            return self.log_posterior(x)

    def transform(self, x):
        """
        Map a parameter vector from the unbounded (sampling) space to the original space.

                Applies each ``TransformedParameter`` forward transform to the
                corresponding element of ``x``.  Elements without a transformation
                are passed through unchanged.  Returns ``x`` unmodified when the
                model has no transformations.

        Args:
            x: Parameter vector in the unbounded sampling space.

        Returns:
            NumPy array of the same length as ``x`` with bounded/original-scale values.
        """

        if self.transformed_model:
            return np.array([
                self.transformations[i](x[i]) if i in self.transformations else x[i]
                for i in range(self.num_params)
            ])
        else:
            return x

    def inv_transform(self, x):
        """
        Map a parameter vector from the original (bounded) space to the unbounded sampling space.

                Applies each ``TransformedParameter.inv_transform`` to the corresponding
                element of ``x``.  Elements without a transformation are passed through.

        Args:
            x: Parameter vector in the original (bounded) space.

        Returns:
            NumPy array in the unbounded sampling space.
        """

        if self.transformed_model:
            return np.array([
                self.transformations[i].inv_transform(x[i]) if i in self.transformations else x[i]
                for i in range(self.num_params)
            ])
        else:
            return x

    def summary(self, param_subset=None, show_positive=False, show_ESS=True, show_R_hat=True, show_CI=True):

        """
        Build and return a formatted MCMC summary string.

                Renders ``summary_df`` with formatted floating-point values and
                appends two-column header blocks showing chain counts, timing,
                acceptance rate, ESS efficiency, and Gelman-Rubin statistics.
                Optionally suppresses the ``positive``, ESS, R-hat, or credible
                interval columns.  Appends a warning when any R-hat exceeds
                ``THRESHOLD_R_HAT`` or any ESS is below ``THRESHOLD_ESS``.

        Args:
            param_subset: Optional list of parameter names to include; ``None`` shows all.
            show_positive: Whether to include the fraction-positive column.
            show_ESS: Whether to include the ESS column.
            show_R_hat: Whether to include the R-hat column.
            show_CI: Whether to include the credible interval columns.

        Returns:
            Multi-line formatted string ready for printing.
        """
        tbl_copy = self.summary_df.copy()

        def coef_fmt(x):
            """
            Format a scalar value for display in the summary table.

            Args:
                x: Numeric value to format.

            Returns:
                String in scientific notation for very small or very large values, else 4 significant figures.
            """
            if abs(x) < .0001 or abs(x) > 100_000:
                return '%.2e' % x
            else:
                return '%.4g' % x

        for c in ['mean', 'std', 'median', '[.05, ', '.95]', 'positive', 'MCSE', 'Rel MCSE']:
            tbl_copy[c] = tbl_copy[c].apply(coef_fmt)
        if param_subset is not None:
            tbl_copy = tbl_copy[tbl_copy.param.isin(param_subset)]

        tbl_copy['R_hat'] = tbl_copy['R_hat'].apply(lambda x: '%.4f' % x)

        if not show_positive:
            del tbl_copy['positive']
        if not show_ESS:
            del tbl_copy['ESS']
        if not show_R_hat:
            del tbl_copy['R_hat']
        if not show_CI:
            del tbl_copy['[.05, ']
            del tbl_copy['.95]']

        tbl_strs = tbl_copy.to_string().split('\n')
        width = len(tbl_strs[0])

        ret = '═' * width
        ret += '\nMCMC Results'
        if self.specification_name is not None:
            ret += f'\n"{self.specification_name}"'
        ret += '\n' + '─' * width

        ret += '\n\nNum Parameters:     %d' % len(self.param_names)
        ret += '\nMethod:             %s' % self.method

        ret += '\n\n'

        info1 = np.array([
            ('Date:', self.date),
            ('Time:', self.timestamp),
            ('', ''),
            ('Total Iterations:', self.n_chains * (self.n_samples + self.n_burnin)),
            ('MCMC Draw Time:', '%.2fs' % self.mcmc_time),
            ('R_hat Time:', '%.2fs' % self.R_hat_time),
            ('ESS Time:', '%.2fs' % self.ess_time),
            ('Summary Time:', '%.2fs' % self.result_init_time),
            ('', ''),
            ('No. Chains:', self.n_chains),
            ('    Thinning:', self.thinning),
            ('    Iterations:', self.n_iterations),
            ('    Burnin:', self.n_burnin),
            ('    Samples:', self.n_samples),
            ('', ''),
        ])
        info2 = np.array([
            ('Acceptance Rate:', "%.4f" % self.acceptance_rate),
            ('Adaptive:', self.options['do_adaptive']),
            ('', ''),
            # ('Multi ESS (*):', np.sum(self.m_ess)),
            # ('', ''),
            ('Efficiency:', ''),
            ('    Min:', "%.4f" % self.min_efficiency),
            ('    Avg:', "%.4f" % self.avg_efficiency),
            ('    Max:', "%.4f" % self.max_efficiency),
            ('', ''),
            ('Gelman-Rubin:', ''),
            ('    R_hat > 1.01:', f"{np.sum(self.R_hat > 1.01)}/{self.num_params}"),
            ('    Avg Split R_hat:', "%.4f" % self.avg_R_hat),
            ('    Median Split R_hat:', "%.4f" % self.median_R_hat),
            ('    Max Split R_hat:', "%.4f" % self.max_R_hat),
            ('', ''),
            ('Maximum Log Posterior:', "%.4e" % self.max_log_posterior),
        ])
        info_series1 = pd.Series(info1[:, 1], index=info1[:, 0]).to_string().split('\n')
        info_series2 = pd.Series(info2[:, 1], index=info2[:, 0]).to_string().split('\n')
        ret += '\n'.join([f'{i1}        {i2}' for i1, i2 in zip(info_series1, info_series2)])

        ret += '\n'

        ret += '\n' + '═' * width
        ret += '\n' + tbl_strs[0]
        ret += '\n' + '─' * width
        ret += '\n' + '\n'.join(tbl_strs[1:])
        ret += '\n' + '─' * width

        if np.any(self.R_hat > THRESHOLD_R_HAT):
            ret += f'\nSome R_hat are above {THRESHOLD_R_HAT}, the MCMC chains have not converged!'
        if np.any(self.ess.sum(axis=1) < THRESHOLD_ESS):
            if self.fix_params is not None:
                not_fixed_params = np.array([k not in self.fix_params for k in self.param_names])
            else:
                not_fixed_params = True
            if np.any((self.ess.sum(axis=1) < THRESHOLD_ESS) & not_fixed_params):
                ret += f'\nSome effective sample sizes are below {THRESHOLD_ESS}!\n'

        # ret += f"\n{np.sum(np.abs(self.geweke_df.values) > 1.96)}/" \
        #        f"{self.n_chains * (self.num_params - (len(self.fix_params) if self.fix_params else 0))} " \
        #        "approximate Geweke z-scores are stat-sig at 5% level."

        # ret += '\n(*) Multi ESS still experimental\n'
        ret += "\n" + (" " * max(width - 11 - len(self.__version__), 0)) + "[kanly, v=%s]\n" % self.__version__

        return ret

    def __str__(self):
        """
        Return the formatted MCMC summary string (calls ``summary()``).

        Returns:
            String returned by ``self.summary()``.
        """
        return self.summary()

    def __repr__(self):
        """
        Return the formatted MCMC summary string.

        Returns:
            String returned by ``str(self)``.
        """
        return str(self)

    # TODO SWAP OUT?
    # from kanly import __version__
    # from kanly.bayes.mcmc.gelman_rubin_split_rhat import get_rhat
    # from kanly.time_series.auto_correlation_function import auto_correlation_function
    # from kanly.bayes.utils.highest_density_interval import get_highest_density_interval
    # from kanly.bayes.utils.kde_clipped import get_kde_clipped
    # from kanly.utils.function_str_to_callable import _check_func_for_test

    # def diagnostic_plot(self, key, max_points=50_000, scale_exclude_burnin=True, figsize=(10, 5), dpi=130, title=None):
    #
    #     f, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)
    #
    #     vals_w_burnin = self.get_sample(key, include_burnin=True)
    #     n = len(vals_w_burnin)
    #
    #     xx = np.arange(n)
    #     stride_0 = max(1, (n // max_points))
    #     ax[0][0].plot(xx[::stride_0], vals_w_burnin[::stride_0])
    #
    #     if scale_exclude_burnin:
    #         z = vals_w_burnin[~self.sample_info_df.is_burnin][::stride_0]
    #         scale = [z.min(), z.max()]
    #         diff = scale[1] - scale[0]
    #         scale[0] -= .01 * diff
    #         scale[1] += .01 * diff
    #         ax[0][0].set_ylim(scale)
    #
    #     for j in range(self.n_chains):
    #         ax[0][0].axvline(j * (n // self.n_chains), ls=':', color='k')
    #         ax[0][0].axvline(j * (n // self.n_chains) + self.n_burnin, ls='--', color='grey')
    #         ax[0][0].axvspan(j * (n // self.n_chains), j * (n // self.n_chains) + self.n_burnin, alpha=.5, color='grey')
    #     ax[0][0].set_title(f'Trace', fontsize=15)
    #     ax[0][0].set_xlabel('Draw', fontsize=12)
    #     ax[0][0].set_xlim(0, n)
    #
    #     vals = vals_w_burnin[~self.sample_info_df.is_burnin]
    #     xx_overall = np.linspace(*np.quantile(vals, (.0005, .9995)), 250)
    #     kde_overall, _, _ = get_kde_clipped(vals)
    #
    #     ax[0][1].plot(xx_overall, kde_overall(xx_overall), lw=2, color='k')
    #     ax[0][1].hist(vals, density=True, bins=30, alpha=.4)
    #     ax[0][1].set_title(f'Overall KDE\n(ESS = {self.summary_df["ESS"][key] if key in self.param_names else "?"})',
    #                        fontsize=15)
    #
    #     mean, median = np.mean(vals), np.median(vals)
    #     ax[0][1].axvline(mean, color='k', label='mean')
    #     ax[0][1].axvline(median, color='k', ls='--', label='med')
    #     if key in self.param_names:
    #         ax[0][1].axvline(self.map_params[key], color='k', ls=':', label='map')
    #     ax[0][1].legend(loc='best', fontsize=9)
    #
    #     for c in range(self.n_chains):
    #         sample_c = self.get_sample(key, chain=c, include_burnin=False)
    #         kde_c, l, h = get_kde_clipped(sample_c)
    #         x_rng_chn = np.linspace(l, h, 250)
    #         ax[1][1].plot(x_rng_chn, kde_c(x_rng_chn))
    #
    #     ax[1][1].plot(xx_overall, kde_overall(xx_overall), color='k', lw=3)  # TODO
    #     ax[1][1].set_title(f'Chain KDEs\n(R_hat = '
    #                        f'{"%.4f" % self.summary_df.R_hat[key] if key in self.param_names else "?"})',
    #                        fontsize=15)
    #
    #     ax[0][1].set_xlim(ax[1][1].get_xlim())
    #
    #     if key in self.param_names:
    #         for c in range(len(self.chain_results)):
    #             ax[1][0].plot(self.acfs[key][c], marker='.', lw=0)
    #         ax[1][0].axhline(0, color='k')
    #         ax[1][0].set_xlabel('Lag', fontsize=12)
    #         ax[1][0].set_title('Autocorrelation by Chain', fontsize=15)
    #     else:
    #         ax[1][0].set_title('Autocorrelations not Shown\nfor custom functions', fontsize=15)
    #         ax[1][0].axis("off")
    #
    #     plt.suptitle(
    #         (f'"{key}"' +
    #          (f'\n{"%.3e" % self.mean_params[key]}  ({"%.3e" % self.bse[key]})'
    #           if key in self.param_names else ''
    #           )) if title is None else title,
    #         fontsize=17)
    #
    #     plt.tight_layout()
    #
    #     return f

    def equitail_ci(self, key, level):
        """
        Compute the equi-tailed credible interval for a parameter.

        Args:
            key: Parameter name (or callable/index) passed to ``get_sample``.
            level: Coverage probability, e.g. ``0.9`` for a 90% interval.

        Returns:
            2-element array ``[lower, upper]`` of the equi-tailed interval bounds.
        """
        return np.quantile(self.get_sample(key), [(1 - level) / 2, (1 + level) / 2])

    def set_param_names(self, param_names):
        """
        Rename all parameters in-place across every stored DataFrame and dict.

                Updates ``param_names``, ``summary_df`` index, ``cov_params``
                columns/index, ``mean_params``, ``median_params``, ``bse``,
                ``R_hat``, ``map_params``, ``ess`` index, ``acfs``, and
                ``sample_df`` column names.

        Args:
            param_names: New ordered list of parameter names; must have the same length as the current list.
        """
        assert len(param_names) == len(self.param_names)

        old_param_names = self.param_names

        self.param_names = param_names
        self.summary_df.index = param_names
        self.cov_params.columns = param_names
        self.cov_params.index = param_names

        for c in [self.median_params, self.mean_params, self.bse, self.R_hat, self.map_params]:
            c.index = param_names

        self.ess.index = [('ESS', k) for k in param_names]
        self.acfs = {new_k: self.acfs[k] for new_k, k in zip(param_names, old_param_names)}

        self.sample_df.rename(dict(zip(old_param_names, param_names)), axis=1, inplace=True)

    def apply_function_to_sample(self, func):
        """
        Apply an arbitrary function to the post-burn-in sample DataFrame.

                Checks the function signature against ``param_names`` via
                ``_check_func_for_test``, then calls it on the filtered DataFrame.
                Falls back to ``np.vectorize`` if the result length does not match
                the number of post-burn-in rows.

        Args:
            func: Callable that accepts a DataFrame row (or the full DataFrame) and returns a value.

        Returns:
            Array or Series of function outputs aligned to the post-burn-in draws.
        """
        func = _check_func_for_test(func, self.param_names)
        v = func(self.sample_df[~self.sample_info_df.is_burnin])
        if len(v) != np.count_nonzero(~self.sample_info_df.is_burnin):
            func = np.vectorize(func)
            return func(self.sample_df[~self.sample_info_df.is_burnin])
        else:
            return v

    def trace(self, key, scale_exclude_burnin=True, show=False, figsize=None, max_points=5_000):

        """
        Plot per-chain trace plots for a single parameter.

                Creates one subplot row per chain, overlaying horizontal lines at
                the 1st, 5th, 50th, 95th, and 99th percentiles of the post-burn-in
                samples.  The burn-in period is shaded in each subplot.

        Args:
            key: Parameter name (or index/callable).
            scale_exclude_burnin: Whether to set the y-axis limits based only on post-burn-in values.
            show: Whether to call ``plt.show()``.
            figsize: Figure size ``(width, height)``; auto-scaled by chain count when ``None``.
            max_points: Maximum number of draw points to render per chain (sub-sampled if exceeded).

        Returns:
            Matplotlib Figure object.
        """
        if figsize is None:
            figsize = (7, self.n_chains * 1.5)

        fig, ax = plt.subplots(nrows=self.n_chains, dpi=130, figsize=figsize, sharex=True, sharey=True)

        sample_excl_burn = self.get_sample(key, include_burnin=False)
        q01, q05, q50, q95, q99 = np.quantile(sample_excl_burn, [.01, .05, .5, .95, .99])

        for i in range(self.n_chains):
            y = self.get_sample(key, chain=i, include_burnin=True).values
            xrng = np.arange(self.n_iterations)
            if len(y) > max_points:
                stride = int(np.ceil(len(y) / max_points))
                y = y[::stride]
                xrng = xrng[::stride]
            ax[i].plot(xrng, y, alpha=.7)
            ax[i].axhline(q50, color='g', lw=2)
            ax[i].axhline(q05, ls='--', color='g')
            ax[i].axhline(q95, ls='--', color='g')
            ax[i].axhline(q01, ls=':', color='g')
            ax[i].axhline(q99, ls=':', color='g')
            ax[i].axvspan(0, self.n_burnin, color='k', alpha=.05)
            ax[i].axvline(self.n_burnin, color='k', ls=':')
            ax[i].set_ylabel(f'Chain {i}')

        if scale_exclude_burnin:
            l, h = sample_excl_burn.min(), sample_excl_burn.max()
            plt.ylim((l - (h - l) * .05, h + (h - l) * .05))

        plt.xlabel('Draw')
        plt.suptitle(key, fontsize=15)
        plt.tight_layout()
        plt.xlim((0, self.n_iterations))

        if show:
            plt.show()

        return fig

    def get_key(self, key, debug=False):
        """
        Resolve a user-supplied parameter key to a canonical parameter name.

        Args:
            key: Parameter name string, integer index, or callable.
            debug: Whether to print debugging info.

        Returns:
            Resolved parameter key (string, index, or callable).
        """
        return get_key(key, self.param_names, debug)

    def boxplot(self, key, subsample=None, seed=0, show=False, figsize=None):

        """
        Plot per-chain box-style interval plots for a single parameter.

                For each chain, renders a horizontal line plot showing the
                [5, 95] interval, [25, 75] IQR, [33, 67] interval, median (circle),
                and mean (diamond).

        Args:
            key: Parameter name (or index/callable).
            subsample: Maximum number of draws to sample per chain.
            seed: Random seed for sub-sampling.
            show: Whether to call ``plt.show()``.
            figsize: Figure size tuple.

        Returns:
            Matplotlib Figure object.
        """
        if figsize is None:
            figsize = (8, self.n_chains * .5)

        fig = plt.figure(figsize=figsize, dpi=130)

        for i, chn in enumerate(range(self.n_chains)):
            color = None
            x = self.get_sample(key, chain=chn, subsample=subsample, seed=seed)
            z = np.quantile(x, [.05, .25, 1 / 3, .5, 2 / 3, .75, .95])
            temp = plt.plot([z[0], z[-1]], [i, i], lw=2,
                            label='[.05, .95]' if i == 0 else None)
            color = temp[0].get_color()
            plt.plot([z[1], z[-2]], [i, i], lw=4, color=color,
                     label='[.25, .75]' if i == 0 else None)
            plt.plot([z[2], z[-3]], [i, i], lw=6, color=color,
                     label='[.33, .67]' if i == 0 else None)
            a = plt.scatter([z[3]], [i], marker='o', s=90, facecolor='none', edgecolor=color, zorder=3, color=color,
                            label='median' if i == 0 else None)
            a.set_facecolor('white')
            a = plt.scatter([x.mean()], [i], marker='d', s=90, facecolor='none', edgecolor=color, zorder=3, color=color,
                            label='mean' if i == 0 else None)
            a.set_facecolor('white')

        plt.title(f'"{key}"', fontsize=13)
        plt.ylabel("Chain", fontsize=13)
        plt.legend(loc=(1.03, 0))
        if show:
            plt.show()

        return fig

    def scatter(self, key1, key2, subsample=15_000, alpha=.3, figsize=None, dpi=130, seed=0, show=False, chain=None):

        """
        Plot a 2-D scatter of two parameters' post-burn-in draws.

                Draws a scatter of all selected samples in cornflower blue with
                random subsets highlighted in magenta and red to reveal local
                density.  Reports the Pearson correlation in the title.

        Args:
            key1: First parameter name (x-axis).
            key2: Second parameter name (y-axis).
            subsample: Number of draws to use; ``None`` uses all post-burn-in draws.
            alpha: Scatter point transparency.
            figsize: Figure size tuple.
            dpi: Figure resolution.
            seed: Random seed for sub-sampling.
            show: Whether to call ``plt.show()``.
            chain: Optional integer chain index to restrict draws; ``None`` uses all chains.

        Returns:
            Matplotlib Figure object.
        """
        if figsize is None:
            figsize = (5, 5)

        for key in [key1, key2]:
            if self.fix_params and key in self.fix_params:
                raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        fig = plt.figure(figsize=figsize, dpi=dpi)

        y1 = self.get_sample(key1, subsample=subsample, seed=seed, chain=chain, return_array=True)
        y2 = self.get_sample(key2, subsample=subsample, seed=seed, chain=chain, return_array=True)

        plt.scatter(y1, y2, alpha=alpha, color='cornflowerblue')

        rand = np.random.RandomState(seed)
        rindx = rand.randint(0, len(y1), min(1000, len(y1) // 5))
        plt.scatter(y1[rindx], y2[rindx], color='darkmagenta', alpha=alpha)

        rindx = rand.randint(0, len(y1), min(200, len(y1) // 50))
        plt.scatter(y1[rindx], y2[rindx], color='r', alpha=alpha)

        plt.xlabel(key1[:50], fontsize=13)
        plt.ylabel(key2[:50], fontsize=13)

        plt.title('Correlation: %.4f' % np.corrcoef(y1, y2)[0, 1])

        if show:
            plt.show()

        return fig

    def hpdi(self, key, level, num_cdf_interp_points=301, num_lower_bound_search_grid_points=501):
        """
        Compute the Highest Posterior Density Interval (HPDI) for a parameter.

                Delegates to ``get_highest_density_interval`` on the post-burn-in
                draws.

        Args:
            key: Parameter name (or index/callable).
            level: Coverage probability (e.g. 0.9).
            num_cdf_interp_points: Number of grid points for CDF interpolation.
            num_lower_bound_search_grid_points: Search grid size for the optimal lower bound.

        Returns:
            2-tuple ``(lower, upper)`` of HPDI bounds.
        """
        return get_highest_density_interval(self.get_sample(key), level,
                                            num_cdf_interp_points=num_cdf_interp_points,
                                            num_lower_bound_search_grid_points=num_lower_bound_search_grid_points)

    def diagnostic_plot(self, key, max_points=50_000, scale_exclude_burnin=True, figsize=(10, 5), dpi=130, title=None,
                        show=False,
                        clip=True,
                        hist_instead_of_kde=False,  # TODO
                        **kde_options,
                        ):

        """
        Generate a 2×2 diagnostic panel for a single parameter.

                Panel layout:
                - Top-left: full-chain trace (all chains overlaid) with burn-in shaded.
                - Top-right: overall KDE with mean, median, and MAP lines annotated.
                - Bottom-left: ACF per chain (scatter plot).
                - Bottom-right: per-chain KDEs with chain legend.

        Args:
            key: Parameter name (or index/callable).
            max_points: Maximum draw points in the trace panel (sub-sampled if exceeded).
            scale_exclude_burnin: Whether to restrict the trace y-axis to post-burn-in range.
            figsize: Figure size ``(width, height)``.
            dpi: Figure resolution.
            title: Optional suptitle override.
            show: Whether to call ``plt.show()``.
            clip: KDE clip bounds; ``True`` auto-clips to the [0.01%, 99.99%] quantiles.
            hist_instead_of_kde: Not yet implemented; reserved for future use.
            **kde_options: Additional keyword arguments forwarded to the ``kde`` helper.

        Returns:
            Matplotlib Figure object.
        """
        key = self.get_key(key, debug=False)

        if self.fix_params and key in self.fix_params:
            raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        f, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

        vals_w_burnin = self.get_sample(key, include_burnin=True)
        n = len(vals_w_burnin)

        xx = np.arange(n)
        stride_0 = max(1, (n // max_points))
        ax[0][0].plot(xx[::stride_0], vals_w_burnin[::stride_0])

        if scale_exclude_burnin:
            z = vals_w_burnin[~self.sample_info_df.is_burnin][::stride_0]
            scale = [z.min(), z.max()]
            diff = scale[1] - scale[0]
            scale[0] -= .01 * diff
            scale[1] += .01 * diff
            ax[0][0].set_ylim(scale)

        for j in range(self.n_chains):
            ax[0][0].axvline(j * (n // self.n_chains), ls=':', color='k')
            ax[0][0].axvline(j * (n // self.n_chains) + self.n_burnin, ls='--', color='grey')
            ax[0][0].axvspan(j * (n // self.n_chains), j * (n // self.n_chains) + self.n_burnin, alpha=.5, color='grey')
        ax[0][0].set_title(f'Trace', fontsize=15)
        ax[0][0].set_xlabel('Draw', fontsize=12)
        ax[0][0].set_xlim(0, n)

        vals = vals_w_burnin[~self.sample_info_df.is_burnin]
        if isinstance(clip, bool) and clip:
            clip = np.quantile(vals, [.0001, .9999])
        kdeobj = kde(vals, clip=clip,  **kde_options)
        ax[0][1].plot(*kdeobj,  lw=2, color='k')
        ax[0][1].hist(vals, density=True, bins=30, alpha=.4)
        ax[0][1].set_title(f'Overall KDE\n(ESS = {self.summary_df["ESS"][key] if key in self.param_names else "?"})',
                           fontsize=15)

        mean, median = np.mean(vals), np.median(vals)
        ax[0][1].axvline(mean, color='k', label='mean')
        ax[0][1].axvline(median, color='k', ls='--', label='med')
        if key in self.param_names:
            ax[0][1].axvline(self.map_params[key], color='k', ls=':', label='map')
        ax[0][1].legend(loc='best', fontsize=9)

        for c in range(self.n_chains):
            sample_c = self.get_sample(key, chain=c, include_burnin=False)
            kdeobj = kde(sample_c, clip=clip, **kde_options)
            ax[1][1].plot(*kdeobj, label=c)
        ax[1][1].legend(loc='best', fontsize=8, labelspacing=.15)

        #ax[1][1].plot(xx_overall, kde_overall(xx_overall), color='k', lw=3)  # TODO
        ax[1][1].set_title(f'Chain KDEs\n(R_hat = '
                           f'{"%.4f" % self.summary_df.R_hat[key] if key in self.param_names else "?"})',
                           fontsize=15)

        ax[0][1].set_xlim(ax[1][1].get_xlim())

        if key in self.param_names:
            for c in range(len(self.chain_results)):
                ax[1][0].plot(self.acfs[key][c], marker='.', lw=0, label=c)
            ax[1][0].axhline(0, color='k')
            ax[1][0].set_xlabel('Lag', fontsize=12)
            ax[1][0].set_title('Autocorrelation by Chain', fontsize=15)
        else:
            ax[1][0].set_title('Autocorrelations not Shown\nfor custom functions', fontsize=15)
            ax[1][0].axis("off")
        ax[1][0].legend(loc='best', fontsize=8, labelspacing=.15)

        plt.suptitle(
            (f'"{key}"' +
             (f'\nmean = {"%.3e" % self.mean_params[key]}  (std = {"%.3e" % self.bse[key]})'
              if key in self.param_names else ''
              )) if title is None else title,
            fontsize=17)

        plt.tight_layout()

        if show:
            plt.show()

        return f

    def multi_trace(self, keys=None, max_points=5_000, scale_exclude_burnin=True, ncols=3, show=False, figsize=None,
                    dpi=130):

        """
        Plot a compact grid of trace plots for multiple parameters.

                Lays out ``len(keys)`` subplots in a grid with ``ncols`` columns,
                sharing the x-axis.  Burn-in regions are shaded and chain boundaries
                are marked with vertical lines.

        Args:
            keys: Parameter names to plot; defaults to all parameters.
            max_points: Maximum total draw points rendered across the grid.
            scale_exclude_burnin: Whether to set each subplot's y-limits from post-burn-in values.
            ncols: Number of subplot columns.
            show: Whether to call ``plt.show()``.
            figsize: Figure size tuple.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object.
        """
        if keys is None:
            keys = self.param_names

        if isinstance(keys, str):
            keys = [keys]

        p = len(keys)

        ncols = min(ncols, p)
        nrows = int(np.ceil(p / ncols))

        if figsize is None:
            figsize = (2 * ncols, 1.5 + 1.5 * nrows)

        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, dpi=dpi)

        stride = max(len(self.sample_df) // max_points, 1)
        xrng = np.arange(len(self.sample_df))[::stride]

        for i, k in tqdm(enumerate(keys)):
            if p > 3:
                _ax = ax[i // ncols][i % ncols]
            elif p == 1:
                _ax = ax
            else:
                _ax = ax[i % ncols]
            title = k if len(k) < 22 else k[:18] + '...'
            _ax.set_title(title)
            y = self.get_sample(k, include_burnin=True)
            _ax.plot(xrng, y[::stride])
            yax = _ax.axes.get_yaxis()
            yax.set_visible(False)
            for c in range(self.n_chains):
                _ax.axvline(self.n_iterations * c, color='k', ls=':')
                _ax.axvline(self.n_iterations * c + self.n_burnin, color='k', lw=.5, ls=':')
                _ax.axvspan(self.n_iterations * c, self.n_iterations * c + self.n_burnin, color='k', alpha=.1)

            _ax.set_xlim([0, self.n_iterations * self.n_chains])

            if scale_exclude_burnin:
                z = y[~self.sample_info_df.is_burnin]
                scale = [z.min(), z.max()]
                diff = scale[1] - scale[0]
                scale[0] -= .01 * diff
                scale[1] += .01 * diff
                _ax.set_ylim(scale)

        plt.tight_layout()

        if show:
            plt.show()

        return f

    def kde(self, keys=None, seed=0, dpi=130, figsize=None, max_kde_sample_points=10_000, gridsize=500,
            split_chain=False, fontsize=12, ncols=3, show_legend=False, show=False,
            clip=None, **kde_options
            ):

        """
        Plot KDE density curves for one or more parameters.

                For each parameter, plots per-chain KDEs (dashed/dotted) and an
                overall pooled-chain KDE (solid black).  Optionally splits each
                chain into its first and second halves to check within-chain
                stationarity.

        Args:
            keys: List of parameter names to plot.
            seed: Random seed for sub-sampling.
            dpi: Figure resolution.
            figsize: Figure size tuple.
            max_kde_sample_points: Maximum number of draws used for KDE estimation.
            gridsize: Number of grid points in the KDE evaluation.
            split_chain: Whether to plot each chain as two halves (stationarity check).
            fontsize: Title font size.
            ncols: Number of subplot columns.
            show_legend: Whether to show chain legends.
            show: Whether to call ``plt.show()``.
            clip: KDE clip bounds; ``True`` auto-clips.
            **kde_options: Additional keyword arguments forwarded to the ``kde`` helper.

        Returns:
            Matplotlib Figure object.
        """
        for key in keys:
            if self.fix_params and key in self.fix_params:
                raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        if keys is None:
            keys = self.param_names
        if isinstance(keys, (str, int)):
            keys = [keys]

        p = len(keys)
        ncols = min(ncols, p)
        nrows = int(np.ceil(p / ncols))
        if figsize is None:
            figsize = (2 * ncols, 1.5 + 1.5 * nrows)

        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

        for i, key in tqdm(enumerate(keys)):

            if p > 3:
                _ax = ax[i // ncols][i % ncols]
            elif p == 1:
                _ax = ax
            else:
                _ax = ax[i % ncols]

            title = key if len(key) < 22 else key[:18] + '...'
            _ax.set_title(title, fontsize=fontsize)

            # overall kde
            y = self.get_sample(key)
            if isinstance(clip, bool) and clip:
                clip = np.quantile(y, [.0001, .9999])

            # chain kde
            for c in range(self.n_chains):
                if split_chain:
                    ys = [self.get_sample(key, chain=c, half=h) for h in [1, 2]]
                    linestyles = ['--', ':']
                    labels = [f'{c} [half 1]', f'{c} [half 2]']
                else:
                    ys = [self.get_sample(key, chain=c)]
                    linestyles = ['--']
                    labels = [c]

                facecolor = None
                for y, ls, label in zip(ys, linestyles, labels):
                    kdeobj = kde(y, gridsize=gridsize, clip=clip, **kde_options)
                    temp = _ax.plot(*kdeobj, ls=ls, label=f'chain {label}', color=facecolor)
                    facecolor = temp[0].get_color()

            # Overall KDE
            kdeobj = kde(y, gridsize=gridsize, clip=clip, **kde_options)
            _ax.plot(*kdeobj, ls='-', lw=2, color='k', alpha=.8)

            if show_legend:
                _ax.legend(loc='best')

            yax = _ax.axes.get_yaxis()
            yax.set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

        return f

    def hpdi_plot(self, key, level, num_cdf_interp_points=301, num_lower_bound_search_grid_points=501,
                  dpi=130, figsize=(10, 4.6), clip=True, max_kde_sample_points=15_000, num_kde_interp_points=500,
                  show=False, **kde_options):

        """
        Plot the HPDI and equi-tailed interval for a parameter.

                Renders a KDE with histogram overlay, annotates mean, median, MAP,
                HPDI bounds, and equi-tailed interval bounds.

        Args:
            key: Parameter name (or index/callable).
            level: Coverage probability (e.g. 0.9 for a 90% interval).
            num_cdf_interp_points: CDF interpolation grid size.
            num_lower_bound_search_grid_points: HPDI search grid size.
            dpi: Figure resolution.
            figsize: Figure size tuple.
            clip: KDE clip bounds; ``True`` auto-clips.
            max_kde_sample_points: KDE grid size.
            num_kde_interp_points: Unused; reserved.
            show: Whether to call ``plt.show()``.
            **kde_options: Additional keyword arguments forwarded to the ``kde`` helper.

        Returns:
            2-tuple ``(fig, (lower_hpdi, upper_hpdi))``.
        """
        (lp_hpdi, rp_hpdi) = self.hpdi(key, level, num_cdf_interp_points=num_cdf_interp_points,
                                       num_lower_bound_search_grid_points=num_lower_bound_search_grid_points)

        lp_eqt, rp_eqt = self.equitail_ci(key, level)

        f, ax = plt.subplots(dpi=dpi, figsize=figsize)
        data = self.get_sample(key)
        plt.hist(data, density=True, alpha=.2, bins=25, color='b')
        plt.hist(data, density=True, alpha=.2, bins=500, color='b')

        if isinstance(clip, bool) and clip:
            clip = np.quantile(data, [.0001, .9999])

        kdeobj = kde(data, gridsize=max_kde_sample_points, clip=clip, **kde_options)
        plt.plot(*kdeobj, color='b', lw=2.2, label='kde')

        mean, median = data.mean(), np.median(data)
        plt.axvline(mean, lw=1.5, ls='-', color='k',
                    label=f'Mean = {mean:03e}')
        plt.axvline(median, lw=1.5, ls='--', color='g',
                    label=f'Median = {median:03e}')
        if key in self.param_names:
            plt.axvline(self.map_params[key], lw=1.5, ls=':', color='orange',
                        label=f'MAP = {"%.3e" % self.map_params[key]}')

        for i, j in enumerate([lp_hpdi, rp_hpdi]):
            plt.axvline(j, lw=2, ls='--', color='r',
                        label=f'Highest Posterior Density Interval\n[{"%.3e" % lp_hpdi}, {"%.3e" % rp_hpdi}]' if i == 0 else None)

        for i, j in enumerate([lp_eqt, rp_eqt]):
            plt.axvline(j, lw=2, ls=':', color='magenta',
                        label=f'Equitailed Interval\n[{"%.3e" % lp_eqt}, {"%.3e" % rp_eqt}]' if i == 0 else None)

        ax.set_title(f'Parameter "{key}"\nLevel: {np.round(100 * level, 1)}%', fontsize=18)
        plt.legend(loc='best')

        plot_lo, plot_hi = (data.min(), data.max())

        plot_lo -= .025 * (plot_hi - plot_lo)
        plot_hi += .025 * (plot_hi - plot_lo)
        plt.xlim([plot_lo, plot_hi])

        if show:
            plt.show()

        return f, (lp_hpdi, rp_hpdi)

    def multi_scatter(self, keys, subsample=10_000, figsize=(10, 10), bins=25, fontsize=14, seed=0, suptitle=None,
                      show=False, max_axis_label_len=24, chain=None, labels=None):

        """
        Plot a pairwise scatter grid (lower triangle + marginal histograms).

                Renders 2-D histograms for all pairs ``(i, j)`` with ``j <= i`` in
                the lower-left triangle, 1-D marginal histograms on the left column
                and bottom row, and reports pairwise Pearson correlations as subplot
                titles.

        Args:
            keys: List of parameter names to include.
            subsample: Number of draws per parameter.
            figsize: Figure size tuple.
            bins: Number of histogram bins.
            fontsize: Font size for labels and titles.
            seed: Random seed for sub-sampling.
            suptitle: Optional super-title.
            show: Whether to call ``plt.show()``.
            max_axis_label_len: Maximum characters to display for axis labels.
            chain: Optional chain index to restrict draws.
            labels: Optional display labels overriding ``keys``.

        Returns:
            Matplotlib Figure object.
        """
        for key in keys:
            if self.fix_params and key in self.fix_params:
                raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        if labels is None:
            key_strs = [str(k) for k in keys]
        else:
            key_strs = labels
        num_keys = len(keys)

        keys = [self.get_key(k) for k in keys]
        assert len(keys) >= 2

        fig, ax = plt.subplots(nrows=num_keys + 1, ncols=num_keys + 1, figsize=figsize,
                               gridspec_kw={'width_ratios': [.5] + [1] * num_keys,
                                            'height_ratios': [1] * num_keys + [.5]})

        for i in range(len(keys)):
            x_i = self.get_sample(keys[i], subsample=subsample, seed=seed, chain=chain)
            ax[len(keys) - 1 - i][0].hist(x_i, density=True, bins=bins, orientation='horizontal')
            ax[len(keys) - 1 - i][0].set_ylabel(key_strs[i][:max_axis_label_len], fontsize=fontsize)

            ax[len(keys)][i + 1].hist(x_i, density=True, bins=bins)
            ax[len(keys)][i + 1].set_xlabel(key_strs[i][:max_axis_label_len], fontsize=fontsize)

            for j in range(len(keys)):
                if j > i:
                    continue
                elif i == j:
                    x_j = x_i
                else:
                    x_j = self.get_sample(keys[j], subsample=subsample, seed=seed, chain=chain)

                ax[len(keys) - 1 - i][j + 1].hist2d(x_j, x_i, bins=bins, cmap='Blues')
                ax[len(keys) - 1 - i][j + 1].set_title(f'corr={np.round(np.corrcoef(x_i, x_j)[0, 1], 4)}',
                                                       fontsize=fontsize)
                # ax[len(keys) - 1 - i][j + 1].set_xlabel(key_strs[j], fontsize=fontsize)
                # ax[len(keys) - 1 - i][j + 1].set_ylabel(key_strs[i], fontsize=fontsize)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize)

        plt.tight_layout()

        ax[len(keys)][0].axis("off")
        for j in range(len(keys)):
            for i in range(j + 1, len(keys)):
                ax[i][len(keys) - j].axis("off")

        if show:
            plt.show()

        return fig

    def hist(self, key, subsample=10_000, figsize=(6, 2.7), bins=25, fontsize=14, seed=0, title=None,
             max_axis_label_len=24, show=False, dpi=130, include_chains=True, cumulative=False):

        """
        Plot a density histogram for a single parameter.

                Plots one combined histogram (black step line) plus per-chain
                histograms when ``include_chains=True``.  Supports cumulative mode.

        Args:
            key: Parameter name (or index/callable).
            subsample: Number of draws to sample.
            figsize: Figure size tuple.
            bins: Number of histogram bins.
            fontsize: Title font size.
            seed: Random seed.
            title: Optional plot title override.
            max_axis_label_len: Maximum characters to display in the auto-generated title.
            show: Whether to call ``plt.show()``.
            dpi: Figure resolution.
            include_chains: Whether to overlay per-chain histograms.
            cumulative: Whether to render a cumulative distribution.

        Returns:
            Matplotlib Figure object.
        """
        fig = plt.figure(dpi=dpi, figsize=figsize)
        y = self.get_sample(key, subsample=subsample, seed=seed)
        plt.hist(y, color='k', histtype='step', lw=2, bins=bins, density=True,
                 cumulative=cumulative)

        if include_chains:
            for r in range(self.n_chains):
                y = self.get_sample(key, subsample=subsample, seed=seed, chain=r)
                plt.hist(y, histtype='step', lw=1, bins=bins, label=f'chn {r}', density=True,
                         cumulative=cumulative)

        if title is None:
            title = key[:max_axis_label_len]
            if len(title) < len(key):
                title += '...'

        plt.title(title, fontsize=fontsize)
        plt.legend(loc='best')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def multi_hist(self, keys, subsample=10_000, figsize=None, bins=25, fontsize=14, seed=0, suptitle=None,
                   ncols=3, max_axis_label_len=24, show=False, dpi=130, cumulative=False, labels=None,
                   sharex=False, sharey=False):

        """
        Plot a grid of histograms for multiple parameters.

                Lays out histograms in a grid with ``ncols`` columns.  Raises if
                any requested parameter is fixed.

        Args:
            keys: List of parameter names.
            subsample: Number of draws per parameter.
            figsize: Figure size tuple.
            bins: Histogram bin count.
            fontsize: Axis label font size.
            seed: Random seed for sub-sampling.
            suptitle: Optional super-title.
            ncols: Number of subplot columns.
            max_axis_label_len: Maximum label characters.
            show: Whether to call ``plt.show()``.
            dpi: Figure resolution.
            cumulative: Whether to use cumulative histograms.
            labels: Optional display labels overriding ``keys``.
            sharex: Whether subplots share the x-axis.
            sharey: Whether subplots share the y-axis.

        Returns:
            Matplotlib Figure object.
        """
        for key in keys:
            if self.fix_params and key in self.fix_params:
                raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        if labels is None:
            key_strs = [str(k) for k in keys]
        else:
            key_strs = labels
        num_keys = len(keys)

        keys = [self.get_key(k) for k in keys]
        assert len(keys) >= 2

        ncols = min(ncols, num_keys)
        nrows = int(np.ceil(num_keys / ncols))

        if figsize is None:
            figsize = (ncols * 2, 1.5 + 1.5 * nrows)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, sharex=sharex, sharey=sharey)
        if nrows == 1:
            ax = [ax]

        for i in range(len(keys)):
            x_i = self.get_sample(keys[i], subsample=subsample, seed=seed)

            r_, c_ = i // ncols, i % ncols

            ax[r_][c_].hist(x_i, density=True, bins=bins, cumulative=cumulative)
            ax[r_][c_].set_xlabel(key_strs[i][:max_axis_label_len], fontsize=fontsize)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def rank_plot(self, key, show=False, bins=10, figsize=None, dpi=130, fontsize=14, suptitle=None):

        """
        Plot per-chain rank histograms for a single parameter.

                Ranks all post-burn-in draws across chains together, then histograms
                each chain's rank distribution.  Under convergence, all histograms
                should be approximately uniform.

        Args:
            key: Parameter name (or index/callable).
            show: Whether to call ``plt.show()``.
            bins: Number of rank histogram bins.
            figsize: Figure size tuple.
            dpi: Figure resolution.
            fontsize: Suptitle font size.
            suptitle: Optional suptitle override.

        Returns:
            Matplotlib Figure object.
        """
        if figsize is None:
            figsize = (7, self.n_chains * 1.5)

        fig, ax = plt.subplots(nrows=self.n_chains, dpi=dpi, figsize=figsize, sharex=True, sharey=True)
        sample = self.get_sample(key, include_burnin=False)

        chains = np.array(self.sample_info_df.chain[~self.sample_info_df.is_burnin])
        index = np.argsort(sample)

        for i, c in enumerate(range(self.n_chains)):
            temp = index[chains == i]
            ax[i].hist(temp, bins=bins, alpha=.5, density=False)
            ax[i].axhline(1 / bins * len(temp), ls='--', color='k')
            ax[i].set_ylabel(f'Chain {i}')


        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize)
        else:
            plt.suptitle(f"Rank Plot\n'{key}'", fontsize=fontsize)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def multi_rank_plot(self, keys, figsize=None, bins=10, fontsize=14, suptitle=None,
                        ncols=3, max_axis_label_len=24, show=False, dpi=130):

        """
        Plot a grid of rank histograms for multiple parameters.

                Raises if any requested parameter is fixed.  Each subplot shows
                per-chain rank histograms with a reference uniform-density line.

        Args:
            keys: List of parameter names.
            figsize: Figure size tuple.
            bins: Number of rank bins per histogram.
            fontsize: Font size for subplot titles.
            suptitle: Optional super-title.
            ncols: Number of subplot columns.
            max_axis_label_len: Maximum characters for subplot titles.
            show: Whether to call ``plt.show()``.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object.
        """
        for key in keys:
            if self.fix_params and key in self.fix_params:
                raise Exception(f"No diagnostic plot available for fixed parameter {key}!")

        key_strs = [str(k) for k in keys]
        num_keys = len(keys)

        keys = [self.get_key(k) for k in keys]
        assert len(keys) >= 2

        ncols = min(ncols, num_keys)
        nrows = int(np.ceil(num_keys / ncols))

        if figsize is None:
            figsize = (ncols * 2.2, 1.5 + 1.5 * nrows)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
        if nrows == 1:
            ax = [ax]

        chains = np.array(self.sample_info_df.chain[~self.sample_info_df.is_burnin])
        for i in range(len(keys)):
            r_, c_ = i // ncols, i % ncols
            x_i = self.get_sample(keys[i])
            index = np.argsort(x_i)

            for c in range(self.n_chains):
                ax[r_][c_].hist(index[chains == c], density=False, bins=bins, lw=2, histtype='step', label=f'{c}')
                ax[r_][c_].set_title(f'ranks "{key_strs[i][:max_axis_label_len]}"', fontsize=fontsize)

            ax[r_][c_].axhline(len(x_i) / (self.n_chains * bins), ls=':', color='k', lw=2)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_chain_full_sample(self, chain, include_burnin=False):
        """
        Return a DataFrame of all draws for a single chain, optionally including burn-in.

        Args:
            chain: Integer chain index (0-based).
            include_burnin: Whether to include burn-in draws.

        Returns:
            DataFrame slice of ``sample_df`` for the requested chain.
        """
        idx = self.sample_info_df.chain == chain
        if not include_burnin:
            idx &= ~self.sample_info_df.is_burnin
        return self.sample_df.loc[idx]

    def get_samples(self, keys, chain=None, subsample=None, seed=0, include_burnin=False, half=None):
        """
        Return a dict of post-burn-in draw arrays for multiple parameters.

        Args:
            keys: Iterable of parameter names (or callables/indices).
            chain: Optional chain index to restrict draws.
            subsample: Optional maximum number of draws to sample.
            seed: Random seed for sub-sampling.
            include_burnin: Whether to include burn-in draws.
            half: Optional ``1`` or ``2`` to restrict to the first or second half of draws.

        Returns:
            Dict mapping each key to its corresponding draw array or Series.
        """
        return {k: self.get_sample(k, chain, subsample, seed, include_burnin=include_burnin, half=half)
                for k in keys}

    def get_sample(self, key, chain=None, subsample=None, seed=0, include_burnin=False, half=None, return_array=False):

        """
        Return post-burn-in draws for a single parameter (or callable).

                Filters by chain and/or half if specified, applies random sub-sampling
                with ``seed`` if ``subsample < len(data)``, and evaluates callable
                keys row-wise against the draw DataFrame.

        Args:
            key: Parameter name string, integer index, the literal ``'log_posterior'``, or a callable.
            chain: Optional integer chain index (0-based).
            subsample: Optional maximum number of draws; ``None`` returns all.
            seed: Random seed for reproducible sub-sampling.
            include_burnin: Whether to include burn-in draws.
            half: ``1`` or ``2`` to return only the first or second half; ``None`` for all.
            return_array: Whether to return a NumPy array instead of a Series.

        Returns:
            Series or NumPy array of draw values for the requested parameter.
        """
        if include_burnin:
            idx = np.full(len(self.sample_df), True)
        else:
            idx = ~self.sample_info_df.is_burnin

        if chain is not None:
            assert isinstance(chain, int) and 0 <= chain < self.n_chains
            idx &= self.sample_info_df.chain == chain

        key = self.get_key(key)

        if isinstance(key, str):
            if key == 'log_posterior':
                data = self.sample_info_df.log_posterior[idx]
            else:
                data = self.sample_df[key][idx]
        else:
            data = self.sample_df.loc[idx]

        if half is not None:
            if half == 1:
                data = data[:len(data) // 2]
            elif half == 2:
                data = data[len(data) // 2:]
            else:
                raise Exception("`half` must be `None`, `1` or `2`!")

        if subsample is not None and subsample < len(data):
            rand = random.Random(seed)
            data = data.iloc[rand.sample(range(len(data)), k=subsample)]

        if isinstance(key, Callable):
            try:
                data = pd.Series(data.apply(key, axis=1))
            except:
                data = np.array([key(np.asarray(data.loc[x])) for x in data.index])

        if return_array and not isinstance(data, np.ndarray):
            data = np.array(data)
        else:
            data = data.copy()

        return data

    def __getitem__(self, key):
        """
        Return the posterior mean of parameter ``key`` (equivalent to ``mean_params[key]``).

        Args:
            key: Parameter name string or index.

        Returns:
            Scalar posterior mean value.
        """
        return self.mean_params[key]

    def __setitem__(self, key):
        """
        Not implemented; ``MCMCResults`` objects are read-only after construction.

        Args:
            key: Unused.
        """
        raise NotImplementedError

    def corr_params(self, keys=None):
        """
        Compute the posterior correlation matrix.

                Derived from ``cov_params`` by normalizing each row and column by the
                corresponding standard deviations.

        Args:
            keys: Optional subset of parameter names; ``None`` returns the full matrix.

        Returns:
            DataFrame correlation matrix indexed and columned by parameter names.
        """
        X = self.cov_params.copy(deep=True)
        std = np.diag(X) ** .5
        X /= std.reshape((-1, 1))
        X /= std.reshape((1, -1))
        if keys is None:
            return X
        else:
            return X.loc[keys, keys]

    def to_string(self, keys=None):
        """
        Return a pretty-printed string representation of selected object attributes.

        Args:
            keys: Optional list of attribute names to include; ``None`` serializes everything.

        Returns:
            String from ``pprint.pformat`` of the (possibly filtered) ``__dict__``.
        """
        self_dict = self.__dict__
        if keys is not None:
            self_dict = {k: self_dict.get(k, None) for k in keys}
        return pprint.pformat(self_dict)

    def _get_samples_split_by_chain(self, include_burnin=False, copy=False):
        """
        Return a list of per-chain sample arrays, optionally excluding burn-in.

                Each element is a NumPy array view (or copy) of shape
                ``(n_iterations [- n_burnin], num_params)`` for one chain.

        Args:
            include_burnin: Whether to include the leading burn-in rows.
            copy: Whether to return independent copies instead of views.

        Returns:
            List of ``n_chains`` arrays, each of shape ``(n_samples, num_params)``.
        """
        sample_refs = []
        for i in range(self.n_chains):
            li, hi = (
                (0 if include_burnin else (i + 1) * self.n_burnin),
                self.n_iterations * (i + 1)
            )
            sample_refs.append(self.sample_df.values[li:hi])
        if copy:
            sample_refs = [c.copy() for c in sample_refs]
        return sample_refs

    def geweke(self, frac1=.1, frac2=.5, bm_taus=None, debug=False):
        """
        Compute approximate Geweke z-scores for convergence assessment.

                Compares the mean of the first ``frac1`` fraction of post-burn-in
                draws to the mean of the last ``frac2`` fraction of draws using
                batched-means variance estimates.  Under convergence the z-scores
                should be within ±1.96 for ~95% of parameters.

        Args:
            frac1: Fraction of post-burn-in draws to use as the 'early' window.
            frac2: Fraction of post-burn-in draws to use as the 'late' window.
            bm_taus: Optional batch-mean tau values; auto-selected when ``None``.
            debug: Whether to print timing.

        Returns:
            DataFrame of shape ``(num_params, n_chains)`` containing per-chain Geweke z-scores.
        """
        t = time.time()
        if debug:
            print("Setting approximate Geweke z-scores...", end="")
        z_scores = geweke_approx(self._get_samples_split_by_chain(include_burnin=False),
                                 frac1=frac1, frac2=frac2, bm_taus=bm_taus)
        if debug:
            print(f"done! ({time.time() - t:0.2f}s)")
        return pd.DataFrame(z_scores, index=self.param_names, columns=range(self.n_chains))

    def get_ess_from_batched_means(self, taus=None, selection=DEFAULT_BATCHED_MEANS_SELECTION):
        """
        Compute ESS using the batched-means estimator for all parameters.

                Delegates to ``get_ess_batched_means_for_chains`` on the post-burn-in
                chain arrays.  Fixed parameters are zeroed out in the result.

        Args:
            taus: Optional batch-size candidates; auto-selected when ``None``.
            selection: Strategy for selecting the batch size (e.g. ``'min'`` or ``'max'``).

        Returns:
            DataFrame of shape ``(num_params, n_chains)`` with batched-means ESS.
        """
        ess = get_ess_batched_means_for_chains(self._get_samples_split_by_chain(include_burnin=False),
                                               taus=taus, selection=selection)
        result = pd.DataFrame(ess, index=self.param_names, columns=range(self.n_chains))
        if self.fix_params:
            result.loc[[p for p in self.param_names if p in self.fix_params]] = 0
        return result

    def set_multi_ess(self, debug=False):
        """
        Compute and store the multivariate ESS in ``self.m_ess``.

        Args:
            debug: Whether to print timing.
        """
        self.m_ess = self.get_multi_ess(debug=debug)

    def get_multi_ess(self, taus=None, debug=False):
        """
        Compute the multivariate ESS for each chain.

                Uses ``multi_ess`` with the pooled posterior covariance
                ``self.cov_params`` as the reference.

        Args:
            taus: Optional batch-size candidates for the batched-means estimator.
            debug: Whether to print timing.

        Returns:
            Array of shape ``(n_chains,)`` with the multivariate ESS for each chain.
        """
        _t = time.time()
        if debug:
            print("Computing multi ESS ...", end="")
        mess = np.array([
            multi_ess(C, taus, cov_indep=self.cov_params)
            for C in tqdm(self._get_samples_split_by_chain(), disable=not debug)
        ])
        if debug:
            print(f" done! ({time.time() - _t:.2f}s)")
        return mess

    def amha(self, n_samples=40_000, inplace=True, debug=False, max_subchain_draws_sample=None,
             user_prompt_for_more_iters=False,
             ):

        # TODO fix circular import!
        """
        Extend sampling by running additional AMH draws from the last chain states.

                Restarts the adaptive Metropolis sampler from the final draw of each
                chain with ``n_burnin=0`` (no additional burn-in), forwarding the
                current proposal covariance (``step_cov``) and scaler from
                ``other_info``.  When ``inplace=True`` merges the new draws into this
                object via ``_merge``; otherwise returns the new ``MCMCResults``.

        Args:
            n_samples: Number of additional post-burn-in draws to collect per chain.
            inplace: Whether to merge the new draws into ``self`` (True) or return a new object (False).
            debug: Whether to pass debug=True to the sampler.
            max_subchain_draws_sample: Override for the maximum subchain draw count; uses the stored option when ``None``.
            user_prompt_for_more_iters: Whether to prompt the user to continue sampling.

        Returns:
            ``None`` when ``inplace=True``; new ``MCMCResults`` object when ``inplace=False``.
        """
        from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha

        assert self.method == AMH_METHOD

        if max_subchain_draws_sample is None:
            max_subchain_draws_sample = self.options['max_subchain_draws_sample']

        # start chains where they left off
        x0 = []
        for c in range(self.n_chains):
            i_c = self.sample_info_df.index[self.sample_info_df.chain == c][-1]
            x0.append(self.sample_df.iloc[i_c].values.copy())

        # run new MCMC
        fit_new = amha(
            self.log_posterior, x0, start_params_is_original_scale=True,
            log_posterior_jacobian_adjustment=self.log_posterior_jacobian_adjustment,
            starting_iter=self.n_iterations,

            step_cov=self.other_info['step_cov'],
            step_cov_initial_samples=self.n_chains * self.n_samples,

            scaler0=self.other_info['scaler'],
            param_names=self.param_names, fix_params=self.fix_params,
            specification_name=self.specification_name, debug=debug,
            n_chains=self.n_chains, n_burnin=0, n_samples=n_samples,
            transformations=self.transformations,
            max_subchain_draws_sample=max_subchain_draws_sample,
            user_prompt_for_more_iters=user_prompt_for_more_iters,
            diff_evolution_past_samples=self.other_info['diff_evolution_past_samples'],
            **{k: self.options[k]
               for k in [
                   'max_processes', 'seed', 'target_acceptance_rate', 'draw_size',
                   'min_scaler', 'max_scaler', 'scaler_adjust_rate', 'scaler_adjust_denom_power',
                   'thinning', 'pbar_update_cadence', 'bounds', 'do_adaptive',
                   'max_subchain_draws_burnin', 'do_parallel',
                   # 'resample_k',
                   'proposal_df', 'normalize_step_cov',
                   # 'step_cov_adjust_rate',

                   'do_diff_evolution_mc', 'diff_evolution_frac_burnin', 'diff_evolution_max_draws',
                   'diff_evolution_weight', 'diff_evolution_jump_cadence',

                   'scalar_jitter_bounds',

                   'stop_adaptation_after_burnin', 'callback_function'
               ]}
        )

        if inplace:
            self._merge(fit_new, debug=debug)

        else:
            return fit_new

    def get_last_sample(self, n_chains=None):
        """
        Return the last draw of each chain as a list of parameter Series.

        Args:
            n_chains: Number of chains to return; defaults to ``self.n_chains``.

        Returns:
            List of ``n_chains`` Series, each containing the final draw of one chain.
        """
        if n_chains is None:
            n_chains = self.n_chains
        return [self.sample_df.loc[(i % (self.n_chains) + 1) * (self.n_iterations) - 1].copy()
                for i in range(n_chains)]

    def get_inv_transform_draws(self, size=None, seed=0, return_cov=True):
        """
        Draw from the posterior and optionally apply the inverse transformation.

                Collects ``size`` draws (or all draws when ``size`` is None) for each
                parameter and applies ``inv_transform`` to parameters that have a
                bounded-to-unbounded transformation, converting them back to the
                unbounded sampling space.

        Args:
            size: Number of draws to sample; ``None`` uses all post-burn-in draws.
            seed: Random seed for sub-sampling.
            return_cov: Whether to also return the covariance matrix of the transformed draws.

        Returns:
            Array of shape ``(num_draws, num_params)`` of transformed draws, or a 2-tuple ``(draws, draws_cov)`` when ``return_cov=True``.
        """
        num_draws = self.n_chains * self.n_samples
        if size is not None:
            if size >= self.n_chains * self.n_samples:
                size = None
            else:
                num_draws = size
        draws = np.zeros((num_draws, self.num_params))
        for i, p in tqdm(enumerate(self.param_names)):
            draws[:, i] = self.get_sample(p, subsample=size, seed=seed)
            if self.transformed_model and i in self.transformations:
                draws[:, i] = self.transformations[i].inv_transform(draws[:, i])

        if return_cov:
            draws_cov = np.cov(draws, rowvar=False)
            return draws, draws_cov
        else:
            return draws

    def _merge(self, fit_new, debug=False):
        """
        Merge a new ``MCMCResults`` object's draws into this one in-place.

                Interleaves the old and new chain draws into a single contiguous
                DataFrame (old draws first, new draws appended, per chain),
                aggregates the per-chain covariance estimates, and recomputes all
                summary statistics.  Modifies ``sample_df``, ``sample_info_df``,
                ``n_iterations``, ``n_samples``, ``chain_results``, and
                ``acceptance_rate`` in place.

        Args:
            fit_new: New ``MCMCResults`` object to merge; must have the same number of chains.
            debug: Whether to print timing.
        """

        _t = time.time()
        if debug:
            print("\nMerging old and new MCMC draws...\n")

        assert isinstance(fit_new, MCMCResults)
        assert self.n_chains == fit_new.n_chains

        sample_df_new = pd.DataFrame(
            index=range(self.n_chains * (self.n_iterations + fit_new.n_iterations)),
            columns=self.param_names,
            dtype=float
        )

        sample_info_df_new = pd.DataFrame(
            index=range(self.n_chains * (self.n_iterations + fit_new.n_iterations)),
            columns=self.sample_info_df.columns,
        )

        for c in range(self.n_chains):
            a1, a2 = (c * (self.n_iterations + fit_new.n_iterations),
                      c * fit_new.n_iterations + (c + 1) * self.n_iterations)
            a3 = (c + 1) * (self.n_iterations + fit_new.n_iterations)

            for _df1, _df2, _df3 in [(self.sample_df, fit_new.sample_df, sample_df_new),
                                     (self.sample_info_df, fit_new.sample_info_df, sample_info_df_new)]:
                _df3.values[a1:a2, :] = _df1[c * self.n_iterations:(c + 1) * self.n_iterations]
                _df3.values[a2:a3, :] = _df2[c * fit_new.n_iterations:(c + 1) * fit_new.n_iterations]

        self.sample_df = sample_df_new
        self.sample_info_df = sample_info_df_new
        self.pids = np.hstack([self.pids, fit_new.pids.copy()])

        for df in [self.sample_df, self.sample_info_df]:
            df.reset_index(drop=True, inplace=True)

        self.n_iterations = len(self.sample_df) // self.n_chains
        self.n_samples = self.n_samples + fit_new.n_samples
        self.set_n_burnin(self.n_burnin, debug=debug)

        self.map_idx, self.max_log_posterior, self.map_params = self.get_map()

        for C, C_new in zip(self.chain_results, fit_new.chain_results):
            C['cov_draws'], C['mean_draws'] = aggregate_covs(
                [temp['cov_draws'] for temp in (C, C_new)],
                [temp['mean_draws'] for temp in (C, C_new)],
                [temp['n_samples'] + temp['n_burnin'] for temp in (C, C_new)]
            )
            for k in ['n_burnin', 'n_samples', 'itr', 'lp_time', 'cov_time', 'draw_time', 'accept_decision_time',
                      'fit_elapsed', 'setup_time']:
                C[k] += C_new[k]
            for k in ['accepteds', 'scalers', 'acceptance_probs']:
                C[k] = np.hstack([C[k], C_new[k]])

        self.acceptance_rate = np.average([self.acceptance_rate, fit_new.acceptance_rate],
                                          weights=[self.n_iterations - fit_new.n_iterations, fit_new.n_iterations])

        self.mcmc_time += fit_new.mcmc_time
        self.other_info = fit_new.other_info.copy()

        gc.collect()

        if debug:
            print(f"\nDone merging MCMC draws! ({time.time() - _t:.2f}s)\n")

    def __call__(self, *args, **kwargs):
        """
        Convenience shorthand for ``get_sample(*args, **kwargs)``.

        Args:
            *args: Positional arguments forwarded to ``get_sample``.
            **kwargs: Keyword arguments forwarded to ``get_sample``.

        Returns:
            Whatever ``get_sample`` returns.
        """
        return self.get_sample(*args, **kwargs)

    def plot_credible_intervals(self, param_names, title=None, figsize=(10, 5),
                                dpi=130, show=False, level=.9, plot_horizontal_line=False, midpoint=None,
                                sample_size=10_000):
        """
        Plot symmetric equi-tailed credible intervals for a set of parameters.

                For each parameter draws a sample, computes the equi-tailed interval
                at ``level``, and delegates to ``plot_confidence_intervals`` for
                rendering a horizontal error-bar plot.

        Args:
            param_names: List of parameter names to plot.
            title: Optional plot title.
            figsize: Figure size tuple.
            dpi: Figure resolution.
            show: Whether to call ``plt.show()``.
            level: Coverage probability (e.g. 0.9 for 90% intervals).
            plot_horizontal_line: Whether to draw a reference horizontal line at zero.
            midpoint: Point estimate to mark: ``'mean'`` or ``'median'``.
            sample_size: Number of draws used for the quantile calculation.

        Returns:
            Matplotlib Figure object from ``plot_confidence_intervals``.
        """


        if midpoint is None:
            midpoint = 'mean'
        else:
            midpoint = midpoint.lower()
            assert midpoint in ('mean', 'median')

        point_estims = []
        conf_ints = []

        for p in param_names:
            x = self.get_sample(p, return_array=True, subsample=sample_size)
            point_estims.append(x.mean() if midpoint == 'mean' else np.median(x))
            conf_ints.append(np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2]))

        return plot_confidence_intervals(
            conf_ints, point_estims=point_estims, labels=param_names, title=title, figsize=figsize,
            dpi=dpi, show=show, level=level, plot_horizontal_line=plot_horizontal_line)
