"""
Mid-run MCMC convergence diagnostics and progress-reporting utilities.

Provides functions for generating human-readable diagnostic messages during
sampling, including split R-hat summaries, log-posterior trace ASCII plots,
optional Geweke z-score reports, and a callback integration point.  These
are called by ``amha`` and ``mala`` after each sub-chain to report progress.
"""
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from kanly.bayes.mcmc.diagnostics.gelman_rubin_split_rhat import get_rhat
from kanly.bayes.mcmc.diagnostics.geweke import geweke_approx
from kanly.plot.ascii_plotlib import plot
import time


def get_diagnostic_update_message(
        chain_results_master, param_names, n_chains, time_start, time_mcmc_draws, transformation_function,
        fix_param_idx, thinning, window_start=None, callback_function=None, title_string=None):

    """Build a full mid-run convergence diagnostic message.

    Generates a printable string containing:

    - Maximum log-posterior observed so far across all chains.
    - ASCII log-posterior trace for each chain (post ``window_start``).
    - Gelman-Rubin split R-hat table (via ``get_diagnostic_rhat_update_message``).
    - Optional callback function output.
    - Summary of total draws, time elapsed, and time per sample.

    Args:
        chain_results_master: List of per-chain result dicts, each containing
            ``'samples'`` (2-D draw array) and ``'log_posterior'``.
        param_names: List of parameter name strings.
        n_chains: Number of parallel chains.
        time_start: Wall-clock time (``time.time()``) when sampling began.
        time_mcmc_draws: Wall-clock time when the most recent sub-chain started
            (used for sub-chain timing).
        transformation_function: Callable mapping unbounded-space means to the
            original parameter space for display.
        fix_param_idx: Iterable of fixed-parameter indices (R-hat set to NaN
            for these).
        thinning: Thinning factor (reported in total-draws count).
        window_start: Index from which to start the trace plot and R-hat
            window.  Defaults to the last 75% of draws when ``None``.
        callback_function: Optional callable invoked with the current chain
            means for user-defined monitoring (e.g., a model prediction).
        title_string: Optional header printed above the message.

    Returns:
        Formatted multi-line string ready for printing.
    """
    if title_string is not None:
        print('\n' + title_string)

    message = "\nMaximum Log Posterior (so far..):\n"
    message += pd.DataFrame(
        {'max lp': [c['log_posterior'].max() for c in chain_results_master]}
    ).transpose().to_string()

    len_chn_result = len(chain_results_master[0]['samples'])
    if window_start is None:
        window_start = len_chn_result // 4

    for num, c in enumerate(chain_results_master):
        message += '\n\n' + plot(
            c['log_posterior'][window_start:],
            ylabel=f'log_posterior [chain {num}]', xlabel=f'sample {window_start} to {len_chn_result}',
            ncols=140, nrows=20, do_print=False, coverage=.9)

    r_hat_message, r_hat_df = get_diagnostic_rhat_update_message(
        [C['samples'] for C in chain_results_master],
        param_names, n_chains, time_start, transformation_function,
        fix_param_idx_subset=fix_param_idx, window_start=window_start
    )

    message += '\n' + r_hat_message

    if callback_function is not None:
        try:
            if '{orig scale}' in r_hat_df.columns:
                x = r_hat_df['orig_scale']
            else:
                x = r_hat_df['chain mean']
            s_callback = '\n\t' + '\n\t'.join(str(callback_function(transformation_function(x))).split('\n'))
            message += '\nCallback Function:\n' + s_callback + '\n'
        except:
            message += '\nCallback function call failed!'

    mcmc_time = time.time() - time_mcmc_draws
    total_time = time.time() - time_start
    message += (f'\n\nMCMC Drawing So Far... '
                f'\n\tTotal draws:       {thinning * chain_results_master[0]["samples"].shape[0] * n_chains}'
                f'\n\tTotal samples:     {chain_results_master[0]["samples"].shape[0] * n_chains}'
                f'\n\tMCMC draw time:    {"%.2fs" % mcmc_time}'
                f'\n\tTime per sample:   {1000 * mcmc_time / (chain_results_master[0]["samples"].shape[0] * n_chains):.3f}ms'
                f'\n\tTotal time:        {"%.2fs" % total_time}\n\n')

    return message


def get_diagnostic_rhat_update_message(chain_results_master, param_names, n_chains, _time_master,
                                       transformation_function,
                                       fix_param_idx_subset=None, do_geweke=False,
                                       geweke_args=None, window_start=None):

    """Build an R-hat (and optional Geweke) convergence summary string.

    Computes the split Gelman-Rubin R-hat via ``get_diagnostic_df``, formats
    it as a DataFrame string, and appends summary statistics (average, median,
    max R-hat, and counts exceeding 1.01/1.05/1.10 thresholds).  Optionally
    appends Geweke z-score statistics.

    Args:
        chain_results_master: List of per-chain draw arrays of shape
            ``(n_draws, num_params)``, or list of dicts whose first element
            is the draw array.
        param_names: List of parameter name strings.
        n_chains: Number of parallel chains (informational).
        _time_master: Wall-clock time when sampling began (for elapsed-time
            reporting).
        transformation_function: Callable mapping unbounded-space means to the
            original parameter space (shown as an extra column).
        fix_param_idx_subset: Iterable of fixed-parameter indices; R-hat and
            SD are set to NaN for these.  Defaults to empty list.
        do_geweke: Whether to compute and append Geweke z-scores.
        geweke_args: Keyword arguments forwarded to ``geweke_approx``.
        window_start: Index defining the start of the diagnostic window.
            Defaults to the last 75% of draws when ``None``.

    Returns:
        2-tuple ``(message_string, r_hat_df)`` where ``message_string`` is a
        formatted multi-line string and ``r_hat_df`` is the underlying
        DataFrame of R-hat statistics.
    """
    if geweke_args is None:
        geweke_args = dict()

    if fix_param_idx_subset is None:
        fix_param_idx_subset = []

    num_samps = len(chain_results_master[0])
    if window_start is None:
        window_start = num_samps // 4

    r_hat_df = get_diagnostic_df(chain_results_master, param_names, window_start,
                                 transformation_function)

    if do_geweke:
        geweke_arr = geweke_approx(chain_results_master, **geweke_args)
        r_hat_df['max|z_Geweke|'] = np.abs(geweke_arr).max(axis=1).round(2)
        r_hat_df['|z_G| > 1.96'] = (np.abs(geweke_arr) > 1.96).sum(axis=1).round(0).astype(str)
    num_params = len(r_hat_df)

    for i in fix_param_idx_subset:
        r_hat_df.loc[param_names[i],
        ['chain sd', 'R_hat'] + (['max|z_Geweke|', '|z_G| > 1.96'] if do_geweke else [])] = np.nan

    r_hat_df_str = '\n\t'.join(r_hat_df.to_string().split('\n'))

    prompt_message = f'\n\nMCMC ran for {num_samps} draws per chain on {n_chains} chains.'
    prompt_message += f'\n(Current run time is {"%.2fs" % (time.time() - _time_master)})'
    prompt_message += (f'\n\nR_hat (on unconstrained parameter space) are shown belows for draws {window_start} to {num_samps}:'
                       f'\n\n\t{r_hat_df_str}')
    prompt_message += (f'\n\nThe average R-hat is {"%.4f" % r_hat_df.R_hat.mean()},'
                       f' with a median R-hat of {"%.4f" % np.median([r_hat_df.R_hat.iloc[i] for i in range(num_params) if i not in fix_param_idx_subset])}'
                       f' and a maximum R-hat of {"%.4f" % r_hat_df.R_hat.max()}.')
    prompt_message += (f"\n\t{np.sum(r_hat_df.R_hat > 1.1)}/{len(r_hat_df)} R-hat's greater than 1.1, "
                       f"{np.sum(r_hat_df.R_hat > 1.05)}/{len(r_hat_df)} R-hat's greater than 1.05, "
                       f"and {np.sum(r_hat_df.R_hat > 1.01)}/{len(r_hat_df)} R-hat's greater than 1.01.\n")

    if do_geweke:
        geweke_arr = geweke_arr[[i for i, p in enumerate(param_names) if i not in fix_param_idx_subset]]
        prompt_message += (
            f'\n{(100 * np.sum(np.abs(geweke_arr) > 1.96) / np.prod(geweke_arr.shape)):.2f}%'
            f' of Geweke z-stats above 5% critical value.  [Mean={geweke_arr.mean():.2f}, StdDev={geweke_arr.std():.2f}]')
        for i in range(len(chain_results_master)):
            prompt_message += f'\n\tChain {i:2d}:   {np.sum(np.abs(geweke_arr[:, i]) > 1.96)}/{geweke_arr.shape[0]}'

    prompt_message += '\n'

    return prompt_message, r_hat_df


def get_diagnostic_df(chain_samples, param_names, start_index, transformation_function=None):
    """Build a per-parameter diagnostic DataFrame with R-hat and chain statistics.

    Computes the split Gelman-Rubin R-hat, per-parameter chain mean, and
    between-chain standard deviation.  Optionally adds a column showing
    parameter means mapped back to the original (bounded) scale.

    Args:
        chain_samples: List of 2-D arrays, one per chain, each of shape
            ``(n_draws, num_params)``.
        param_names: List of parameter name strings (DataFrame index).
        start_index: Integer row index; only rows ``[start_index:]`` are used
            for the R-hat calculation.
        transformation_function: Optional callable that maps a Series of
            unbounded-space means to the original parameter space; adds an
            ``'{orig space}'`` column when provided.

    Returns:
        DataFrame indexed by ``param_names`` with columns:
        ``'chain mean'``, optionally ``'{orig space}'``, ``'chain sd'``,
        and ``'R_hat'`` (rounded to 3 decimal places).
    """
    samps = [c[start_index:] for c in chain_samples]
    R_hat, n_eff, _, chn_means, chn_sd = get_rhat(samps)

    r_hat_df = pd.DataFrame({
        'chain mean': chn_means,
        'chain sd': chn_sd,
        'R_hat': R_hat,
    }, index=param_names)
    r_hat_df['R_hat'] = r_hat_df['R_hat'].apply(lambda z: np.round(z, 3))

    if transformation_function is not None:
        r_hat_df.insert(loc=1, column='{orig space}', value=transformation_function(r_hat_df['chain mean']))

    return r_hat_df


if __name__ == '__main__':

    from kanly.api import bayes_nlls_model
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(0)
    n = 100
    x = 1.56 * np.random.randn(n)
    z = np.random.rand(n)
    y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
    wts = .01 + np.random.rand(n)
    data = {'x': x, 'y': y, 'z': z, 'wts': wts}

    fit1 = bayes_nlls_model(
        '[y] ~ {a}+{b}*[x] + {c}*[z]', data, bounds={'b': [5, 8], 'a': [1, 6]}
    ).sample(
        [2, 7, 1., 1.], user_prompt_for_more_iters=False, debug=False, n_samples=50_000,
        #max_subchain_draws=30_000,
        #show_r_hat_ever_subchain=True,
        fix_params={'a': 2.6}
    )

    print(fit1)
    print(fit1.get_ess_from_batched_means())
    print(fit1.get_ess_from_batched_means(selection='coordinate'))
    print(fit1.ess)
