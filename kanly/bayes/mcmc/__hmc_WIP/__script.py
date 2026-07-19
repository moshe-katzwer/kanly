cov = fit.other_info['cov_params_unbounded_space'].copy()
x0_start = model.inv_transform(
    fit.mean_params.copy())
# fit.mean_params.copy())

lp_func = model.log_posterior_transformed
lp_jac_adj = model.log_posterior_jacobian_adjustment

FIX_PARAMS = fit.fix_params.copy()

fix_params_ind = np.array([model.param_2_idx[k] for k in FIX_PARAMS])
fix_params_val = np.array([model.transformations[k].inv_transform(v)
                           if k in model.transformations
                           else v
                           for k, v in FIX_PARAMS.items()])


def potential_energy(x, return_lp=True):
    """potential_energy function.

    Args:
        x: TODO.
        return_lp: TODO.
    """
    lp_val = lp_func(x)
    if return_lp:
        return -(lp_val + lp_jac_adj(x)), lp_val
    else:
        return -(lp_val + lp_jac_adj(x))


def grad_pe(x):
    """grad_pe function.

    Args:
        x: TODO.
    """
    xcopy1 = x.copy()
    xcopy2 = x.copy()
    g = np.zeros(num_params)
    # f0 = potential_energy(x)
    for i in range(num_params):
        dx = min(100, max(1, abs(x[i]))) * 1e-8
        xcopy1[i] += dx
        xcopy2[i] -= dx
        g[i] = (
                       potential_energy(xcopy1, return_lp=False)
                       - potential_energy(xcopy2, return_lp=False)
               ) / (2 * dx)
        xcopy1[i] -= dx
        xcopy2[i] += dx
    g[fix_params_ind] = 0.0

    return g


#     xcopy1 = x.copy()
#     f0 = potential_energy(x, return_lp=False)
#     g = np.zeros(num_params)
#     for i in range(num_params):
#         dx = min(100, max(1, abs(x[i]))) * 1e-8
#         xcopy1[i] += dx
#         g[i] = (
#             potential_energy(xcopy1, return_lp=False)
#             - f0
#         ) / dx
#         xcopy1[i] -= dx
#     g[fix_params_ind] = 0.0

#     return g

num_params = model.num_params

from scipy.stats import multivariate_normal

# M = np.linalg.pinv(cov.copy())
# M += np.diag(np.diag(M)*1.1)
M = np.diag(1.0 / np.diag(cov.copy()))

for k in FIX_PARAMS:
    M[model.param_2_idx[k], :] = 0
    M[:, model.param_2_idx[k]] = 0
    M[model.param_2_idx[k], model.param_2_idx[k]] = 1.0

M_inv = np.linalg.pinv(M)

for k in FIX_PARAMS:
    M_inv[model.param_2_idx[k], :] = 0
    M_inv[:, model.param_2_idx[k]] = 0
    M_inv[model.param_2_idx[k], model.param_2_idx[k]] = 1.0

ke_dist = multivariate_normal(mean=[0] * num_params, cov=M, allow_singular=True)


def kinetic_energy(p):
    """kinetic_energy function.

    Args:
        p: TODO.
    """
    p2 = p.copy()
    p2[fix_params_ind] = 0
    return 0.5 * np.dot(p2, M_inv).dot(p2)


def grad_ke(p):
    """grad_ke function.

    Args:
        p: TODO.
    """
    p2 = p.copy()
    p2[fix_params_ind] = 0
    return np.dot(M_inv, p2)


import time
import ray
from ray.experimental.tqdm_ray import tqdm


@ray.remote
def hmc_chain(x0, kinetic_energy, grad_ke, potential_energy, grad_pe,
              seed, eps, L, target_accept, n_samples, bloc):
    """hmc_chain function.

    Args:
        x0: TODO.
        kinetic_energy: TODO.
        grad_ke: TODO.
        potential_energy: TODO.
        grad_pe: TODO.
        seed: TODO.
        eps: TODO.
        L: TODO.
        target_accept: TODO.
        n_samples: TODO.
        bloc: TODO.
    """
    pbar = tqdm(total=n_samples, position=seed)

    x0 = x0.copy()
    x0[fix_params_ind] = fix_params_val
    current_U, cur_lp = potential_energy(x0)

    cur_grad = grad_pe(x0)

    rand = np.random.RandomState(seed=bloc * 100 + seed)
    momentums = ke_dist.rvs(size=n_samples, random_state=rand)

    samples = np.zeros((n_samples, num_params))
    log_posteriors = np.zeros(n_samples)
    n_accepted = 0

    time0 = time.time()

    for i in range(n_samples):

        p = momentums[i].copy()
        p[fix_params_ind] = 0
        p0 = p
        x1 = x0
        current_K = kinetic_energy(p)

        # do leapfrog integration
        try:

            proposed_grad = cur_grad
            x1 = x0

            for l in range(L):
                p = p - eps * proposed_grad / 2
                p[fix_params_ind] = 0

                x1 = x1 + eps * grad_ke(p)
                x1[fix_params_ind] = fix_params_val
                proposed_grad = grad_pe(x1)

                p = p - eps * proposed_grad / 2
                p[fix_params_ind] = 0

            proposed_U, proposed_lp = potential_energy(x1)
            proposed_K = kinetic_energy(p)

        except KeyboardInterrupt as e:
            if isinstance(e, KeyboardInterrupt):
                print('break')
            else:
                print('failed')
                proposed_K = np.inf

        accept = np.log(rand.rand()) < current_U - proposed_U + current_K - proposed_K
        n_accepted += accept
        if accept:
            x0, current_U, cur_lp, cur_grad = x1, proposed_U, proposed_lp, proposed_grad

        eps *= (1 + .25 * (accept - target_accept))
        samples[i] = x0
        log_posteriors[i] = cur_lp

        desc = ''.join(("%5d" % i, "%10.2e" % eps, "%10.2e" % (L * eps), "%6s" % accept,
                        "%5.2f" % (n_accepted / (i + 1)), "%13.4e" % cur_lp,
                        "%8.2fs" % (time.time() - time0)))

        pbar.update(i - pbar._x)
        pbar.set_description(
            desc)

    return {'samples': samples, 'log_posterior': log_posteriors, 'time': time.time() - time0,
            'options': {'eps': eps, 'L': L}}


try:
    ray.shutdown()
except:
    pass

from kanly.bayes.mcmc.diagnostics.diagnostics import get_diagnostic_update_message
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method


eps = .12
L = 8
target_accept = .8
n_chains = 8
n_samples = 150
bloc = 0

try:
    ray.shutdown()
except:
    pass

time0 = time.time()


result_master = None

n_samples_total = 0

x0s = [x0_start.copy()] * n_chains

ray.init(num_cpus=n_chains, ignore_reinit_error=True)

n_samples = 50

while True:

    result = ray.get([
        hmc_chain.remote(x0, kinetic_energy, grad_ke, potential_energy, grad_pe,
                         seed, eps, L, target_accept, n_samples, bloc)
        for x0, seed in zip(x0s, range(n_chains))
    ])
    n_samples_total += n_samples
    bloc += 1

    if result_master is None:
        result_master = result
    else:
        result_master = [
            {
                'samples': np.vstack([r_old['samples'], r_new['samples']]),
                'log_posterior': np.hstack([r_old['log_posterior'], r_new['log_posterior']]),
                'time': r_old['time'] + r_new['time'],
            } for r_old, r_new in zip(result_master, result)
        ]

    message = get_diagnostic_update_message(
        result_master,
        model.param_names, n_chains,
        time0, time.time() - time0,
        window_start=n_samples_total // 6,
        transformation_function=model.transform,
        fix_param_idx=fix_params_ind, thinning=1, callback_function=None)

    n_samples_new = user_prompt_for_more_iters_method(message, do_prompt=True, assert_even=True)

    if n_samples_new <= 0 or n_samples_new is None or n_samples_new == '':
        break

    n_samples = n_samples_new

    eps = np.mean([r['options']['eps'] for r in result])
    x0s = [r['samples'][-1] for r in result_master]

print(time.time() - time0)

ray.shutdown()
