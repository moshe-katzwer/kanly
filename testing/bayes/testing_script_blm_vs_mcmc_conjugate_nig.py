import pandas as pd
import numpy as np
from kanly.api import lm, blm, build_data_model
from numpy.linalg import pinv
from numpy.testing import assert_allclose

n = 100
k = 3
np.random.seed(0)

X = np.random.randn(n, k)
beta = np.random.rand(k)
y = X.dot(beta) + .15 * np.random.randn(n)

df = pd.DataFrame(X, columns=[f'x{j}' for j in range(k)])
df['y'] = y
df['wts'] = np.random.rand(n) * (2 + np.abs(X[:, 0]))

for jeffreys_prior in [True, False]:

    # PRIOR
    a0 = 0 if jeffreys_prior else 5
    b0 = 0 if jeffreys_prior else 10

    mu0 = -np.ones(k)
    Lambda0 = np.eye(k) * 50
    Lambda0[0, 0] = .000_0000_1
    Lambda0[1, 2] = Lambda0[2, 1] = -10

    # CHECK LM
    f1 = lm('y ~ x0 + x1 + x2 - 1 $ wts', df)
    f2 = blm('y ~ x0 + x1 + x2 - 1 $ wts', df)
    assert_allclose(f1.params, f2.coef)

    # SOLVE
    fblm = fblm = blm('y ~ x0 + x1 + x2 - 1 $ wts', df,
                      a0=a0, b0=b0,
                      mu0=mu0, Lambda0=Lambda0)

    # CHECK ANALYTICAL
    w = df.wts

    XtX = X.T @ np.diag(w) @ X

    beta_hat = pinv(XtX).dot(X.T @ np.diag(w) @ y)
    Lambda_n = XtX + Lambda0
    mu_n = pinv(Lambda_n) @ (XtX @ beta_hat + Lambda0 @ mu0)
    mu_n, Lambda_n

    assert_allclose(mu_n, fblm.posterior.mu)
    assert_allclose(Lambda_n, fblm.posterior.Lambda)

    # if not jeffreys_prior:
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (np.sum(y * y * w) + mu0 @ Lambda0 @ mu0 - mu_n @ Lambda_n @ mu_n)

    assert_allclose(a_n, fblm.posterior.a)
    assert_allclose(b_n, fblm.posterior.b)

    # CHECK ANALYTICAL AGAINST MCMC QUANTILES

    data_code = '''
    self.y = `y`
    self.x0 = `x0`
    self.x1 = `x1`
    self.x2 = `x2`
    self.wts = `wts`
    self.rt_wts = np.sqrt(self.wts)
    '''

    model_code = f'''
    $x[3]$
    
    $__sigma2<0,np.inf>$
    sigma = $__sigma2$ ** .5
        
    pred = x0 * $x$[0] + x1 * $x$[1] + x2 * $x$[2]
    llf = logpdf_norm(y, pred, sigma / rt_wts).sum()
    prior = (
        logpdf_multivariate_normal($x$, mu0, tau=Lambda0 / $__sigma2$)
        {"-np.log($__sigma2$)" if jeffreys_prior else "+ logpdf_invgamma($__sigma2$, a=a0, scale=b0)"}
    )
    
    return llf + prior
    '''

    model = build_data_model(data_code, model_code, df,
                             other_variables={'mu0': mu0, 'Lambda0': Lambda0,
                                              'a0': a0, 'b0': b0})

    fitmcmc = model.to_bayesian_model().sample(
        {'__sigma2': 1.},
        n_burnin=5_000,
        max_subchain_draws_burnin=1_000,
        n_samples=100_000, n_chains=10, debug=False,
        thinning=3,
    )

    for j in range(k):
        qblm = fblm.posterior_marginal_rv[f'x{j}'].ppf([.01, .1, .25, .5, .75, .9, .99])
        qmcmc = np.quantile(fitmcmc(f'x[{j}]'), [.01, .1, .25, .5, .75, .9, .99])

        print(f'x[{j}]', np.round((qblm - qmcmc) / (np.clip(np.abs(qblm), a_min=1, a_max=np.inf)), 3))
        assert np.all(np.abs(np.round((qblm - qmcmc) / (np.clip(np.abs(qblm), a_min=1, a_max=np.inf))) < .002))

    qblm = fblm.posterior_marginal_rv['__sigma2'].ppf([.01, .1, .25, .5, .75, .9, .99])
    qmcmc = np.quantile(fitmcmc('__sigma2'), [.01, .1, .25, .5, .75, .9, .99])

    print(f'__sigma2', np.round((qblm - qmcmc) / (np.clip(np.abs(qblm), a_min=1, a_max=np.inf)), 3))
    assert np.all(np.abs(np.round((qblm - qmcmc) / (np.clip(np.abs(qblm), a_min=1, a_max=np.inf))) < .002))
