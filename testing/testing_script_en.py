import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from kanly.api import elastic_net, ELASTIC_NET, bfgs_pqn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import pprint

n = 10_000
k = 50
np.random.seed(0)

df = pd.DataFrame(index=range(n))
X = np.hstack([np.random.randn(n, 1) for i in range(k)])
X = np.dot(X, np.random.randint(0, 5, (k, k)))
X[:, :10] *= 3
X[:, :10] += 50

weights = np.exp(np.random.randn(n))
weights_normalized = weights * n / sum(weights)
weights_none = np.ones(n)

coefs = np.zeros(k)
coefs[[2, 3, 10]] = 1.5
coefs[[9, 22]] = -1.5

y = -2.6 + X.dot(coefs) + .3 * np.random.randn(n)

df.loc[:, ['x%d' % i for i in range(k)]] = X
df['y'] = y
df['wts'] = weights

passed_coef = 0
passed_intercept = 0
passed_obj_func = 0
failed = 0
failed_tests = dict()

for normalize in [False, True]:
    for fit_intercept in [True, False]:
        for is_weighted in [False, True]:
            for positive in [True, False]:
                for alpha in [0.1, .01]:
                    for l1_ratio in [0.05, 0.5, 0.95]:

                            key = f'{normalize=}, {fit_intercept=}, {is_weighted=}, {positive=}, {alpha=}, {l1_ratio=}'
                            print(f'\n{key=}', end='...')

                            if normalize and not fit_intercept:
                                print('invalid setup, skipped!')
                                continue

                            formula = 'y ~ ' + ' + '.join(['x%d' % i for i in range(k)]) + (" $ wts" if is_weighted else "")

                            fit = elastic_net(formula, df, alpha=alpha, l1_ratio=l1_ratio, debug=False,
                                              specification_name='example elastic net',
                                              selection='random',
                                              seed=0,
                                              normalize=normalize,
                                              fit_intercept=fit_intercept,
                                              positive=positive,
                                              ftol=1e-16,
                                              xtol=1e-10, active_set=True, max_iter=30_000,
                                              )

                            fit_array = ELASTIC_NET(y, X, weights=weights if is_weighted else None,
                                                    alpha=alpha, l1_ratio=l1_ratio,
                                                    debug=False, specification_name='example elastic net',
                                                    selection='random',
                                                    seed=0,
                                                    normalize=normalize,
                                                    fit_intercept=fit_intercept,
                                                    positive=positive,
                                                    ftol=1e-16,
                                                    xtol=1e-10, active_set=True, max_iter=30_000,
                                                    )

                            if normalize:
                                ss = StandardScaler()
                                X_scaled = ss.fit_transform(X.copy(), sample_weight=weights if is_weighted else None)
                                x_scales = ss.scale_
                                x_means = ss.mean_

                            else:
                                X_scaled = X
                                x_scales = np.ones(k)
                                x_means = np.zeros(k)

                            EN_sk = ElasticNet(
                                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, tol=1e-10, max_iter=30_000,
                                selection='cyclic',
                                positive=positive,
                            )

                            fit_sk = EN_sk.fit(
                               X_scaled, y, sample_weight=(weights if is_weighted else None)
                            )

                            def objective_function(x):
                                if fit_intercept:
                                    b0 = x[0]
                                else:
                                    b0 = 0
                                coef = x[1:]
                                wts = weights_normalized if is_weighted else weights_none
                                resid = y - b0 - X_scaled.dot(coef)
                                ssr = (wts * resid ** 2).sum() / (2 * n)
                                penalty = alpha * (
                                        l1_ratio * np.abs(coef).sum()
                                        + (1 - l1_ratio) / 2 * (coef ** 2).sum()
                                )

                                return ssr + penalty

                            # optres = bfgs_pqn(objective_function, fit.params)

                            coef_tbl = (pd.DataFrame(
                                {'kanly': fit.coef_,
                                 'kanly_array': fit_array.coef_,
                                 'sklearn': fit_sk.coef_.flatten() / x_scales,
                                 #'bfgs': optres.x[fit_intercept:] / x_scales,
                                 }
                            ))

                            intercepts = pd.DataFrame({
                                'kanly': [fit.intercept_, ],
                                'kanly_array': [fit_array.intercept_, ],
                                'sklearn': [(
                                                    fit_sk.intercept_ - (
                                                (fit_sk.coef_.flatten() / x_scales * x_means
                                                ).sum())).item()
                                            if fit_intercept else
                                            0.0
                                            ]
                            })

                            obj_func_values = {
                                k: objective_function(x)
                                for k, x in zip(
                                    ['kanly', 'kanly_array', 'sklearn'],
                                    [np.hstack([intercepts[k], coef_tbl[k]]) for k in ['kanly', 'kanly_array', 'sklearn']]
                                )
                            }

                            try:
                                assert_allclose(coef_tbl.kanly, coef_tbl.sklearn, atol=1e-4, rtol=1e-2)
                                assert_allclose(coef_tbl.kanly, coef_tbl.kanly_array, atol=1e-4, rtol=1e-2)
                                passed_coef += 1

                                assert_allclose(intercepts.kanly, intercepts.sklearn, atol=1e-4, rtol=1e-2)
                                assert_allclose(intercepts.kanly, intercepts.kanly_array, atol=1e-4, rtol=1e-2)
                                passed_intercept += 1

                                print('passed!')
                            except Exception as e:
                                print('failed', str(e))
                                failed_tests[key] = {'coef': coef_tbl, 'intercept': intercepts, 'obj_func_values': obj_func_values}
                                failed += 1

                            try:
                                assert obj_func_values['kanly'] <= obj_func_values['sklearn'] + 1e-4
                                passed_obj_func += 1
                            except:
                                failed_tests[key] = {'coef': coef_tbl, 'intercept': intercepts, 'obj_func_values': obj_func_values}



print('\n\n', '='*10, '\n\n')

pprint.pprint(failed_tests)

print('\n\n', '='*10, '\n\n')

print(f'{passed_coef=}')
print(f'{passed_intercept=}')
print(f'{passed_obj_func=}')
print(f'{failed=}')


for k, v in failed_tests.items():
    ofv = v['obj_func_values']
    if ofv['kanly'] > ofv['sklearn'] + 1e-4:
        print()
        print(k)
        print(ofv)