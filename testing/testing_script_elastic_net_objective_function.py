import numpy as np
from numpy.testing import assert_allclose

from kanly.regression.linear_models.penalized.elastic_net_objective_function import ElasticNetObjectiveFunction

n = 500
p = 20
s = .01
np.random.seed(0)
X = np.random.randn(n, p).dot((s * np.eye(p) + (1 - s) * np.ones((p, p))))
z = np.ones(4)
beta = np.hstack([z, np.zeros(p - len(z))])
y = 3 + X.dot(beta) + np.random.randn(n)

alpha, l1_ratio = .5, .5

wts = np.random.rand(n) / 3
wts_norm = wts / sum(wts) * n

for wts in [
    np.ones(n),
    wts_norm,

    np.ones(n) / 3,
    wts,
]:
    print("\n" * 3, wts[:5], '\n')


    def ssr_by_hand(intercept_, coef_):
        return (
            np.sum(wts * (y - intercept_ - X.dot(coef_)) ** 2)
        )


    def obj_func_by_hand(intercept_, coef_):
        return (
                ssr_by_hand(intercept_, coef_) / (2 * n)
                + alpha * (
                        l1_ratio * np.abs(coef_).sum()
                        + (1 - l1_ratio) / 2 * np.sum(coef_ ** 2)
                )
        )


    enf, ssr_quad_form = ElasticNetObjectiveFunction.build_elastic_net_objective_function(
        X, y, weights=wts, l1_penalties=alpha * l1_ratio, l2_penalties=alpha * (1 - l1_ratio) / 2
    )
    enf2 = enf.add_intercept()

    """
    f0: y'Wy
    df_db: -2 * X' W y
    d2f_db2: X' W X
    """
    assert_allclose(ssr_quad_form.f0, np.dot(y * wts, y))
    assert_allclose(ssr_quad_form.df_db, -2 * X.T.dot(y * wts))
    assert_allclose(ssr_quad_form.d2f_db2, X.T.dot(np.diag(wts)).dot(X))

    beta0 = np.random.randn(p)
    intercept0 = 1.4

    print("> ", sum(wts))
    print(ssr_by_hand(intercept0, beta0))
    print(enf.ssr_func(intercept0, beta0))
    print(enf2.ssr_func(0, np.hstack([[intercept0], np.hstack(beta0)])))
    print('diff = ', ssr_by_hand(intercept0, beta0) - enf.ssr_func(intercept0, beta0))
    print('diff_rel = ', ssr_by_hand(intercept0, beta0) / enf.ssr_func(intercept0, beta0) - 1)
    assert_allclose(ssr_by_hand(intercept0, beta0), enf.ssr_func(intercept0, beta0))
    assert_allclose(ssr_by_hand(intercept0, beta0), enf2.ssr_func(0, np.hstack([[intercept0], np.hstack(beta0)])))

    print(obj_func_by_hand(intercept0, beta0))
    print(enf(intercept0, beta0))
    print(enf2(0, np.hstack([[intercept0], np.hstack(beta0)])))
    print('diff = ', obj_func_by_hand(intercept0, beta0) - enf(intercept0, beta0))
    assert_allclose(obj_func_by_hand(intercept0, beta0), enf(intercept0, beta0))
    assert_allclose(obj_func_by_hand(intercept0, beta0), enf2(0, np.hstack([[intercept0], np.hstack(beta0)])))