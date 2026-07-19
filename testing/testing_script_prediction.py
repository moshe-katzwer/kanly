import pandas as pd
pd.set_option('display.max_columns', 20)
import numpy as np
from numpy.testing import assert_almost_equal

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = np.exp(1.2 - 0.5 * df['x'] + .2 * np.random.randn(n))

formula = 'y ~ x*w + C(grp)'

################
# LINEAR MODEL #
# QUANTILE REG #
# ROBUST REG   #
################

from kanly.api import lm, qr, rlm

for fitting_method, kwargs in zip(
        [lm, rlm, qr],
        [dict(), dict(), {'tau': .9}]
):

    # if fitting_method == qr:
    #     continue

    fit = fitting_method(formula, df.iloc[:100], debug=False, **kwargs)
    print(fit)

    for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:100]]:
        print(fit.__class__, X.__class__ if X is not None else None)
        try:
            assert_almost_equal(fit.predict(data=X), fit.fittedvalues, decimal=6)
        except Exception as e:
            import matplotlib.pyplot as plt
            plt.scatter(df.iloc[:100].y, fit.fittedvalues)
            plt.scatter(df.iloc[:100].y, fit.predict(data=X))
            plt.show()

            print("\n>>>> ", fitting_method, type(X), X.shape)
            raise e

    expected = fit.model.exog.toarray().dot([1.5, 1., -.3] + [.15] + [0] * 11)
    X = fit.model.exog
    for params in [
        [1.5, 1., -.3] + [.15] + [0] * 11,
        {'Intercept': 1.5, 'x': 1., 'w': -.3, 'C(grp)[1]': .15},
        pd.Series(index=['Intercept', 'x', 'w', 'C(grp)[1]'], data=[1.5, 1., -.3, .15])
    ]:
        assert_almost_equal(fit.predict(data=X, params=params), expected)

    params = [1.5, 1., -.3] + [.15] + [0] * 11
    for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:100]]:
        assert_almost_equal(fit.predict(data=X, params=params), expected)

    model_full = fitting_method(formula, df, **kwargs).model
    expected = model_full.exog.toarray().dot(fit.params)

    for X in [model_full.exog, model_full.exog.toarray(), df]:
        assert_almost_equal(fit.predict(X), expected)

    df_null = df.copy()
    scale = 0.0
    df_null['x'] = df['x'] * scale
    predicted = fit.predict(df_null)
    expected = fit.predict(df) - (1 - scale) * fit['x'] * df.x.values - (1 - scale) * fit['w:x'] * (df.x * df.w)
    assert_almost_equal(predicted, expected)


###############
# ELASTIC NET #
###############

from kanly.api import elastic_net
fit = elastic_net(formula, df.iloc[:50], alpha=.01)
print(fit)

for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:50]]:
    assert_almost_equal(fit.predict(data=X), fit.fittedvalues, decimal=3)

expected = 1.5 + fit.model.exog.toarray().dot([1., -.3] + [0, .15] + [0.] * 11)
for params in [
    [1.5, 1., -.3] + [0, .15] + [0] * 11,
    {'Intercept': 1.5, 'x': 1., 'w': -.3, 'C(grp)[1]': .15},
    pd.Series(index=['Intercept', 'x', 'w', 'C(grp)[1]'], data=[1.5, 1., -.3, .15])
]:
    assert_almost_equal(fit.predict(data=X, params=params), expected)

params = [1.5, 1., -.3] + [0, .15] + [0] * 11
for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:50]]:
    assert_almost_equal(fit.predict(data=X, params=params), expected)


model_full = elastic_net(formula, df, alpha=.01).model
expected = fit.params.iloc[0] + model_full.exog.toarray().dot(fit.params[1:])
for X in [model_full.exog, model_full.exog.toarray(), df]:
    assert_almost_equal(fit.predict(X), expected)

df_null = df.copy()
scale = 0.0
df_null['x'] = df['x'] * scale
predicted = fit.predict(df_null)
expected = fit.predict(df) - (1 - scale) * fit['x'] * df.x.values - (1 - scale) * fit['w:x'] * (df.x * df.w)
assert_almost_equal(predicted, expected)


############################
# GENERALIZED LINEAR MODEL #
############################

from kanly.api import glm
from kanly.regression.generalized_linear_models.links import Log

fit = glm(formula, df.iloc[:100], debug=False, family='Poisson', link='log')
print(fit)

for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:100]]:
    assert_almost_equal(fit.predict(data=X), fit.fittedvalues)
    assert_almost_equal(fit.predict(data=X, link=Log()), fit.fittedvalues)
    assert_almost_equal(fit.predict(data=X, link='log'), fit.fittedvalues)

expected = np.exp(fit.model.exog.toarray().dot([1.5, 1., -.3] + [.15] + [0] * 11))
for params in [
    [1.5, 1., -.3] + [.15] + [0] * 11,
    {'Intercept': 1.5, 'x': 1., 'w': -.3, 'C(grp)[1]': .15},
    pd.Series(index=['Intercept', 'x', 'w', 'C(grp)[1]'], data=[1.5, 1., -.3, .15])
]:
    assert_almost_equal(fit.predict(data=X, params=params), expected)

for X in [None, fit.model.exog, fit.model.exog.toarray(), df.iloc[:100]]:
    params = [1.5, 1., -.3] + [.15] + [0] * 11
    assert_almost_equal(fit.predict(data=X, params=params), expected)

model_full = lm(formula, df).model
expected = np.exp(model_full.exog.toarray().dot(fit.params))
for X in [model_full.exog, model_full.exog.toarray(), df]:
    assert_almost_equal(fit.predict(X), expected)

########
# NLLS #
########

from kanly.api import nlls
fit = nlls('[y] ~ np.exp({a}+{b}*[x])', df)
print(fit)

assert_almost_equal(fit.predict(), fit.fittedvalues)
assert_almost_equal(fit.predict(params=fit.params), fit.fittedvalues)
assert_almost_equal(fit.predict(data=df), fit.fittedvalues)

expected = np.exp(df.x * 1.2)
assert_almost_equal(fit.predict(params={'b': 1.2}), expected)
assert_almost_equal(fit.predict(params=[0, 1.2]), expected)

expected = np.exp(df.x.iloc[:20] * fit.params['b'] + fit.params['a'])
assert_almost_equal(fit.predict(data=df.iloc[:20]), expected)

df2 = df.copy(deep=True)
df2['x'] = df2['x'] * 2.5
expected = np.exp(df2.x * fit['b'] + fit['a'])
assert_almost_equal(fit.predict(data=df2), expected)

df_null = df.copy()
scale = 0.0
df_null['x'] = df['x'] * scale
predicted = fit.predict(df_null)
expected = [np.exp(fit.params['a'])] * len(df_null)
assert_almost_equal(predicted, expected)


####################################
# Linear Model, with absorb and IV #
####################################

fit_iv = lm('y ~ x|w', df)
print(fit_iv)

exception_thrown = False
try:
    y = fit_iv.predict(df)
except Exception:
    exception_thrown = True
assert exception_thrown

assert_almost_equal(fit_iv.predict(df, override_iv_error=True), fit_iv.model.exog.toarray().dot(fit_iv.params))

fit_abs = lm('y ~ x', df, absorb='grp')
print(fit_abs)

exception_thrown = False
try:
    y = fit_abs.predict(df)
except Exception:
    exception_thrown = True
assert exception_thrown

assert_almost_equal(fit_abs.predict(df, override_absorb_error=True), fit_abs.model.exog.toarray().dot(fit_abs.params))


