from kanly.api import lm, nlls, elastic_net, glm, qr, gmm, rlm
import pandas as pd
import numpy as np

np.random.seed(0)
n = 20

df = pd.DataFrame()
df['x'] = np.random.randn(n)
df['w'] = np.exp(np.random.randn(n))
df['y'] = 1.32 + df.x + np.random.randn(n)

df.loc[0, 'x'] = np.nan
df.loc[10, 'y'] = np.nan
df.loc[13, 'w'] = np.nan

for index, expected in zip(
        [None, np.arange(10), np.arange(20) < 10],
        [set(np.arange(n)) - {0, 10, 13}, set(np.arange(10)) - {0},  set(np.arange(10)) - {0}]
):

    for method in [lm, qr, rlm, elastic_net, glm]:
        fit = lm('y ~ x $ w', df, index=index)
        vor = fit.model.valid_obs_rows
        assert set(vor) == expected

    fit = nlls('[y] ~ {b}+{a}*[x] $ [w]', df, index=index)
    vor = fit.model.valid_obs_rows
    assert set(vor) == expected

# TODO
# fit = gmm(
#     [
#         ('[y] - {b}+{a}*[x]',),
#         ('[y] - {b}+{a}*[x]', '[x]')
#     ], df)
# vor = fit.model.valid_obs_rows
# assert set(vor) == set(np.arange(n)) - {0, 10, 13}
