from kanly.api import rlm

import pandas as pd
import numpy as np
from statsmodels.formula.api import rlm as rlm_sm
from statsmodels.robust.norms import HuberT, TrimmedMean, AndrewWave, LeastSquares, TukeyBiweight, RamsayE
from numpy.testing import assert_allclose
from kanly.regression.linear_models.robust.robust_norm_functions import get_norm

n = 50
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.rand(n),
    'e': np.random.randn(n),
})
df.loc[np.random.choice(df.index, 5, replace=False), 'e'] += 15
df['y'] = 1.2 - .8 * df['x'] + df['e']

for m, M in {
    'hubert': HuberT,
    'trimmed_mean': TrimmedMean,
    'andrew_wave': AndrewWave,
    'least_squares': LeastSquares,
    'TukeyBiweight': TukeyBiweight,
    'RamsayE': RamsayE,
}.items():

    print(m, end='')

    try:
        fit = rlm('y ~ x', df, M=m)
        fit_sm = rlm_sm('y ~ x', df, M=M()).fit()
        assert_allclose(fit.params, fit_sm.params)

        norm = get_norm(m)
        cnt = 0
        for attr in ['rho', 'psi', 'psi_deriv', 'weights']:
            try:
                assert_allclose(getattr(norm, attr)(df.x), getattr(M(), attr)(df.x))
            except:
                cnt += 1
                print(f'\t{attr} failed!,  ', end='')
        if cnt:
            raise Exception
        print('...passed')
    except Exception as e:
        print('...failed!')
