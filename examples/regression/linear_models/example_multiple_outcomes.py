import numpy as np
import pandas as pd

from kanly.api import lm, compare_results, LM

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'z': np.random.randn(n),
    'e': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
    'obs': np.random.randint(1, 6, n),
})
df['x'] = .5 * df['z'] + .7 * df['e']
df['y'] = 1.2 - 0.3 * df['x'] + .2 * df['e'] / df['obs']

fits = lm('y + I(y+1) ~ x | z  $ obs', df, specification_name='WLS-IV-Absorb')
print(compare_results(fits.values()))
