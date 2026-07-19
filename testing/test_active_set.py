import pandas as pd
import numpy as np
from kanly.api import elastic_net, compare_results

n = 201040
n_g = 20
n_z2 = 300

np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randn(n) ** 2,
    'g': np.random.randint(0, n_g, n),
})

Z = np.random.randn(n, 10)
Z = Z.dot(np.random.randn(10, n_z2))

for j in range(n_z2):
    df[f'z{j}'] = Z[:, j]

df['y'] = -2 + np.exp(1.2 * df.x) + .3 * df.x * df.w + 0.6 * df.x ** 2 + np.random.randn(n) * .3 + .54 * np.sqrt(
    df.g + 5)

fits = []
for active_set in (True, False):
    fits.append(
        elastic_net(
            'y ~ x + w + C(g)'
            + '+' + ' + '.join(f'z{j}' for j in range(n_z2))
            ,
            df, alpha=.00025, active_set=True,
            selection='random',
            specification_name=f'active set: {active_set}'
        )
    )

print(compare_results(fits))
for t in fits:
    print(t.specification_name, ', time = ', t.fit_elapsed)
