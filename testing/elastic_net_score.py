# from kanly.api import lm, elastic_net
# import pandas as pd
# import numpy as np
#
# np.random.seed(0)
# n = 100
# df = pd.DataFrame({
#     'x': np.random.randn(n),
#     'g': np.random.randint(0,5,n),
#     'w': .2+np.exp(np.random.randn(n)),
# })
# df['y'] = 12.5 + .33 * df.x + np.random.randn(n) * (1 + 1 / df.w)
#
# print(lm('y ~ x +C(g) -1 $ w', df))
# print((fit := elastic_net('y ~ x + C(g) $ w', df, alpha=.000001, l1_ratio=.8)).summary(show_only_non_zero=False))
#
