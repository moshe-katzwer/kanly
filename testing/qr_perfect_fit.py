# import pandas as pd
# import numpy as np
# from kanly.api import qr, lm
# from statsmodels.formula.api import quantreg
#
# import matplotlib.pyplot as plt
#
# n = 100
# np.random.seed(0)
# df = pd.DataFrame({
#     'x': (x := np.random.randn(n)),
#     'y': (1.2 + 2.3 * x),
# })
#
# print(qr('y ~ x', df, .5, debug=True))
