import pandas as pd
from statsmodels.formula.api import ols as ols_sm
from kanly.api import lm
import numpy as np

df = pd.DataFrame({'grp': np.arange(10), 'y': np.arange(10)})

print(ols_sm('y ~ C(grp)-1', df).fit().summary())
print(lm('y ~ C(grp)-1', df))