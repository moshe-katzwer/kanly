from textwrap import wrap

import numpy as np
import pandas as pd

from kanly.api import sparse_dmatrix
from kanly.api import LM_fast

np.random.seed(0)
n = 300_000
x = np.random.randn(n)
w = 1.5 + np.random.randn(n)
df = pd.DataFrame({'x': x,
                   'w': w,
                   'z': np.random.randn(n),
                   'g': np.random.randint(0, 1_200, n),
                   'y': 10 + x * 1.2 + np.random.randn(n)
                   })

exog_data_object = sparse_dmatrix('x*C(g) + L(x) + I(x**2) + poly(z, 3) + center(w)', df, debug=True)

col_names_sub = "\n".join(wrap(str(exog_data_object.column_names[:20]), 80, subsequent_indent="\t"))

print(f'The sparse matrix returned has shape {exog_data_object.values.shape}'
      f' and nnz {exog_data_object.values.nnz} ({"%.2f" % (100 * exog_data_object.values.nnz / np.prod(exog_data_object.values.shape))}%).'
      f'\nThe first 20 column labels are\n\t'
      f' {col_names_sub}.'
      )

null_rows = exog_data_object.null_rows
valid_index = np.delete(np.arange(len(df)), list(null_rows))

fit = LM_fast(df.y.iloc[valid_index],
              exog_data_object.values[valid_index,:],
              exog_names=exog_data_object.column_names)
print(fit)
