import numpy as np
import matplotlib.pyplot as plt
from kanly.api import kde

n = 500
x = np.hstack([np.random.randn(n), 2+.5*np.random.randn(n), 4 + .25*np.random.randn(n)])

plt.hist(x, density=True, alpha=.5, bins=40)
for a in [1, .5, 1.5]:
    plt.plot(*kde(x, return_arrays=True, adjust=a), lw=2, label=f'kde(adjust={a})')
plt.legend(loc='best')
plt.title('kde example')
plt.show()