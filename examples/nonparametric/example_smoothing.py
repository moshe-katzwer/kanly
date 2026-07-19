import numpy as np
from kanly.api import LOWESS, gaussian_kernel_smooth
import matplotlib.pyplot as plt

n = 100
x = np.random.rand(n)
y = np.sin(x * 6) + .1 * np.random.randn(n) + .1 * x

x_lowess, y_lowess = LOWESS(y, x, it=0, degree=1, return_arrays=True)
x_gauss, y_gauss = gaussian_kernel_smooth(x, y, return_arrays=True, adjust=.5)

plt.scatter(x, y, color='grey')
plt.plot(x_lowess, y_lowess, label='Lowess')
plt.plot(x_gauss, y_gauss, label='Gaussian Kernel')
plt.legend(loc='best')
plt.title('smoothing')
plt.show()
