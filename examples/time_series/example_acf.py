from kanly.api import simulate_sarima, acf, pacf
from kanly.time_series.sarimax.arma_innovation_functions import get_acf
import numpy as np
import matplotlib.pyplot as plt

# simulate an ARMA model
n = 20_000
ar = [.8, -.3]
ma = [0, 0, 0, 0, .5]
u = simulate_sarima(n, ar=ar, ma=ma, seed=0, sigma2=1.3)
y = u + 1.5

# estimate the autocovariance functions
acf_values = acf(y)
pacf_values = pacf(y)

plt.plot(acf_values[:30], marker='.', label='acf from values', lw=.5)
plt.plot(pacf_values[:30], marker='.', label='pacf from values', lw=.5)
plt.plot(get_acf(ar=ar, ma=ma)[:30], label='acf from ar/ma coefs', marker='.', lw=.5)
plt.axhline(0, c='k')
plt.legend(loc='best')
plt.show()
