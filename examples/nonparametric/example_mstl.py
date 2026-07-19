import numpy as np
from kanly.api import mstl, plot_mstl

n = 365 * 3
t = np.arange(n)
weekly = 2.0 * np.sin(2 * np.pi * t / 7)
yearly = 5.0 * np.sin(2 * np.pi * t / 365)
trend_true = 0.005 * t
noise = np.random.randn(n)
y = weekly + yearly + trend_true + noise

trend, seasonalities, resid = mstl(y, period=[7, 365])
plot_mstl(y, trend, seasonalities, resid, show=True,
          period_labels=['Weekly (7)', 'Yearly (365)'])
