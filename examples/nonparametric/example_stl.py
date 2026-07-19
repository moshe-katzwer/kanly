import numpy as np
from kanly.api import stl, plot_stl

n = 12*6
x = np.random.randn(n)
t = np.arange(n)
s = np.sin(2* np.pi * t / 12)
y = 1 * s + .15 * t + x

trend, seasonality, resid = stl(y, period=12, twindow=18)
plot_stl(y, trend, seasonality, resid, show=True)