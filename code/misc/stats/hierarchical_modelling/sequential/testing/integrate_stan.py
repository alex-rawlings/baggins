import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import cmdstanpy


x = np.linspace(0.1, 4, 100)
y = x

data = dict(
    N_tot = len(x),
    x = x,
    y = y
)

stan_model = cmdstanpy.CmdStanModel(stan_file="integrate_stan.stan")
fit = stan_model.sample(data=data)
print(fit.summary())
print(fit.diagnose())


integrated_x = fit.stan_variable("integrated_x")
plt.plot(x, np.nanmean(integrated_x, axis=0), "-o")
plt.semilogy(x, scipy.integrate.cumulative_trapezoid(1/y, x, initial=0))
plt.show()