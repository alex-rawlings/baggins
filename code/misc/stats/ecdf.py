import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs

bgs.plotting.check_backend()

rng = np.random.default_rng(42)

x = rng.gamma(2, 0.5, size=1000)
x[23] = np.nan

fig, ax = plt.subplots()
for label, weights in zip(("no weights", "weights"), (None, x**2)):
    ecdf = bgs.mathematics.EmpiricalCDF(x=x, weights=weights)
    ecdf.plot(ci_prob=0.94, ci_kwargs={"alpha":0.3}, ax=ax, label=label)
ax.legend()
bgs.plotting.savefig("ecdf.png")