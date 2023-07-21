import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


def b_param(t):
    return 1/np.tan(t/2)

data_path = "/users/arawling/projects/collisionless-merger-sample/papers/paper-eccentricity/scripts/data"
data = [
    os.path.join(data_path, "deflection_angles_e0-0.900.pickle"),
    os.path.join(data_path, "deflection_angles_e0-0.990.pickle"),
]

if True:
    stat_func = np.std
    ylab = "Std Dev"
else:
    stat_func = cmf.mathematics.iqr
    ylb = IQR


cols = cmf.plotting.mplColours()
fig, ax = plt.subplots(1,2, sharex="all")
for i in range(2): 
    ax[i].set_xlabel("Mass Res")
    ax[i].set_xscale("log")
ax[0].set_ylabel(f"{ylab} deflection angle")
ax[1].set_ylabel(f"{ylab} $b/b_{{90}}$")


for i, d in enumerate(data):
    datadict = cmf.utils.load_data(d)
    ts = np.array(datadict["thetas"])
    bs = b_param(ts)
    mr = np.unique(datadict["mass_res"])
    for _mr in mr:
        mask = datadict["mass_res"] == _mr
        ax[0].scatter(_mr, stat_func(ts[mask]), c=cols[i])
        ax[1].scatter(_mr, stat_func(bs[mask]), c=cols[i])
plt.show()