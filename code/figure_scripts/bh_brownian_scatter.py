#plot the BH Brownian motion
# TODO ensure subplots are the same size

import matplotlib.pyplot as plt
import cm_functions as cmf

def plot_circle(ax, r=1, center=(0,0), **kwargs):
    circle = plt.Circle(center, r, fill=False, **kwargs)
    ax.add_artist(circle)

data_dict = cmf.utils.load_data("../analysis_scripts/pickle/bh_perturb/NGCa4291_bhperturb.pickle")


fig, ax = plt.subplots(1,2, figsize=(6, 1.05*6/2), subplot_kw={"aspect":1})
for axi in ax:
    axi.scatter(0,0, c="tab:red", marker="X", zorder=10, s=30, label="Stellar CoM")
ax[0].scatter(data_dict["diff_x"][:,0], data_dict["diff_x"][:,2], c="k", marker=".", label="BH")
cmf.plotting.draw_sizebar(ax[0], 1e-2, units="pc", unitconvert="kilo2base")
plot_circle(ax[0], 1.11e-2, color="tab:blue", ls="--", label=r"$1\sigma$")
ax[1].scatter(data_dict["diff_v"][:,0], data_dict["diff_v"][:,2], c="k", marker=".")
cmf.plotting.draw_sizebar(ax[1], 10, units="km/s")
plot_circle(ax[1], 12.8, color="tab:blue", ls="--")
ax[0].legend(loc="upper left")
plt.show()