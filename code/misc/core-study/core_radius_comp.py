import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs


data1 = bgs.literature.LiteratureTables.load_thomas_2016_data()
data2 = bgs.literature.LiteratureTables.load_dullo_2019_data()

ax = None
for dat in (data1, data2):
    ax = dat.plot_lin_regress("BH_mass", "core_radius_kpc", fit_in_log=True, xhat_method=np.geomspace, ax=ax, mask=dat.table.loc[:,"core_radius_kpc"]<0.5, itype="pred")

# add mergers
kicks = dict(
        v0000 = 0.580,
        v0120 = 0.737,
        v0240 = 0.832,
        v0300 = 1.130,
        v0600 = 1.330,
        v1020 = 1.640
    )
my_BH_mass = 5.86e9
for rb in kicks.values():
    ax.scatter(my_BH_mass, rb, marker="s")

ax.set_xlabel("BH Mass")
ax.set_ylabel("rb/kpc")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

plt.show()