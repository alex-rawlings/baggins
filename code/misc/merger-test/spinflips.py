import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


ketju_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.005/perturbations/009/output/ketju_bhs_cp.hdf5"

bh1, bh2, merged = cmf.analysis.get_bound_binary(ketju_file)
myr = ketjugw.units.yr * 1e6
orbit_params = ketjugw.orbital_parameters(bh1, bh2)

fig, ax = plt.subplots(3,1, sharex="all", sharey="all")
ax[0].set_ylim(-1, 1)
for i, axi in enumerate(ax):
    axi.plot(orbit_params["t"]/myr, orbit_params["plane_normal"][:,i])
    axi.axhline(0, c="k", alpha=0.6)
plt.show()