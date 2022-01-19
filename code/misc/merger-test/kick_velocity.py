import numpy as np
import scipy.constants as C
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


ketju_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.001/perturbations/005/output/ketju_bhs_cp.hdf5"

bh1, bh2 = ketjugw.data_input.load_hdf5(ketju_file).values()
bh1, bh2, merged = cmf.analysis.get_bound_binary(ketju_file)
print(merged)


quit()
#orbit_params = ketjugw.orbital_parameters(bh1, bh2)
#start_idx = -100


"""fig, ax = plt.subplots(1,2)
for bh in (bh1, bh2):
    for i, (x,y) in enumerate(zip((0,2), (0,1))):
        ax[i].plot((bh.x[start_idx:,x]-orbit_params["x_CM"][start_idx:,x])/ketjugw.units.pc, (bh.x[start_idx:,y]-orbit_params["x_CM"][start_idx:,y])/ketjugw.units.pc, alpha=0.5, markevery=[-1], marker="o")
        #ax[i].plot(bh.v[start_idx:,x]/ketjugw.units.km_per_s, bh.v[start_idx:,y]/ketjugw.units.km_per_s, alpha=0.5, markevery=[-1], marker="o")
ax[0].set_xlabel(r"$v_x$ / km/s")
ax[1].set_xlabel(r"$v_x$ / km/s")
ax[0].set_ylabel(r"$v_z$ / km/s")
ax[1].set_ylabel(r"$v_y$ / km/s")
"""

myr = ketjugw.units.yr * 1e6

fig, ax = plt.subplots(2,3, sharex="all", sharey="row")
for bh in (bh1, bh2):
    for i in range(3):
        ax[0, i].plot(bh.t/myr, bh.x[:,i]/ketjugw.units.pc, "-o", markevery=[-1])
        ax[1, i].plot(bh.t/myr, bh.v[:,i]/ketjugw.units.km_per_s, "-o", markevery=[-1])

ax[1, 0].set_xlabel("t/Myr")
ax[0, 0].set_ylabel("x/pc")
ax[1, 0].set_ylabel("v/km/s")

plt.show()