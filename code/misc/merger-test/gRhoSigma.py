import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
import ketjugw


data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/000/output/"
snaplist = bgs.utils.get_snapshots_in_dir(data_dir)
#ketjufile = os.path.join(data_dir, "ketju_bhs_cp.hdf5")
#bh1, bh2, merged = bgs.analysis.get_bound_binary(ketjufile)
#orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)

Myr = ketjugw.units.yr * 1e6
kpc = ketjugw.units.pc * 1e3

g_rho_sigma = {"t": np.full_like(snaplist, np.nan, dtype=float),
               "r_infl": np.full_like(snaplist, np.nan, dtype=float),
               "Gps": np.full_like(snaplist, np.nan, dtype=float)}

bh_binary = bgs.analysis.BHBinary(ketjufile, snaplist, 15)
bh_binary.get_influence_and_hard_radius()

for i, snapfile in enumerate(snaplist):
    print("Complete: {:.1f}%                               ".format(i/(len(snaplist)-1)*100), end="\r")
    snap = pygad.Snapshot(snapfile, physical=True)
    g_rho_sigma["t"][i] = bgs.general.convert_gadget_time(snap, "Myr")
    g_rho_sigma["Gps"][i] = bgs.analysis.get_G_rho_per_sigma(snaplist, g_rho_sigma["t"][i], extent=bh_binary.r_infl)
    del snap 

plt.plot(g_rho_sigma["t"], g_rho_sigma["Gps"], "-o")
plt.axvline(bh_binary.r_hard_time, c="tab:red")
plt.scatter(bh_binary.r_hard_time, bgs.analysis.get_G_rho_per_sigma(snaplist, bh_binary.r_hard_time, extent=bh_binary.r_infl))
plt.xlabel("t/Myr")
plt.ylabel(r"$G\rho/\sigma$")
plt.show()