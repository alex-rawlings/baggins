import os
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad
import ketjugw


data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/000/output/"
snaplist = cmf.utils.get_snapshots_in_dir(data_dir)
ketjufile = os.path.join(data_dir, "ketju_bhs_cp.hdf5")
bh1, bh2, merged = cmf.analysis.get_bound_binary(ketjufile)
orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)

Gyr = ketjugw.units.yr * 1e9
kpc = ketjugw.units.pc * 1e3

g_rho_sigma = {"t": np.full_like(snaplist, np.nan, dtype=float),
               "Gps": np.full_like(snaplist, np.nan, dtype=float)}
for i, snapfile in enumerate(snaplist):
    print("Complete: {:.1f}%                               ".format(i/(len(snaplist)-1)*100), end="\r")
    snap = pygad.Snapshot(snapfile, physical=True)
    g_rho_sigma["t"][i] = cmf.general.convert_gadget_time(snap)
    if i==0:
        #determine influence radius
        rh = pygad.UnitScalar(list(cmf.analysis.influence_radius(snap).values())[0], "kpc")
        r_h_time = cmf.general.xval_of_quantity(rh, orbit_params["t"]/Gyr, orbit_params["a_R"]/kpc, xsorted=True)
    g_rho_sigma["Gps"][i] = cmf.analysis.get_G_rho_per_sigma(snap, extent=rh)
    del snap 

plt.plot(g_rho_sigma["t"], g_rho_sigma["Gps"], "-o")
plt.axvline(r_h_time, c="tab:red")
plt.xlabel("t/Gyr")
plt.ylabel(r"$G\rho/\sigma$")
plt.show()