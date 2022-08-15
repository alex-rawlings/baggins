import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad



snapfiles = dict(
                ketju = "/scratch/pjohanss/mannerko/gadget4-ketju_testing/plummer_sinking/mass_ratio_10000_soft_0.005/output_ketju/snapshot_007.hdf5",
                gadget = "/scratch/pjohanss/mannerko/gadget4-ketju_testing/plummer_sinking/mass_ratio_10000_soft_0.005/output_no_ketju/snapshot_007.hdf5"
)

fig, ax = plt.subplots(1,1)
ax.set_xlabel("v/(km/s)")


for k, v in snapfiles.items():
    snap = pygad.Snapshot(v, physical=True)
    xcom = cmf.analysis.get_com_of_each_galaxy(snap, method="ss", family="stars")
    vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
    ax.hist(pygad.utils.geo.dist(snap.stars["vel"], list(vcom.values())[0]), bins=100, density=True, histtype="step", label=k)
ax.legend()

plt.show()
