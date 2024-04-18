import numpy as np
import matplotlib.pyplot as plt
import pygad
import ketjugw
import baggins as bgs

pc = ketjugw.units.pc
myr = ketjugw.units.yr * 1e6
kms = ketjugw.units.km_per_s

bhfile = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa0524/output/ketju_bhs.hdf5"
snapdir = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa0524/output/"

snap_files = bgs.utils.get_snapshots_in_dir(snapdir)

bhs = ketjugw.data_input.load_hdf5(bhfile)

fig, ax = plt.subplots(1,2)
ax[0].set_xlabel('x/kpc')
ax[0].set_ylabel('z/kpc')
ax[1].set_xlabel(r'v$_\mathrm{x}$/kms$^{-1}$')
ax[1].set_ylabel(r'v$_\mathrm{z}$/kms$^{-1}$')
for i, bh in enumerate(bhs.values()):
    ax[0].plot(bh.x[:,0]/pc, bh.x[:,2]/pc, markevery=[-1], marker='o')
    ax[1].plot(bh.v[:,0]/kms, bh.v[:,2]/kms , markevery=[-1], marker='o')
for i, snap_file in enumerate(snap_files):
    print("Reading: {}".format(snap_file))
    snap = pygad.Snapshot(snap_file)
    snap.to_physical_units()
    xcom = pygad.analysis.shrinking_sphere(snap.stars, snap.bh['pos'], 10)
    xcom *= 1e3
    ax[0].scatter(xcom[0], xcom[2], c='tab:red')
    snap.delete_blocks()
plt.tight_layout()
plt.savefig("/users/arawling/figures/perturb-test/stability-perturb.png", dpi=300)
plt.show()