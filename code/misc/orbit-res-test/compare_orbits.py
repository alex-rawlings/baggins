import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf


main_path = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/"
res = "x10"
orbits = ["0-001", "0-005", "0-030", "0-180", "1-000"]

kpc = ketjugw.units.pc * 1e3

cols = cmf.plotting.mplColours()
fig, ax = plt.subplots(1, 1)
ax.set_aspect("equal")
ax.set_xlabel('x/kpc')
ax.set_ylabel('z/kpc')
for ind, orbit in enumerate(orbits):
    bhs = ketjugw.data_input.load_hdf5("{}{}/{}/output/ketju_bhs.hdf5".format(main_path, res, orbit))
    for ind2, bh in enumerate(bhs.values()):
        ax.plot(bh.x[:,0]/kpc, bh.x[:,2]/kpc, markevery=[-1], label=(orbit if ind2==0 else ""), c=cols[ind], marker='o')
plt.legend()
plt.tight_layout()
plt.savefig("/users/arawling/figures/res-test/orbit-compare.png", dpi=300)
plt.show()