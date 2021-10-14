import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import os
import cm_functions as cmf


mainpath = "/scratch/pjohanss/arawling/collisionless_merger/r0-test"
datapaths = ["A-C-0.1-0.001", "A-C-0.5-0.001", "A-C-1.0-0.001", "A-C-3.0-0.001", "A-C-5.0-0.001", "A-C-10.0-0.001"]

kpc = ketjugw.units.pc * 1e3
myr = ketjugw.units.yr * 1e6

fig, ax = plt.subplots(1,1)
cols = cmf.plotting.mplColours()
for i, datapath in enumerate(datapaths):
    try:
        bhfile = os.path.join(mainpath, datapath, "output/ketju_bhs.hdf5")
        label = datapath.split("-")[2]
        print(bhfile)
        bhs = ketjugw.data_input.load_hdf5(bhfile)
    except BlockingIOError:
        bhfile = os.path.join(mainpath, datapath, "output/ketju_bhs_cp.hdf5")
        label = datapath.split("-")[2]
        print(bhfile)
        bhs = ketjugw.data_input.load_hdf5(bhfile)
    for j, bh in enumerate(bhs.values()):
        ax.plot(bh.x[:,0]/kpc, bh.x[:,2]/kpc, markevery=[-1], c=cols[i], label=(label if j==0 else ""))
ax.legend()
ax.set_xlabel("x/kpc")
ax.set_ylabel("z/kpc")
plt.show()