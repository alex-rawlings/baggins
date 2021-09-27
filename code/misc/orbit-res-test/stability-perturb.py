import numpy as np
import matplotlib.pyplot as plt
import pygad
import ketjugw

pc = ketjugw.units.pc
myr = ketjugw.units.yr * 1e6
kms = ketjugw.units.km_per_s

bhfile = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa0524/output/ketju_bhs.hdf5"

bhs = ketjugw.data_input.load_hdf5(bhfile)

fig, ax = plt.subplots(1,2)
for i, bh in enumerate(bhs.values()):
    ax[0].plot(bh.x[:,0]/pc, bh.x[:,2]/pc, markevery=[-1], marker='o')
    ax[1].plot(bh.v[:,0]/kms, bh.v[:,2]/kms , markevery=[-1], marker='o')
plt.show()