import numpy as np
import matplotlib.pyplot as plt
import ketjugw

bh_file = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/1-000/output/ketju_bhs.hdf5"
bhs = ketjugw.data_input.load_hdf5(bh_file)
bh1, bh2 = bhs.values()

myr = 1e6 * ketjugw.units.yr
kpc = 1e3 * ketjugw.units.pc

plt.plot(bh1.t / myr, bh1.x[:,2] / kpc)
plt.plot(bh2.t / myr, bh2.x[:,2] / kpc)
plt.xlabel('t/Myr')
plt.ylabel('z/kpc')
plt.show()
