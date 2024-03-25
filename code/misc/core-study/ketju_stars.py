import numpy as np
import matplotlib.pyplot as plt
import h5py
import baggins as bgs


kfiles = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick")

fig, ax = plt.subplots(1,1)
cmapper, sm = bgs.plotting.create_normed_colours(0, 2000)
for i, kf in enumerate(kfiles):
    if i*60 > 900: break
    print(f"Reading {kf}")
    with h5py.File(kf, "r") as f:
        tsteps = f["/timesteps"]["physical_time"][:]
        numparts = np.zeros(len(tsteps), dtype=float)
        for k, bh in f["/BHs"].items():
            numparts[:len(bh["mass"])] += bh["num_particles_in_region"][:]
        ax.plot(tsteps[::500], numparts[::500])
plt.show()