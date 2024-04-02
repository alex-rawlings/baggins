import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_002.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

radii = [0.01, 0.1, 0.5, 1., 2., 5.]
t = np.full_like(radii, np.nan)
for i, r in enumerate(radii):
    t[i] = bgs.analysis.relax_time(snap, r)
    print(f"For {r}: {t[i]:.2e} Gyr")
plt.loglog(radii, t, "-o")
plt.xlabel("r / kpc")
plt.ylabel("t_relax / Gyr")
bgs.plotting.savefig("t_relax.png")
plt.show()