import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
kpc = cmf.general.units.kpc
Myr = cmf.general.units.Myr

kfiles = cmf.utils.get_ketjubhs_in_dir(data_dir)

fig, ax = plt.subplots(1,1)

for i, kf in enumerate(kfiles):
    if "1200" in kf: break
    bhs = ketjugw.load_hdf5(kf)
    print(f"Reading {kf}")
    bhid = max(bhs, key=lambda k: len(bhs[k]))
    r = cmf.mathematics.radial_separation(bhs[bhid].x/kpc)
    print(f"    For {i*60}, max distance is {max(r):.1f}")
    ls = ":" if np.any(r>30) else "-"
    ax.plot(bhs[bhid].t[::10]/Myr, r[::10], ls=ls, label=i*60)

ax.set_xlabel("t/Myr")
ax.set_ylabel("r/kpc")
ax.legend()
plt.show()