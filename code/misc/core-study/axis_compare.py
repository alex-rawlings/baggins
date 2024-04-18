import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import ketjugw


data_dir1 = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0300"
data_dir2 = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/along_y_axis"
kpc = bgs.general.units.kpc
Myr = bgs.general.units.Myr

kfiles = bgs.utils.get_ketjubhs_in_dir(data_dir1)
kfiles.extend(bgs.utils.get_ketjubhs_in_dir(data_dir2))

fig, ax = plt.subplots(2,1, sharex="all")

# get the BHs
bh_list = []
for i, kf in enumerate(kfiles):
    bhs = ketjugw.load_hdf5(kf)
    print(f"Reading {kf}")
    bhid = max(bhs, key=lambda k: len(bhs[k]))
    bh_list.append(bhs[bhid])

endidx = min([len(b) for b in bh_list])

for a, bh in zip("xy", bh_list):
    r = bgs.mathematics.radial_separation(bh.x/kpc)
    print(f"    For {a}, max distance is {max(r):.4f}")
    ls = ":" if np.any(r>30) else "-"
    ax[0].plot(bh.t[:endidx:10]/Myr, r[:endidx:10], ls=ls, label=a)
ax[1].plot(bh_list[0].t[:endidx:10]/Myr, 
           bgs.mathematics.radial_separation(bh_list[0].x[:endidx:10])/bgs.mathematics.radial_separation(bh_list[1].x[:endidx:10])
           )

ax[-1].set_xlabel("t/Myr")
ax[0].set_ylabel("r/kpc")
ax[1].set_ylabel("rx/ry")
ax[0].legend()
bgs.plotting.savefig("axis_compare.png")
plt.show()