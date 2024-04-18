import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


kv = "1200"


snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv}/output/snap_016.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)
ball_mask = pygad.BallMask(20)
slab_mask = pygad.DiscMask(zmax=1)

# align the snapshot with the BH motion along the LOS
rot1 = pygad.rot_to_z(snap.bh["vel"].flatten())
rot1.apply(snap, total=True)
rot2 = pygad.rot_from_axis_angle([1,0,0], np.pi/2)
rot2.apply(snap, total=True)

mask = ball_mask & slab_mask

fig, ax, *_ = pygad.plotting.image(snap.stars[mask], "mass", showcbar=False)

dens, xe, ye, bn = binned_statistic_2d(
                x = snap.stars[mask]["pos"][:,0],
                y = snap.stars[mask]["pos"][:,1], 
                values = None,
                statistic = "count",
                bins = 50)
xb = bgs.mathematics.get_histogram_bin_centres(xe)
yb = bgs.mathematics.get_histogram_bin_centres(ye)

#dens = np.log10(dens)

CS = ax.contour(xb, yb, dens, 10, lw=2, cmap="Reds")
ax.clabel(CS, inline=True)
ax.scatter(snap.bh["pos"][0,0], snap.bh["pos"][0,1], marker="o", color="k", zorder=2, ec="w")

ax.set_aspect("equal")
ax.set_facecolor("k")

# add the initial BH position
if True:
    snapfile0 = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv}/output/snap_000.hdf5"
    snap0 = pygad.Snapshot(snapfile0, physical=True)
    rot1.apply(snap0, total=True)
    rot2.apply(snap0, total=True)
    ax.scatter(snap0.bh["pos"][0,0], snap0.bh["pos"][0,1], marker="o", color="r", zorder=2, ec="w")

if True:
    ball_mask_0 = pygad.BallMask(0.1, snap0.bh["pos"][0])
    id_mask = pygad.IDMask(snap0.stars[ball_mask_0]["ID"])
    new_mask = id_mask & slab_mask
    ax.plot(snap.stars[new_mask]["pos"][:,0], snap.stars[new_mask]["pos"][:,1], ls="", marker=".", c="r", ms=2)
plt.savefig("isophotes.png")
plt.show()
