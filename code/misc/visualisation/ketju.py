import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pygad
import baggins as bgs


def bh_inset(ax, bounds, bhs, bh_num, stars):
    # set up the inset axis
    axins = ax.inset_axes(bounds)
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor((0.1, 0.1, 0.1))
    delta = 2e-3
    axins.set_xlim(bhs["pos"][bh_num, 0]-delta, bhs["pos"][bh_num, 0]+delta)
    axins.set_ylim(bhs["pos"][bh_num, 2]-delta, bhs["pos"][bh_num, 2]+delta)
    ax.indicate_inset_zoom(axins, edgecolor="w", linewidth=2)
    # add stars
    star_mask = pygad.BallMask(delta, bhs["pos"][bh_num, :])
    axins.scatter(stars[star_mask]["pos"][:,0], stars[star_mask]["pos"][:,2], color=(0.8, 0.5, 0.2), marker="o", s=20)
    axins.scatter(stars[star_mask]["pos"][:,0], stars[star_mask]["pos"][:,2], color="#FAC205", marker="o", s=2)
    # add the bh
    axins.scatter(bhs["pos"][:, 0], bhs["pos"][:, 2], color="k", edgecolor="w", linewidth=0.5, s=50)
    # add the Ketju region
    circle = Circle((bhs["pos"][bh_num, 0], bhs["pos"][bh_num, 2]), delta/2, ec="w", fc="none", ls="--")
    axins.add_patch(circle)




snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/4M/D_4M_a-D_4M_b-3.720-0.279/output/snap_003.hdf5"
#snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/output/A-C-3.0-0.05_030.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

fig, ax = plt.subplots(1,1)
ax.set_facecolor("k")
ax.set_aspect("equal")
plot_kwargs = {"cbartitle":r"$\log_{10}\left( \Sigma / (\mathrm{M}_\odot \mathrm{kpc}^{-2}) \right)$", "Npx":800, "qty":"mass", "fontsize":10, "xaxis":0, "yaxis":2, "cmap":"cividis"}
ball_mask = pygad.BallMask(30, snap.bh["pos"][0,:])
pygad.plotting.image(snap.stars[ball_mask], ax=ax, **plot_kwargs)

# note that to make the sketch "pretty", the ketju radius is not the same as
# what it is in the true simulation. As this is just a sketch, I think it's ok
bh_inset(ax, [0.05, 0.5, 0.3, 0.3], snap.bh, 0, snap.stars)
bh_inset(ax, [0.65, 0.02, 0.3, 0.3], snap.bh, 1, snap.stars)

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "ketju.pdf"), fig=fig, force_ext=True)
plt.show()