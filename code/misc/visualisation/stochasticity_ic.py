import os.path
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/4M/D_4M_a-D_4M_b-3.720-0.279/D_4M_a-D_4M_b-3.720-0.279.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

id_masks = cmf.analysis.get_all_id_masks(snap)
radial_masks = cmf.analysis.get_all_radial_masks(snap, 5, id_masks=id_masks)

mask = pygad.BallMask(1e-4)
for v in radial_masks.values():
    mask = mask | v


fig, ax, im, cbar = pygad.plotting.image(snap.stars[mask], qty="mass", cmap="cividis", yaxis=2, cbartitle=r"$\log_{10}\left( \Sigma / (\mathrm{M}_\odot \mathrm{kpc}^{-2}) \right)$")
ax.scatter(snap.bh["pos"][:,0], snap.bh["pos"][:,2], marker="o", c="k", s=50, ec="w", lw=0.5)
ax.set_facecolor("k")

cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "stochasticity_ic.pdf"), fig=fig, force_ext=True)
plt.show()