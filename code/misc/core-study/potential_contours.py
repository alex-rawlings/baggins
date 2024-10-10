import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad


bgs.plotting.check_backend()

snap_num = 24 #267

snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-2000/output/snap_{snap_num:03d}.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

ball_mask = pygad.BallMask(30)
xcom = pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars[ball_mask]), 30
)
trans = pygad.Translation(-xcom)
trans.apply(snap, total=True)

print(f"BH is located at {snap.bh['pos']}")

snap["pot_mag"] = np.abs(snap["pot"])

# mask to a slice
mask = pygad.masks.ExprMask("abs(pos[:,1])<0.2")


fig, ax, im, cbar = pygad.plotting.image(snap.stars[mask], "mass", zero_is_white=False, xaxis=0, yaxis=2)
ax.set_facecolor("k")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "core-study/v2000-potential/pot_contour.png"))

plt.close()

idxs = np.argsort(snap.stars[mask]["r"])
plt.loglog(snap.stars[mask]["r"][idxs], pygad.utils.geo.dist(snap.stars[mask]["angmom"][idxs]))
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "core-study/v2000-potential/L_1D.png"))