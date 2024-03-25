import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.set_publishing_style()

gal_file = "/scratch/pjohanss/arawling/collisionless_merger/galaxies/hernquist/hernquist.hdf5"

snap = pygad.Snapshot(gal_file, physical=True)

rad_bins = dict(
    stars = np.geomspace(5e-2, 5e2, 101),
    dm = np.geomspace(0.8, 5e2, 101)
)

rad_bin_centres = {}
for k, v in rad_bins.items():
    rad_bin_centres[k] = bgs.mathematics.get_histogram_bin_centres(v)

fig, ax = plt.subplots(1,1)
star_dens_3d = pygad.analysis.profile_dens(snap.stars, "mass", r_edges=rad_bins["stars"])
dm_dens_3d = pygad.analysis.profile_dens(snap.dm, "mass", r_edges=rad_bins["dm"])

ax.loglog(rad_bin_centres["stars"], star_dens_3d, lw=2, label="stars")
ax.loglog(rad_bin_centres["dm"], dm_dens_3d, lw=2, label="DM")

ax.set_xlabel("Radius [kpc]")
ax.set_ylabel(r"Density [M$_\odot$/kpc$^3$]")
ax.legend()

plt.savefig(os.path.join(bgs.FIGDIR, "demo_density.pdf"))
plt.show()