import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

filenames = ["/users/arawling/projects/collisionless-merger-sample/parameters/parameters-galaxies/dm-compare/hernquist/hernquist.hdf5", "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-galaxies/dm-compare/nfw/nfw.hdf5", ]
labels = ["hernquist", "nfw"]

radial_edges = np.logspace(1, 4, 25)
radial_centres = bgs.mathematics.get_histogram_bin_centres(radial_edges)

fig, ax = plt.subplots(1,2, sharex="all", figsize=(7, 4))
for i, (snapfile, label) in enumerate(zip(filenames, labels)):
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    centre = pygad.analysis.shrinking_sphere(snap.stars, snap.bh['pos'], 10)
    pygad.Translation(-centre).apply(snap)
    virial_radius, virial_mass = pygad.analysis.virial_info(snap)
    print("{} Mvir: {:.3e}".format(label, virial_mass))
    print("{} fraction: {:.3f}".format(label, virial_mass/np.sum(snap["mass"])))
    
    l, = ax[0].loglog(radial_centres, pygad.analysis.profile_dens(snap.dm, qty="mass", r_edges=radial_edges), label=label)
    for axi in ax:
        axi.axvline(virial_radius, c=l.get_color(), ls="--", lw=0.8)
        if label == "nfw":
            axi.axvline(7*virial_radius, c=l.get_color(), ls=":", lw=0.8)
        axi.axvline(8000, c="k", ls="-", lw=0.8)
    ax[1].loglog(radial_centres, pygad.analysis.profile_dens(snap.dm, qty="mass", r_edges=radial_edges, proj=1), label=label)
ax[0].set_xlabel("R/kpc")
ax[1].set_xlabel("R/kpc")
ax[0].set_ylabel(r"3D Density (M$_\odot$/kpc$^3$)")
ax[1].set_ylabel(r"Projected Density (M$_\odot$/kpc$^2$)")
ax[0].legend()
plt.tight_layout()
#plt.savefig("/users/arawling/figures/dm-comp.png", dpi=300)
plt.show()