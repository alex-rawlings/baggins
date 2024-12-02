import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad


data_dir = "/scratch/pjohanss/arawling/antti-cores/raw-data"

snapfiles = bgs.utils.get_files_in_dir(data_dir)

fig, ax = plt.subplots(5, 1, sharex="all", sharey="all")
fig.set_figheight(1.5 * fig.get_figheight())
axdict = {"100": ax[0], "125":ax[1], "150":ax[2], "175":ax[3], "200":ax[4]}
redges = np.geomspace(1e-1, 15, 11)
r = bgs.mathematics.get_histogram_bin_centres(redges)
cmapper, sm = bgs.plotting.create_normed_colours(1e8, 2e9, "crest", norm="LogNorm")

gammas = []

for i, snapfile in enumerate(snapfiles):
    # load and centre snap
    gamma_key = int(os.path.basename(snapfile).replace("snapshot_last_gamma_", "")[:3])
    print(f"Doing {gamma_key}")
    gamma = float(gamma_key)/100
    if gamma not in gammas:
        gammas.append(gamma)
    snap = pygad.Snapshot(snapfile, physical=True)
    pygad.Translation(
        -pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30
        )
    ).apply(snap, total=True)
    pygad.Boost(
        -pygad.analysis.mass_weighted_mean(
            snap.stars[pygad.BallMask(1)],
            "vel"
            )
        ).apply(snap, total=True)

    beta, ppb = bgs.analysis.velocity_anisotropy(
        snap.stars,
        redges
    )
    axdict[str(gamma_key)].semilogx(r, beta, c=cmapper(np.sum(snap.bh["mass"])), lw=2)

    snap.delete_blocks()
    pygad.gc_full_collect()
    del snap

ax[-1].set_xlabel(r"$r/\mathrm{kpc}$")
for axi in ax:
    axi.set_ylabel(r"$\beta$")
    axi.set_ylim(-1.5, 1)
    axi.axhline(0, ls=":", c="gray", zorder=0.01)
for axi, gamma in zip(ax, gammas):
    axi.text(0.5, 0.5, f"$\gamma_0 = {gamma:.2f}$")
plt.colorbar(sm, ax=ax.flat, label=r"$M_\bullet/M_\odot$")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "antti-core-all/beta.pdf"), force_ext=True)