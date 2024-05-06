import matplotlib.pyplot as plt
import pygad
from baggins.plotting import savefig
import figure_config


upper_bound = 10

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].set_ylabel(r"$y/\mathrm{kpc}$")

for kv, axi in zip(("0000", "1500"), ax):

    snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv}/output/snap_016.hdf5"

    snap = pygad.Snapshot(snapfile, physical=True)

    mask = (
        pygad.ExprMask(f"pos[:,1]<{upper_bound}")
        & pygad.ExprMask(f"pos[:,1]>{-upper_bound}")
        & pygad.ExprMask(f"pos[:,2]<{upper_bound}")
        & pygad.ExprMask(f"pos[:,2]>{-upper_bound}")
    )  # & pygad.ExprMask("pos[:,0]>0")

    pygad.plotting.vec_field(
        snap.stars[mask],
        qty="vel",
        extent=[[0, 5], [-5, 5]],
        ax=axi,
        streamlines=True,
        Npx=10,
        linewidth=1,
        color=figure_config.col_list[1],
    )
    # set to True to add the BH
    if False:
        axi.plot(
            snap.bh["pos"][:, 0], snap.bh["pos"][:, 1], c="tab:red", marker="o", ls=""
        )
        # add arrow showing motion
        axi.arrow(
            snap.bh["pos"][0, 0],
            snap.bh["pos"][0, 1],
            snap.bh["vel"][0, 0] * pygad.UnitScalar(0.01, "Gyr"),
            snap.bh["vel"][0, 1] * pygad.UnitScalar(0.01, "Gyr"),
            color="tab:red",
            width=0.1,
            head_width=0.5,
            zorder=2,
        )
    axi.set_xlabel(r"$x/\mathrm{kpc}$")
    axi.set_title(
        f"$v_\mathrm{{kick}}={float(kv):.0f} \,\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$"
    )

savefig(figure_config.fig_path("wake.pdf"), force_ext=True)

plt.show()
