import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot orbit families, based on `script_freq.py`",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

# create logger
SL = bgs.setup_logger("script", args.verbosity)

bgs.plotting.check_backend()


orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()


labels = [
    r"$\pi\mathrm{-box}$",
    r"$\mathrm{boxlet}$",
    r"$x\mathrm{-tube}$",
    "",
    r"$z\mathrm{-tube}$",
    r"$\mathrm{rosette}$",
    r"$\mathrm{irregular}$",
    r"$\mathrm{unclassified}$",
]


# figure 1: plots of different orbital families
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.set_figwidth(2 * fig.get_figwidth())
vkcols = figure_config.VkickColourMap()


# figure 2: plots of different kick velocities
fig2, ax2 = plt.subplots(5, 4, sharex=True, sharey=True)
fig2.set_figwidth(2 * fig2.get_figwidth())
fig2.set_figheight(2.5 * fig2.get_figheight())

for j, (axj, orbitfilebase) in enumerate(zip(ax2.flat, orbitfilebases)):
    try:
        orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[
            0
        ]
        (
            meanrads,
            classfrequency,
            rad_len,
            classids,
            peri,
            apo,
            minang,
        ) = bgs.analysis.radial_frequency(orbitcl, returnextra=True)
        rosette_mask = classids == 4
        for dist, arr in zip(("Apocentre", "Pericentre"), (apo, peri)):
            SL.info(
                f"{dist} IQR for rosettes: {np.nanquantile(arr[rosette_mask], 0.25):.2e} - {np.nanquantile(arr[rosette_mask], 0.75):.2e} (median: {np.median(arr[rosette_mask]):.2e})"
            )
    except:  # noqa
        # ongoing analysis
        SL.error(f"Unable to read {orbitfilebase}: skipping")
        # continue
    vkick = float(orbitfilebase.split("/")[-1].split("-")[-1])
    cfi = 0
    for i, axi in enumerate(ax.flat):
        if i == 3:
            continue
        axi.semilogx(
            meanrads,
            classfrequency[:, cfi],
            label=vkick,
            c=vkcols.get_colour(vkick),
            ls="-",
        )
        axj.semilogx(meanrads, classfrequency[:, cfi], label=labels[i])
        cfi += 1
    axj.text(
        0.95,
        0.9,
        f"${vkick:.0f}\, \mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
        ha="right",
        va="center",
        transform=axj.transAxes,
    )

# for first figure:
# make axis labels nice
for i in range(ax.shape[0]):
    ax[i, 0].set_ylabel(r"$f_\mathrm{orbit}$")
for i in range(ax.shape[1]):
    ax[1, i].set_xlabel(r"$r/\mathrm{kpc}$")
for axi, label in zip(ax.flat, labels):
    axi.text(0.05, 0.86, label, ha="left", va="center", transform=axi.transAxes)

# add the colour bar in the top right subplot, hiding that subplot
vkcols.make_cbar(ax[0, -1], pad=-1.075, fraction=0.5, aspect=10)
ax[0, 3].set_visible(False)

# for second figure
for i in range(ax2.shape[0]):
    ax2[i, 0].set_ylabel(r"$f_\mathrm{orbit}$")
for i in range(ax2.shape[1]):
    ax2[-1, i].set_xlabel(r"$r/\mathrm{kpc}$")
bbox = ax2[-1, -1].get_position()
fig2.legend(
    *ax2[0, 0].get_legend_handles_labels(),
    loc="center left",
    bbox_to_anchor=(bbox.x0 + bbox.width / 4, bbox.y0 + bbox.height / 4),
)
ax2[-1, -1].axis("off")

bgs.plotting.savefig(figure_config.fig_path("orbits.pdf"), fig=fig, force_ext=True)
bgs.plotting.savefig(figure_config.fig_path("orbits2.pdf"), fig=fig2, force_ext=True)
