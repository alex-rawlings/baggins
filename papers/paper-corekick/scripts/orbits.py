import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import gadgetorbits as go
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
    choices=cmf.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

# create logger
SL = cmf.setup_logger("script", args.verbosity)


orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and d.name != "fast"
]
orbitfilebases.sort()


mergemask = [
    6,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    0,
    5,
    6,
    0,
    0,
    4,
]
labels = [
    r"$\pi\mathrm{-box}$",
    r"$\mathrm{boxlet}$",
    r"$x\mathrm{-tube}$",
    "",
    r"$z\mathrm{-tube}$",
    r"$\mathrm{Keplerian}$",
    r"$\mathrm{irregular}$",
    r"$\mathrm{unclassified}$",
]


def radial_frequency(ofb, minrad=0.2, maxrad=30.0, nbin=10, returnextra=False):
    orbitcl = cmf.utils.get_files_in_dir(ofb, ext=".cl", recursive=True)[0]
    SL.info(f"Reading: {orbitcl}")
    (
        orbitids,
        classids,
        rad,
        rot_dir,
        energy,
        denergy,
        inttime,
        b92class,
        pericenter,
        apocenter,
        meanposrad,
        minangmom,
    ) = go.loadorbits(orbitcl, mergemask=mergemask, addextrainfo=True)
    radbins = np.geomspace(minrad, maxrad, nbin + 1)
    meanrads = cmf.mathematics.get_histogram_bin_centres(radbins)
    possibleclasses = np.arange(np.max(classids) + 1).astype(int)
    classfrequency = np.zeros((nbin, len(possibleclasses)))
    rad_len = []
    for i in np.arange(nbin) + 1:
        radcond = np.logical_and(radbins[i - 1] < rad, rad < radbins[i])
        radclassids = classids[radcond]
        rad_len.append(float(len(radclassids)))

        if rad_len[-1] > 0:
            for cl in possibleclasses:
                classfrequency[i - 1, cl] = (
                    float(len(radclassids[radclassids == cl])) / rad_len[-1]
                )
        else:
            SL.debug("Warning: no particles in current radial bin")
    rad_len = np.array(rad_len)

    if returnextra:
        return (
            meanrads,
            classfrequency,
            rad_len,
            b92class,
            pericenter,
            apocenter,
            minangmom,
        )
    else:
        return meanrads, classfrequency, rad_len


# figure 1: plots of different orbital families
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.set_figwidth(2 * fig.get_figwidth())
cmapper, sm = cmf.plotting.create_normed_colours(0, 900, cmap="custom_Blues")


# figure 2: plots of different kick velocities
fig2, ax2 = plt.subplots(4, 4, sharex=True, sharey=True)
fig2.set_figwidth(2 * fig2.get_figwidth())
fig2.set_figheight(2 * fig2.get_figheight())

for axj, orbitfilebase in zip(ax2.flat, orbitfilebases):
    try:
        meanrads, classfrequency, rad_len = radial_frequency(orbitfilebase)
    except:  # noqa
        # ongoing analysis
        continue
    vkick = float(orbitfilebase.split("/")[-1].split("-")[-1])
    cfi = 0
    for i, axi in enumerate(ax.flat):
        if i == 3:
            continue
        axi.semilogx(
            meanrads, classfrequency[:, cfi], label=vkick, c=cmapper(vkick), ls="-"
        )
        axj.semilogx(meanrads, classfrequency[:, cfi], label=labels[i])
        cfi += 1
    axj.text(
        0.05,
        0.9,
        f"${vkick:.0f}\, \mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
        ha="left",
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
    axi.text(0.05, 0.9, label, ha="left", va="center", transform=axi.transAxes)

# add the colour bar in the top right subplot, hiding that subplot
plt.colorbar(
    sm,
    ax=ax[0, -1],
    pad=-1.075,
    fraction=0.5,
    aspect=10,
    label=r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$",
)
ax[0, 3].set_visible(False)


# for second figure
for i in range(ax2.shape[0]):
    ax2[i, 0].set_ylabel(r"$f_\mathrm{orbit}$")
for i in range(ax2.shape[1]):
    ax2[-1, i].set_xlabel(r"$r/\mathrm{kpc}$")
fig2.subplots_adjust(bottom=0.16, top=0.98)
ax2[-1, 1].legend(loc="upper center", bbox_to_anchor=(1.1, -0.45), ncol=len(labels))


cmf.plotting.savefig(figure_config.fig_path("orbits.pdf"), fig=fig, force_ext=True)
cmf.plotting.savefig(figure_config.fig_path("orbits2.pdf"), fig=fig2, force_ext=True)
plt.show()
