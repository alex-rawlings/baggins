import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import dask
from tqdm.dask import TqdmCallback
import baggins as bgs
from v_anisotropy import main
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot beta profile and box-tube ratio",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract orbit data", action="store_true", dest="extract"
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


SL = bgs.setup_logger("script", args.verbosity)

@dask.delayed
def dask_extractor(orbitcl, vkey, mergemask, N, rng):
    """
    Helper function to extract box tube ratio using dask parallelism

    Parameters
    ----------
    orbitcl : str, path-like
        orbit classifiction file
    vkey : str
        kick velocity key to read core data
    mergemask : bgs.analysis.MergeMask
        how to merge different orbits
    N : int
        number of samples
    rng : np.random.Generator
        random number generator

    Returns
    -------
    ratio_med : np.array
        median values of box tube ratio
    ratio_err : np.array
        lower and upper errors of box tube ratio
    """
    # construct the classifier
    classifier = bgs.analysis.OrbitClassifier(orbitcl, mergemask=mergemask)

    # array to the hold ratios
    ratios = np.full((N, 2), np.nan)
    rbs = np.full(N, np.nan)
    Nbox = np.full(N, np.nan)
    Ntube = np.full(N, np.nan)
    for i in range(N):
        rbs[i] = rng.choice(core_data["rb"][vkey].flatten())
        ratios[i, :] = classifier.box_tube_ratio([0, rbs[i], 1000], box_names=["box"], tube_names=["tube"])
        Nbox[i] = classifier.family_size_in_radius("box", rbs[i])
        Ntube[i] = classifier.family_size_in_radius("tube", rbs[i])
    ratio_med, ratio_err = bgs.mathematics.quantiles_relative_to_median(ratios, axis=0)
    ratio_r = classifier.box_tube_ratio(box_names=["box"], tube_names=["tube"])
    return ratio_med, ratio_err, classifier.meanrads, ratio_r, rbs, Nbox, Ntube

orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()
core_data = bgs.utils.load_data("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle")
data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/box_tube_ratio.pickle"
rng = np.random.default_rng(5488947)
N_rb_samples = 250

mergemask = bgs.analysis.MergeMask.make_box_tube_mask()

fig, ax = plt.subplots(2, 1)
fig.set_figheight(1.5 * fig.get_figheight())

'''# run Max's velocity anisotropy code
main(ax=ax[0], read_betas=1)
ax[0].set_xlabel("")'''

ax[0].set_xlabel(r"$v_{\mathrm{kick}}/\mathrm{km \, s^{-1}}$")
ax[0].set_ylabel(r"$N_\mathrm{box}/N_\mathrm{tube}$")
ax[0].axhline(1, c="k", lw=1, ls=":")
labels = [r"$r \leq r_{\mathrm{b}}$", r"$r > r_{\mathrm{b}}$"]

if args.extract:
    data = {"vkick":[], "ratio_med":[], "ratio_err":[], "radius":[], "ratio_r":[], "rb":[], "Nbox":[], "Ntube":[]}
    dask_res = []
    for j, orbitfilebase in enumerate(orbitfilebases):
        orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[0]
        try:
            vkey = f"{orbitfilebase.split('/')[-1].split('-')[-1]}"
            # test that we have a valid key
            core_data["rb"][vkey]
            vkick = float(vkey)
            # store kick velocities to data
            data["vkick"].append(vkick)
        except KeyError:
            SL.debug(f"No key for {vkey}")
            break
        SL.info(f"Reading {orbitcl}")

        # parallel delayed computation
        dask_res.append(dask_extractor(orbitcl, vkey, mergemask, N_rb_samples, rng))
    # compute
    with TqdmCallback(desc="Computing samples"):
        dask_res = dask.compute(dask_res)

    # store data
    data["ratio_med"] = [r[0] for r in dask_res[0]]
    data["ratio_err"] = [r[1] for r in dask_res[0]]
    data["radius"] = [r[2] for r in dask_res[0]]
    data["ratio_r"] = [r[3] for r in dask_res[0]]
    data["rb"] = [r[4] for r in dask_res[0]]
    data["Nbox"] = [r[5] for r in dask_res[0]]
    data["Ntube"] = [r[6] for r in dask_res[0]]
    bgs.utils.save_data(data, data_file)
else:
    data = bgs.utils.load_data(data_file)

for j, (vkick, ratio_med, ratio_err) in enumerate(zip(data["vkick"], data["ratio_med"], data["ratio_err"])):
    for i in range(2):
        ax[0].errorbar(vkick, ratio_med[i], yerr=np.atleast_2d(ratio_err[:,i]).T, fmt="o", c=bgs.plotting.mplColours()[i], label=(labels[i] if j==0 else ""), mec="k", capthick=1)

vkcols = figure_config.VkickColourMap()
'''ax[1].set_xlabel(r"$r/r_\mathrm{b}$")
ax[1].set_ylabel(r"$N_\mathrm{box}/N_\mathrm{tube}$")
ax[1].axhline(1, c="k", lw=1, ls=":")
for i, (vk, _rad, _ratior, _rb) in enumerate(zip(data["vkick"], data["radius"], data["ratio_r"], data["rb"])):
    ax[1].loglog(_rad/np.nanmedian(_rb), _ratior, ls="-", c=vkcols.get_colour(vk))
vkcols.make_cbar(ax=ax[1])'''

ax[1].set_xlabel(r"$N_\mathrm{box}$")
ax[1].set_ylabel(r"$N_\mathrm{tube}$")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].plot()
for vk, Nb, Nt in zip(data["vkick"], data["Nbox"], data["Ntube"]):
    #print(Nb.shape)
    xm, xerr = bgs.mathematics.quantiles_relative_to_median(Nb, 0.45, 0.55)
    ym, yerr = bgs.mathematics.quantiles_relative_to_median(Nt, 0.45, 0.55)
    #ax[1].errorbar(xm, ym, xerr=xerr, yerr=yerr, fmt=".", ls="", capthick=1, c=vkcols.get_colour(vk))
    ax[1].scatter(xm, ym, color=vkcols.get_colour(vk))
xlims = ax[1].get_xlim()
ylims = ax[1].get_ylim()
ax[1].plot([1e2, 1e6], [1e2, 1e6], ls=":", c="k", zorder=0.5)
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)
vkcols.make_cbar(ax=ax[1])

bgs.plotting.savefig(figure_config.fig_path("orbit_bt.pdf"), force_ext=True)
