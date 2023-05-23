import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cm_functions as cmf
import figure_config
from figure_config import plotter


# path to processed data files
e90_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.279"
e99_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_4M-D_4M-3.720-0.028"

# some analysis parameters
analysis_params = cmf.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")


def inverse_e(e, zero=1e-4):
    """
    Helper function to transform eccentricity to 1-e

    Parameters
    ----------
    e : float, array-like
        eccentricity values
    zero : float, optional
        how to represent 0, by default 1e-4

    Returns
    -------
    array-like
        converted eccentricity values
    """
    res = np.atleast_1d(1-e)
    res[res<0] = zero
    return res


# initialise the figure
fig, ax = plt.subplots(1,2,
                    figsize=np.array(rcParams["figure.figsize"])*np.array([2,1]), 
                    sharex="all", sharey="all"
                    )

for axi in ax: axi.set_xlabel(r"$t'/\mathrm{Myr}$")
ax[0].set_ylabel(r"$1-e$")

# zoom in for right panel
axins = ax[1].inset_axes([0.3, 0.1, 0.6, 0.3])


for i, (data_path, axi, title) in enumerate(zip((e90_data_path, e99_data_path), ax, (r"$e_0=0.90$", r"$e_0=0.99$"))):
    # set ylimits
    axi.set_ylim(1e-3, 1)
    axi.set_yscale("log")
    axi.set_title(title)

    # extract data
    HMQ_files = cmf.utils.get_files_in_dir(data_path)
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)

    # plot the time series for each run
    for t, ecc in zip(km.obs["t"], km.obs["e"]):
        plotter.plot(t-t[0], inverse_e(ecc), ax=axi)
        if i==1:
            plotter.plot(t-t[0], inverse_e(ecc), ax=axins)
    # plot the mean of each time series' mean
    eccs = np.array([np.nanmean(ecc) for ecc in km.obs["e"]])
    mean_ecc = np.nanmean(eccs)
    axi.axhline(inverse_e(mean_ecc), c="k", ls="--")
    # plot the SD of eacg time series' mean
    sd_ecc = np.nanstd(eccs)
    axi.axhspan(inverse_e(mean_ecc-sd_ecc), inverse_e(mean_ecc+sd_ecc), fc="k", ec="k", alpha=0.3)

axins.set_xlim(0.08, 0.12)
axins.set_ylim(0.16, 0.21)
axins.set_xticklabels([])
axins.set_xticks([])
ax[1].indicate_inset_zoom(axins, edgecolor="k")


cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)
plt.show()