import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ketjugw
import cm_functions as cmf
import figure_config


# path to processed data files
e90_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.279"
e99_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_4M-D_4M-3.720-0.028"

# path to raw data
e90_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/500K"
e99_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/4M"

# some analysis parameters
analysis_params = cmf.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")


def t_shift(t):
    """
    Apply a time shift

    Parameters
    ----------
    t : array-like
        times

    Returns
    -------
    : array-like
        shifted times
    """
    return t - np.nanmean(t)


# initialise the figure
fig, axdict = plt.subplot_mosaic(
                                """
                                AAAA
                                BBCC
                                """,
                                figsize=(6,5)
                                )

for k in axdict.keys():
    axdict[k].set_yscale("eccentricity")
axdict["B"].sharex(axdict["C"])
axdict["B"].sharey(axdict["C"])

# plot the full eccentricity data for some runs
axdict["A"].set_xlabel(r"$t/\mathrm{Myr}$")
axdict["A"].set_ylabel(r"$e$")
axdict["A"].set_xscale("log")
for suite, marker in zip((e90_data_raw, e99_data_raw), figure_config.marker_cycle.by_key()["marker"]):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(suite)
    for i, kf in enumerate(ketju_files):
        if i > 2: break
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf)
        op = ketjugw.orbital_parameters(bh1, bh2)
        axdict["A"].plot(op["t"]/cmf.general.units.Myr, op["e_t"], marker=marker, markevery=[-1], zorder=10)


for axi in (axdict["B"], axdict["C"]): 
    axi.set_xlabel(r"$t'/\mathrm{Myr}$")
    axi.set_ylabel(r"$e$")


for i, (data_path, axi, label) in enumerate(zip((e90_data_path, e99_data_path), (axdict["B"], axdict["C"]), (r"$e_0=0.90$", r"$e_0=0.99$"))):
    # extract data
    HMQ_files = cmf.utils.get_files_in_dir(data_path)
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)
    mean_t_h = np.mean([np.mean(t) for t in km.obs["t"]])
    axdict["A"].axvline(mean_t_h, c="silver")
    axdict["A"].annotate(label, (mean_t_h, 1.5e-3), xytext=(mean_t_h*1.02,0.4), horizontalalignment="left", verticalalignment="bottom")
    axi.set_title(label)


    # plot the time series for each run
    for t, ecc in zip(km.obs["t"], km.obs["e"]):
        axi.plot(t_shift(t), ecc)
    # plot the mean of each time series' mean
    eccs = np.array([np.nanmean(ecc) for ecc in km.obs["e"]])
    mean_ecc = np.nanmean(eccs)
    axi.axhline(mean_ecc, c="k", ls="--")
    # plot the SD of each time series' mean
    sd_ecc = np.nanstd(eccs)
    axi.axhspan(mean_ecc-2.5*sd_ecc, mean_ecc+2.5*sd_ecc, alpha=0.2)

cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)