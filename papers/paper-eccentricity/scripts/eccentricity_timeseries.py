import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ketjugw
import cm_functions as cmf
import figure_config
from figure_config import plotter


# path to processed data files
e90_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.279"
e99_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_4M-D_4M-3.720-0.028"

# path to raw data
e90_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/500K"
e99_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/4M"

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
'''fig, ax = plt.subplots(1,2,
                    figsize=np.array(rcParams["figure.figsize"])*np.array([2,1]), 
                    sharex="all", sharey="all"
                    )'''
fig = plt.figure(figsize=np.array(rcParams["figure.figsize"])*np.array([7/3,5/3]))
ax1 = plt.subplot(2,2,(1,2))
ax2 = plt.subplot(2,2,3, sharey=ax1)
ax3 = plt.subplot(2,2,4, sharex=ax2, sharey=ax2)

# plot the full eccentricity data for some runs
ax1.set_xlabel(r"$t/\mathrm{Myr}$")
ax1.set_ylabel(r"$1-e$")
ax1.set_xscale("log")
ax1.set_yscale("log")
for suite, ls in zip((e90_data_raw, e99_data_raw), ("-", "--")):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(suite)
    for i, kf in enumerate(ketju_files):
        if i > 2: break
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf)
        op = ketjugw.orbital_parameters(bh1, bh2)
        plotter.plot(op["t"]/cmf.general.units.Myr, inverse_e(op["e_t"]), ax=ax1, ls=ls)
    plotter.reset_line_count(ax1)


for axi in (ax2, ax3): axi.set_xlabel(r"$t'/\mathrm{Myr}$")
ax2.set_ylabel(r"$1-e$")

# zoom in for right panel
axins = ax3.inset_axes([0.3, 0.03, 0.6, 0.4])

# define a textbox
textbox_dict = {"boxstyle":"square", "fc":"w", "ec":"k"}


for i, (data_path, axi, label) in enumerate(zip((e90_data_path, e99_data_path), (ax2, ax3), (r"$e_0=0.90$", r"$e_0=0.99$"))):
    # set ylimits
    axi.set_ylim(1e-3, 1)
    axi.set_yscale("log")

    # extract data
    HMQ_files = cmf.utils.get_files_in_dir(data_path)
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)
    mean_t_h = np.mean([np.mean(t) for t in km.obs["t"]])
    ax1.axvline(mean_t_h, c="k")
    ax1.annotate(label, (mean_t_h, 1.5e-3), xytext=(mean_t_h*1.02,1.5e-3), horizontalalignment="left", verticalalignment="bottom", bbox=textbox_dict)
    axi.annotate(label, (0.02,0.97), xycoords=axi.transAxes, bbox=textbox_dict, verticalalignment="top", horizontalalignment="left", clip_on=True)


    # plot the time series for each run
    for t, ecc in zip(km.obs["t"], km.obs["e"]):
        plotter.plot(t_shift(t), inverse_e(ecc), ax=axi)
        if i==1:
            plotter.plot(t_shift(t), inverse_e(ecc), ax=axins)
    # plot the mean of each time series' mean
    eccs = np.array([np.nanmean(ecc) for ecc in km.obs["e"]])
    mean_ecc = np.nanmean(eccs)
    axi.axhline(inverse_e(mean_ecc), c="k", ls="--")
    # plot the SD of eacg time series' mean
    sd_ecc = np.nanstd(eccs)
    axi.axhspan(inverse_e(mean_ecc-sd_ecc), inverse_e(mean_ecc+sd_ecc), alpha=0.3)

axins.set_xlim(-0.03, 0.03)
axins.set_ylim(0.17, 0.21)
axins.set_xticklabels([])
axins.set_xticks([])
ax3.indicate_inset_zoom(axins, edgecolor="k")


cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)