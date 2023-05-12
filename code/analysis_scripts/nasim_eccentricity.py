import argparse
import os.path
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import matplotlib_fname
import cm_functions as cmf



parser = argparse.ArgumentParser(description="Perform the eccentricity dispersion calculation in Nasim et al. 2020", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory of HMQ files", dest="path")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories of HMQ files to compare")
parser.add_argument("-g", "--groups", choices=["e", "res"], help="Data groups", default="e", dest="groups")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbosity)

if args.publish:
    cmf.plotting.set_publishing_style()
    legend_kwargs = {"ncol":2, "fontsize":"x-small"}
else:
    legend_kwargs = {}

hmq_dirs = []
hmq_dirs.append(args.path)
if args.extra_dirs:
    hmq_dirs.extend(args.extra_dirs)
    SL.logger.debug(f"Directories are: {hmq_dirs}")

# read in the analysis parameters
analysis_params = cmf.utils.read_parameters(args.apf)

fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$N_\star$")
ax.set_ylabel(r"$\sigma_e$")
ax.set_xscale("log")
ax.set_yscale("log")

# list to store the values to plot
e_ini = []
mass_res = []
sigma_e = []
sim_count = 0


for d in hmq_dirs:
    # we can hack into the Kepler HM classes to extract the data
    HMQ_files = cmf.utils.get_files_in_dir(d)
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)
    sigma_e.append(np.nanstd(
        [np.nanmean(ecc) for ecc in km.obs["e"]]
    ))
    e_ini.extend(km.obs["e_ini"])
    min_bh_mass = min(min(km.obs["mass1"]), min(km.obs["mass2"]))
    try:
        assert np.allclose(np.diff(np.concatenate(km.obs["star_mass"])), np.zeros(km.num_groups-1))
    except AssertionError:
        SL.logger.exception(f"Non-unique stellar masses! A fair comparison cannot be made. Stellar masses are {km.obs['star_mass']}", exc_info=True)
        raise
    mass_res.append(min_bh_mass/km.obs["star_mass"][0])
    sim_count += km.num_groups

e_ini = np.concatenate(e_ini)
mass_res = np.concatenate(mass_res)

if args.groups == "e":
    try:
        assert np.allclose(np.diff(mass_res), np.zeros(sim_count-1))
    except AssertionError:
        SL.logger.exception(f"Mass resolution must be constant when varying initial eccentricity!", exc_info=True)
        raise
    x = e_ini
    fig_prefix = f"res-{mass_res[0]:.1e}"
else:
    try:
        assert np.allclose(np.diff(e_ini), np.zeros(sim_count-1))
    except AssertionError:
        SL.logger.exception(f"Initial eccentricity must be constant when varying mass resolution!", exc_info=True)
        raise
    x = mass_res
    fig_prefix = f"e0-{e_ini[0]:.3f}"

ax.scatter(x, sigma_e, lw=0.5, ec="k", label="Sims.", zorder=10)

# add the Nasim line if plotting resolution on x axis
if args.groups == "res":
    xseq = np.geomspace(0.9*min(x), 1.1*max(x))
    nasim_line_1 = lambda n,k: k/np.sqrt(n+k**2)
    nasim_line_2 = lambda n,k: k/np.sqrt(n**2+k**2)
    for nl, lab, ls in zip((nasim_line_1, nasim_line_2), (r"$\propto 1/\sqrt{N}$", r"$\propto 1/N$"), cmf.plotting.mplLines()):
        popt, pcov = scipy.optimize.curve_fit(nl, x, sigma_e)
        ax.plot(xseq, nl(xseq, *popt), c="k", label=lab, ls=ls)
cmf.plotting.nice_log10_scale(ax, "x")
ax.legend()

if args.save:
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/nasim_scatter_{fig_prefix}.png"))

plt.show()
