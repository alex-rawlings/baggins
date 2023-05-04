import argparse
import numpy as np
import matplotlib.pyplot as plt
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
x = []
sigma_e = []


for d in hmq_dirs:
    # we can hack into the Kepler HM classes to extract the data
    # TODO may need to fix place holder figname parameter
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(d, analysis_params)
    sigma_e.append(np.nanstd(
        [np.nanmean(ecc) for ecc in km.obs["e"]]
    ))
    if args.groups == "e":
        x.append(km.obs["e_ini"])
    else:
        min_bh_mass = min(min(km.obs["mass1"]), min(km.obs["mass2"]))
        try:
            assert np.allclose(np.diff([m for m in km.obs["star_mass"]]))
        except AssertionError:
            SL.logger.exception(f"Non-unique stellar masses! A fair comparison cannot be made. Stellar masses are {km.obs['star_mass']}", exc_info=True)
            raise
        x.append(min_bh_mass/km.obs["star_mass"][0])

ax.scatter(x, sigma_e, lw=0.5, ec="k")

plt.show()
