import argparse
import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Plot deflection angle and median eccentricity of merger runs")
parser.add_argument(type=str, help="path to HMQ directory", dest="path")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other HMQ directories to compare")
parser.add_argument("-a", "--angle", type=float, dest="angle", help="minimum deflection angle", default=30)
parser.add_argument("-o", "--orbits", type=int, dest="orbits", help="number of orbits to determine eccentricity over", default=5)
parser.add_argument("-l", "--label", type=str, dest="label", help="labelling method", choices=["e", "res"], default=None)
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-sd", "--save-data", dest="save_data", type=str, default=None, help="directory to save data to")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbosity", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbosity)

if args.publish:
    cmf.plotting.set_publishing_style()
    legend_kwargs = {"ncol":2, "fontsize":"small"}
try:
    assert args.angle >=0 and args.angle <= 180
    angle_defl = args.angle * np.pi / 180
except AssertionError:
    SL.logger.exception(f"Deflection angle must be in range 0<t<180!", exc_info=True)
    raise

data_dirs = []
data_dirs.append(args.path)
if args.extra_dirs:
    data_dirs.extend(args.extra_dirs)
    SL.logger.debug(f"Directories are: {data_dirs}")
    labels = cmf.general.get_unique_path_part(data_dirs)
    SL.logger.debug(f"Labels are: {labels}")
else:
    labels = [""]


# arrays to hold data
data = dict(
    thetas = [],
    median_eccs = [],
    low_iqr_ecc = [],
    high_iqr_ecc = [],
    e_ini = [],
    mass_res = [],
    threshold_angle = args.angle
)


for j, datdir in enumerate(data_dirs):
    SL.logger.info(f"Reading from directory: {datdir}")
    HMQfiles = cmf.utils.get_files_in_dir(datdir)
    N = len(HMQfiles)

    thetas = np.full(N, np.nan)
    median_eccs = np.full(N, np.nan)
    iqr_eccs = np.full((2,N), np.nan)

    # loop through each bh file in the directory
    for i, HMQfile in enumerate(HMQfiles):
        SL.logger.debug(f"Reading file: {HMQfile}")
        hmq = cmf.analysis.HMQuantitiesData.load_from_file(HMQfile)
        if i==0 and args.label is not None:
            if args.label == "e":
                labels[j] = f"{hmq.initial_galaxy_orbit['e0']:.3f}"
                legend_title = r"$e_\mathrm{ini}$"
            else:
                labels[j] = cmf.general.represent_numeric_in_scientific(hmq.mass_resolution())
                legend_title = r"$M_\bullet/m_\star$"
        
        # save some data already
        data["e_ini"].append(hmq.initial_galaxy_orbit["e0"])
        data["mass_res"].append(hmq.mass_resolution())

        thetas[i], theta_idx = cmf.analysis.first_major_deflection_angle(hmq.prebound_deflection_angles, angle_defl)
        if theta_idx is None:
            SL.logger.warning(f"No hard scattering in file {i}, skipping")
            continue
        
        # determine the eccentricity
        try:
            _, period_idxs = cmf.analysis.find_idxs_of_n_periods(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis, hmq.binary_separation, num_periods=args.orbits)
        except AssertionError:
            SL.logger.error(f"Unable to determine hardening radius for file {i}, skipping...")
            continue
        SL.logger.debug(f"Period idxs: {period_idxs}")
        m, iqr = cmf.mathematics.quantiles_relative_to_median(hmq.eccentricity[period_idxs[0]:period_idxs[1]])
        median_eccs[i] = m
        iqr_eccs[0,i], iqr_eccs[1,i] = iqr
    # save data
    data["thetas"].extend(thetas*180/np.pi)
    data["median_eccs"].extend(median_eccs)
    data["low_iqr_ecc"].extend(iqr_eccs[0,:])
    data["high_iqr_ecc"].extend(iqr_eccs[1,:])


    plt.errorbar(thetas*180/np.pi, median_eccs, xerr=None, yerr=iqr_eccs, fmt="o", capsize=2, mec="k", mew=0.5, label=labels[j])

if args.extra_dirs:
    plt.legend(title=legend_title, **legend_kwargs)
plt.xlabel(r"$\theta\degree_\mathrm{defl}$")
plt.ylabel("$e$")
plt.title(f"$\\theta_\mathrm{{defl,min}}={args.angle:.1f}\degree$")
plt.ylim(0,1)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.save:
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"deflection_angles/deflection_angles_{now}.png"))
else:
    SL.logger.warning("Figure will not be saved!")

# save the data if desired
if args.save_data:
    try:
        dat_len = [len(v) for v in data.values() if isinstance(v, list)]
        assert np.allclose(np.diff(dat_len), np.zeros(len(dat_len)-1))
    except AssertionError:
        SL.logger.exception(f"Data must have equal length arrays, but has lengths {dat_len}!", exc_info=True)
        raise
    cmf.utils.save_data(data, os.path.join(args.save_data, f"deflection_angles_{now}.pickle"))

plt.show()
