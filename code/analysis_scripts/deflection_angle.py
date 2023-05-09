import argparse
import os.path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ketjugw
import pygad
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Plot deflection angle and median eccentricity of merger runs")
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories to compare")
parser.add_argument("-a", "--angle", type=float, dest="angle", help="minimum deflection angle", default=90)
parser.add_argument("-o", "--orbits", type=int, dest="orbits", help="number of orbits to determine eccentricity over", default=11)
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbosity", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbosity)

if args.publish:
    cmf.plotting.set_publishing_style()

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

# for unit conversions
myr = cmf.general.units.Myr
kpc = cmf.general.units.kpc

# arrays to hold data
thetas =  np.array([])
median_eccs = np.array([])
iqr_eccs = np.array([[],[]])
total_N = 0

for j, datdir in enumerate(data_dirs):
    bhfiles = cmf.utils.get_ketjubhs_in_dir(datdir)
    N = len(bhfiles)

    thetas = np.concatenate((thetas, np.full(N, np.nan)))
    median_eccs = np.concatenate((median_eccs, np.full(N, np.nan)))
    iqr_eccs = np.hstack((iqr_eccs, np.full((2,N), np.nan)))

    # loop through each bh file in the directory
    for i, bhfile in enumerate(bhfiles):
        bh1, bh2 = cmf.analysis.get_binary_before_bound(bhfile)
        bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)

        try:
            bh1_bound, bh2_bound, merged = cmf.analysis.get_bound_binary(bhfile)
        except IndexError:
            SL.logger.warning(f"Skipping file {i}: no bound binary formed")
            continue
        bh1_bound, bh2_bound = cmf.analysis.move_to_centre_of_mass(bh1_bound, bh2_bound)

        _, idxs, sep = cmf.analysis.find_pericentre_time(bh1, bh2, return_sep=True, prominence=0.005)

        theta_d = cmf.analysis.deflection_angle(bh1, bh2, idxs)
        thetas[total_N+i], theta_idx = cmf.analysis.first_major_deflection_angle(theta_d, angle_defl)
        if theta_idx is None:
            SL.logger.warning(f"No hard scattering in file {i}, skipping")
            continue
        
        # determine the eccentricity
        op = ketjugw.orbital_parameters(bh1_bound, bh2_bound)
        snapfiles = cmf.utils.get_snapshots_in_dir(os.path.dirname(bhfile))
        # TODO more robust determination of hardening radius? 
        snap = pygad.Snapshot(snapfiles[int(len(snapfiles)/2)], physical=True)
        rinfl = max(list(cmf.analysis.influence_radius(snap).values()))
        SL.logger.debug(f"Influence radius: {rinfl}")
        ahard = cmf.analysis.hardening_radius(snap.bh["mass"], rinfl)
        SL.logger.debug(f"Hardening radius: {ahard} kpc")
        ahard_idx = cmf.general.get_idx_in_array(ahard, op["a_R"]/kpc)
        SL.logger.debug(f"Hardening index: {ahard_idx} / {len(op['a_R'])}")
        _, period_idxs = cmf.analysis.find_idxs_of_n_periods(op["t"][ahard_idx], op["t"], cmf.mathematics.radial_separation(bh1_bound.x, bh2_bound.x), num_periods=args.orbits)
        SL.logger.debug(f"Period idxs: {period_idxs}")
        m, iqr = cmf.mathematics.quantiles_relative_to_median(op["e_t"][period_idxs[0]:period_idxs[1]])
        median_eccs[total_N+i] = m
        iqr_eccs[0,total_N+i], iqr_eccs[1,total_N+i] = iqr
    total_N += N

plt.errorbar(thetas*180/np.pi, median_eccs, xerr=None, yerr=iqr_eccs, fmt="o", capsize=2, mec="k", mew=0.5)
plt.xlabel(r"$\theta_\mathrm{defl}$")
plt.ylabel("e")
plt.ylim(0,1)

if args.save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"deflection_angles_{now}.png"))
else:
    SL.logger.warning("Figure will not be saved!")
plt.show()
