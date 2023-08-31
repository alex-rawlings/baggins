import argparse
import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import dask
import cm_functions as cmf
import ketjugw


parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters for several runs", allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
parser.add_argument("-t0", "--time0", type=float, help="Initial time value (Myr)", default=None, dest="t0")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-o", "--orbits", action="store_true", dest="orbits", help="plot binary orbits")
parser.add_argument("-logt", action="store_true", dest="logt", help="log time axis")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories to compare")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbosity)

if args.publish:
    cmf.plotting.set_publishing_style()
    legend_kwargs = {"ncol":2, "fontsize":"x-small"}
    fig_kwargs = {"transparent":True}
else:
    legend_kwargs = {}
    fig_kwargs = {}

ketju_dirs = []
ketju_dirs.append(args.path)
if args.extra_dirs:
    ketju_dirs.extend(args.extra_dirs)
    SL.logger.debug(f"Directories are: {ketju_dirs}")
    labels = cmf.general.get_unique_path_part(ketju_dirs)
    SL.logger.debug(f"Labels are: {labels}")

ax = None
if args.orbits:
    fig2, ax2 = plt.subplots(2,2,sharex="row")
    for i, (s, u) in enumerate(zip(" v", ("kpc", "[km/s]"))):
        ax2[i,0].set_xlabel(f"{s}x/{u}".lstrip())
        ax2[i,0].set_ylabel(f"{s}z/{u}".lstrip())
        ax2[i,1].set_xlabel(f"{s}x/{u}".lstrip())
        ax2[i,1].set_xlabel(f"{s}y/{u}".lstrip())

myr = cmf.general.units.Myr
kpc = cmf.general.units.kpc
cols = cmf.plotting.mplColours()
linestyles = cmf.plotting.mplLines()
num_dirs = len(ketju_dirs)
total_sim_count = 0
SL.logger.debug(f"We will be plotting {num_dirs} different families...")

@dask.delayed
def dask_helper(kf, i, j, ax, ax2):
    SL.logger.debug(f"Reading: {kf}")
    try:
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf)
    except:
        SL.logger.warning(f"No binaries found in: {kf} --> skipping...")
        return 0
    if args.t0 is None:
        toffset = 0
    else:
        toffset = args.t0 - bh1.t[0]/myr
    if args.mask is not None:
        bh1 = bh1[bh1.t/myr+toffset < args.mask]
        bh2 = bh2[bh2.t/myr+toffset < args.mask]
    op = ketjugw.orbital_parameters(bh1, bh2)
    zorder = 0.1*(num_dirs * j + i)

    def _orbit_plotter(bh, c):
        pos = (bh.x - op["x_CM"])/kpc
        vel = (bh.v - op["v_CM"])/ketjugw.units.km_per_s
        for axidx, q in enumerate((pos, vel)):
            ax2[axidx,0].plot(q[:,0], q[:,2], c=c, alpha=0.7, markevery=[-1], marker="o", zorder=zorder)
            ax2[axidx,1].plot(q[:,0], q[:,1], c=c, alpha=0.7, markevery=[-1], marker="o", zorder=zorder)

    if num_dirs == 1:
        cmf.plotting.binary_param_plot(op, ax=ax, label=f"{kf.split('/')[-3]}", ls=linestyles[line_count//len(cols)], toffset=toffset, zorder=zorder)
        if args.orbits:
            c = cols[i%len(cols)]
            for bh in (bh1, bh2):
                _orbit_plotter(bh,c)
    else:
        cmf.plotting.binary_param_plot(op, ax=ax, label=(labels[j] if i==0 else ""), c=cols[j], alpha=0.6, markevery=1000, ls=linestyles[line_count//len(cols)], toffset=toffset, zorder=zorder)
        if args.orbits:
            c = cols[j%len(cols)]
            for bh in (bh1, bh2):
                _orbit_plotter(bh,c)
    return 1

# initialise the plot
ax = cmf.plotting.binary_param_plot({"t":np.nan, "a_R":np.nan, "e_t":np.nan}, ax)
for axi in ax:
    axi.set_prop_cycle(None)

for j, d in enumerate(ketju_dirs):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(d)
    line_count = 0
    bound_bhs_present = False
    res = []
    for i, k in enumerate(ketju_files):
        res.append(dask_helper(k,i,j, ax, ax2))
    res_sum = sum(dask.compute(res)[0])
    line_count += res_sum
    total_sim_count += res_sum
    bound_bhs_present = bool(res_sum) or bound_bhs_present
    if not bound_bhs_present:
        SL.logger.warning(f"No bound BHs present in {d}")

try:
    ax[0].legend(loc="upper right", **legend_kwargs)
    if args.logt: ax[0].set_xscale("log")
    if args.save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries_{now}.png"), save_kwargs=fig_kwargs)
    plt.show()
except (IndexError, TypeError):
    SL.logger.error("No bound BHs found!")
