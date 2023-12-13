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
parser.add_argument("-i", "--interp", action="store_true", dest="interp", help="interpolate BH data if needed")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-o", "--orbits", action="store_true", dest="orbits", help="plot binary orbits")
parser.add_argument("-logt", action="store_true", dest="logt", help="log time axis")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories to compare")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.setup_logger("script", args.verbosity)

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
    SL.debug(f"Directories are: {ketju_dirs}")
    labels = cmf.general.get_unique_path_part(ketju_dirs)
    SL.debug(f"Labels are: {labels}")

ax = None
if args.orbits:
    fig2, ax2 = plt.subplots(2,2,sharex="row")
    for i, (s, u) in enumerate(zip(" v", ("kpc", "[km/s]"))):
        ax2[i,0].set_xlabel(f"{s}x/{u}".lstrip())
        ax2[i,0].set_ylabel(f"{s}z/{u}".lstrip())
        ax2[i,1].set_xlabel(f"{s}x/{u}".lstrip())
        ax2[i,1].set_ylabel(f"{s}y/{u}".lstrip())
else:
    ax2 = None

myr = cmf.general.units.Myr
kpc = cmf.general.units.kpc
cols = cmf.plotting.mplColours()
linestyles = cmf.plotting.mplLines()
num_dirs = len(ketju_dirs)
total_sim_count = 0
SL.debug(f"We will be plotting {num_dirs} different families...")


@dask.delayed
def dask_data_loader(kf):
    """Dask helper to load data and calculate quantities"""
    SL.debug(f"Reading: {kf}")
    try:
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf, interp=args.interp)
        if merged.merged:
            SL.info(f"Merger in {kf}")
            SL.info(merged)
    except:
        SL.warning(f"No binaries found in: {kf} --> skipping...")
        return [None, None, None, None]
    if args.t0 is None:
        toffset = 0
    else:
        toffset = args.t0 - bh1.t[0]/myr
    if args.mask is not None:
        bh1 = bh1[bh1.t/myr+toffset < args.mask]
        bh2 = bh2[bh2.t/myr+toffset < args.mask]
    op = ketjugw.orbital_parameters(bh1, bh2)
    return [bh1, bh2, op, toffset]


def plotter(d, ax, ax2, j):
    """Serial plotting"""
    kfs = cmf.utils.get_ketjubhs_in_dir(d)
    results = []
    line_count = 0
    for i, k in enumerate(kfs):
        results.append(dask_data_loader(k))
    results = dask.compute(*results)
    for i, (r, kf) in enumerate(zip(results, kfs)):
        bh1, bh2 = r[0], r[1]
        op = r[2]
        toffset = r[3]
        if any([x is None for x in r]):
            continue
        def _orbit_plotter(bh1, bh2, c):
            pos = (bh1.x - bh2.x)/kpc
            vel = (bh1.v - bh2.v)/ketjugw.units.km_per_s
            for axidx, q in enumerate((pos, vel)):
                ax2[axidx,0].plot(q[:,0], q[:,2], c=c, alpha=0.7, markevery=[-1], marker="o")
                ax2[axidx,1].plot(q[:,0], q[:,1], c=c, alpha=0.7, markevery=[-1], marker="o")
        if num_dirs == 1:
            cmf.plotting.binary_param_plot(op, ax=ax, label=f"{kf.split('/')[-3]}", ls=linestyles[line_count//len(cols)], toffset=toffset)
            if args.orbits:
                c = cols[i%len(cols)]
                _orbit_plotter(bh1, bh2, c)
        else:
            cmf.plotting.binary_param_plot(op, ax=ax, label=(labels[j] if i==0 else ""), c=cols[j], alpha=0.6, markevery=1000, ls=linestyles[line_count//len(cols)], toffset=toffset)
            if args.orbits:
                c = cols[j%len(cols)]
                _orbit_plotter(bh1, bh2 ,c)
        line_count += 1
    return line_count

for kd in ketju_dirs:
    kfs = cmf.utils.get_ketjubhs_in_dir(kd)
    for kf in kfs:
        bh1, bh2, merged = cmf.analysis.get_bh_particles(kf)
        plt.plot(ketjugw.orbital_energy(bh1, bh2))
plt.show()
quit()

# initialise the plot
ax = cmf.plotting.binary_param_plot({"t":np.nan, "a_R":np.nan, "e_t":np.nan}, ax)
for axi in ax:
    axi.set_prop_cycle(None)

for j, d in enumerate(ketju_dirs):
    try:
        assert os.path.exists(d)
    except AssertionError:
        SL.exception(f"Path {d} does not exist!", exc_info=True)
        raise
    bound_bhs_present = False
    lc = plotter(d, ax, ax2, j)
    total_sim_count += lc
    bound_bhs_present = bool(lc) or bound_bhs_present
    if not bound_bhs_present:
        SL.warning(f"No bound BHs present in {d}")

try:
    fig = ax[0].get_figure()
    try:
        ax[0].legend(loc="best", ncol=total_sim_count//5, columnspacing=1)
    except:
        ax[0].legend()
    if args.logt: ax[0].set_xscale("log")
    if args.save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries_{now}.png"), fig=fig, save_kwargs=fig_kwargs)
        if args.orbits:
            cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries_{now}_orbit.png"), fig=fig2, save_kwargs=fig_kwargs)
    plt.show()
except (IndexError, TypeError):
    SL.error("No bound BHs found!")
