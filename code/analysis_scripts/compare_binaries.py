import argparse
import os.path
from datetime import datetime
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters for several runs", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-o", "--orbits", action="store_true", dest="orbits", help="plot binary orbits")
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
    fig2, ax2 = plt.subplots(1,2,sharex="all")
    ax2[0].set_xlabel("x/kpc")
    ax2[0].set_ylabel("z/kpc")
    ax2[1].set_xlabel("x/kpc")
    ax2[1].set_xlabel("y/kpc")

myr = cmf.general.units.Myr
kpc = cmf.general.units.kpc
cols = cmf.plotting.mplColours()
linestyles = cmf.plotting.mplLines()
num_dirs = len(ketju_dirs)
total_sim_count = 0
SL.logger.debug(f"We will be plotting {num_dirs} different families...")

for j, d in enumerate(ketju_dirs):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(d)
    line_count = 0
    bound_bhs_present = False
    for i, k in enumerate(ketju_files):
        SL.logger.debug(f"Reading: {k}")
        try:
            bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
            bound_bhs_present = True
        except:
            SL.logger.warning(f"No binaries found in: {k} --> skipping...")
            continue
        if args.mask is not None:
            bh1 = bh1[bh1.t/myr < args.mask]
            bh2 = bh2[bh2.t/myr < args.mask]
        op = ketjugw.orbital_parameters(bh1, bh2)
        if num_dirs == 1:
            ax = cmf.plotting.binary_param_plot(op, ax=ax, label=f"{k.split('/')[-3]}", ls=linestyles[line_count//len(cols)])
            if args.orbits:
                for bh in (bh1, bh2):
                    l = ax2[0].plot((bh.x[:,0]-op["x_CM"][:,0])/kpc, (bh.x[:,2]-op["x_CM"][:,2])/kpc, alpha=0.7)
                    ax2[1].plot((bh.x[:,0]-op["x_CM"][:,0])/kpc, (bh.x[:,1]-op["x_CM"][:,1])/kpc, c=l[0].get_color(), alpha=0.7)
        else:
            ax = cmf.plotting.binary_param_plot(op, ax=ax, label=(labels[j] if i==0 else ""), c=cols[j], alpha=0.6, markevery=1000, ls=linestyles[line_count//len(cols)])
            if args.orbits:
                for bh in (bh1, bh2):
                    ax2[0].plot((bh.x[:,0]-op["x_CM"][:,0])/kpc, (bh.x[:,2]-op["x_CM"][:,2])/kpc, alpha=0.7, c=cols[j])
                    ax2[1].plot((bh.x[:,0]-op["x_CM"][:,0])/kpc, (bh.x[:,1]-op["x_CM"][:,1])/kpc, c=cols[j], alpha=0.7)
        line_count += 1
        total_sim_count += 1
    if not bound_bhs_present:
        SL.logger.warning(f"No bound BHs present in {d}")

try:
    #if total_sim_count < 10:
    ax[0].legend(loc="upper right", **legend_kwargs)
    ax[0].set_xscale("log")
    if args.save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries_{now}.png"), save_kwargs=fig_kwargs)
    plt.show()
except (IndexError, TypeError):
    SL.logger.error("No bound BHs found!")
