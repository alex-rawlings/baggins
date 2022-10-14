import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters for several runs", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories to compare")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbosity)

if args.publish:
    cmf.plotting.set_publishing_style()
    legend_kwargs = {"ncol":2, "fontsize":"x-small"}
else:
    legend_kwargs = {}

ketju_dirs = []
ketju_dirs.append(args.path)
if args.extra_dirs:
    ketju_dirs.extend(args.extra_dirs)
    SL.logger.debug(f"Directories are: {ketju_dirs}")
    labels = cmf.general.get_string_unique_part(ketju_dirs)
    SL.logger.debug(f"Labels are: {labels}")

ax = None
myr = ketjugw.units.yr * 1e6
cols = cmf.plotting.mplColours()
num_dirs = len(ketju_dirs)
SL.logger.debug(f"We will be plotting {num_dirs} different families...")

for j, d in enumerate(ketju_dirs):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(d)
    for i, k in enumerate(ketju_files):
        SL.logger.debug(f"Reading: {k}")
        bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
        if args.mask is not None:
            mask1 = bh1.t/myr < args.mask
            mask2 = bh2.t/myr < args.mask
            op = ketjugw.orbital_parameters(bh1[mask1], bh2[mask2])
        else:
            op = ketjugw.orbital_parameters(bh1, bh2)
        if num_dirs == 1:
            ax = cmf.plotting.binary_param_plot(op, ax=ax, label=f"{k.split('/')[-3]}")
        else:
            ax = cmf.plotting.binary_param_plot(op, ax=ax, label=(labels[j] if i==0 else ""), c=cols[j], alpha=0.6, markevery=1000)
ax[0].legend(loc="upper right", **legend_kwargs)
ax[0].set_xscale("log")
if args.save:
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries.png"))
plt.show()
