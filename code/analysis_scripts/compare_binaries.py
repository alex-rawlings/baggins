import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters for several runs", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
parser.add_argument("-s", "--save", action="store_true", dest="save", help="save figure")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.CustomLogger("script", args.verbosity)

ketjufiles = cmf.utils.get_ketjubhs_in_dir(args.path)
ax = None
myr = ketjugw.units.yr * 1e6

for i, k in enumerate(ketjufiles):
    SL.logger.info(f"Reading: {k}")
    bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    if args.mask is not None:
        mask1 = bh1.t/myr < args.mask
        mask2 = bh2.t/myr < args.mask
        op = ketjugw.orbital_parameters(bh1[mask1], bh2[mask2])
    else:
        op = ketjugw.orbital_parameters(bh1, bh2)
    ax = cmf.plotting.binary_param_plot(op, ax=ax, label=f"{k.split('/')[-3]}")
ax[0].legend()
if args.save:
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"merger/compare_binaries.png"))
plt.show()
