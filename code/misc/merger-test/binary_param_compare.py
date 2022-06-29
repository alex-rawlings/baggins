import argparse
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw



parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
args = parser.parse_args()

ketjufiles = cmf.utils.get_ketjubhs_in_dir(args.path)
ax = None

for i, k in enumerate(ketjufiles):
    print(k)
    bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    if args.mask is not None:
        mask1 = bh1.t/ketjugw.units.yr < args.mask
        mask2 = bh2.t/ketjugw.units.yr < args.mask
        op = ketjugw.orbital_parameters(bh1[mask1], bh2[mask2])
    else:
        op = ketjugw.orbital_parameters(bh1, bh2)
    ax = cmf.plotting.binary_param_plot(op, ax=ax, label=f"{k.split('/')[-3]}")
ax[0].legend()
plt.show()
