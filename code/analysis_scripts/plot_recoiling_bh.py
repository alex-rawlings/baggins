import argparse
import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ketjugw.units import km_per_s
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Plot orbit of recoiling SMBH",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="simulation output path", dest="path")
parser.add_argument(
    "-m",
    "--masking",
    type=float,
    help="mask to times less than this (Myr)",
    default=None,
    dest="mask",
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    action="append",
    default=[],
    dest="extra_dirs",
    help="other directories to compare",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

output_paths = [args.path]
if args.extra_dirs is not None:
    output_paths.extend(args.extra_dirs)

fig, ax = plt.subplots(1, 2, sharex="all")

for j, p in enumerate(output_paths):
    SL.info(f"Doing {p}")
    kf = bgs.utils.get_ketjubhs_in_dir(p)[0]

    bh = bgs.analysis.get_bh_after_merger(kf)
    bh.x /= bgs.general.units.kpc
    bh.v /= km_per_s
    vkick = np.linalg.norm(bh.v[0, :])
    SL.debug(f"BH recoils with {vkick:.3e} km/s")
    if args.mask is not None:
        bh = bh[bh.t / bgs.general.units.Myr < args.mask]

    for i, (axi, lab) in enumerate(zip(ax, "yz"), start=1):
        axi.plot(
            bh.x[:, 0],
            bh.x[:, i],
            markevery=[-1],
            marker="o",
            label=f"{vkick:.1f}" if i == 1 else "",
        )
        if j == 0:
            axi.set_xlabel("x/kpc")
            axi.set_ylabel(f"{lab}/kpc")

fig.legend(loc="outside upper center", ncols=4, title="Recoil velocity [km/s]")
now = datetime.now().strftime("%Y%m%d_%H%M%S")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"merger/recoil_trajectory_{now}.png"))
