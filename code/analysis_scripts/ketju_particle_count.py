import argparse
import os.path
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Quickly check number of Ketju particles",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="path to directory", dest="path")
parser.add_argument(
    "-s", "--save", action="store_true", dest="save", help="save figure"
)
parser.add_argument(
    "-t", "--thin", type=int, dest="thin", default=100, help="thin data"
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

ketju_dirs = []
ketju_dirs.append(args.path)
if args.extra_dirs:
    ketju_dirs.extend(args.extra_dirs)
    SL.debug(f"Directories are: {ketju_dirs}")
    labels = bgs.general.get_unique_path_part(ketju_dirs)
    SL.debug(f"Labels are: {labels}")

num_dirs = len(ketju_dirs)
total_sim_count = 0
lstyles = bgs.plotting.mplLines()
SL.debug(f"We will be plotting {num_dirs} different families...")


fig, ax = plt.subplots(1, 1)
ax.set_xlabel(f"Index/{args.thin}")
ax.set_ylabel("Number of Ketju particles")
for i, d in enumerate(ketju_dirs):
    try:
        assert os.path.exists(d)
    except AssertionError:
        SL.exception(f"Path {d} does not exist!", exc_info=True)
        raise
    kfiles = bgs.utils.get_ketjubhs_in_dir(d)
    for kf in kfiles:
        with h5py.File(kf, "r") as f:
            for j, bh in enumerate(f["/BHs"].values()):
                # TODO how to handle BHs that form at different times?
                if j == 0:
                    _line = ax.plot(
                        bh["num_particles_in_region"][:][:: args.thin], ls=lstyles[j]
                    )
                else:
                    ax.plot(
                        bh["num_particles_in_region"][:][:: args.thin],
                        c=_line[0].get_color(),
                        ls=lstyles[j],
                    )
if ax.get_ylim()[1] > 100:
    ax.set_yscale("log")
if args.save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    bgs.plotting.savefig(
        os.path.join(bgs.FIGDIR, f"run_diagnostics/ketju_particles{now}.png"),
        fig=fig,
    )
plt.show()
