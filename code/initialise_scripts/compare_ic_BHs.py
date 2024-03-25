import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


parser = argparse.ArgumentParser(
    "Compare BH positions and velocities between different ICs", allow_abbrev=False
)
parser.add_argument(type=str, help="directory of merger ICs", dest="path")
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


ic_dirs = []
ic_dirs.append(args.path)
if args.extra_dirs:
    ic_dirs.extend(args.extra_dirs)
    SL.debug(f"Directories are: {ic_dirs}")
    labels = bgs.general.get_unique_path_part(ic_dirs)
    SL.debug(f"Labels are: {labels}")
else:
    labels = [os.path.basename(args.path)]


fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title("BH 1")
ax[0, 1].set_title("BH 2")
for i in range(2):
    ax[0, i].set_xlabel(r"$x/\mathrm{kpc}$")
    ax[0, i].set_ylabel(r"$z/\mathrm{kpc}$")
    ax[1, i].set_xlabel(r"$v_x/\mathrm{km/s}$")
    ax[1, i].set_xlabel(r"$v_z/\mathrm{km/s}$")

for d, label in zip(ic_dirs, labels):
    ic_files = bgs.utils.get_files_in_dir(d, recursive=True)
    for f in ic_files:
        if "output" in f:
            continue
        SL.debug(f"Reading {f}")
        snap = pygad.Snapshot(f)
        for i, bhid in enumerate(snap.bh["ID"]):
            id_mask = pygad.IDMask(bhid)
            ax[0, i].plot(
                snap.bh[id_mask]["pos"][0, 0],
                snap.bh[id_mask]["pos"][0, 2],
                marker="o",
                ls="",
                label=label,
            )
            ax[1, i].plot(
                snap.bh[id_mask]["vel"][0, 0],
                snap.bh[id_mask]["vel"][0, 2],
                marker="o",
                ls="",
            )
for axi in np.concatenate(ax).flatten():
    axi.ticklabel_format(useOffset=False)
plt.show()
