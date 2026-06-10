import argparse
import os
from datetime import datetime
import pygad
import baggins as bgs

parser = argparse.ArgumentParser(
    description="View the initial conditions or snapshot.",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    help="snapshot to view",
    type=str,
    dest="snap",
)
parser.add_argument(
    "-o",
    "--orientate",
    help="orientate the galaxy",
    dest="orientate",
    choices=["red I", "L"],
    default=None,
)
parser.add_argument(
    "-SE",
    "--StarExtent",
    type=float,
    help="extent of the stellar plot",
    dest="starextent",
    default=600,
)
parser.add_argument(
    "-HE",
    "--HaloExtent",
    type=float,
    help="extent of the dm halo plot",
    dest="haloextent",
    default=8000,
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

# load the snapshot
SL.info("Reading from a user-defined snapshot...")
snap = pygad.Snapshot(args.snap, physical=True)

SL.debug(f"Available families for this snapshot: {snap.families()}")

extent = dict(
    stars={"xz": args.starextent, "xy": args.starextent},
    dm={"xz": args.haloextent, "xy": args.haloextent},
)
fig, ax = bgs.plotting.plot_galaxies_with_pygad(
    snap, extent=extent, orientate=args.orientate, overplot_bhs=True
)
now = datetime.now().strftime(bgs.PARAMS["date_format"])
os.makedirs(os.path.join(bgs.FIGDIR, "snapshot_view"), exist_ok=True)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"snapshot_view/snap_{now}.png"))
