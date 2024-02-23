import os.path
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf

parser = cmf.utils.argparse_for_initialise(
    description="View the initial conditions or snapshot."
)
parser.add_argument(
    "-s",
    "--snap",
    help="view an arbitrary snapshot",
    action="store_true",
    dest="snapview",
)
parser.add_argument(
    "-V", "--View", help="view the snapshot", action="store_true", dest="view"
)
parser.add_argument(
    "-o",
    "--orientate",
    help="orientate the galaxy",
    dest="orientate",
    choices=["ri", "L"],
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
    choices=cmf.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

print("\nRunning view_gal.py\n")

SL = cmf.setup_logger("script", args.verbosity)

if args.snapview:
    # load the snapshot
    SL.info("Reading from a user-defined snapshot...")
    snap = pygad.Snapshot(args.paramFile, physical=True)
else:
    # get the parameter file
    SL.info("Reading from a parameter file...")
    pfv = cmf.utils.read_parameters(args.paramFile)
    fig_loc = os.path.join(pfv.saveLocation, pfv.galaxyName, pfv.figureLocation)
    snap = pygad.Snapshot(
        os.path.join(pfv.saveLocation, pfv.galaxyName, f"{pfv.galaxyName}.hdf5"),
        physical=True,
    )

SL.debug(f"Available families for this snapshot: {snap.families()}")

if args.orientate is not None:
    if args.orientate == "ri":
        orientate_snap = "red I"
    else:
        orientate_snap = "L"
else:
    orientate_snap = None

extent = dict(
    stars={"xz": args.starextent, "xy": args.starextent},
    dm={"xz": args.haloextent, "xy": args.haloextent},
)
fig, ax = cmf.plotting.plot_galaxies_with_pygad(
    snap, extent=extent, orientate=orientate_snap, overplot_bhs=True
)
if not args.snapview:
    cmf.plotting.savefig(os.path.join(fig_loc, pfv.galaxyName))
if args.view:
    plt.show()
