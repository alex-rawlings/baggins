import argparse
import os.path
import baggins as bgs

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Make IFU map",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    help="snapshot directory",
    type=str,
    dest="snapdir",
    default=None,
)
parser.add_argument(
    "-f",
    "--family",
    dest="family",
    help="particle family to map",
    default="stars",
    type=str,
)
parser.add_argument(
    "--xy", dest="axes", help="position axes", type=int, nargs=2, default=[0, 1]
)
parser.add_argument(
    "-m", "--moment", type=int, help="moment to fit to", dest="moment", default=4
)
parser.add_argument(
    "-z", "--redshift", type=float, help="redshift", dest="redshift", default=0
)
parser.add_argument("--SN", type=int, help="signal-noise ratio", dest="SN", default=200)
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


# will use MUSE, but can be switched for other instruments as API is the same
ifu = bgs.analysis.MUSE_NFM(z=args.redshift)
SL.info(ifu)

# get the list of snapshots we want to do
snap_gen = bgs.analysis.SnapshotIterator(snapdir=args.snapdir)
snap_gen.limit_to_snaps(0, -1)  # let's do the first and last snaps
for i, t, snap in snap_gen.make_generator(hide_prog=True):
    # create the mock observation -> this doesn't plot anything yet
    ifu.make_observation(
        snap=getattr(snap, args.family),
        xaxis=args.axes[0],
        yaxis=args.axes[1],
        part_per_bin=args.SN,
        moment=args.moment,
    )
    # option to save data with
    # bgs.utils.save_data(ifu.voronoi.dump_to_dict(), "/path/to/save/dir")
    ifu.voronoi.plot_kinematic_maps(cbar="inset")
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"ifu_t{t:.3f}.png"))
