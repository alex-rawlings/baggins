import argparse
import os.path
import psutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import dask
import pygad
import baggins as bgs


# set up command line arguments
parser = argparse.ArgumentParser(
    description="Determine radially-dependent axis ratios for a range of snapshots",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="path to snapshots or data", dest="path")
parser.add_argument(
    "-n",
    "--num",
    dest="num",
    type=int,
    help="Number of snapshots to analyse",
    default=None,
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    help="method of determining inertia tensor",
    dest="method",
    choices=["shell", "ball"],
    default="shell",
)
parser.add_argument(
    "-f",
    "--family",
    type=str,
    help="particle family",
    dest="family",
    choices=["dm", "stars"],
    default="stars",
)
parser.add_argument(
    "-S",
    "--statistic",
    type=str,
    help="statistic",
    dest="stat",
    choices=["median", "mean", "last"],
    default="median",
)
parser.add_argument(
    "-r",
    "--radii",
    type=bgs.utils.cl_str_2_space,
    help="radii to calculate inertia tensor at",
    dest="radii",
    default=None,
)
parser.add_argument(
    "-sd",
    "--savedir",
    type=str,
    help="save directory",
    dest="savedir",
    default="pickle/triaxiality",
)
parser.add_argument(
    "-s", "--save", action="store_true", dest="save", help="save figure"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()


SL = bgs.setup_logger("script", args.verbose)
bgs.plotting.check_backend()


@dask.delayed
def dask_helper(s, r):
    """Helper function to parallelise axis ratio calculation over radial values"""
    mask = list(bgs.analysis.get_all_radial_masks(s, r, family=args.family).values())[0]
    if len(s[mask]) < 1000:
        SL.warning(f"Only {len(s[mask])} particles in {r}")
        return (np.nan, np.nan)
    rats = bgs.analysis.get_galaxy_axis_ratios(s, bin_mask=mask, family=args.family)
    del mask
    return rats


def memory_helper():
    """Print the memory being used"""
    proc = psutil.Process()
    SL.debug(f"Total memory usage (GB): {proc.memory_info().rss/2**30:.3f}")


if os.path.isfile(args.path) and os.path.splitext(args.path)[1] == ".pickle":
    data = bgs.utils.load_data(args.path)
    figname = os.path.splitext(os.path.basename(args.path))[0]
    times = data["times"]
    ratios = data["ratios"]
else:
    # analyse a new dataset
    if os.path.os.path.isfile(args.path):
        snaplist = [args.path]
    else:
        snaplist = bgs.utils.get_snapshots_in_dir(args.path)
    if args.num is not None:
        if args.num < 0:
            # assume that negative number means that many from the end
            snaplist = snaplist[args.num :]
        else:
            snaplist = snaplist[: args.num]

    # set the default radial scaling in units of Rvir
    memory_helper()
    if args.radii is None:
        N_rad = 20
        if args.family == "dm":
            args.radii = np.geomspace(0.1, 6, N_rad)
        else:
            args.radii = np.geomspace(0.001, 0.1, N_rad)
        SL.info(
            f"Using a default radial scaling of Rvir*({args.radii[0]}-{args.radii[-1]} in {len(args.radii)}) bins"
        )
    else:
        N_rad = len(args.radii)
    if args.method == "shell":
        N_rad -= 1
    # instantiate arrays
    times = np.full_like(snaplist, np.nan, dtype=float)
    ratios = dict(
        r=np.full((len(snaplist), N_rad), np.nan),
        ba=np.full((len(snaplist), N_rad), np.nan),
        ca=np.full((len(snaplist), N_rad), np.nan),
    )

    # loop through all snapshots
    for i, snapfile in enumerate(snaplist):
        SL.info(f"Reading: {snapfile}")
        memory_helper()
        snap = pygad.Snapshot(snapfile, physical=True)
        times[i] = bgs.general.convert_gadget_time(snap)

        # recentre snapshot to CoM
        xcom = bgs.analysis.get_com_of_each_galaxy(
            snap, method="ss", family=args.family
        )
        translation = pygad.Translation(list(xcom.values())[0])
        translation.apply(snap, total=True)

        # determine radial binning method
        Rvir = pygad.analysis.virial_info(snap)[0]
        SL.debug(f"Virial radius: {Rvir:.2f}")
        radii = args.radii * Rvir
        if args.method == "shell":
            radii = list(zip(radii[:-1], radii[1:]))

        # loop through each radii
        results = []
        # delay the data once so we don't send multiple copies
        # from https://docs.dask.org/en/stable/delayed-best-practices.html
        _snap = dask.delayed(snap)
        for r in radii:
            results.append(dask_helper(_snap, r))
        results = dask.compute(*results)
        for j, (r, res) in enumerate(zip(radii, results)):
            ratios["r"][i, j] = np.mean(r) if isinstance(r, tuple) else r
            ratios["ba"][i, j] = res[0]
            ratios["ca"][i, j] = res[1]

        # clean up
        snap.delete_blocks()
        pygad.gc_full_collect()
        del snap
        del _snap
        memory_helper()
    data = dict(times=times, ratios=ratios)
    figname = f"triax_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.savedir, exist_ok=True)
    bgs.utils.save_data(data, os.path.join(args.savedir, f"{figname}.pickle"))


fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
for axi in ax:
    axi.set_xlabel(r"$r/\mathrm{kpc}$")
ax[0].set_ylabel(r"$b/a$")
ax[1].set_ylabel(r"$c/a$")

# create a colour scale
cmapper, sm = bgs.plotting.create_normed_colours(min(times), max(times))

for i, t in enumerate(times):
    ax[0].semilogx(ratios["r"][i, :], ratios["ba"][i, :], c=cmapper(t))
    ax[1].semilogx(ratios["r"][i, :], ratios["ca"][i, :], c=cmapper(t))
ax[0].set_ylim(0, 1)
cbar = plt.colorbar(sm, ax=ax[-1], label=r"$t/\mathrm{Gyr}$")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "triaxiality", figname))
plt.show()
