import argparse
import os.path
import dask
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Determine if a merger remnant has settled",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="Path to simulation output", dest="path", type=str)
parser.add_argument(
    "-t",
    "--threshold",
    help="velocity threshold",
    type=float,
    dest="threshold",
    default=10,
)
parser.add_argument(
    "-n",
    "--num",
    help="number of snapshots to analyse",
    type=int,
    dest="num",
    default=30,
)
parser.add_argument("-s", "--save", help="save data", action="store_true", dest="save")
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbosity",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

# set up a logger
SL = bgs.setup_logger("script", args.verbosity)


def _cleanup(sf):
    # helper function to keep memory manageable
    sf.delete_blocks()
    pygad.gc_full_collect()


# get a list of snapshots
snapfiles = bgs.utils.get_snapshots_in_dir(args.path)
LENGTH = len(snapfiles)


@dask.delayed
def _helper(_snap, i):
    SL.debug(f"Reading snapshot {_snap.filename}")
    t = bgs.general.convert_gadget_time(_snap)
    # recentre
    xcom = bgs.analysis.get_com_of_each_galaxy(_snap, method="ss", family="stars")
    vcom = bgs.analysis.get_com_velocity_of_each_galaxy(_snap, xcom)
    xtrans = pygad.Translation(-list(xcom.values())[0])
    vtrans = pygad.Boost(-list(vcom.values())[0])
    xtrans.apply(_snap, total=True)
    vtrans.apply(_snap, total=True)
    r = _snap.bh["r"]
    r_edges = [0.9 * r, 1.1 * r]
    rho = pygad.analysis.profile_dens(_snap.stars, "mass", r_edges=r_edges)
    # ensure density is always non-zero
    threshold = pygad.UnitQty(1e-12, units=rho.units)
    if rho[0] < threshold and r < np.nanquantile(_snap.stars["r"], 0.1):
        while rho[0] < threshold:
            r_edges = [0.9 * r_edges[0], 1.1 * r_edges[1]]
            rho = pygad.analysis.profile_dens(_snap.stars, "mass", r_edges=r_edges)
    ve = bgs.analysis.escape_velocity(_snap)(r)
    vm = pygad.utils.geo.dist(_snap.bh["vel"])
    _cleanup(_snap)
    return [i, t, vm[0], ve[0], r[0], rho[0][0]]


res = []
if args.num < 0:
    start_snap = 0
else:
    start_snap = max(LENGTH - args.num, 0)

# read the first snapshot to see if we can enable parallelism
for i, s in enumerate(snapfiles[start_snap:], start=start_snap):
    snap = pygad.Snapshot(s, physical=True)
    if bgs.analysis.determine_if_merged(snap)[0]:
        SL.info(f"First snapshot to analyse: {i:03d}")
        res.append(_helper(snap, i))
        start_i = i + 1
        break
    _cleanup(snap)
results = list(dask.compute(*res))
res = []


for i, s in enumerate(snapfiles[start_i:], start=start_i):
    snap = pygad.Snapshot(s, physical=True)
    SL.debug(f"Adding snapshot {i:03d} to dask queue...")
    res.append(_helper(snap, i))
SL.info(f"{len(res)+1} snapshots to analyse")
results.extend(dask.compute(*res))

results = np.array(results)
idx = np.argsort(results[:, 0])
results = results[idx, :]

idx_minus01 = np.argmax(results[-1, 1] - 0.1 < results[:, 1])
med_vel = np.median(results[idx_minus01:, 2])
has_settled = False
has_escaped = False

if np.any(results[:, 4] > 30):
    SL.warning("BH has escaped the system (r>30kpc)")
    has_escaped = True

if not has_escaped and results.shape[0] > 5 and med_vel < args.threshold:
    SL.warning("System has settled!")
    has_settled = True
    analyse_idx_r = 1
    max_iter = min(args.num - 1, LENGTH)
    while analyse_idx_r < max_iter:
        med_vel = np.median(results[idx_minus01 - analyse_idx_r : -analyse_idx_r, 2])
        SL.debug(
            f"Median velocity from {results[idx_minus01-analyse_idx_r,1]:.2f} to {results[-analyse_idx_r,1]:.2f} (snap {results[idx_minus01-analyse_idx_r,0]}-{results[-analyse_idx_r,0]}) is {med_vel:.2f} km/s"
        )
        if med_vel > args.threshold:
            SL.warning(f"Snapshot to analyse: {LENGTH-analyse_idx_r}")
            break
        analyse_idx_r += 1
    if analyse_idx_r == max_iter:
        SL.error(
            f"Max iterations {analyse_idx_r} have been reached, and no clear snapshot to analyse! Try increasing the number of snapshots analysed from {args.num}"
        )
else:
    SL.warning(
        f"System has not settled! Median velocity over the past 0.1 Gyr is {med_vel:.2f} km/s"
    )
    analyse_idx_r = np.nan


if args.save:
    bgs.utils.save_data(
        dict(
            data_dir=args.path,
            results=results,
            chosen_snap=LENGTH - analyse_idx_r,
            has_settled=has_settled,
            has_escaped=has_escaped,
        ),
        os.path.join(args.path, "../processed_data.pickle"),
    )

# plot
fig, ax = plt.subplots(2, 1, sharex="all")
# velocity of BH
ax_twin1 = bgs.plotting.twin_axes_from_samples(ax[0], results[:, 1], results[:, 0])
ax[0].set_xlabel(r"$t/\mathrm{Gyr}$")
ax[0].set_ylabel(r"$v/\mathrm{kms}^{-1}$")
ax_twin1.set_xlabel("Snap number")
ax[0].plot(results[:, 1], results[:, 2], marker="o", ls="", mec="k", mew=0.5)
if has_settled:
    ax[0].scatter(
        results[-analyse_idx_r, 1],
        results[-analyse_idx_r, 2],
        marker="s",
        s=90,
        c="tab:orange",
        ec="k",
        lw=0.5,
        label="Chosen",
    )
ax[0].axhline(args.threshold, c="tab:red", lw=2, label="Threshold")
ax[0].legend(fontsize="small")
if results[0, 2] > 50:
    ax[0].set_yscale("log")

# position of BH, coloured by the stellar density in a shell about its position
ax_twin2 = bgs.plotting.twin_axes_from_samples(ax[1], results[:, 1], results[:, 0])
ax[1].set_xlabel(r"$t/\mathrm{Gyr}$")
ax[1].set_ylabel(r"$r/\mathrm{kpc}$")
ax_twin2.set_xlabel("Snap number")
cmap, sm = bgs.plotting.create_normed_colours(
    10 ** np.floor(np.log10(min(results[:, 5]))),
    10 ** np.ceil(np.log10(min(results[:, 5]))),
    norm="LogNorm",
)
ax[1].scatter(results[:, 1], results[:, 4], c=cmap(results[:, 5]), ec="k", lw=0.5)
cbar = plt.colorbar(
    sm, ax=ax[1], label=r"$\rho_\star / \mathrm{M}_\odot/\mathrm{kpc}^{-3}$"
)
if max(results[:, 4]) > 5e-2:
    ax[1].set_yscale("log")
plt.show()
