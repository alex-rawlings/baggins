import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os.path
import pygad
import baggins as bgs


# set up the command line options
parser = argparse.ArgumentParser(
    description="Compare the orbit of the stellar centre of mass for different runs",
    allow_abbrev=False,
)
parser.add_argument(type=str, help="base path to parent directory", dest="pathbase")
parser.add_argument(
    "-sd", "--subdir", type=str, help="name of subdirectory", dest="subdir", default=""
)
parser.add_argument(
    "-ld",
    "--loopdir",
    type=str,
    help="directories to loop over",
    dest="loopdir",
    action="extend",
    nargs="+",
)
parser.add_argument(
    "-ldf",
    "--loopdirfirst",
    help="loop directory before subdirectory?",
    dest="loopdirfirst",
    action="store_true",
)
parser.add_argument(
    "-od",
    "--outputdir",
    type=str,
    help="directory containing snapshots",
    dest="outputdir",
    default="output",
)
parser.add_argument(
    "-fd",
    "--figdir",
    type=str,
    help="directory for figures",
    dest="figdir",
    default=bgs.FIGDIR,
)
parser.add_argument(
    "-rc",
    "--relcomp",
    help="compare relative difference in orbits?",
    dest="relcomp",
    action="store_true",
)
parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="verbose printing in script",
)
args = parser.parse_args()


time_dict = {}
xcom_dict = {}


for ind1, ld in enumerate(args.loopdir):
    if args.loopdirfirst:
        snap_location = os.path.join(args.pathbase, ld, args.subdir, args.outputdir)
    else:
        snap_location = os.path.join(args.pathbase, args.subdir, ld, args.outputdir)
    snap_files = bgs.utils.get_snapshots_in_dir(snap_location)

    for ind2, snapfile in enumerate(snap_files):
        print("Reading: {}".format(snapfile))
        snap = pygad.Snapshot(snapfile)
        snap.to_physical_units()

        if ind2 == 0:
            if args.verbose:
                print("Creating ID masks...")
            id_masks = dict(
                stars=bgs.analysis.get_all_id_masks(snap),
                dm=bgs.analysis.get_all_id_masks(snap, family="dm"),
            )
            time_dict[ld] = np.full_like(snap_files, np.nan, dtype=float)
            xcom_dict[ld] = {}
            for bhid in snap.bh["ID"]:
                xcom_dict[ld][bhid] = np.full((len(snap_files), 3), np.nan)
        time_dict[ld][ind2] = bgs.general.convert_gadget_time(snap)
        this_xcom = bgs.analysis.get_com_of_each_galaxy(
            snap, method="ss", masks=id_masks["stars"], verbose=args.verbose
        )
        for k in this_xcom.keys():
            xcom_dict[ld][k][ind2, :] = this_xcom[k]

# set up interpolation dicts
base_interp = dict()
base_bhid_keys = list(xcom_dict[args.loopdir[0]].keys())
if xcom_dict[args.loopdir[0]][base_bhid_keys[0]][0, 2] < 0:
    # the BH with the smaller ID is in the lower corner
    place_in_system = ("lower", "upper")
else:
    # the BH with the smaller ID is in the upper corner
    place_in_system = ("upper", "lower")

for inds, place in enumerate(place_in_system):
    base_interp[place] = dict(
        x=scipy.interpolate.interp1d(
            time_dict[args.loopdir[0]],
            xcom_dict[args.loopdir[0]][base_bhid_keys[inds]][:, 0],
        ),
        z=scipy.interpolate.interp1d(
            time_dict[args.loopdir[0]],
            xcom_dict[args.loopdir[0]][base_bhid_keys[inds]][:, 2],
        ),
    )

min_common_time = max([np.min(v) for v in time_dict.values()])
max_common_time = min([np.max(v) for v in time_dict.values()])
timestep = 0.01
common_time = np.arange(min_common_time, max_common_time, timestep)
divisions = 0
while len(common_time) < 2 and divisions < 10:
    timestep /= 2
    common_time = np.arange(min_common_time, max_common_time, timestep)
    divisions += 1
if divisions == 10:
    raise ValueError("Timestep too large!")

if args.relcomp:
    fig = plt.figure(constrained_layout=True, figsize=(5, 6))
    gridspec = fig.add_gridspec(ncols=2, nrows=5)
    fig.add_subplot(gridspec[:-2, :])
    fig.add_subplot(gridspec[-2:, 0])
    fig.add_subplot(gridspec[-2:, 1])
    ax = fig.axes
else:
    fig, ax = plt.subplots(1, 1)
    ax = [ax]
for xk in xcom_dict.keys():
    # plot the orbit
    for indp, k in enumerate(xcom_dict[xk].keys()):
        this_gal_interp = dict(
            x=scipy.interpolate.interp1d(time_dict[xk], xcom_dict[xk][k][:, 0]),
            z=scipy.interpolate.interp1d(time_dict[xk], xcom_dict[xk][k][:, 2]),
        )
        if indp == 0:
            (m,) = ax[0].plot(
                this_gal_interp["x"](common_time),
                this_gal_interp["z"](common_time),
                marker="o",
                markevery=[-1],
                label=xk,
                alpha=0.7,
            )
        else:
            ax[0].plot(
                this_gal_interp["x"](common_time),
                this_gal_interp["z"](common_time),
                marker="s",
                markevery=[-1],
                c=m.get_color(),
                alpha=0.7,
            )
        extra_time = np.arange(max_common_time, np.max(time_dict[xk]), timestep)
        try:
            ax[0].plot(
                this_gal_interp["x"](extra_time),
                this_gal_interp["z"](extra_time),
                ls=":",
                c=m.get_color(),
                alpha=0.7,
            )
        except ValueError:
            print(
                "Interpolation error for plotting non-common time for file {}, BH {}. Skipping...".format(
                    xk, indp
                )
            )

        time_min = max([np.min(time_dict[args.loopdir[0]]), np.min(time_dict[xk])])
        time_max = min([np.max(time_dict[args.loopdir[0]]), np.max(time_dict[xk])])

        if args.relcomp:
            interp_time = np.arange(time_min, time_max, timestep)
            deviations = dict()
            if xcom_dict[xk][k][0, 2] < 0:
                for axis in ("x", "z"):
                    deviations[axis] = this_gal_interp[axis](interp_time) - base_interp[
                        "lower"
                    ][axis](interp_time)
            else:
                for axis in ("x", "z"):
                    deviations[axis] = this_gal_interp[axis](interp_time) - base_interp[
                        "upper"
                    ][axis](interp_time)
            for inda, axis in enumerate(("x", "z"), start=1):
                ax[inda].plot(
                    interp_time,
                    deviations[axis],
                    marker=("o" if indp == 0 else "s"),
                    c=m.get_color(),
                    alpha=0.7,
                    markevery=[-1],
                )
                if xk != args.loopdir[0]:
                    max_dev_idx = np.argmax(np.abs(deviations[axis]))
                    xpoint = this_gal_interp["x"](interp_time[max_dev_idx])
                    zpoint = this_gal_interp["z"](interp_time[max_dev_idx])
                    ax[0].scatter(
                        xpoint, zpoint, marker="*", c=m.get_color(), zorder=20
                    )
                    ax[0].text(
                        xpoint,
                        zpoint,
                        axis,
                        fontdict={"color": m.get_color()},
                        horizontalalignment="right",
                        verticalalignment="bottom",
                    )

ax[0].legend()
ax[0].set_xlabel("x/kpc")
ax[0].set_ylabel("z/kpc")
if args.relcomp:
    ax[1].set_xlabel("t/Gyr")
    ax[1].set_ylabel(r"x$_i$ - x$_\mathrm{0.001}$ [kpc]")
    ax[2].set_xlabel("t/Gyr")
    ax[2].set_ylabel(r"z$_i$ - z$_\mathrm{0.001}$ [kpc]")
figpath = os.path.join(args.figdir, "orbit-compare-{}.png".format(args.subdir))
plt.savefig(figpath, dpi=300)
plt.show()
