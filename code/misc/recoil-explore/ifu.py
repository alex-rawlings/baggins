import argparse
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt
from PIL import Image
import baggins as bgs
import pygad


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot IFU maps as a function of time for 420 km/s kick", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(dest="orientation", choices=["para", "ortho"], help="orientation")
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
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

kv = 600
data_file = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/recoil-explore/kinematics_{kv:04d}.pickle"
snapdir = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv:04d}/output"

if args.extract:
    seeing = {"num": 25, "sigma": 0.3}
    extent = 5

    data = dict(
        time = [],
        rhalf = [],
        bound_stars_all = [],
        original_bound_stars = [],
        para = dict(
            ifu = [],
            bh_pos_x = [],
            bh_pos_y = [],
            dens = []
        ),
        ortho = dict(
            ifu = [],
            bh_pos_x = [],
            bh_pos_y = [],
            dens = []
        )
    )

    snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
    first_snap_done = False
    for i, snapfile in enumerate(snapfiles):
        snap = pygad.Snapshot(snapfile, physical=True)
        if len(snap.bh) > 1:
            SL.warning(f"Two BHs present in snapshot {i} -> skipping")
            continue
        if i == 73:
            SL.warning("Final snapshot: breaking")
            break
        SL.info(f"Doing snapshot {i:03d}")
        centre = pygad.analysis.shrinking_sphere(
            snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
        )
        data["time"].append(bgs.general.convert_gadget_time(snap))
        # move to CoM frame
        pygad.Translation(-centre).apply(snap, total=True)
        pre_ball_mask = pygad.BallMask(30)
        vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
        pygad.Boost(-vcom).apply(snap, total=True)

        rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
        data["rhalf"].append(rhalf)
        if not first_snap_done:
            bound_stars_0 = bgs.analysis.find_individual_bound_particles(snap)
            data["bound_stars_all"].append(len(bound_stars_0))
            data["original_bound_stars"].append(len(bound_stars_0))
        else:
            bound_stars = bgs.analysis.find_individual_bound_particles(snap)
            data["bound_stars_all"].append(len(bound_stars))
            data["original_bound_stars"].append(len(set(bound_stars_0).intersection(set(bound_stars))))
        n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

        for orientation, x_axis, LOS_axis in zip(("para", "ortho"), (1, 0), (0, 1)):
            ifu_mask = pygad.ExprMask(f"abs(pos[:,{x_axis}]) <= {extent}") & pygad.ExprMask(f"abs(pos[:,2]) <= {extent}")
            extent_multiplier = 4
            density_mask = pygad.ExprMask(f"abs(pos[:,{x_axis}]) <= {extent_multiplier*extent}") & pygad.ExprMask(f"abs(pos[:,2]) <= {extent_multiplier*extent}")

            # all stars
            voronoi_stats_all = bgs.analysis.voronoi_binned_los_V_statistics(
                x=snap.stars[ifu_mask]["pos"][:, x_axis],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
                m=snap.stars[ifu_mask]["mass"],
                Npx=n_regular_bins,
                part_per_bin=2000 * seeing["num"],
                seeing=seeing,
            )
            data[orientation]["ifu"].append(voronoi_stats_all)
            data[orientation]["bh_pos_x"].append(snap.bh["pos"][:,x_axis])
            data[orientation]["bh_pos_y"].append(snap.bh["pos"][:,2])

            # plot the density and save
            _fig, _ax, _aximage, _cbar = pygad.plotting.image(
                snap.stars[density_mask],
                qty="mass",
                xaxis=x_axis,
                yaxis=2
            )
            data[orientation]["dens"].append(_aximage)
    
        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
        if not first_snap_done:
            first_snap_done = True

    bgs.utils.save_data(data, data_file, exist_ok=True)
else:
    data = bgs.utils.load_data(data_file)


fig_path = "/scratch/pjohanss/arawling/collisionless_merger/visualisations/recoil-explore"
N_frames = len(data["time"])

# set axis and colour limits
max_V = max([np.max(np.abs(v["img_V"])) for v in data[args.orientation]["ifu"]])
max_s = max([np.max(v["img_sigma"]) for v in data[args.orientation]["ifu"]])
min_s = min([np.min(v["img_sigma"]) for v in data[args.orientation]["ifu"]])
max_h3 = max([np.max(np.abs(v["img_h3"])) for v in data[args.orientation]["ifu"]])
max_h4 = max([np.max(np.abs(v["img_h4"])) for v in data[args.orientation]["ifu"]])
max_dens = max([np.nanmax(im.get_array()) for im in data[args.orientation]["dens"]])
min_dens = min([np.nanmin(im.get_array()) for im in data[args.orientation]["dens"]])
max_bound_all = max(data["bound_stars_all"])
max_bound_original = max(data["original_bound_stars"])


def make_plot_and_save(i):
    fig, ax = plt.subplot_mosaic(
    """
    ABEG
    CDFH
    """,
    figsize=(8,4)
    )

    fig.suptitle(f"{data['time'][i]:.3f} Gyr")

    # voronoi plots
    bgs.plotting.voronoi_plot(
        data[args.orientation]["ifu"][i],
        clims={"V":[max_V], "sigma":[min_s, max_s], "h3":[max_h3], "h4":[max_h4]},
        ax = np.array([ax["A"], ax["B"], ax["C"], ax["D"]])
        )
    for k in "ABCD":
        ax[k].scatter(data[args.orientation]["bh_pos_x"][i], data[args.orientation]["bh_pos_y"][i], lw=2, s=100, ec="green", fc='none')
        ax[k].set_xlim(-3, 3)
        ax[k].set_ylim(-3, 3)
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")

    # density plots
    dens = data[args.orientation]["dens"][i]
    for k in "EF":
        ax[k].imshow(dens.get_array(), extent=dens.get_extent(), cmap=dens.get_cmap(), vmax=max_dens, vmin=min_dens, origin="lower")
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")
    ax["E"].scatter(data[args.orientation]["bh_pos_x"][i], data[args.orientation]["bh_pos_y"][i], lw=2, s=100, ec="red", fc='none')
    # TODO add contours

    # bound star plots
    ax["G"].plot(data["times"][:i], data["bound_stars_all"][:i], lw=2)
    ax["H"].plot(data["times"][:i], data["original_bound_stars"][:i], lw=2)
    for k in "GH":
        ax[k].set_xlabel("t/Gyr")
    ax["G"].set_ylabel("All bound stars")
    ax["G"].set_ylim(0, max_bound_all)
    ax["H"].set_ylabel("Bound stars that were also originally bound")
    ax["H"].set_ylim(0, max_bound_original)

    fig.tight_layout()
    os.makedirs(os.path.join(fig_path, args.orientation), exist_ok=True)
    bgs.plotting.savefig(os.path.join(fig_path, args.orientation, f"ifu_{i:02d}.png"), fig=fig)

for i in range(N_frames):
    make_plot_and_save(i)
frames = bgs.utils.get_files_in_dir(os.path.join(fig_path, args.orientation), ".png")

frames = [Image.open(img) for img in frames]     # Load images
frames.extend([frames[-1]] * 10)

# Save as an animated GIF
frames[0].save(os.path.join(fig_path, args.orientation, f"animation_{args.orientation}.gif"), format="GIF", append_images=frames[1:], 
               save_all=True, duration=200, loop=0)
