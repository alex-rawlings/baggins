import argparse
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use

    use("Agg")
    import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import baggins as bgs
import pygad


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot IFU maps as a function of time for 420 km/s kick",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="orientation", choices=["para", "ortho"], help="orientation")
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
parser.add_argument(
    "-a", "--animate", action="store_true", dest="animate", help="make animation"
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


def make_data_file(n):
    return f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/kinematics_{args.kv:04d}_part_{n}.pickle"


snapdir = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
animation_path = (
    "/scratch/pjohanss/arawling/collisionless_merger/visualisations/recoil-explore"
)
save_data_interval = 80


def make_data_dict():
    return dict(
        time=[],
        rhalf=[],
        bound_stars_all=[],
        original_bound_stars=[],
        para=dict(ifu=[], bh_pos_x=[], bh_pos_y=[], dens=[]),
        ortho=dict(ifu=[], bh_pos_x=[], bh_pos_y=[], dens=[]),
    )


if args.extract:
    seeing = {"num": 25, "sigma": 0.3}
    extent = 7
    if args.kv <= 600:
        extent_multiplier = 4
    else:
        extent_multiplier = 8
    data = make_data_dict()

    snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
    first_snap_done = False
    for i, snapfile in enumerate(snapfiles):
        snap = pygad.Snapshot(snapfile, physical=True)
        if len(snap.bh) > 1:
            SL.warning(f"Two BHs present in snapshot {i} -> skipping")
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            continue
        # if i == 73:
        if i == 199:
            SL.warning("Final snapshot: breaking")
            break
        SL.info(f"Doing snapshot {i:03d}")
        snap_time = bgs.general.convert_gadget_time(snap)
        # we don't need all the snapshots as the BH is thermalising
        if snap_time > 1 and i % 4 != 0:
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            continue
        data["time"].append(snap_time)

        # move to CoM frame
        pre_ball_mask = pygad.BallMask(5)
        centre = pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30,
        )
        SL.debug(f"Centre is {centre}")
        vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
        pygad.Translation(-centre).apply(snap, total=True)
        pygad.Boost(-vcom).apply(snap, total=True)
        SL.debug(f"BH is now position: {snap.bh['pos']}")

        rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
        data["rhalf"].append(rhalf)
        if not first_snap_done:
            bound_stars_0 = bgs.analysis.find_individual_bound_particles(snap)
            data["bound_stars_all"].append(len(bound_stars_0))
            data["original_bound_stars"].append(len(bound_stars_0))
            first_snap_done = True
        else:
            bound_stars = bgs.analysis.find_individual_bound_particles(snap)
            data["bound_stars_all"].append(len(bound_stars))
            data["original_bound_stars"].append(
                len(set(bound_stars_0).intersection(set(bound_stars)))
            )
        n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

        for orientation, x_axis, LOS_axis in zip(("para", "ortho"), (1, 0), (0, 1)):
            ifu_mask = pygad.ExprMask(
                f"abs(pos[:,{x_axis}]) <= {extent}"
            ) & pygad.ExprMask(f"abs(pos[:,2]) <= {extent}")
            density_mask = pygad.ExprMask(
                f"abs(pos[:,{x_axis}]) <= {extent_multiplier*extent}"
            ) & pygad.ExprMask(f"abs(pos[:,2]) <= {extent_multiplier*extent}")

            data[orientation]["bh_pos_x"].append(snap.bh["pos"][:, x_axis])
            data[orientation]["bh_pos_y"].append(snap.bh["pos"][:, 2])

            voronoi_stats_all = bgs.analysis.voronoi_binned_los_V_statistics(
                x=snap.stars[ifu_mask]["pos"][:, x_axis],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
                m=snap.stars[ifu_mask]["mass"],
                Npx=n_regular_bins,
                part_per_bin=5000 * seeing["num"],
                seeing=seeing,
            )
            data[orientation]["ifu"].append(voronoi_stats_all)

            # plot the density and save
            _fig, _ax, _aximage, _cbar = pygad.plotting.image(
                snap.stars[density_mask],
                qty="mass",
                xaxis=x_axis,
                yaxis=2,
                scaleind="labels",
            )
            data[orientation]["dens"].append(_aximage)

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()

        if i % save_data_interval == 0:
            SL.warning("Dumping data")
            bgs.utils.save_data(
                data, make_data_file(i // save_data_interval - 1), exist_ok=True
            )
            data = make_data_dict()
else:
    data = bgs.utils.load_data(make_data_file(0))

N_frames = len(data["time"])

# set axis and colour limits
max_V = max([np.max(np.abs(v["img_V"])) for v in data[args.orientation]["ifu"]])
max_s = max([np.max(v["img_sigma"]) for v in data[args.orientation]["ifu"]])
min_s = min([np.min(v["img_sigma"]) for v in data[args.orientation]["ifu"]])
max_h3 = max([np.max(np.abs(v["img_h3"])) for v in data[args.orientation]["ifu"]])
max_h4 = max([np.max(np.abs(v["img_h4"])) for v in data[args.orientation]["ifu"]])
max_dens = max([np.nanmax(im.get_array()) for im in data[args.orientation]["dens"]])
min_dens = min([np.nanmin(im.get_array()) for im in data[args.orientation]["dens"]])


def make_plot_and_save_for_gif(i):
    fig, ax = plt.subplot_mosaic(
        """
    ABE
    CDF
    """,
        figsize=(8, 4),
    )

    fig.suptitle(f"{data['time'][i]-data['time'][0]:.3f} Gyr")

    # voronoi plots
    ifu_half_extent = 6
    bgs.plotting.voronoi_plot(
        data[args.orientation]["ifu"][i],
        clims={"V": [max_V], "sigma": [min_s, max_s], "h3": [max_h3], "h4": [max_h4]},
        ax=np.array([ax["A"], ax["B"], ax["C"], ax["D"]]),
    )
    for k in "ABCD":
        ax[k].scatter(
            data[args.orientation]["bh_pos_x"][i],
            data[args.orientation]["bh_pos_y"][i],
            lw=2,
            s=100,
            ec="green",
            fc="none",
        )
        ax[k].set_xlim(-ifu_half_extent, ifu_half_extent)
        ax[k].set_ylim(-ifu_half_extent, ifu_half_extent)
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")

    # density plots
    dens = data[args.orientation]["dens"][i]
    for k in "EF":
        ax[k].imshow(
            # np.flip(dens.get_array()),
            dens.get_array(),
            extent=dens.get_extent(),
            cmap=dens.get_cmap(),
            vmax=max_dens,
            vmin=min_dens,
            origin="lower",
        )
        # make an "aperture" rectangle to show IFU footprint
        ifu_rect = Rectangle(
            (-ifu_half_extent, -ifu_half_extent),
            2 * ifu_half_extent,
            2 * ifu_half_extent,
            fc="none",
            ec="k",
            fill=False,
        )
        ax[k].add_artist(ifu_rect)
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")
    ax["E"].scatter(
        data[args.orientation]["bh_pos_x"][i],
        data[args.orientation]["bh_pos_y"][i],
        lw=2,
        s=100,
        ec="red",
        fc="none",
    )
    # TODO add contours

    fig.tight_layout()
    os.makedirs(os.path.join(animation_path, args.orientation), exist_ok=True)
    bgs.plotting.savefig(
        os.path.join(animation_path, args.orientation, f"frame_ifu_{i:02d}.png"),
        fig=fig,
    )


if args.animate:
    # make the gif
    for i in range(N_frames):
        make_plot_and_save_for_gif(i)
    frames = bgs.utils.get_files_in_dir(
        os.path.join(animation_path, args.orientation), ".png"
    )

    frames = [Image.open(img) for img in frames if "frame_ifu_" in img]  # Load images
    frames.extend([frames[-1]] * 10)

    # Save as an animated GIF
    frames[0].save(
        os.path.join(
            animation_path,
            args.orientation,
            f"animation_{args.orientation}_{args.kv:04d}.gif",
        ),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
    )

    plt.close()
else:
    # make still panel of select times
    fig, ax = plt.subplot_mosaic(
        """
    AAAA
    BCDE
    FGHI
    """,
        figsize=(10, 6),
    )
    for k in "CDEFGHI":
        ax[k].sharex(ax["B"])
        ax[k].sharey(ax["B"])
    ifu_half_extent = 6
    for k in "BF":
        ax[k].set_ylabel("z/kpc")
    for k in "FGHI":
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
    specific_snaps = dict(ortho=[0, 2, 13, 45], para=[0, 2, 18, 35])
    for i, vax, sax in zip(specific_snaps[args.orientation], "BCDE", "FGHI"):
        bgs.plotting.voronoi_plot(
            data[args.orientation]["ifu"][i],
            clims={
                "V": [max_V],
                "sigma": [min_s, max_s],
                "h3": [max_h3],
                "h4": [max_h4],
            },
            ax=np.array([ax[vax], ax[sax]]),
        )
        for k in (vax, sax):
            ax[k].scatter(
                data[args.orientation]["bh_pos_x"][i],
                data[args.orientation]["bh_pos_y"][i],
                lw=1.5,
                s=100,
                ec="green",
                fc="none",
            )
    t = np.array(data["time"]) - data["time"][0]
    r = np.sqrt(
        np.array(data[args.orientation]["bh_pos_x"]) ** 2
        + np.array(data[args.orientation]["bh_pos_y"]) ** 2
    )
    ax["A"].plot(t, r, marker="o", markevery=specific_snaps[args.orientation])
    ax["A"].set_xlabel("t/Gyr")
    ax["A"].set_ylabel("r/kpc")
    fig.tight_layout()
    os.makedirs(os.path.join(animation_path, args.orientation), exist_ok=True)
    bgs.plotting.savefig(
        os.path.join(
            animation_path, args.orientation, f"IFU_panel_{args.orientation}.png"
        ),
        fig=fig,
    )
